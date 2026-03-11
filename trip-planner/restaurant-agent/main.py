import json
import os
import sys
from pathlib import Path
from typing import Optional

from anyio import Path as AnyioPath
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from pydantic import BaseModel, ConfigDict, ValidationError
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments, kernel_function
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers import create_rpc_error, get_structured_output_settings, is_schema_response_format_unsupported
from memory import RESTAURANT_DB

app = FastAPI()


class TaskRequestParams(BaseModel):
    city: str
    cuisine_type: str


class TaskRequest(BaseModel):
    jsonrpc: str
    method: str
    params: TaskRequestParams
    id: Optional[int] = None


class RestaurantItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    price_range: str


class RestaurantResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    restaurants: list[RestaurantItem]
    note: str | None = None


BASE_DIR = Path(__file__).parent
AGENT_CARD_PATH = BASE_DIR / "agent_card.json"
INSTRUCTIONS_PATH = BASE_DIR / "instructions.md"
AGENT_CARD_CONTENT: dict = {}
SYSTEM_INSTRUCTIONS: str = ""

load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env", override=True)


class RestaurantSearchPlugin:
    @staticmethod
    def _normalize(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def _resolve_city_key(city: str) -> Optional[str]:
        target = city.strip().lower()
        for key in RESTAURANT_DB:
            if key.lower() == target:
                return key
        return None

    @kernel_function(description="Restituisce ristoranti filtrati per citta e tipo di cucina.")
    async def get_restaurants(self, city: str, cuisine: str) -> list[dict]:
        city_key = self._resolve_city_key(city)
        if not city_key:
            return []

        restaurants = RESTAURANT_DB.get(city_key, [])
        normalized_cuisine = self._normalize(cuisine)
        if normalized_cuisine in {"", "any", "all"}:
            return restaurants

        return [
            item
            for item in restaurants
            if normalized_cuisine in self._normalize(item.get("cuisine_type", ""))
            or normalized_cuisine in self._normalize(item.get("type", ""))
        ]


@app.on_event("startup")
async def startup_event() -> None:
    global AGENT_CARD_CONTENT, SYSTEM_INSTRUCTIONS

    card_raw = await AnyioPath(AGENT_CARD_PATH).read_text(encoding="utf-8")
    instructions_raw = await AnyioPath(INSTRUCTIONS_PATH).read_text(encoding="utf-8")

    AGENT_CARD_CONTENT = json.loads(card_raw)
    SYSTEM_INSTRUCTIONS = instructions_raw


@app.get("/.well-known/agent-card.json")
async def get_agent_card() -> Response:
    if AGENT_CARD_CONTENT:
        return Response(content=json.dumps(AGENT_CARD_CONTENT, indent=4), media_type="application/json")
    return Response(status_code=404, content="Scheda agente non trovata.")


@app.post("/task")
async def suggest_restaurant(request: TaskRequest):
    city = request.params.city
    cuisine_type = request.params.cuisine_type

    if RestaurantSearchPlugin._resolve_city_key(city) is None:
        return create_rpc_error(-32602, "Citta non supportata", request.id)

    kernel = Kernel()

    ai_service = AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION"),
    )
    kernel.add_service(ai_service)
    kernel.add_plugin(RestaurantSearchPlugin(), plugin_name="RestaurantSearch")

    chat_function = kernel.add_function(
        function_name="chat",
        plugin_name="RestaurantAgent",
        prompt=SYSTEM_INSTRUCTIONS,
    )

    try:
        execution_settings = AzureChatPromptExecutionSettings(
            tool_choice="auto",
            parallel_tool_calls=False,
            response_format=get_structured_output_settings(RestaurantResponse),
            function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
        )

        try:
            result = await kernel.invoke(
                chat_function,
                KernelArguments(settings=execution_settings, city=city, cuisine=cuisine_type),
            )
        except Exception as schema_exc:
            if not is_schema_response_format_unsupported(schema_exc):
                raise

            print(f"WARNING: json_schema response_format non supportato dal deployment: {schema_exc}")
            fallback_settings = AzureChatPromptExecutionSettings(
                tool_choice="auto",
                parallel_tool_calls=False,
                function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
            )
            result = await kernel.invoke(
                chat_function,
                KernelArguments(settings=fallback_settings, city=city, cuisine=cuisine_type),
            )

        result_str = str(result)
    except Exception as e:
        print(f"CRITICAL KERNEL ERROR: {type(e).__name__}: {str(e)}")
        if hasattr(e, "inner_exception"):
            print(f"INNER ERROR: {e.inner_exception}")

        return create_rpc_error(-32603, f"Agente non disponibile: {str(e)}", request.id)

    try:
        selected = json.loads(result_str)
        validated_response = RestaurantResponse.model_validate(selected)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Malformed response: {result_str}")

        error_payload = create_rpc_error(
            -32603,
            "Errore interno: impossibile decodificare il JSON dalla risposta AI.",
            request.id,
        )
        error_payload["error"]["data"] = {"raw_response": result_str}
        return error_payload
    except ValidationError as e:
        print(f"ValidationError: {e}")
        error_payload = create_rpc_error(
            -32603,
            "Errore interno: schema di risposta non valido nell'output AI.",
            request.id,
        )
        error_payload["error"]["data"] = {"validation_errors": e.errors(include_url=False)}
        return error_payload

    return {
        "jsonrpc": "2.0",
        "result": validated_response.model_dump(mode="json"),
        "id": request.id,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
