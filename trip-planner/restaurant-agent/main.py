import json
import os
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

    @kernel_function(description="Returns restaurants filtered by city and cuisine.")
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
    return Response(status_code=404, content="Agent card not found.")


@app.post("/task")
async def suggest_restaurant(request: TaskRequest):
    city = request.params.city
    cuisine_type = request.params.cuisine_type

    if RestaurantSearchPlugin._resolve_city_key(city) is None:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": "City not supported"},
            "id": request.id,
        }

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
            response_format={"type": "json_object"},
            function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
        )

        result = await kernel.invoke(
            chat_function,
            KernelArguments(settings=execution_settings, city=city, cuisine=cuisine_type),
        )
        result_str = str(result)
    except Exception as e:
        print(f"CRITICAL KERNEL ERROR: {type(e).__name__}: {str(e)}")
        if hasattr(e, "inner_exception"):
            print(f"INNER ERROR: {e.inner_exception}")

        return {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": f"Agent failed: {str(e)}"},
            "id": request.id,
        }

    try:
        if "```json" in result_str:
            result_str = result_str.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in result_str:
            result_str = result_str.split("```", 1)[1].split("```", 1)[0].strip()

        selected = json.loads(result_str)
        validated_response = RestaurantResponse.model_validate(selected)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Malformed response: {result_str}")

        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": "Internal error: Failed to decode JSON from AI response.",
            },
            "id": request.id,
        }
    except ValidationError as e:
        print(f"ValidationError: {e}")
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": "Internal error: Invalid response schema from AI output.",
            },
            "id": request.id,
        }

    return {
        "jsonrpc": "2.0",
        "result": validated_response.model_dump(mode="json"),
        "id": request.id,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
