import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from anyio import Path as AnyioPath
from fastapi import FastAPI, Response
from pydantic import BaseModel, ConfigDict, ValidationError
import uvicorn
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from dotenv import load_dotenv
from semantic_kernel.functions import KernelArguments, kernel_function

from memory import ACTIVITIES_DB

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Pydantic Models ---
class TaskRequestParams(BaseModel):
    city: str
    weather: str

class TaskRequest(BaseModel):
    jsonrpc: str
    method: str
    params: TaskRequestParams
    id: Optional[int] = None


class ActivityItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    description: str


class ActivityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activities: list[ActivityItem]
    note: str | None = None


def _normalize_schema_for_structured_outputs(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                # Azure/OpenAI strict schema expects every property listed as required.
                node["required"] = list(properties.keys())
                node.setdefault("additionalProperties", False)

            for key in ("$defs", "definitions", "properties", "patternProperties", "dependentSchemas"):
                child_map = node.get(key)
                if isinstance(child_map, dict):
                    for child in child_map.values():
                        walk(child)

            for key in ("items", "contains", "if", "then", "else", "not", "propertyNames"):
                child = node.get(key)
                if isinstance(child, dict):
                    walk(child)

            for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
                child_list = node.get(key)
                if isinstance(child_list, list):
                    for child in child_list:
                        walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(normalized)
    return normalized


def build_response_format_from_model(model: type[BaseModel], schema_name: str) -> dict[str, Any]:
    normalized_schema = _normalize_schema_for_structured_outputs(model.model_json_schema())
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": normalized_schema,
        },
    }


def is_schema_response_format_unsupported(exc: Exception) -> bool:
    def collect_messages(error: BaseException, seen: set[int]) -> list[str]:
        if id(error) in seen:
            return []
        seen.add(id(error))

        messages = [str(error), repr(error)]
        inner = getattr(error, "inner_exception", None)
        if isinstance(inner, BaseException):
            messages.extend(collect_messages(inner, seen))
        cause = getattr(error, "__cause__", None)
        if isinstance(cause, BaseException):
            messages.extend(collect_messages(cause, seen))
        context = getattr(error, "__context__", None)
        if isinstance(context, BaseException):
            messages.extend(collect_messages(context, seen))
        return messages

    error_text = " ".join(collect_messages(exc, set())).lower()
    markers = (
        "response_format",
        "json_schema",
        "schema",
        "unsupported",
        "not supported",
        "invalid",
        "bad request",
        "required",
    )
    return any(marker in error_text for marker in markers)


def rpc_error_response(
    rpc_id: int | None,
    code: int,
    message: str,
    data: dict[str, Any] | list[Any] | str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        payload["data"] = data
    return {"jsonrpc": "2.0", "error": payload, "id": rpc_id}

# --- File Paths ---
BASE_DIR = Path(__file__).parent
AGENT_CARD_PATH = BASE_DIR / "agent_card.json"
INSTRUCTIONS_PATH = BASE_DIR / "instructions.md"
AGENT_CARD_CONTENT: dict = {}
SYSTEM_PROMPT: str = ""


class ActivitySearchPlugin:
    @staticmethod
    def _normalize(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def _resolve_city_key(city: str) -> Optional[str]:
        target = city.strip().lower()
        for key in ACTIVITIES_DB:
            if key.lower() == target:
                return key
        return None

    @kernel_function(description="Restituisce attivita filtrate per citta e meteo.")
    async def get_activities(self, city: str, weather: str) -> list[dict]:
        city_key = self._resolve_city_key(city)
        if not city_key:
            return []

        normalized_weather = self._normalize(weather)
        activities = ACTIVITIES_DB.get(city_key, [])
        neutral_weather_values = {"unknown", "sconosciuto", "n/a", "na", "none"}

        bad_weather_markers = {"pioggia", "temporale", "neve", "tuono", "vento", "grandine"}
        good_weather_markers = {"sole", "sereno", "caldo"}

        if normalized_weather in neutral_weather_values:
            return activities

        if any(marker in normalized_weather for marker in bad_weather_markers):
            return [item for item in activities if item.get("type", "").lower() == "al chiuso"]

        if any(marker in normalized_weather for marker in good_weather_markers):
            return [item for item in activities if item.get("type", "").lower() == "all'aperto"]

        return activities

# --- Load Environment Variables ---
load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env", override=True)

# --- Endpoints ---
@app.on_event("startup")
async def startup_event() -> None:
    global AGENT_CARD_CONTENT, SYSTEM_PROMPT

    card_raw = await AnyioPath(AGENT_CARD_PATH).read_text(encoding="utf-8")
    instructions_raw = await AnyioPath(INSTRUCTIONS_PATH).read_text(encoding="utf-8")

    AGENT_CARD_CONTENT = json.loads(card_raw)
    SYSTEM_PROMPT = instructions_raw


@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    """
    Serves the agent's A2A card for discovery.
    """
    if AGENT_CARD_CONTENT:
        return Response(content=json.dumps(AGENT_CARD_CONTENT, indent=4), media_type="application/json")
    return Response(status_code=404, content="Scheda agente non trovata.")

@app.post("/task")
async def suggest_activity(request: TaskRequest):
    """
    Suggests activities based on city and weather.
    """
    city = request.params.city
    weather = request.params.weather

    # 1. Init Kernel
    kernel = Kernel()

    # 2. Add AI Service (Verified Stable v1.x Syntax)
    ai_service = AzureChatCompletion(
        deployment_name= os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key= os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION")
    )
    # Register the service directly. The kernel handles it as the default.
    kernel.add_service(ai_service)
    kernel.add_plugin(ActivitySearchPlugin(), plugin_name="ActivitySearch")

    # --- Error Handling for Unsupported City ---
    if ActivitySearchPlugin._resolve_city_key(city) is None:
        return rpc_error_response(request.id, -32602, "Citta non supportata")

    # 3. Register Function from prompt
    chat_function = kernel.add_function(
        function_name="chat",
        plugin_name="ActivityPlugin",
        prompt=SYSTEM_PROMPT,
    )

    try:
        execution_settings = AzureChatPromptExecutionSettings(
            tool_choice="auto",
            parallel_tool_calls=False,
            response_format=build_response_format_from_model(ActivityResponse, "activity_response"),
            function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
        )

        try:
            result = await kernel.invoke(
                chat_function,
                KernelArguments(settings=execution_settings, city=city, weather=weather),
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
                KernelArguments(settings=fallback_settings, city=city, weather=weather),
            )

        result_str = str(result)
    except Exception as e:
        # LOG THE FULL ERROR TO CONSOLE
        print(f"CRITICAL KERNEL ERROR: {type(e).__name__}: {str(e)}")
        # If it's a wrapper, try to get the inner message
        if hasattr(e, 'inner_exception'):
            print(f"INNER ERROR: {e.inner_exception}")

        return rpc_error_response(request.id, -32603, f"Agente non disponibile: {str(e)}")

    # --- Format and Return Response ---
    try:
        suggested_activities = json.loads(result_str)
        validated_response = ActivityResponse.model_validate(suggested_activities)

    except json.JSONDecodeError as e:
        # Log the error for debugging
        print(f"JSONDecodeError: {e}")
        print(f"Malformed response: {str(result)}")

        return rpc_error_response(
            request.id,
            -32603,
            "Errore interno: impossibile decodificare il JSON dalla risposta AI.",
            data={"raw_response": result_str},
        )
    except ValidationError as e:
        print(f"ValidationError: {e}")
        return rpc_error_response(
            request.id,
            -32603,
            "Errore interno: schema di risposta non valido nell'output AI.",
            data={"validation_errors": e.errors(include_url=False)},
        )

    return {
        "jsonrpc": "2.0",
        "result": validated_response.model_dump(mode="json"),
        "id": request.id
    }

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
