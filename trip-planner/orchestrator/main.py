import asyncio
import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, ValidationError
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments

try:
    from orchestrator.plugins.discovery_plugin import DiscoveryPlugin
except ImportError:
    from plugins.discovery_plugin import DiscoveryPlugin

LOGGER = logging.getLogger("trip_orchestrator")
SYSTEM_INSTRUCTIONS = (
    "Sei il Direttore del Viaggio. "
    "Devi rispondere sempre in italiano. "
    "1. Ottieni il meteo. "
    "2. Ottieni attivita. "
    "3. Ottieni suggerimenti ristorante. "
    "Devi tentare di ottenere i dati meteo dal tool WeatherMcp prima di chiamare ActivityAgent. "
    "Per il parametro weather di ActivityAgent usa solo e soltanto l'output esatto del tool meteo, con le descrizioni meteo in italiano fornite dal tool stesso. "
    "Se WeatherMcp restituisce un errore di limite previsione o qualsiasi altro errore, non fermarti. "
    "Prosegui chiamando ActivityAgent con weather='Sconosciuto' in una richiesta JSON-RPC valida e chiama normalmente RestaurantAgent. "
    "Nella risposta finale usa rigorosamente il contratto TripDirectorResponse. "
    "4. Mappa i risultati nei campi obbligatori: weather_data, activity_suggestions e restaurant_recommendations. "
    "activity_suggestions deve contenere esattamente il payload restituito da ActivityAgent: "
    "oppure {activities, note opzionale} oppure {error}. "
    "restaurant_recommendations deve contenere esattamente il payload restituito da RestaurantAgent: "
    "oppure {restaurants, note opzionale} oppure {error}. "
    "Se un agente/tool restituisce un oggetto error, preservalo nel campo corrispondente senza trasformarlo. "
    "I dati meteo possono essere 'Sconosciuto'; fornisci comunque attivita e ristoranti e aggiungi una nota esplicita."
)


class RpcError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: int | str
    message: str
    data: dict[str, Any] | list[Any] | str | None = None


class AgentErrorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: RpcError


class ActivityItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    description: str


class ActivityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activities: list[ActivityItem]
    note: str | None = None


class RestaurantItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    price_range: str


class RestaurantResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    restaurants: list[RestaurantItem]
    note: str | None = None


class TripDirectorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weather_data: str | dict[str, Any] | list[Any]
    activity_suggestions: ActivityResponse | AgentErrorPayload
    restaurant_recommendations: RestaurantResponse | AgentErrorPayload
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


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_environment() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    LOGGER.info("Loaded environment", extra={"env_path": str(env_path)})


def build_kernel(travel_services_plugin: DiscoveryPlugin) -> Kernel:
    kernel = Kernel()

    chat_service = AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION"),
    )
    kernel.add_service(chat_service)

    kernel.add_plugin(travel_services_plugin, plugin_name="TravelServices")

    return kernel


def build_system_instructions() -> str:
    now = datetime.now()
    return (
        f"{SYSTEM_INSTRUCTIONS} "
        f"La data di oggi e {now.strftime('%Y-%m-%d')}. "
        f"L'ora locale corrente e {now.strftime('%H:%M:%S')}."
    )


def present_itinerary(raw_json: str) -> None:
    try:
        parsed = TripDirectorResponse.model_validate_json(raw_json)
    except ValidationError as exc:
        try:
            fallback = json.loads(raw_json)
        except json.JSONDecodeError:
            print("----- RAW AGENT OUTPUT -----")
            print(raw_json)
            return

        print("----- ERRORE VALIDAZIONE SCHEMA ORCHESTRATOR -----")
        print(str(exc))
        print(json.dumps(fallback, indent=2, ensure_ascii=True))
        return
    weather_data = parsed.weather_data
    activities_data = parsed.activity_suggestions
    restaurants_data = parsed.restaurant_recommendations
    note = parsed.note

    activities: list[ActivityItem] = []
    restaurants: list[RestaurantItem] = []
    weather_text = weather_data if isinstance(weather_data, str) else json.dumps(weather_data, ensure_ascii=True)

    print("--- IL TUO VIAGGIO ---")
    print(f"Meteo attuale: {weather_text}")

    if isinstance(activities_data, ActivityResponse):
        note = activities_data.note or note
        activities = activities_data.activities
    if isinstance(restaurants_data, RestaurantResponse):
        note = restaurants_data.note or note
        restaurants = restaurants_data.restaurants

    if note is not None:
        print(f"Nota: {note}")

    if isinstance(activities_data, AgentErrorPayload):
        print("Attivita consigliate:")
        print(f"- ERRORE servizio attivita: {activities_data.error.message} ({activities_data.error.code})")
    else:
        print("Attivita consigliate:")
        if not activities:
            print("- Nessuna attivita disponibile.")
        else:
            for item in activities:
                print(f"- {item.name} [{item.type}] - {item.description}")

    if isinstance(restaurants_data, AgentErrorPayload):
        print("Ristoranti consigliati:")
        print(f"- ERRORE servizio ristoranti: {restaurants_data.error.message} ({restaurants_data.error.code})")
        return

    print("Ristoranti consigliati:")
    if not restaurants:
        print("- Nessun suggerimento ristorante disponibile.")
        return

    for item in restaurants:
        print(f"- {item.name} [{item.type}] - Prezzo: {item.price_range}")


async def run_console() -> None:
    configure_logging()
    load_environment()

    travel_services_plugin = DiscoveryPlugin()
    kernel = build_kernel(travel_services_plugin)

    weather_server_script = Path(__file__).resolve().parents[2] / "mcp-weather-server" / "server.py"
    python_executable = str((Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe").resolve())

    mcp_plugin = MCPStdioPlugin(
        name="WeatherMcp",
        command=python_executable,
        args=[str(weather_server_script)],
        request_timeout=30,
    )

    try:
        await mcp_plugin.connect()
        await mcp_plugin.load_tools()
        kernel.add_plugin(mcp_plugin)
    except Exception:
        LOGGER.exception("Failed to initialize MCP weather plugin")
        print(
            "Impossibile avviare lo strumento meteo (sottoprocesso MCP non riuscito). "
            "Controlla mcp-weather-server e riprova."
        )
        return

    try:
        runtime_system_instructions = build_system_instructions()
        agent = ChatCompletionAgent(
            name="TripOrchestrator",
            instructions=runtime_system_instructions,
            kernel=kernel,
            function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
        )

        print("Trip Orchestrator pronto. Inserisci la tua richiesta (oppure 'exit').")
        while True:
            user_input = input("\nTu> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Arrivederci.")
                return
            if not user_input:
                continue

            execution_settings = AzureChatPromptExecutionSettings(
                tool_choice="auto",
                parallel_tool_calls=False,
                response_format=build_response_format_from_model(TripDirectorResponse, "trip_director_response"),
            )

            try:
                response = await agent.get_response(
                    messages=user_input,
                    arguments=KernelArguments(settings=execution_settings),
                )
            except Exception as schema_exc:
                if not is_schema_response_format_unsupported(schema_exc):
                    raise

                LOGGER.warning(
                    "json_schema response_format non supportato dal deployment, fallback a validazione post-invoke: %s",
                    schema_exc,
                )
                fallback_settings = AzureChatPromptExecutionSettings(
                    tool_choice="auto",
                    parallel_tool_calls=False,
                )
                response = await agent.get_response(
                    messages=user_input,
                    arguments=KernelArguments(settings=fallback_settings),
                )

            content = str(response.message.content)
            print("\n----- RAW JSON RESPONSE -----")
            print(content)
            print()
            present_itinerary(content)
    finally:
        try:
            await mcp_plugin.close()
        except Exception:
            LOGGER.warning("MCP weather plugin closed with errors.")
        try:
            await travel_services_plugin.close()
        except Exception:
            LOGGER.warning("Travel services plugin closed with errors.")


if __name__ == "__main__":
    asyncio.run(run_console())
