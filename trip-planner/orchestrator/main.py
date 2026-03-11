import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from helpers import get_structured_output_settings

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


def build_weather_mcp_plugin() -> MCPStdioPlugin:
    project_root = Path(__file__).resolve().parents[2]
    app_root = Path(__file__).resolve().parents[1]
    weather_server_script = project_root / "mcp-weather-server" / "server.py"
    python_executable = app_root / ".venv" / "Scripts" / "python.exe"

    if not python_executable.exists():
        raise FileNotFoundError(f"Interprete del virtual environment non trovato in: {python_executable}")
    if not weather_server_script.exists():
        raise FileNotFoundError(f"Server meteo MCP non trovato in: {weather_server_script}")

    return MCPStdioPlugin(
        name="WeatherMcp",
        command=str(python_executable),
        args=[str(weather_server_script)],
        request_timeout=30,
    )


def build_orchestrator_agent(kernel: Kernel) -> ChatCompletionAgent:
    return ChatCompletionAgent(
        name="TripOrchestrator",
        instructions=build_system_instructions(),
        kernel=kernel,
        function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
    )


def build_execution_settings() -> AzureChatPromptExecutionSettings:
    return AzureChatPromptExecutionSettings(
        tool_choice="auto",
        parallel_tool_calls=False,
        response_format=get_structured_output_settings(TripDirectorResponse),
    )
