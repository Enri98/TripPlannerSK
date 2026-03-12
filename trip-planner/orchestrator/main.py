import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.connectors.mcp import MCPStdioPlugin

from helpers import get_structured_output_settings

LOGGER = logging.getLogger("trip_orchestrator")

PLANNER_SYSTEM_PROMPT = (
    "Sei l'analista. Chiama il tool meteo per la citta. "
    "Genera SOLO un JSON interno (non all'utente) con: city, weather, cuisine (opzionale), budget (opzionale). "
    "Estrai city, cuisine e budget dalla richiesta utente quando presenti. "
    "Chiama sempre il tool meteo per la city estratta. "
    "Se il meteo non e disponibile o il tool fallisce, imposta weather a 'Sconosciuto'. "
    "Rispondi solo con il JSON richiesto."
)

SYNTHESIZER_SYSTEM_PROMPT = (
    "Sei il Direttore di Viaggio. Ricevi il meteo, le risposte dell'agente attivita e dell'agente ristoranti. "
    "Unisci tutto in un discorso fluido, formattato in Markdown, coerente e diretto all'utente finale."
)


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    city: str
    weather: str
    cuisine: str | None = None
    budget: str | None = None


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


def _build_chat_service() -> AzureChatCompletion:
    return AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION"),
    )


def build_planner_kernel(weather_plugin: MCPStdioPlugin) -> Kernel:
    kernel = Kernel()
    kernel.add_service(_build_chat_service())
    kernel.add_plugin(weather_plugin)
    return kernel


def build_synthesizer_kernel() -> Kernel:
    kernel = Kernel()
    kernel.add_service(_build_chat_service())
    return kernel


def build_planner_instructions() -> str:
    now = datetime.now()
    return (
        f"{PLANNER_SYSTEM_PROMPT} "
        f"La data di oggi e {now.strftime('%Y-%m-%d')} e l'ora locale e {now.strftime('%H:%M:%S')}."
    )


def build_planner_agent(kernel: Kernel) -> ChatCompletionAgent:
    return ChatCompletionAgent(
        name="PlannerAgent",
        instructions=build_planner_instructions(),
        kernel=kernel,
        function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
    )


def build_synthesizer_agent(kernel: Kernel) -> ChatCompletionAgent:
    return ChatCompletionAgent(
        name="SynthesizerAgent",
        instructions=SYNTHESIZER_SYSTEM_PROMPT,
        kernel=kernel,
    )


def build_planner_execution_settings() -> AzureChatPromptExecutionSettings:
    return AzureChatPromptExecutionSettings(
        tool_choice="auto",
        parallel_tool_calls=False,
        response_format=get_structured_output_settings(PlannerOutput),
    )


def build_synthesizer_execution_settings() -> AzureChatPromptExecutionSettings:
    return AzureChatPromptExecutionSettings()


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
