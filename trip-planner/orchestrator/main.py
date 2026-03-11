import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
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
    "Nella risposta finale indica chiaramente che i dati meteo non erano disponibili per la data richiesta. "
    "4. Restituisci un oggetto JSON rigorosamente strutturato contenente TUTTE e tre le categorie: "
    "weather_data, activity_suggestions e restaurant_recommendations. "
    "Imposta activity_suggestions con l'oggetto risultato esatto di ActivityAgent e preserva esattamente le sue chiavi "
    "(activities e note opzionale). "
    "Imposta restaurant_recommendations con l'oggetto risultato esatto di RestaurantAgent e preserva esattamente le sue chiavi "
    "(restaurants e note opzionale). "
    "Se un agente/tool restituisce un oggetto error, preserva e inoltra quel blocco error nella stessa sezione "
    "senza trasformarlo. "
    "I dati meteo possono essere 'Sconosciuto'; fornisci comunque attivita e ristoranti. "
    "Non scrivere prosa. "
    "Se un tool non meteo fallisce, riporta l'errore JSON."
)


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
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        print("----- RAW AGENT OUTPUT -----")
        print(raw_json)
        return

    if isinstance(parsed, dict) and "error" in parsed:
        print("----- ERRORE PIANIFICAZIONE VIAGGIO -----")
        print(json.dumps(parsed, indent=2, ensure_ascii=True))
        return

    weather_data = parsed.get("weather_data", parsed.get("weather", "N/A")) if isinstance(parsed, dict) else "N/A"
    activities_data = parsed.get("activity_suggestions", []) if isinstance(parsed, dict) else []
    restaurants_data = parsed.get("restaurant_recommendations", []) if isinstance(parsed, dict) else []
    note = parsed.get("note") if isinstance(parsed, dict) else None
    city = "Sconosciuta"

    if isinstance(parsed, dict):
        city = (
            parsed.get("city")
            or parsed.get("destination")
            or parsed.get("trip_city")
            or parsed.get("location")
            or parsed.get("city_name")
            or "Sconosciuta"
        )

    if isinstance(activities_data, dict):
        note = activities_data.get("note") or note
        activities = activities_data.get("activities", [])
    else:
        activities = activities_data

    if isinstance(restaurants_data, dict):
        restaurants = restaurants_data.get("restaurants", [])
    else:
        restaurants = restaurants_data

    print(f"--- IL TUO VIAGGIO A {city} ---")
    print(f"Meteo attuale: {weather_data}")
    if note:
        print(f"Nota: {note}")
    print("Attivita consigliate:")

    if not isinstance(activities, list) or not activities:
        print("- Nessuna attivita disponibile.")
        return

    for item in activities:
        if isinstance(item, dict):
            name = item.get("name", "Attivita senza nome")
            kind = item.get("type", "Sconosciuto")
            description = item.get("description", "Nessuna descrizione")
            print(f"- {name} [{kind}] - {description}")
        else:
            print(f"- {item}")

    print("Ristoranti consigliati:")
    if not isinstance(restaurants, list) or not restaurants:
        print("- Nessun suggerimento ristorante disponibile.")
        return

    for item in restaurants:
        if isinstance(item, dict):
            name = item.get("name", "Ristorante senza nome")
            kind = item.get("type", "Sconosciuto")
            price_range = item.get("price_range", "N/A")
            print(f"- {name} [{kind}] - Prezzo: {price_range}")
        else:
            print(f"- {item}")


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
            )
            response = await agent.get_response(
                messages=user_input,
                arguments=KernelArguments(settings=execution_settings),
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
