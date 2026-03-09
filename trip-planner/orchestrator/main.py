import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from plugins.discovery_plugin import DiscoveryPlugin

LOGGER = logging.getLogger("trip_orchestrator")
SYSTEM_INSTRUCTIONS = (
    "You are a strictly data-driven Orchestrator. "
    "1. Call WeatherMcp for the city. "
    "2. Pass EXACT weather string and city to ActivityAgent. "
    "3. Output ONLY a strictly structured JSON object with keys weather_data and activity_suggestions. "
    "Do not write prose. "
    "If the user asks for a trip on a specific date other than today, check if the Weather Tool supports it. "
    "If not, inform the user you can only plan for today. "
    "If a tool fails, report the error JSON. NEVER generate activities yourself."
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


def build_kernel() -> Kernel:
    kernel = Kernel()

    chat_service = AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION"),
    )
    kernel.add_service(chat_service)

    kernel.add_plugin(DiscoveryPlugin(), plugin_name="ActivityService")

    return kernel


def present_itinerary(raw_json: str) -> None:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        print("----- RAW AGENT OUTPUT -----")
        print(raw_json)
        return

    if isinstance(parsed, dict) and "error" in parsed:
        print("----- TRIP PLANNING ERROR -----")
        print(json.dumps(parsed, indent=2, ensure_ascii=True))
        return

    weather_data = parsed.get("weather_data", parsed.get("weather", "N/A")) if isinstance(parsed, dict) else "N/A"
    activities = parsed.get("activity_suggestions", []) if isinstance(parsed, dict) else []
    city = "Unknown"

    if isinstance(parsed, dict):
        city = parsed.get("city") or parsed.get("destination") or "Unknown"

    print(f"--- YOUR TRIP TO {city} ---")
    print(f"Current Weather: {weather_data}")
    print("Recommended Activities:")

    if not isinstance(activities, list) or not activities:
        print("- No activities available.")
        return

    for item in activities:
        if isinstance(item, dict):
            name = item.get("name", "Unnamed activity")
            kind = item.get("type", "Unknown")
            description = item.get("description", "No description")
            print(f"- {name} [{kind}] - {description}")
        else:
            print(f"- {item}")


async def run_console() -> None:
    configure_logging()
    load_environment()

    kernel = build_kernel()

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
            "Unable to start the weather tool (MCP subprocess failed). "
            "Check mcp-weather-server and try again."
        )
        return

    try:
        agent = ChatCompletionAgent(
            name="TripOrchestrator",
            instructions=SYSTEM_INSTRUCTIONS,
            kernel=kernel,
            function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
        )

        print("Trip Orchestrator ready. Type your request (or 'exit').")
        while True:
            user_input = input("\nYou> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
                return
            if not user_input:
                continue

            response = await agent.get_response(messages=user_input)
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


if __name__ == "__main__":
    asyncio.run(run_console())
