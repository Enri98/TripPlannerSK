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

from plugins.discovery_plugin import DiscoveryPlugin

LOGGER = logging.getLogger("trip_orchestrator")
SYSTEM_INSTRUCTIONS = (
    "You are the Trip Director. "
    "1. Get weather. "
    "2. Get activities. "
    "3. Get restaurant recommendations. "
    "You must attempt to obtain weather data from the WeatherMcp tool before calling the ActivityAgent. "
    "Use only the exact output from the weather tool for the ActivityAgent weather parameter. "
    "If the WeatherMcp tool returns a forecast limit error or any other error, do not stop. "
    "Proceed by calling the ActivityAgent with weather='Unknown' and call the RestaurantAgent normally. "
    "In your final response, clearly state that weather data was unavailable for the requested date. "
    "4. Output a strictly structured JSON object containing ALL three categories: "
    "weather_data, activity_suggestions, and restaurant_recommendations. "
    "Do not write prose. "
    "If a non-weather tool fails, report the error JSON."
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
        f"Today's date is {now.strftime('%Y-%m-%d')}. "
        f"Current local time is {now.strftime('%H:%M:%S')}."
    )


def present_itinerary(raw_json: str) -> None:
    cleaned = raw_json.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
        cleaned = cleaned.split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1]
        cleaned = cleaned.split("```", 1)[0].strip()

    try:
        parsed = json.loads(cleaned)
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
    restaurants = parsed.get("restaurant_recommendations", []) if isinstance(parsed, dict) else []
    city = "Unknown"

    if isinstance(parsed, dict):
        city = (
            parsed.get("city")
            or parsed.get("destination")
            or parsed.get("trip_city")
            or parsed.get("location")
            or parsed.get("city_name")
            or "Unknown"
        )

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

    print("Recommended Restaurants:")
    if not isinstance(restaurants, list) or not restaurants:
        print("- No restaurant recommendations available.")
        return

    for item in restaurants:
        if isinstance(item, dict):
            name = item.get("name", "Unnamed restaurant")
            kind = item.get("type", "Unknown")
            price_range = item.get("price_range", "N/A")
            print(f"- {name} [{kind}] - Price: {price_range}")
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
            "Unable to start the weather tool (MCP subprocess failed). "
            "Check mcp-weather-server and try again."
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

        print("Trip Orchestrator ready. Type your request (or 'exit').")
        while True:
            user_input = input("\nYou> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
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
