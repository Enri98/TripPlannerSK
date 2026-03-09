import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

try:
    from semantic_kernel.functions.kernel_plugin_mcp import MCPStdioPlugin
except ImportError:
    from semantic_kernel.connectors.mcp import MCPStdioPlugin

from plugins.discovery_plugin import DiscoveryPlugin

LOGGER = logging.getLogger("trip_orchestrator")
SYSTEM_INSTRUCTIONS = (
    "You are the Trip Orchestrator. When a user asks for a trip, first use the Weather Tool "
    "to check conditions. Then, pass that weather and city to the Activity Specialist. "
    "Finally, present a cohesive travel plan. Do not hallucinate; only use data from the tools."
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

    activity_card_path = Path(__file__).resolve().parents[1] / "activity-agent" / "agent_card.json"
    kernel.add_plugin(DiscoveryPlugin(card_path=activity_card_path), plugin_name="DiscoveryPlugin")

    return kernel


async def run_console() -> None:
    configure_logging()
    load_environment()

    kernel = build_kernel()

    weather_server_script = Path(__file__).resolve().parents[2] / "mcp-weather-server" / "server.py"
    python_executable = str((Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe").resolve())

    mcp_plugin = MCPStdioPlugin(
        name="WeatherTool",
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
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
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
            print(f"\nPlanner> {response.message.content}")
    finally:
        try:
            await mcp_plugin.close()
        except Exception:
            LOGGER.warning("MCP weather plugin closed with errors.")


if __name__ == "__main__":
    asyncio.run(run_console())
