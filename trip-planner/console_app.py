import asyncio
import json
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.functions import KernelArguments

from orchestrator.main import (
    DiscoveryPlugin,
    build_kernel,
    build_system_instructions,
    configure_logging,
    load_environment,
)


class ActivityItem(BaseModel):
    name: str
    type: str
    description: str


class RpcError(BaseModel):
    code: int | str
    message: str


class ActivityResponse(BaseModel):
    activities: list[ActivityItem] | None = None
    error: RpcError | dict | None = None
    note: str | None = None


class RestaurantItem(BaseModel):
    name: str
    type: str
    price_range: str


class RestaurantResponse(BaseModel):
    restaurants: list[RestaurantItem] | None = None
    error: RpcError | dict | None = None
    note: str | None = None


class TripDirectorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    weather_data: str | dict | list
    activity_suggestions: ActivityResponse
    restaurant_recommendations: RestaurantResponse
    note: str | None = None


async def initialize_runtime(console: Console) -> tuple[
    ChatCompletionAgent,
    DiscoveryPlugin,
    MCPStdioPlugin,
    httpx.AsyncClient,
]:
    configure_logging()
    load_environment()

    shared_client = httpx.AsyncClient(timeout=20.0)
    discovery_plugin = DiscoveryPlugin(timeout_seconds=20.0, client=shared_client)
    mcp_plugin: MCPStdioPlugin | None = None
    try:
        kernel = build_kernel(discovery_plugin)

        weather_server_script = Path(__file__).resolve().parents[1] / "mcp-weather-server" / "server.py"
        python_executable = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"

        if not python_executable.exists():
            raise FileNotFoundError(f"Virtual environment interpreter not found at: {python_executable}")
        if not weather_server_script.exists():
            raise FileNotFoundError(f"Weather MCP server not found at: {weather_server_script}")

        mcp_plugin = MCPStdioPlugin(
            name="WeatherMcp",
            command=str(python_executable),
            args=[str(weather_server_script)],
            request_timeout=30,
        )

        await mcp_plugin.connect()
        await mcp_plugin.load_tools()
        kernel.add_plugin(mcp_plugin)

        agent = ChatCompletionAgent(
            name="TripOrchestrator",
            instructions=build_system_instructions(),
            kernel=kernel,
            function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
        )
        console.print("[bold green]Trip Planner ready.[/bold green] Type 'exit' to quit.")

        return agent, discovery_plugin, mcp_plugin, shared_client
    except Exception:
        if mcp_plugin is not None:
            try:
                await mcp_plugin.close()
            except Exception:
                pass
        if not shared_client.is_closed:
            await shared_client.aclose()
        raise


async def shutdown_runtime(
    discovery_plugin: DiscoveryPlugin,
    mcp_plugin: MCPStdioPlugin,
    shared_client: httpx.AsyncClient,
) -> None:
    try:
        await mcp_plugin.close()
    except Exception:
        pass

    try:
        await discovery_plugin.close()
    except Exception:
        pass

    if not shared_client.is_closed:
        await shared_client.aclose()


async def present_itinerary(console: Console, raw_json: str) -> None:
    try:
        itinerary = TripDirectorResponse.model_validate_json(raw_json)
    except ValidationError as exc:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            console.print(
                Panel(
                    "Trip Director returned non-JSON output.",
                    title="Parsing Error",
                    border_style="red",
                )
            )
            console.print(raw_json)
            return

        if isinstance(parsed, dict) and "error" in parsed:
            console.print(
                Panel(
                    json.dumps(parsed["error"], indent=2, ensure_ascii=True),
                    title="Trip Planning Error",
                    border_style="red",
                )
            )
            return

        console.print(
            Panel(
                str(exc),
                title="Schema Validation Error",
                border_style="red",
            )
        )
        console.print(json.dumps(parsed, indent=2, ensure_ascii=True))
        return

    weather = itinerary.weather_data
    weather_text = weather if isinstance(weather, str) else json.dumps(weather, indent=2, ensure_ascii=True)
    console.print(Panel(weather_text, title="Weather", border_style="cyan"))

    note = itinerary.activity_suggestions.note or itinerary.restaurant_recommendations.note or itinerary.note
    if note:
        console.print(Panel(note, title="Note", border_style="yellow"))

    if itinerary.activity_suggestions.error:
        error_obj = itinerary.activity_suggestions.error
        error_text = (
            json.dumps(error_obj.model_dump(mode="json"), indent=2, ensure_ascii=True)
            if isinstance(error_obj, RpcError)
            else json.dumps(error_obj, indent=2, ensure_ascii=True)
        )
        console.print(Panel(error_text, title="Service Unavailable: Activities", border_style="red"))
    else:
        activities_table = Table(title="Activities")
        activities_table.add_column("Name", style="bold")
        activities_table.add_column("Type")
        activities_table.add_column("Description")

        if itinerary.activity_suggestions.activities:
            for activity in itinerary.activity_suggestions.activities:
                activities_table.add_row(activity.name, activity.type, activity.description)
        else:
            activities_table.add_row("-", "-", "No activities available.")
        console.print(activities_table)

    if itinerary.restaurant_recommendations.error:
        error_obj = itinerary.restaurant_recommendations.error
        error_text = (
            json.dumps(error_obj.model_dump(mode="json"), indent=2, ensure_ascii=True)
            if isinstance(error_obj, RpcError)
            else json.dumps(error_obj, indent=2, ensure_ascii=True)
        )
        console.print(Panel(error_text, title="Service Unavailable: Restaurants", border_style="red"))
    else:
        restaurants_table = Table(title="Restaurants")
        restaurants_table.add_column("Name", style="bold")
        restaurants_table.add_column("Cuisine")
        restaurants_table.add_column("Price Range")

        if itinerary.restaurant_recommendations.restaurants:
            for restaurant in itinerary.restaurant_recommendations.restaurants:
                restaurants_table.add_row(restaurant.name, restaurant.type, restaurant.price_range)
        else:
            restaurants_table.add_row("-", "-", "No restaurants available.")
        console.print(restaurants_table)


async def read_destination() -> str:
    return await asyncio.to_thread(Prompt.ask, "[bold blue]Where would you like to go?")


async def run_console_app() -> None:
    console = Console()
    agent: ChatCompletionAgent | None = None
    discovery_plugin: DiscoveryPlugin | None = None
    mcp_plugin: MCPStdioPlugin | None = None
    shared_client: httpx.AsyncClient | None = None

    try:
        agent, discovery_plugin, mcp_plugin, shared_client = await initialize_runtime(console)

        while True:
            destination = (await read_destination()).strip()
            if destination.lower() in {"exit", "quit"}:
                return
            if not destination:
                continue

            user_message = f"Plan a trip to {destination}."
            execution_settings = AzureChatPromptExecutionSettings(
                tool_choice="auto",
                parallel_tool_calls=False,
                response_format={"type": "json_object"},
            )

            with console.status("[bold green]Planning your trip..."):
                response = await agent.get_response(
                    messages=user_message,
                    arguments=KernelArguments(settings=execution_settings),
                )
            await present_itinerary(console, str(response.message.content))
    except Exception as exc:
        console.print(Panel(str(exc), title="Startup/Runtime Error", border_style="red"))
    finally:
        if discovery_plugin and mcp_plugin and shared_client:
            await shutdown_runtime(discovery_plugin, mcp_plugin, shared_client)


if __name__ == "__main__":
    asyncio.run(run_console_app())
