import asyncio
import json

import httpx
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.functions import KernelArguments

from helpers import is_schema_response_format_unsupported
from orchestrator.main import (
    ActivityResponse,
    AgentErrorPayload,
    DiscoveryPlugin,
    RestaurantResponse,
    TripDirectorResponse,
    build_execution_settings,
    build_kernel,
    build_orchestrator_agent,
    build_weather_mcp_plugin,
    configure_logging,
    load_environment,
)


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
        mcp_plugin = build_weather_mcp_plugin()

        await mcp_plugin.connect()
        await mcp_plugin.load_tools()
        kernel.add_plugin(mcp_plugin)

        agent = build_orchestrator_agent(kernel)
        console.print("[bold green]Trip Planner pronto.[/bold green] Scrivi 'exit' per uscire.")

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
    payload = raw_json.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            payload = "\n".join(lines[1:-1]).strip()

    try:
        itinerary = TripDirectorResponse.model_validate_json(payload)
    except ValidationError as exc:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            console.print(
                Panel(
                    "Il Direttore Viaggio ha restituito un output non JSON.",
                    title="Errore di parsing",
                    border_style="red",
                )
            )
            console.print(payload)
            return

        if isinstance(parsed, dict) and "error" in parsed:
            console.print(
                Panel(
                    json.dumps(parsed["error"], indent=2, ensure_ascii=True),
                    title="Errore pianificazione viaggio",
                    border_style="red",
                )
            )
            return

        console.print(
            Panel(
                str(exc),
                title="Errore validazione schema",
                border_style="red",
            )
        )
        console.print(json.dumps(parsed, indent=2, ensure_ascii=True))
        return

    weather = itinerary.weather_data
    weather_text = weather if isinstance(weather, str) else json.dumps(weather, indent=2, ensure_ascii=True)
    console.print(Panel(weather_text, title="Meteo", border_style="cyan"))

    activity_note = itinerary.activity_suggestions.note if isinstance(itinerary.activity_suggestions, ActivityResponse) else None
    restaurant_note = (
        itinerary.restaurant_recommendations.note
        if isinstance(itinerary.restaurant_recommendations, RestaurantResponse)
        else None
    )
    note = activity_note or restaurant_note or itinerary.note
    if note:
        console.print(Panel(note, title="Nota", border_style="yellow"))

    if isinstance(itinerary.activity_suggestions, AgentErrorPayload):
        error_text = json.dumps(itinerary.activity_suggestions.error.model_dump(mode="json"), indent=2, ensure_ascii=True)
        console.print(Panel(error_text, title="Servizio non disponibile: attivita", border_style="red"))
    else:
        activities_table = Table(title="Attivita consigliate")
        activities_table.add_column("Nome", style="bold")
        activities_table.add_column("Tipo")
        activities_table.add_column("Descrizione")

        if itinerary.activity_suggestions.activities:
            for activity in itinerary.activity_suggestions.activities:
                activities_table.add_row(activity.name, activity.type, activity.description)
        else:
            activities_table.add_row("-", "-", "Nessuna attivita disponibile.")
        console.print(activities_table)

    if isinstance(itinerary.restaurant_recommendations, AgentErrorPayload):
        error_text = json.dumps(
            itinerary.restaurant_recommendations.error.model_dump(mode="json"),
            indent=2,
            ensure_ascii=True,
        )
        console.print(Panel(error_text, title="Servizio non disponibile: ristoranti", border_style="red"))
    else:
        restaurants_table = Table(title="Ristoranti consigliati")
        restaurants_table.add_column("Nome", style="bold")
        restaurants_table.add_column("Cucina")
        restaurants_table.add_column("Fascia di prezzo")

        if itinerary.restaurant_recommendations.restaurants:
            for restaurant in itinerary.restaurant_recommendations.restaurants:
                restaurants_table.add_row(restaurant.name, restaurant.type, restaurant.price_range)
        else:
            restaurants_table.add_row("-", "-", "Nessun ristorante disponibile.")
        console.print(restaurants_table)


async def read_destination() -> str:
    return await asyncio.to_thread(Prompt.ask, "[bold blue]Dove vorresti andare?")


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

            user_message = f"Pianifica un viaggio a {destination}."
            execution_settings = build_execution_settings()

            with console.status("[bold green]Sto pianificando il tuo viaggio..."):
                try:
                    response = await agent.get_response(
                        messages=user_message,
                        arguments=KernelArguments(settings=execution_settings),
                    )
                except Exception as schema_exc:
                    if not is_schema_response_format_unsupported(schema_exc):
                        raise
                    fallback_settings = AzureChatPromptExecutionSettings(
                        tool_choice="auto",
                        parallel_tool_calls=False,
                    )
                    response = await agent.get_response(
                        messages=user_message,
                        arguments=KernelArguments(settings=fallback_settings),
                    )
            await present_itinerary(console, str(response.message.content))
    except Exception as exc:
        console.print(Panel(str(exc), title="Errore avvio/esecuzione", border_style="red"))
    finally:
        if discovery_plugin and mcp_plugin and shared_client:
            await shutdown_runtime(discovery_plugin, mcp_plugin, shared_client)


if __name__ == "__main__":
    asyncio.run(run_console_app())
