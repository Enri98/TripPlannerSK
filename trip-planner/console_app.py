import asyncio
import json

import httpx
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.functions import KernelArguments

from helpers import is_schema_response_format_unsupported
from orchestrator.main import (
    PlannerOutput,
    build_planner_agent,
    build_planner_execution_settings,
    build_planner_kernel,
    build_synthesizer_agent,
    build_synthesizer_execution_settings,
    build_synthesizer_kernel,
    build_weather_mcp_plugin,
    configure_logging,
    load_environment,
)
from orchestrator.plugins.discovery_plugin import DiscoveryPlugin


async def initialize_runtime(console: Console) -> tuple[
    ChatCompletionAgent,
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
        mcp_plugin = build_weather_mcp_plugin()

        await mcp_plugin.connect()
        await mcp_plugin.load_tools()

        planner_kernel = build_planner_kernel(mcp_plugin)
        synthesizer_kernel = build_synthesizer_kernel()
        planner_agent = build_planner_agent(planner_kernel)
        synthesizer_agent = build_synthesizer_agent(synthesizer_kernel)

        console.print("[bold green]Trip Planner pronto.[/bold green] Scrivi 'exit' per uscire.")

        return planner_agent, synthesizer_agent, discovery_plugin, mcp_plugin, shared_client
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


def parse_planner_output(raw_text: str) -> PlannerOutput:
    payload = raw_text.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            payload = "\n".join(lines[1:-1]).strip()

    try:
        return PlannerOutput.model_validate_json(payload)
    except (ValidationError, json.JSONDecodeError):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            start = payload.find("{")
            end = payload.rfind("}")
            if start == -1 or end <= start:
                raise ValueError("Il Planner non ha restituito un JSON valido.")
            parsed = json.loads(payload[start : end + 1])

        return PlannerOutput.model_validate(parsed)


async def read_request() -> str:
    return await asyncio.to_thread(Prompt.ask, "[bold blue]Che viaggio vuoi pianificare?")


async def run_console_app() -> None:
    console = Console()
    planner_agent: ChatCompletionAgent | None = None
    synthesizer_agent: ChatCompletionAgent | None = None
    discovery_plugin: DiscoveryPlugin | None = None
    mcp_plugin: MCPStdioPlugin | None = None
    shared_client: httpx.AsyncClient | None = None

    try:
        (
            planner_agent,
            synthesizer_agent,
            discovery_plugin,
            mcp_plugin,
            shared_client,
        ) = await initialize_runtime(console)

        while True:
            user_request = (await read_request()).strip()
            if user_request.lower() in {"exit", "quit"}:
                return
            if not user_request:
                continue

            planner_settings = build_planner_execution_settings()
            with console.status("[bold green]Sto analizzando la richiesta..."):
                try:
                    planner_response = await planner_agent.get_response(
                        messages=user_request,
                        arguments=KernelArguments(settings=planner_settings),
                    )
                except Exception as schema_exc:
                    if not is_schema_response_format_unsupported(schema_exc):
                        raise
                    fallback_settings = AzureChatPromptExecutionSettings(
                        tool_choice="auto",
                        parallel_tool_calls=False,
                    )
                    planner_response = await planner_agent.get_response(
                        messages=user_request,
                        arguments=KernelArguments(settings=fallback_settings),
                    )

            try:
                planner_data = parse_planner_output(str(planner_response.message.content))
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                console.print(Panel(str(exc), title="Errore Planner", border_style="red"))
                continue

            city = planner_data.city.strip()
            weather = (planner_data.weather or "Sconosciuto").strip() or "Sconosciuto"
            cuisine = (planner_data.cuisine or "any").strip() or "any"
            budget = (planner_data.budget or "any").strip() or "any"

            with console.status("[bold green]Sto interrogando gli agenti specializzati..."):
                activity_reply, restaurant_reply = await asyncio.gather(
                    discovery_plugin.call_activity_agent(city=city, weather=weather),
                    discovery_plugin.call_restaurant_agent(
                        city=city,
                        cuisine_preference=cuisine,
                        budget=budget,
                    ),
                )

            synthesis_input = (
                f"Contesto: Meteo: {weather}. "
                f"Attivita: {activity_reply}. "
                f"Ristoranti: {restaurant_reply}"
            )

            with console.status("[bold green]Sto sintetizzando l'itinerario finale..."):
                final_response = await synthesizer_agent.get_response(
                    messages=synthesis_input,
                    arguments=KernelArguments(settings=build_synthesizer_execution_settings()),
                )

            final_text = str(final_response.message.content).strip()
            if not final_text:
                console.print(Panel("Risposta finale vuota dal SynthesizerAgent.", border_style="red"))
                continue

            console.print(Markdown(final_text))
    except Exception as exc:
        console.print(Panel(str(exc), title="Errore avvio/esecuzione", border_style="red"))
    finally:
        if discovery_plugin and mcp_plugin and shared_client:
            await shutdown_runtime(discovery_plugin, mcp_plugin, shared_client)


if __name__ == "__main__":
    asyncio.run(run_console_app())
