import json
import logging

import httpx
from semantic_kernel.functions import kernel_function

LOGGER = logging.getLogger(__name__)


class AgentCardResolver:
    """Resolves a generic A2A agent task endpoint from its card URL."""

    def __init__(self, card_url: str) -> None:
        self._card_url = card_url

    async def resolve_endpoint(self, client: httpx.AsyncClient) -> str:
        response = await client.get(self._card_url)
        response.raise_for_status()
        card = response.json()
        endpoint = card.get("endpoint")
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError(f"Agent card at {self._card_url} is missing a valid endpoint")

        LOGGER.info("Resolved A2A endpoint", extra={"card_url": self._card_url, "endpoint": endpoint})
        return endpoint


class DiscoveryPlugin:
    """A2A discovery and caller plugin for Activity and Restaurant agents."""

    def __init__(self, timeout_seconds: float = 20.0, client: httpx.AsyncClient | None = None) -> None:
        self._timeout_seconds = timeout_seconds
        self._client = client or httpx.AsyncClient(timeout=self._timeout_seconds)
        self._owns_client = client is None
        self._activity_resolver = AgentCardResolver("http://localhost:8081/.well-known/agent-card.json")
        self._restaurant_resolver = AgentCardResolver("http://localhost:8082/.well-known/agent-card.json")

    def _rpc_error(self, code: str, message: str, rpc_id: int | str | None = 1) -> str:
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "error": {"code": code, "message": message},
                "id": rpc_id,
            },
            ensure_ascii=True,
        )

    async def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    async def _post_task(self, resolver: AgentCardResolver, payload: dict, error_prefix: str) -> str:
        rpc_id = payload.get("id")
        try:
            endpoint = await resolver.resolve_endpoint(self._client)
            response = await self._client.post(endpoint, json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.ConnectError as exc:
            LOGGER.exception("%s is offline", error_prefix, extra={"error": str(exc)})
            return self._rpc_error(
                code=f"{error_prefix.lower()}_offline",
                message=f"{error_prefix} non raggiungibile: {exc}",
                rpc_id=rpc_id,
            )
        except httpx.TimeoutException as exc:
            LOGGER.exception("%s timed out", error_prefix, extra={"error": str(exc)})
            return self._rpc_error(
                code=f"{error_prefix.lower()}_timeout",
                message=f"Timeout su {error_prefix}: {exc}",
                rpc_id=rpc_id,
            )
        except httpx.HTTPStatusError as exc:
            LOGGER.exception("%s returned HTTP error", error_prefix, extra={"status_code": exc.response.status_code})
            return self._rpc_error(
                code=f"{error_prefix.lower()}_http_error",
                message=f"Errore HTTP {exc.response.status_code} su {error_prefix}",
                rpc_id=rpc_id,
            )
        except httpx.RequestError as exc:
            LOGGER.exception("%s request failed", error_prefix, extra={"error": str(exc)})
            return self._rpc_error(
                code=f"{error_prefix.lower()}_request_error",
                message=f"Richiesta non riuscita: {exc}",
                rpc_id=rpc_id,
            )
        except (ValueError, KeyError) as exc:
            LOGGER.exception("%s discovery/response validation failed", error_prefix, extra={"error": str(exc)})
            return self._rpc_error(
                code=f"{error_prefix.lower()}_invalid_response",
                message=str(exc),
                rpc_id=rpc_id,
            )

        if "error" in body:
            LOGGER.error("%s returned RPC error", error_prefix, extra={"error": body.get("error")})
            return json.dumps({"jsonrpc": "2.0", "error": body["error"], "id": rpc_id}, ensure_ascii=True)

        result = body.get("result")
        if result is None:
            LOGGER.error("%s response missing result", error_prefix, extra={"body": body})
            return self._rpc_error(
                code=f"{error_prefix.lower()}_missing_result",
                message="Risultato mancante nella risposta.",
                rpc_id=rpc_id,
            )

        return json.dumps(result, ensure_ascii=True)

    @kernel_function(description="Chiama ActivityAgent con contesto di citta e meteo.")
    async def call_activity_agent(self, city: str, weather: str) -> str:
        payload = {
            "jsonrpc": "2.0",
            "method": "suggest_activity",
            "params": {
                "city": city,
                "weather": weather,
            },
            "id": 1,
        }
        return await self._post_task(self._activity_resolver, payload, "activity_agent")

    @kernel_function(description="Chiama RestaurantAgent con citta e preferenza di cucina.")
    async def call_restaurant_agent(self, city: str, cuisine_preference: str) -> str:
        payload = {
            "jsonrpc": "2.0",
            "method": "suggest_restaurant",
            "params": {
                "city": city,
                "cuisine_type": cuisine_preference,
            },
            "id": 1,
        }
        return await self._post_task(self._restaurant_resolver, payload, "restaurant_agent")
