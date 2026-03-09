import json
import logging

import httpx
from semantic_kernel.functions import kernel_function

LOGGER = logging.getLogger(__name__)


class ActivityCardResolver:
    """Resolves ActivityAgent metadata from the ActivityAgent card endpoint."""

    def __init__(self, card_url: str = "http://localhost:8081/.well-known/agent-card.json") -> None:
        self._card_url = card_url

    async def resolve_endpoint(self, client: httpx.AsyncClient) -> str:
        response = await client.get(self._card_url)
        response.raise_for_status()
        card = response.json()
        endpoint = card.get("endpoint")
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("Activity agent card is missing a valid 'endpoint'.")

        LOGGER.info("Resolved ActivityAgent endpoint", extra={"endpoint": endpoint})
        return endpoint


class DiscoveryPlugin:
    """A2A discovery and caller plugin for the ActivityAgent."""

    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self._resolver = ActivityCardResolver()
        self._timeout_seconds = timeout_seconds

    @kernel_function(description="Calls the Activity Specialist with city and weather context.")
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

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                endpoint = await self._resolver.resolve_endpoint(client)
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.ConnectError as exc:
            LOGGER.exception("ActivityAgent is offline", extra={"error": str(exc)})
            return json.dumps(
                {"error": {"code": "activity_agent_offline", "message": f"ActivityAgent offline: {exc}"}},
                ensure_ascii=True,
            )
        except httpx.HTTPStatusError as exc:
            LOGGER.exception(
                "ActivityAgent returned HTTP error",
                extra={"status_code": exc.response.status_code},
            )
            return json.dumps(
                {
                    "error": {
                        "code": "activity_agent_http_error",
                        "message": f"ActivityAgent HTTP error {exc.response.status_code}",
                    }
                },
                ensure_ascii=True,
            )
        except httpx.RequestError as exc:
            LOGGER.exception("ActivityAgent request failed", extra={"error": str(exc)})
            return json.dumps(
                {"error": {"code": "activity_agent_request_error", "message": f"Request failed: {exc}"}},
                ensure_ascii=True,
            )
        except (ValueError, KeyError) as exc:
            LOGGER.exception("ActivityAgent discovery/response validation failed", extra={"error": str(exc)})
            return json.dumps(
                {"error": {"code": "activity_agent_invalid_response", "message": str(exc)}},
                ensure_ascii=True,
            )

        if "error" in body:
            LOGGER.error("ActivityAgent returned RPC error", extra={"error": body.get("error")})
            return json.dumps({"error": body["error"]}, ensure_ascii=True)

        result = body.get("result")
        if result is None:
            LOGGER.error("ActivityAgent response missing result", extra={"body": body})
            return json.dumps(
                {"error": {"code": "activity_agent_missing_result", "message": "Missing result in response."}},
                ensure_ascii=True,
            )

        return json.dumps(result, ensure_ascii=True)
