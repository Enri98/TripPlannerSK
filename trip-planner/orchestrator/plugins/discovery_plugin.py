import json
import logging
from pathlib import Path
from typing import Any

import httpx
from semantic_kernel.functions import kernel_function

LOGGER = logging.getLogger(__name__)


class ActivityCardResolver:
    """Resolves ActivityAgent metadata from a local A2A card."""

    def __init__(self, card_path: Path) -> None:
        self._card_path = card_path

    def load_card(self) -> dict[str, Any]:
        if not self._card_path.exists():
            raise FileNotFoundError(f"Activity agent card not found: {self._card_path}")

        with self._card_path.open("r", encoding="utf-8") as handle:
            card = json.load(handle)

        if not isinstance(card, dict):
            raise ValueError("Activity agent card must be a JSON object.")

        endpoint = card.get("endpoint")
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("Activity agent card is missing a valid 'endpoint'.")

        return card


class DiscoveryPlugin:
    """A2A discovery and caller plugin for the ActivityAgent."""

    def __init__(self, card_path: Path, timeout_seconds: float = 20.0) -> None:
        self._resolver = ActivityCardResolver(card_path=card_path)
        self._timeout_seconds = timeout_seconds

    def _resolve_endpoint(self) -> str:
        card = self._resolver.load_card()
        endpoint = card["endpoint"]
        LOGGER.info("Resolved ActivityAgent endpoint", extra={"endpoint": endpoint})
        return endpoint

    @kernel_function(description="Calls the Activity Specialist with city and weather context.")
    async def call_activity_agent(self, city: str, weather: str) -> str:
        endpoint = self._resolve_endpoint()
        payload = {
            "jsonrpc": "2.0",
            "method": "suggest_activities",
            "params": {
                "city": city,
                "weather": weather,
            },
            "id": 1,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            LOGGER.exception(
                "ActivityAgent returned HTTP error",
                extra={"status_code": exc.response.status_code, "endpoint": endpoint},
            )
            return (
                f"ActivityAgent HTTP error {exc.response.status_code}. "
                "The Activity Specialist may be unavailable right now."
            )
        except httpx.RequestError as exc:
            LOGGER.exception("ActivityAgent is offline or unreachable", extra={"endpoint": endpoint})
            return f"ActivityAgent is offline or unreachable: {exc}"
        except ValueError:
            LOGGER.exception("Invalid JSON response from ActivityAgent", extra={"endpoint": endpoint})
            return "ActivityAgent returned invalid JSON."

        if "error" in body:
            LOGGER.error("ActivityAgent returned RPC error", extra={"error": body.get("error")})
            return f"ActivityAgent error: {body['error']}"

        result = body.get("result")
        if result is None:
            LOGGER.error("ActivityAgent response missing result", extra={"body": body})
            return "ActivityAgent response did not include a result."

        return json.dumps(result, ensure_ascii=True)
