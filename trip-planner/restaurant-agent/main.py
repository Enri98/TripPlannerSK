import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from anyio import Path as AnyioPath
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments, kernel_function
import uvicorn

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from data_contracts import TaskRequest
from helpers import create_rpc_error
from memory import RESTAURANT_DB

app = FastAPI()


BASE_DIR = Path(__file__).parent
AGENT_CARD_PATH = BASE_DIR / "agent_card.json"
INSTRUCTIONS_PATH = BASE_DIR / "instructions.md"
AGENT_CARD_CONTENT: dict = {}
SYSTEM_INSTRUCTIONS: str = ""

load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env", override=True)


class RestaurantSearchPlugin:
    @staticmethod
    def _normalize(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def _resolve_city_key(city: str) -> Optional[str]:
        target = city.strip().lower()
        for key in RESTAURANT_DB:
            if key.lower() == target:
                return key
        return None

    @staticmethod
    def _extract_price_bounds(price_range: str) -> tuple[int, int] | None:
        values = [int(match) for match in re.findall(r"\d+", price_range)]
        if not values:
            return None
        if len(values) == 1:
            return values[0], values[0]
        return values[0], values[-1]

    def _matches_budget(self, price_range: str, budget: str) -> bool:
        normalized_budget = self._normalize(budget)
        if normalized_budget in {"", "any", "all", "none", "qualsiasi", "non specificato"}:
            return True

        bounds = self._extract_price_bounds(price_range)
        if bounds is None:
            return True

        min_price, max_price = bounds

        if any(token in normalized_budget for token in {"economico", "cheap", "basso", "low"}):
            return max_price <= 25
        if any(token in normalized_budget for token in {"medio", "moderato", "medium"}):
            return 18 <= min_price <= 45
        if any(token in normalized_budget for token in {"alto", "premium", "lusso", "high"}):
            return max_price >= 45

        number_match = re.search(r"\d+", normalized_budget)
        if number_match:
            target = int(number_match.group(0))
            return min_price <= target and max_price <= target + 10

        return True

    @kernel_function(description="Restituisce ristoranti filtrati per citta, tipo di cucina e budget.")
    async def get_restaurants(self, city: str, cuisine: str, budget: str) -> list[dict]:
        city_key = self._resolve_city_key(city)
        if not city_key:
            return []

        restaurants = RESTAURANT_DB.get(city_key, [])
        normalized_cuisine = self._normalize(cuisine)
        if normalized_cuisine in {"", "any", "all"}:
            cuisine_filtered = restaurants
        else:
            cuisine_filtered = [
                item
                for item in restaurants
                if normalized_cuisine in self._normalize(item.get("cuisine_type", ""))
                or normalized_cuisine in self._normalize(item.get("type", ""))
            ]

        return [
            item
            for item in cuisine_filtered
            if self._matches_budget(item.get("price_range", ""), budget)
        ]


@app.on_event("startup")
async def startup_event() -> None:
    global AGENT_CARD_CONTENT, SYSTEM_INSTRUCTIONS

    card_raw = await AnyioPath(AGENT_CARD_PATH).read_text(encoding="utf-8")
    instructions_raw = await AnyioPath(INSTRUCTIONS_PATH).read_text(encoding="utf-8")

    AGENT_CARD_CONTENT = json.loads(card_raw)
    SYSTEM_INSTRUCTIONS = instructions_raw


@app.get("/.well-known/agent-card.json")
async def get_agent_card() -> Response:
    if AGENT_CARD_CONTENT:
        return Response(content=json.dumps(AGENT_CARD_CONTENT, indent=4), media_type="application/json")
    return Response(status_code=404, content="Scheda agente non trovata.")


@app.post("/task")
async def suggest_restaurant(request: TaskRequest):
    question = request.params.question

    kernel = Kernel()

    ai_service = AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION"),
    )
    kernel.add_service(ai_service)
    kernel.add_plugin(RestaurantSearchPlugin(), plugin_name="RestaurantSearch")

    chat_function = kernel.add_function(
        function_name="chat",
        plugin_name="RestaurantAgent",
        prompt=SYSTEM_INSTRUCTIONS,
    )

    try:
        execution_settings = AzureChatPromptExecutionSettings(
            tool_choice="auto",
            parallel_tool_calls=False,
            function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
        )

        result = await kernel.invoke(
            chat_function,
            KernelArguments(
                settings=execution_settings,
                question=question,
            ),
        )
        result_str = str(result)
    except Exception as e:
        print(f"CRITICAL KERNEL ERROR: {type(e).__name__}: {str(e)}")
        if hasattr(e, "inner_exception"):
            print(f"INNER ERROR: {e.inner_exception}")

        return create_rpc_error(-32603, f"Agente non disponibile: {str(e)}", request.id)

    return {
        "jsonrpc": "2.0",
        "result": {"reply": result_str},
        "id": request.id,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
