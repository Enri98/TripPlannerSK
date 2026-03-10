import json
import os
from pathlib import Path
from typing import Optional

from anyio import Path as AnyioPath
from fastapi import FastAPI, Response
from pydantic import BaseModel
import uvicorn
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from dotenv import load_dotenv
from semantic_kernel.functions import KernelArguments, kernel_function

from memory import ACTIVITIES_DB

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Pydantic Models ---
class TaskRequestParams(BaseModel):
    city: str
    weather: str

class TaskRequest(BaseModel):
    jsonrpc: str
    method: str
    params: TaskRequestParams
    id: Optional[int] = None

# --- File Paths ---
BASE_DIR = Path(__file__).parent
AGENT_CARD_PATH = BASE_DIR / "agent_card.json"
INSTRUCTIONS_PATH = BASE_DIR / "instructions.md"
AGENT_CARD_CONTENT: dict = {}
SYSTEM_PROMPT: str = ""


class ActivitySearchPlugin:
    @staticmethod
    def _normalize(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def _resolve_city_key(city: str) -> Optional[str]:
        target = city.strip().lower()
        for key in ACTIVITIES_DB:
            if key.lower() == target:
                return key
        return None

    @kernel_function(description="Returns activities filtered by city and weather.")
    async def get_activities(self, city: str, weather: str) -> list[dict]:
        city_key = self._resolve_city_key(city)
        if not city_key:
            return []

        normalized_weather = self._normalize(weather)
        activities = ACTIVITIES_DB.get(city_key, [])

        bad_weather_markers = {"rain", "storm", "snow", "thunder", "wind"}
        good_weather_markers = {"sun", "clear", "warm", "hot"}

        if any(marker in normalized_weather for marker in bad_weather_markers):
            return [item for item in activities if item.get("type", "").lower() == "indoor"]

        if any(marker in normalized_weather for marker in good_weather_markers):
            return [item for item in activities if item.get("type", "").lower() == "outdoor"]

        return activities

# --- Load Environment Variables ---
load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env", override=True)

# --- Endpoints ---
@app.on_event("startup")
async def startup_event() -> None:
    global AGENT_CARD_CONTENT, SYSTEM_PROMPT

    card_raw = await AnyioPath(AGENT_CARD_PATH).read_text(encoding="utf-8")
    instructions_raw = await AnyioPath(INSTRUCTIONS_PATH).read_text(encoding="utf-8")

    AGENT_CARD_CONTENT = json.loads(card_raw)
    SYSTEM_PROMPT = instructions_raw


@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    """
    Serves the agent's A2A card for discovery.
    """
    if AGENT_CARD_CONTENT:
        return Response(content=json.dumps(AGENT_CARD_CONTENT, indent=4), media_type="application/json")
    return Response(status_code=404, content="Agent card not found.")

@app.post("/task")
async def suggest_activity(request: TaskRequest):
    """
    Suggests activities based on city and weather.
    """
    city = request.params.city
    weather = request.params.weather

    # 1. Init Kernel
    kernel = Kernel()

    # 2. Add AI Service (Verified Stable v1.x Syntax)
    ai_service = AzureChatCompletion(
        deployment_name= os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key= os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION")
    )
    # Register the service directly. The kernel handles it as the default.
    kernel.add_service(ai_service)
    kernel.add_plugin(ActivitySearchPlugin(), plugin_name="ActivitySearch")

    # --- Error Handling for Unsupported City ---
    if ActivitySearchPlugin._resolve_city_key(city) is None:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": "City not supported"},
            "id": request.id
        }

    # 3. Register Function from prompt
    chat_function = kernel.add_function(
        function_name="chat",
        plugin_name="ActivityPlugin",
        prompt=SYSTEM_PROMPT,
    )

    try:
        execution_settings = AzureChatPromptExecutionSettings(
            tool_choice="auto",
            parallel_tool_calls=False,
            function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
        )

        # Invoke with variable binding
        result = await kernel.invoke(
            chat_function,
            KernelArguments(settings=execution_settings, city=city, weather=weather),
        )
        result_str = str(result)
    except Exception as e:
        # LOG THE FULL ERROR TO CONSOLE
        print(f"CRITICAL KERNEL ERROR: {type(e).__name__}: {str(e)}")
        # If it's a wrapper, try to get the inner message
        if hasattr(e, 'inner_exception'):
            print(f"INNER ERROR: {e.inner_exception}")

        return {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": f"Agent failed: {str(e)}"},
            "id": request.id
        }

    # --- Format and Return Response ---
    try:
        # Clean the response string
        if '```json' in result_str:
            result_str = result_str.split('```json')[1].split('```')[0].strip()

        suggested_activities = json.loads(result_str)

    except json.JSONDecodeError as e:
        # Log the error for debugging
        print(f"JSONDecodeError: {e}")
        print(f"Malformed response: {str(result)}")

        # Return a JSON-RPC error response
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": "Internal error: Failed to decode JSON from AI response."
            },
            "id": request.id
        }

    return {
        "jsonrpc": "2.0",
        "result": suggested_activities,
        "id": request.id
    }

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
