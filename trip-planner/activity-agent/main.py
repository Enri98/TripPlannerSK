import json
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Response
from pydantic import BaseModel
import uvicorn
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from dotenv import load_dotenv
from semantic_kernel.functions import KernelArguments

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

# --- Load Environment Variables ---
load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env", override=True)

# --- Endpoints ---

@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    """
    Serves the agent's A2A card for discovery.
    """
    if AGENT_CARD_PATH.exists():
        with open(AGENT_CARD_PATH, "r") as f:
            agent_card_content = json.load(f)
        return Response(content=json.dumps(agent_card_content, indent=4), media_type="application/json")
    return Response(status_code=404, content="Agent card not found.")

@app.post("/task")
async def suggest_activity(request: TaskRequest):
    """
    Suggests activities based on city and weather.
    """
    city = request.params.city
    weather = request.params.weather

    # 1. Init Kernel
    kernel = sk.Kernel()

    # 2. Add AI Service (Verified Stable v1.x Syntax)
    ai_service = AzureChatCompletion(
        deployment_name= os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key= os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION")
    )
    # Register the service directly. The kernel handles it as the default.
    kernel.add_service(ai_service)

    # --- Error Handling for Unsupported City ---
    if city not in ACTIVITIES_DB:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": "City not supported"},
            "id": request.id
        }

    # --- Load System Prompt ---
    with open(INSTRUCTIONS_PATH, "r") as f:
        system_prompt = f.read()

    # 3. Register Function from prompt
    chat_function = kernel.add_function(
        function_name="chat",
        plugin_name="ActivityPlugin",
        prompt=system_prompt,
    )

    try:
        # Prepare context
        activities = ACTIVITIES_DB.get(city, [])
        user_input_str = f"The weather in {city} is {weather}. Activities: {json.dumps(activities)}"

        # Invoke with variable binding
        result = await kernel.invoke(
            chat_function,
            KernelArguments(user_message=user_input_str),
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
