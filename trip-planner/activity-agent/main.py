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
load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env")

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
    
    # --- Semantic Kernel Initialization ---
    kernel = sk.Kernel()

    # --- AI Service Configuration ---
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_model = os.getenv("AZURE_OPENAI_MODEL")
    
    if not all([azure_deployment, azure_endpoint, azure_api_key, azure_model]):
        raise ValueError("Azure OpenAI credentials are not fully configured in environment variables.")
    
    ai_service = AzureChatCompletion(
        deployment_name=azure_deployment,
        endpoint=azure_endpoint,
        api_key=azure_api_key,
    )

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

    # --- Create Chat Function ---
    chat_function = kernel.create_function_from_prompt(
        prompt=system_prompt,
        function_name="chat",
    )

    # --- Prepare Kernel Arguments ---
    activities = ACTIVITIES_DB.get(city, [])
    
    # --- Run Kernel ---
    result = await kernel.invoke(
        chat_function,
        user_message=f"The weather in {city} is {weather}. Here is the list of available activities: {json.dumps(activities)}",
    )

    # --- Format and Return Response ---
    try:
        suggested_activities = json.loads(str(result))
    except json.JSONDecodeError:
        suggested_activities = [] # fallback to an empty list if JSON is malformed

    return {
        "jsonrpc": "2.0",
        "result": suggested_activities,
        "id": request.id
    }

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
