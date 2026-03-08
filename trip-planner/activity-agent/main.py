import json
from pathlib import Path
from fastapi import FastAPI, Response
from pydantic import BaseModel
import uvicorn

from memory import ACTIVITIES_DB

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Pydantic Models ---
class TaskRequest(BaseModel):
    city: str
    weather: str

# --- File Paths ---
BASE_DIR = Path(__file__).parent
AGENT_CARD_PATH = BASE_DIR / "agent_card.json"

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
    city = request.city
    weather = request.weather

    if city not in ACTIVITIES_DB:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": "City not supported"},
            "id": None
        }

    activities = ACTIVITIES_DB[city]
    
    indoor_weather = ['Rain', 'Overcast', 'Fog', 'Drizzle']
    outdoor_weather = ['Clear', 'Sunny']

    if any(w in weather for w in indoor_weather):
        suggested_activities = [act for act in activities if act["type"] == "Indoor"]
    elif any(w in weather for w in outdoor_weather):
        suggested_activities = [act for act in activities if act["type"] == "Outdoor"]
    else:
        suggested_activities = []

    return {
        "jsonrpc": "2.0",
        "result": suggested_activities,
        "id": None
    }

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
