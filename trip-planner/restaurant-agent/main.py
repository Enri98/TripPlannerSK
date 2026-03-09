import json
from pathlib import Path
from typing import Optional

from anyio import Path as AnyioPath
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from pydantic import BaseModel
import uvicorn

from memory import RESTAURANT_DB

app = FastAPI()


class TaskRequestParams(BaseModel):
    city: str
    cuisine_type: str


class TaskRequest(BaseModel):
    jsonrpc: str
    method: str
    params: TaskRequestParams
    id: Optional[int] = None


BASE_DIR = Path(__file__).parent
AGENT_CARD_PATH = BASE_DIR / "agent_card.json"
INSTRUCTIONS_PATH = BASE_DIR / "instructions.md"
AGENT_CARD_CONTENT: dict = {}
SYSTEM_INSTRUCTIONS: str = ""

load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env", override=True)


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
    return Response(status_code=404, content="Agent card not found.")


@app.post("/task")
async def suggest_restaurant(request: TaskRequest):
    city = request.params.city
    cuisine_type = request.params.cuisine_type

    if city not in RESTAURANT_DB:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": "City not supported"},
            "id": request.id,
        }

    restaurants = RESTAURANT_DB.get(city, [])
    if cuisine_type and cuisine_type.lower() not in {"any", "all"}:
        selected = [
            item
            for item in restaurants
            if cuisine_type.lower() in item.get("cuisine_type", "").lower()
            or cuisine_type.lower() in item.get("type", "").lower()
        ]
    else:
        selected = restaurants

    if not selected:
        selected = restaurants

    return {
        "jsonrpc": "2.0",
        "result": selected,
        "id": request.id,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)