# Trip Planner Agentico

Trip planner multi-agente basato su Semantic Kernel `1.40` con orchestrazione a due fasi:
- `PlannerAgent` (con tool meteo MCP)
- `SynthesizerAgent` (solo testo, nessun tool)
- due agenti A2A specializzati (`ActivityAgent`, `RestaurantAgent`)
- interfaccia console (`trip-planner/console_app.py`)

La comunicazione A2A e in linguaggio naturale: i worker ricevono solo `question`.

## Architettura attuale

### 1. Planner
- Estrae la citta dalla richiesta utente.
- Chiama `WeatherMcp.get_weather`.
- Produce JSON strutturato interno con:
  - `weather_context`
  - `activity_question`
  - `restaurant_question`

### 2. Worker A2A
- `DiscoveryPlugin` risolve gli endpoint via `/.well-known/agent-card.json`.
- Invia JSON-RPC ai worker su `/task` con payload:
  - `{"params": {"question": "..."} }`
- I worker rispondono con:
  - `{"result": {"reply": "testo naturale"}}`

### 3. Synthesizer
- Riceve: meteo + reply attività + reply ristoranti.
- Genera risposta finale Markdown per l'utente.

## Struttura progetto

- `trip-planner/console_app.py`
  - Entry point CLI.
  - Esegue il flusso Planner -> Worker -> Synthesizer.
  - Render finale con `rich.Markdown`.
- `trip-planner/orchestrator/main.py`
  - Definisce prompt, modelli e factory per:
    - `PlannerAgent`
    - `SynthesizerAgent`
    - kernel e settings di esecuzione.
- `trip-planner/orchestrator/plugins/discovery_plugin.py`
  - Discovery A2A e chiamate JSON-RPC a `ActivityAgent` e `RestaurantAgent`.
  - Validazione envelope RPC e `result.reply`.
- `trip-planner/activity-agent/`
  - FastAPI su porta `8081`.
  - Endpoint: `/.well-known/agent-card.json`, `/task`.
  - Input `/task`: `question`.
  - Tool locale: `ActivitySearch.get_activities(city, weather)`.
- `trip-planner/restaurant-agent/`
  - FastAPI su porta `8082`.
  - Endpoint: `/.well-known/agent-card.json`, `/task`.
  - Input `/task`: `question`.
  - Tool locale: `RestaurantSearch.get_restaurants(city, cuisine, budget)`.
- `trip-planner/data_contracts.py`
  - Contratti condivisi RPC.
  - `TaskRequestParams` con solo `question: str`.
- `trip-planner/helpers.py`
  - Utility Structured Outputs e gestione errori.
- `mcp-weather-server/server.py`
  - Server MCP con `get_available_cities` e `get_weather`.

## Prerequisiti

- Python 3.11+
- Virtual environment in `trip-planner/.venv`
- Virtual environment in `mcp-weather-server/.venv`
- Variabili `.env` nella root:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_DEPLOYMENT`
  - `AZURE_OPENAI_MODEL`
  - `API_VERSION`

## Installazione dipendenze

Dalla root:

```powershell
trip-planner\.venv\Scripts\python.exe -m pip install -r .\trip-planner\requirements.txt
mcp-weather-server\.venv\Scripts\python.exe -m pip install -r .\mcp-weather-server\requirements.txt
```

## Avvio

Apri 3 terminali dalla root.

### Terminale 1 - Activity Agent

```powershell
cd trip-planner\activity-agent
..\.venv\Scripts\python.exe .\main.py
```

### Terminale 2 - Restaurant Agent

```powershell
cd trip-planner\restaurant-agent
..\.venv\Scripts\python.exe .\main.py
```

### Terminale 3 - Console App

```powershell
cd trip-planner
.\.venv\Scripts\python.exe .\console_app.py
```

Nota: il server MCP meteo non va avviato separatamente; viene avviato via `MCPStdioPlugin`.

## Flusso runtime

1. L'utente inserisce una richiesta libera in console.
2. Il `PlannerAgent` usa il tool meteo e produce:
   - `weather_context`
   - `activity_question`
   - `restaurant_question`
3. `DiscoveryPlugin` invia le due domande testuali agli agenti A2A.
4. Il `SynthesizerAgent` unisce tutto e restituisce il piano in Markdown.

## Citta supportate (dataset locale)

- `Roma`
- `Milano`
- `Venezia`
- `Firenze`
- `Napoli`
