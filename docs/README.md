# Trip Planner Agentico (Semantic Kernel 1.40)

Questo progetto implementa un trip planner multi-agente con orchestrazione automatica tramite Semantic Kernel 1.40.

## Architettura

Componenti principali:

1. `mcp-weather-server`
- Server MCP (FastMCP) per meteo.
- Tool principale: `get_weather(city, date="today")`.
- Vincolo temporale: per date future restituisce `Error: I can only provide weather for today.`.

2. `trip-planner/activity-agent`
- Agente A2A (FastAPI, porta `8081`).
- Espone `/.well-known/agent-card.json` e `/task`.
- Suggerisce attivita in base a citta + meteo.

3. `trip-planner/restaurant-agent`
- Agente A2A (FastAPI, porta `8082`).
- Espone `/.well-known/agent-card.json` e `/task`.
- Suggerisce ristoranti in base a citta + preferenza cucina.

4. `trip-planner/orchestrator`
- Console app con `ChatCompletionAgent` (SK 1.40).
- Usa `FunctionChoiceBehavior.Auto(auto_invoke=True)`.
- Usa MCP per il meteo + DiscoveryPlugin per chiamare ActivityAgent e RestaurantAgent.

## Prerequisiti

- Python 3.11+
- Virtual environment condiviso in `trip-planner/.venv`
- Variabili in `.env` (root progetto):
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_DEPLOYMENT`
  - `AZURE_OPENAI_MODEL`
  - `API_VERSION`

## Avvio rapido (Windows / PowerShell)

Apri 4 terminali dalla root del progetto `ESAME/`.

### Terminale 1: Weather MCP Server

```powershell
cd mcp-weather-server
..\trip-planner\.venv\Scripts\python.exe .\server.py
```

### Terminale 2: Activity Agent

```powershell
cd trip-planner\activity-agent
..\.venv\Scripts\python.exe .\main.py
```

### Terminale 3: Restaurant Agent

```powershell
cd trip-planner\restaurant-agent
..\.venv\Scripts\python.exe .\main.py
```

### Terminale 4: Orchestrator Console

```powershell
cd trip-planner\orchestrator
..\.venv\Scripts\python.exe .\main.py
```

## Flusso logico

1. L'utente inserisce richiesta viaggio in console.
2. Orchestrator chiama `WeatherMcp`.
3. Orchestrator chiama `ActivityAgent` via discovery A2A (`8081`).
4. Orchestrator chiama `RestaurantAgent` via discovery A2A (`8082`).
5. Orchestrator produce JSON strutturato e lo presenta in formato leggibile.

## Note operative

- Il parser UI dell'orchestrator gestisce anche risposte JSON dentro blocchi Markdown (```json ... ```).
- Se un agente A2A non e raggiungibile, il plugin restituisce un errore JSON strutturato.
- L'orchestrator comunica con gli agenti solo tramite `DiscoveryPlugin`.