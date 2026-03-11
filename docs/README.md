# Trip Planner Agentico (Semantic Kernel 1.40)

Questo progetto implementa un trip planner multi-agente con orchestrazione automatica tramite Semantic Kernel 1.40.

## Architettura

Componenti principali:

1. `mcp-weather-server`
- Server MCP (FastMCP) per meteo.
- Tool principale: `get_weather(city, date="today")`.
- Supporta richieste da oggi fino a 14 giorni nel futuro.
- Se la data supera la finestra di previsione restituisce: `{"error": "forecast_limit", "message": "I dati meteo sono disponibili solo per i prossimi 14 giorni."}`.

2. `trip-planner/activity-agent`
- Agente A2A (FastAPI, porta `8081`).
- Espone `/.well-known/agent-card.json` e `/task`.
- Suggerisce attivita in base a citta + meteo.
- Usa Structured Outputs con JSON Schema derivato da Pydantic (`ActivityResponse`).
- Valida sempre l'output LLM con `model_validate` prima di rispondere.

3. `trip-planner/restaurant-agent`
- Agente A2A (FastAPI, porta `8082`).
- Espone `/.well-known/agent-card.json` e `/task`.
- Suggerisce ristoranti in base a citta + preferenza cucina.
- Usa Structured Outputs con JSON Schema derivato da Pydantic (`RestaurantResponse`).
- Valida sempre l'output LLM con `model_validate` prima di rispondere.

4. `trip-planner/orchestrator`
- Console app con `ChatCompletionAgent` (SK 1.40).
- Usa `FunctionChoiceBehavior.Auto(auto_invoke=True)`.
- Usa MCP per il meteo + DiscoveryPlugin per chiamare ActivityAgent e RestaurantAgent.
- Usa Structured Outputs con contratto finale `TripDirectorResponse`.
- La risposta finale contiene sempre: `weather_data`, `activity_suggestions`, `restaurant_recommendations` (piu `note` opzionale).

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
2. Orchestrator chiama `WeatherMcp` per ottenere il meteo.
3. Orchestrator chiama `ActivityAgent` via discovery A2A (`8081`).
4. Orchestrator chiama `RestaurantAgent` via discovery A2A (`8082`).
5. Orchestrator produce JSON strutturato (`TripDirectorResponse`) e lo presenta in formato leggibile.

## Contratti e validazione

- I due agenti A2A usano `response_format={"type":"json_schema","json_schema":...}`.
- Gli schema sono normalizzati per compatibilita Azure/OpenAI strict (`required` include tutte le chiavi in `properties`).
- Se il deployment non supporta `json_schema`, gli agenti effettuano fallback a chiamata senza schema ma mantengono validazione Pydantic post-invoke.
- Nessun parsing markdown: gli output LLM vengono decodificati con `json.loads(...)` direttamente.
- In caso di output non valido, gli agenti restituiscono errore JSON-RPC strutturato (non crashano).

## Discovery Plugin

- `DiscoveryPlugin` risolve dinamicamente gli endpoint leggendo le `agent-card`.
- `_post_task` valida envelope JSON-RPC (`jsonrpc`, `result`/`error`) e schema del `result`.
- Se la risposta agente non rispetta lo schema, il plugin restituisce un errore strutturato (`..._invalid_result_schema`).
- I metodi tool ritornano sempre stringhe JSON serializzate verso l'orchestrator.

## Note operative

- Se il meteo fallisce o supera la finestra supportata, l'orchestrator prosegue con `weather='Sconosciuto'` per ActivityAgent.
- Se un agente A2A non e raggiungibile o restituisce payload non valido, il plugin restituisce un errore JSON strutturato.
- L'orchestrator comunica con gli agenti solo tramite `DiscoveryPlugin`.
