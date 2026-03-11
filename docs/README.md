# Trip Planner Agentico

Trip planner multi-agente con:
- orchestratore Semantic Kernel
- due agenti A2A (attivita e ristoranti)
- tool meteo MCP (FastMCP + Open-Meteo)
- interfaccia utente console (`trip-planner/console_app.py`)

## Struttura progetto

- `trip-planner/console_app.py`
  - Entry point CLI reale.
  - Inizializza orchestratore + `DiscoveryPlugin` + plugin MCP meteo.
  - Mostra output in console con `rich`
- `trip-planner/orchestrator/main.py`
  - Modulo di orchestrazione
  - Costruisce kernel, agent orchestratore, execution settings e plugin MCP.
- `trip-planner/orchestrator/plugins/discovery_plugin.py`
  - Discovery A2A via `/.well-known/agent-card.json`.
  - Chiamata `/task` JSON-RPC e validazione envelope/schema risultato.
- `trip-planner/activity-agent/`
  - FastAPI su porta `8081`.
  - Endpoint: `/.well-known/agent-card.json`, `/task`.
  - Tool locale: `ActivitySearch.get_activities(city, weather)`.
- `trip-planner/restaurant-agent/`
  - FastAPI su porta `8082`.
  - Endpoint: `/.well-known/agent-card.json`, `/task`.
  - Tool locale: `RestaurantSearch.get_restaurants(city, cuisine)`.
- `trip-planner/data_contracts.py`
  - Contratti Pydantic condivisi (RPC + payload agenti + output finale).
- `trip-planner/helpers.py`
  - Normalizzazione JSON Schema strict + utility error handling.
- `mcp-weather-server/server.py`
  - Server MCP con tool `get_available_cities` e `get_weather`.
  - `get_weather` supporta `today` o `YYYY-MM-DD` fino a +14 giorni.


## Prerequisiti

- Python 3.11+
- Virtual environment in `trip-planner/.venv`
- Virtual environment in `mcp-weather-server/.venv`
- Variabili in `.env` (root progetto):
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_DEPLOYMENT`
  - `AZURE_OPENAI_MODEL`
  - `API_VERSION`

File di esempio: `.env.example`.

## Installazione dipendenze

Dalla root del progetto:

```powershell
trip-planner\.venv\Scripts\python.exe -m pip install -r .\trip-planner\requirements.txt

mcp-weather-server\.venv\Scripts\python.exe -m pip install -r .\mcp-weather-server\requirements.txt
```

## Avvio

Apri 3 terminali dalla root

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

### Terminale 3 - Console App (orchestratore + MCP)

```powershell
cd trip-planner
.\.venv\Scripts\python.exe .\console_app.py
```

Nota: non serve avviare `mcp-weather-server/server.py` a parte quando usi `console_app.py`; viene avviato come processo stdio tramite `MCPStdioPlugin`.

## Flusso

1. Utente inserisce destinazione in console.
2. L'orchestratore invoca il tool MCP meteo (`WeatherMcp`).
3. Chiama ActivityAgent e RestaurantAgent tramite `DiscoveryPlugin`.
4. Valida output finale con `TripDirectorResponse`.
5. Renderizza meteo, nota ed elenchi in tabelle `rich`.

## fallback

- Structured Outputs: usati dove possibile con schema Pydantic normalizzato.
- Fallback automatico se `response_format=json_schema` non e supportato dal deployment
- Validazione post-LLM con Pydantic.
- Errori agente/tool conservati in forma strutturata (`error`)
- Se il meteo non e disponibile, il flusso prosegue con `weather='Sconosciuto'`.

## Citta supportate (dataset locale)

- `Roma`
- `Milano`
- `Venezia`
- `Firenze`
- `Napoli`