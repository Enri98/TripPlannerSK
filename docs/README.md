# Documentazione Progetto

Questo progetto è composto da due servizi principali: un server meteo e un pianificatore di viaggi.

## Architettura

L'architettura è basata su microservizi:

1.  **Server Meteo (`mcp-weather-server`)**: Un servizio Python che fornisce dati meteorologici.
2.  **Pianificatore di Viaggi (`trip-planner`)**: Un servizio che include un "Activity Agent" per suggerire attività basate sul meteo.

## Come Eseguire il Progetto

Per eseguire i servizi, segui questi passaggi.

### 1. Eseguire il Server Meteo

Apri un terminale e posizionati nella cartella `mcp-weather-server`:

```bash
cd mcp-weather-server
```

Attiva l'ambiente virtuale:

```bash
# Su Windows
.venv\Scripts\activate
```

Installa le dipendenze:

```bash
pip install -r requirements.txt
```

Avvia il server:

```bash
python server.py
```

### 2. Eseguire il Pianificatore di Viaggi

Apri un secondo terminale e posizionati nella cartella `trip-planner/activity-agent`:

```bash
cd trip-planner\activity-agent
```

Attiva l'ambiente virtuale (condiviso con il trip-planner):

```bash
# Su Windows
..\.venv\Scripts\activate
```

Installa le dipendenze:

```bash
pip install -r ..equirements.txt
```

Avvia il server dell'agente:

```bash
python main.py
```
