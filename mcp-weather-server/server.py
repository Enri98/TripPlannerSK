from datetime import date as dt_date, timedelta

import httpx
from mcp.server.fastmcp import FastMCP

from data import CITIES_DB, WMO_CODES

mcp = FastMCP("WeatherMcp")


@mcp.tool()
async def get_available_cities() -> list[str]:
    """Restituisce la lista delle citta disponibili."""
    return list(CITIES_DB.keys())


@mcp.tool()
async def get_weather(city: str, date: str = "today") -> str:
    """
    Recupera il meteo per una citta.
    Supporta richieste per oggi e per date future fino a 14 giorni.

    Args:
        city: Nome della citta.
        date: Data richiesta in formato YYYY-MM-DD oppure "today". Default: "today".

    Returns:
        Una stringa descrittiva delle condizioni meteo in italiano.
    """
    if city not in CITIES_DB:
        return f"Citta '{city}' non trovata. Citta disponibili: {', '.join(CITIES_DB.keys())}"

    today = dt_date.today()
    requested_date = today
    if date and date.lower() != "today":
        try:
            requested_date = dt_date.fromisoformat(date)
        except ValueError:
            return "Errore: formato data non valido. Usa YYYY-MM-DD oppure 'today'."
    if requested_date > today + timedelta(days=14):
        return '{"error": "forecast_limit", "message": "I dati meteo sono disponibili solo per i prossimi 14 giorni."}'

    city_data = CITIES_DB[city]
    lat = city_data["lat"]
    lon = city_data["lon"]
    requested_date_str = requested_date.isoformat()

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        "&daily=weather_code,temperature_2m_max"
        "&timezone=auto"
        f"&start_date={requested_date_str}"
        f"&end_date={requested_date_str}"
    )

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            weather_data = response.json()

            daily = weather_data.get("daily", {})
            temperatures = daily.get("temperature_2m_max", [])
            weather_codes = daily.get("weather_code", [])

            if not temperatures or not weather_codes:
                return "Impossibile recuperare informazioni meteo complete."

            temperature = temperatures[0]
            weather_code = weather_codes[0]
            weather_description = WMO_CODES.get(weather_code, "condizioni meteo sconosciute")

            return (
                f"Le previsioni per {city} del {requested_date_str} indicano una massima di "
                f"{temperature} C con {weather_description.lower()}."
            )

        except httpx.HTTPStatusError as exc:
            return f"Errore nel recupero dati meteo: {exc.response.status_code}"
        except httpx.RequestError as exc:
            return f"Errore di connessione al servizio meteo: {exc}"


if __name__ == "__main__":
    mcp.run()
