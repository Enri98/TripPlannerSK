from datetime import date as dt_date, timedelta

import httpx
from mcp.server.fastmcp import FastMCP

from data import CITIES_DB, WMO_CODES

mcp = FastMCP("WeatherMcp")


@mcp.tool()
async def get_available_cities() -> list[str]:
    """Returns a list of available cities."""
    return list(CITIES_DB.keys())


@mcp.tool()
async def get_weather(city: str, date: str = "today") -> str:
    """
    Get weather for a given city.
    Supports forecast requests for today and future dates up to 14 days ahead.

    Args:
        city: The name of the city.
        date: Requested date in YYYY-MM-DD format or "today". Defaults to "today".

    Returns:
        A string describing weather conditions.
    """
    if city not in CITIES_DB:
        return f"City '{city}' not found. Available cities: {', '.join(CITIES_DB.keys())}"

    today = dt_date.today()
    requested_date = today
    if date and date.lower() != "today":
        try:
            requested_date = dt_date.fromisoformat(date)
        except ValueError:
            return "Error: Invalid date format. Use YYYY-MM-DD or 'today'."
    if requested_date > today + timedelta(days=14):
        return '{"error": "forecast_limit", "message": "Weather data is only available for the next 14 days."}'

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
                return "Could not retrieve complete weather information."

            temperature = temperatures[0]
            weather_code = weather_codes[0]
            weather_description = WMO_CODES.get(weather_code, "Unknown weather condition")

            return (
                f"The forecast for {city} on {requested_date_str} is a high of "
                f"{temperature} C with {weather_description.lower()}."
            )

        except httpx.HTTPStatusError as exc:
            return f"Error fetching weather data: {exc.response.status_code}"
        except httpx.RequestError as exc:
            return f"Error connecting to the weather service: {exc}"


if __name__ == "__main__":
    mcp.run()
