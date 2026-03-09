from datetime import date as dt_date

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
    Use this only for today's weather.

    Args:
        city: The name of the city.
        date: Requested date in YYYY-MM-DD format or "today". Defaults to "today".

    Returns:
        A string describing weather conditions.
    """
    if city not in CITIES_DB:
        return f"City '{city}' not found. Available cities: {', '.join(CITIES_DB.keys())}"

    today = dt_date.today()
    if date and date.lower() != "today":
        try:
            requested_date = dt_date.fromisoformat(date)
        except ValueError:
            return "Error: Invalid date format. Use YYYY-MM-DD or 'today'."
        if requested_date > today:
            return "Error: I can only provide weather for today."

    city_data = CITIES_DB[city]
    lat = city_data["lat"]
    lon = city_data["lon"]

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            weather_data = response.json()

            current_weather = weather_data.get("current_weather", {})
            temperature = current_weather.get("temperature")
            weather_code = current_weather.get("weathercode")

            if temperature is None or weather_code is None:
                return "Could not retrieve complete weather information."

            weather_description = WMO_CODES.get(weather_code, "Unknown weather condition")

            return f"The current temperature in {city} is {temperature} C with {weather_description.lower()}."

        except httpx.HTTPStatusError as exc:
            return f"Error fetching weather data: {exc.response.status_code}"
        except httpx.RequestError as exc:
            return f"Error connecting to the weather service: {exc}"


if __name__ == "__main__":
    mcp.run()
