import httpx
from mcp.server.fastmcp import FastMCP

from data import CITIES_DB, WMO_CODES

mcp = FastMCP("WeatherMcp")

@mcp.tool()
async def get_available_cities() -> list[str]:
    """Returns a list of available cities."""
    return list(CITIES_DB.keys())

@mcp.tool()
async def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.

    Args:
        city: The name of the city.

    Returns:
        A string describing the current weather.
    """
    if city not in CITIES_DB:
        return f"City '{city}' not found. Available cities: {', '.join(CITIES_DB.keys())}"

    city_data = CITIES_DB[city]
    lat = city_data["lat"]
    lon = city_data["lon"]

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            weather_data = response.json()

            current_weather = weather_data.get("current_weather", {})
            temperature = current_weather.get("temperature")
            weather_code = current_weather.get("weathercode")

            if temperature is None or weather_code is None:
                return "Could not retrieve complete weather information."

            weather_description = WMO_CODES.get(weather_code, "Unknown weather condition")

            return f"The current temperature in {city} is {temperature}°C with {weather_description.lower()}."

        except httpx.HTTPStatusError as e:
            return f"Error fetching weather data: {e.response.status_code}"
        except httpx.RequestError as e:
            return f"Error connecting to the weather service: {e}"


if __name__ == "__main__":
    mcp.run()
