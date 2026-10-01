"""Expose keyless Open-Meteo current conditions as a FastMCP tool."""

import sys
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout
from fastmcp import FastMCP


CURRENT_DIR = Path(__file__).resolve().parent
CH11_DIR = CURRENT_DIR.parent
if str(CH11_DIR) not in sys.path:
    sys.path.insert(0, str(CH11_DIR))

from env_config import load_env  # noqa: E402


load_env()

mcp = FastMCP("mcp-open-meteo")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


async def _get_json(
    session: ClientSession,
    url: str,
    params: dict[str, str | int | float],
) -> dict[str, Any]:
    """Fetch one JSON object and turn upstream failures into readable errors."""

    async with session.get(url, params=params) as response:
        data = await response.json(content_type=None)
        if response.status != 200:
            reason = data.get("reason") if isinstance(data, dict) else str(data)
            raise RuntimeError(f"Open-Meteo returned HTTP {response.status}: {reason}")
        if not isinstance(data, dict):
            raise RuntimeError("Open-Meteo returned an unexpected response.")
        return data


async def _find_location(
    session: ClientSession,
    location: str,
) -> dict[str, Any]:
    """Resolve a place name, retrying without comma-separated qualifiers."""

    candidates = [location]
    short_name = location.split(",", maxsplit=1)[0].strip()
    if short_name and short_name != location:
        candidates.append(short_name)

    for candidate in candidates:
        data = await _get_json(
            session,
            GEOCODING_URL,
            {
                "name": candidate,
                "count": 5,
                "language": "en",
                "format": "json",
            },
        )
        results = data.get("results", [])
        if results:
            return results[0]

    raise ValueError(f"Location not found: {location}")


@mcp.tool(description="Get current weather conditions for a location without an API key.")
async def get_weather_conditions(location: str) -> dict[str, Any]:
    """Resolve a location and return its current Open-Meteo weather conditions."""

    location = location.strip()
    if not location:
        raise ValueError("location must not be empty")

    timeout = ClientTimeout(total=15)
    async with ClientSession(timeout=timeout) as session:
        place = await _find_location(session, location)
        weather = await _get_json(
            session,
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,rain,weather_code,cloud_cover,wind_speed_10m,"
                    "wind_direction_10m,is_day"
                ),
                "timezone": "auto",
            },
        )

    current = weather.get("current", {})
    units = weather.get("current_units", {})
    weather_code = current.get("weather_code")

    return {
        "location": {
            "name": place.get("name"),
            "admin1": place.get("admin1"),
            "country": place.get("country"),
            "country_code": place.get("country_code"),
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
            "timezone": weather.get("timezone"),
        },
        "current_conditions": {
            "observation_time": current.get("time"),
            "temperature": {
                "value": current.get("temperature_2m"),
                "unit": units.get("temperature_2m"),
            },
            "apparent_temperature": {
                "value": current.get("apparent_temperature"),
                "unit": units.get("apparent_temperature"),
            },
            "relative_humidity": {
                "value": current.get("relative_humidity_2m"),
                "unit": units.get("relative_humidity_2m"),
            },
            "precipitation": {
                "value": current.get("precipitation"),
                "unit": units.get("precipitation"),
            },
            "rain": {
                "value": current.get("rain"),
                "unit": units.get("rain"),
            },
            "cloud_cover": {
                "value": current.get("cloud_cover"),
                "unit": units.get("cloud_cover"),
            },
            "wind_speed": {
                "value": current.get("wind_speed_10m"),
                "unit": units.get("wind_speed_10m"),
            },
            "wind_direction": {
                "value": current.get("wind_direction_10m"),
                "unit": units.get("wind_direction_10m"),
            },
            "is_day": bool(current.get("is_day")),
            "weather_code": weather_code,
            "weather_text": WEATHER_CODES.get(weather_code, "Unknown"),
        },
        "attribution": "Weather data by Open-Meteo.com",
    }


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8021,
        path="/weather-mcp-server",
    )
