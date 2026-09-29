"""Small, independent ADK function tools for search and weather."""

from datetime import date

import httpx
from ddgs import DDGS

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 10.0


def search_day_trip_options(query: str) -> dict:
    """Search DuckDuckGo for day-trip transport, prices, hours and attractions.

    Args:
        query: Specific search terms including the place and relevant date.

    Returns:
        Search results with title, URL and snippet, or an error message.
    """
    if not query.strip():
        return {"status": "error", "message": "Search query cannot be empty."}
    try:
        with DDGS(timeout=TIMEOUT_SECONDS) as search:
            raw = list(search.text(query, max_results=5))
    except Exception as exc:
        return {"status": "error", "message": f"DuckDuckGo search unavailable: {type(exc).__name__}"}

    results = [
        {"title": item.get("title", ""), "url": item.get("href", ""), "snippet": item.get("body", "")}
        for item in raw
        if item.get("href")
    ]
    return {"status": "ok" if results else "no_results", "query": query, "results": results}


def get_weather_forecast(city: str, trip_date: str) -> dict:
    """Get the free Open-Meteo daily forecast for a city and trip date.

    Args:
        city: Destination city, preferably including country (e.g. "Bologna, Italy").
        trip_date: Date of the day trip in YYYY-MM-DD format.

    Returns:
        Dated weather data and source URL, or a clear error/availability message.
    """
    try:
        requested_date = date.fromisoformat(trip_date)
    except ValueError:
        return {"status": "error", "message": "trip_date must be YYYY-MM-DD."}

    if requested_date < date.today():
        return {"status": "unavailable", "message": "The trip date is in the past; this tool supplies forecasts only."}
    if not city.strip():
        return {"status": "error", "message": "City cannot be empty."}

    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            geo_response = client.get(
                GEOCODING_URL,
                params={"name": city.strip(), "count": 1, "language": "en"},
            )
            geo_response.raise_for_status()
            places = geo_response.json().get("results") or []
            if not places:
                return {"status": "unavailable", "message": f"City not found: {city}"}

            place = places[0]
            forecast_response = client.get(
                FORECAST_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,wind_speed_10m_max",
                    "timezone": place.get("timezone", "auto"),
                    "forecast_days": 16,
                },
            )
            forecast_response.raise_for_status()
            forecast = forecast_response.json()
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        return {"status": "error", "message": f"Open-Meteo unavailable: {type(exc).__name__}"}

    daily = forecast.get("daily") or {}
    days = daily.get("time") or []
    if trip_date not in days:
        return {
            "status": "unavailable",
            "message": "No forecast for that date; Open-Meteo provides at most 16 days ahead.",
            "available_dates": [days[0], days[-1]] if days else [],
        }

    index = days.index(trip_date)
    fields = (
        "weather_code", "temperature_2m_max", "temperature_2m_min",
        "precipitation_probability_max", "precipitation_sum", "wind_speed_10m_max",
    )
    values = {
        field: (daily.get(field) or [])[index] if len(daily.get(field) or []) > index else None
        for field in fields
    }
    return {
        "status": "ok",
        "location": f"{place['name']}, {place.get('country', '')}",
        "date": trip_date,
        "timezone": forecast.get("timezone", place.get("timezone")),
        **values,
        "source": "https://open-meteo.com/en/docs",
    }
