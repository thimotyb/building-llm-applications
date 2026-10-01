# Open-Meteo MCP server with FastMCP

This alternative to the AccuWeather example provides current weather through
Open-Meteo. It requires no account, trial, API key, or secret in `.env`.

The original AccuWeather server remains available in `ch11/mcp` for comparison.

## Why Open-Meteo

The example uses two public endpoints:

- the [Open-Meteo Geocoding API](https://open-meteo.com/en/docs/geocoding-api)
  to convert a place name to coordinates;
- the [Open-Meteo Forecast API](https://open-meteo.com/en/docs) to retrieve
  current conditions.

Review the [Open-Meteo terms](https://open-meteo.com/en/terms) before using the
service outside this educational example. The MCP response includes attribution
to Open-Meteo.

## 1. Install the environment

From the repository root on Ubuntu or WSL:

```bash
python3 -m venv ch11/.venv
source ch11/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ch11/requirements.txt
```

No provider-specific environment variable is required.

## 2. Start the server

```bash
ch11/.venv/bin/python ch11/mcp_open_meteo/open_meteo_mcp.py
```

The server uses Streamable HTTP at:

```text
http://127.0.0.1:8021/weather-mcp-server
```

The separate port lets this server run alongside the AccuWeather server on
port 8020.

You can alternatively use the FastMCP CLI:

```bash
ch11/.venv/bin/fastmcp run ch11/mcp_open_meteo/open_meteo_mcp.py:mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8021 \
  --path /weather-mcp-server
```

## 3. Run the included client

Keep the server running and execute this in a second terminal:

```bash
ch11/.venv/bin/python ch11/mcp_open_meteo/test_open_meteo_mcp.py
```

The client discovers `get_weather_conditions` and calls it for `Penzance, UK`.
The server accepts comma-qualified names and retries the base place name when
the geocoding endpoint cannot resolve the full string.

## 4. Use MCP Inspector

Start Inspector in another terminal:

```bash
npx --yes @modelcontextprotocol/inspector
```

Configure:

- **Transport Type:** `Streamable HTTP`
- **URL:** `http://127.0.0.1:8021/weather-mcp-server`
- **Connection Type:** `Via Proxy`
- **Authentication:** disabled

Connect, open **Tools**, select `get_weather_conditions`, and invoke it with:

```json
{
  "location": "Penzance, UK"
}
```

## Response shape

The tool returns:

- the resolved place, country, coordinates, and timezone;
- temperature and apparent temperature;
- relative humidity, rain, and total precipitation;
- cloud cover, wind speed, and wind direction;
- the WMO weather code and a readable description;
- Open-Meteo attribution.

## Troubleshooting

- **Connection refused:** keep the server running and use port 8021.
- **404 in Inspector:** use the exact `/weather-mcp-server` path and select
  Streamable HTTP.
- **Location not found:** try a less-qualified name such as `Penzance` instead
  of a full address.
- **Upstream timeout:** retry later; both geocoding and weather data come from
  the public Open-Meteo service.

Stop the server with `Ctrl+C`.
