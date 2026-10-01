# AccuWeather MCP server with FastMCP

This example exposes one AccuWeather function as an MCP tool using FastMCP and
the Streamable HTTP transport. You can call it from the included Python client,
inspect it interactively with MCP Inspector, or use it from `main_07_01.py`.

## What the example exposes

`accuweather_mcp.py` creates the server and registers the tool with the
`@mcp.tool` decorator:

```python
mcp = FastMCP("mcp-accuweather")

@mcp.tool(description="Get weather conditions for a location.")
async def get_weather_conditions(location: str) -> dict:
    ...
```

The server listens at:

```text
http://127.0.0.1:8020/accu-mcp-server
```

Its transport is `streamable-http`, not the legacy SSE transport.

## 1. Install the Python environment

Run the commands from the repository root on Ubuntu or WSL:

```bash
python3 -m venv ch11/.venv
source ch11/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ch11/requirements.txt
```

The Pydantic constraint in `requirements.txt` is intentional: FastMCP 2.10.5
uses behavior that changed in Pydantic 2.12.

## 2. Configure AccuWeather

Create an AccuWeather developer API key and add it to the project-root `.env`:

```dotenv
ACCUWEATHER_API_KEY=replace-with-your-key
```

The server loads this root file through `env_config.load_env()`. Do not put the
key in source code or commit `.env`.

## 3. Start the FastMCP server

The simplest method runs the Python entrypoint:

```bash
ch11/.venv/bin/python ch11/mcp/accuweather_mcp.py
```

The startup banner should report:

```text
Transport:  Streamable-HTTP
Server URL: http://127.0.0.1:8020/accu-mcp-server
```

Alternatively, start the same `mcp` object through the FastMCP CLI:

```bash
ch11/.venv/bin/fastmcp run ch11/mcp/accuweather_mcp.py:mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8020 \
  --path /accu-mcp-server
```

Keep this terminal running.

## 4. Test it with the FastMCP Python client

In a second terminal, from the repository root:

```bash
ch11/.venv/bin/python ch11/mcp/test_accuweather_mcp.py
```

The client connects, lists `get_weather_conditions`, and calls it for
`Penzance, UK`. A `401` or `403` response from AccuWeather means that the MCP
connection works but the external API key is missing, expired, or not enabled.

## 5. Inspect it with MCP Inspector

MCP Inspector requires Node.js and `npx`. Verify them first:

```bash
node --version
npx --version
```

With the Python server still running, launch Inspector from another terminal:

```bash
npx --yes @modelcontextprotocol/inspector
```

In the Inspector interface configure:

- **Transport Type:** `Streamable HTTP`
- **URL:** `http://127.0.0.1:8020/accu-mcp-server`
- **Connection Type:** `Via Proxy`
- **Authentication:** disabled

Select **Connect**, open **Tools**, and list the available tools. Select
`get_weather_conditions` and invoke it with:

```json
{
  "location": "Penzance, UK"
}
```

Inspector should show both the MCP request and the structured result returned
by the FastMCP server.

## 6. Use the server from the chapter agent

Keep the MCP server running, then start the chapter 11 MCP agent in another
terminal:

```bash
ch11/.venv/bin/python ch11/main_07_01.py
```

`main_07_01.py` uses `MultiServerMCPClient` to discover the remote AccuWeather
tool and combines it with the local `search_travel_info` RAG tool.

Example prompt:

> Suggest a destination in Cornwall and check the current weather there.

## Troubleshooting

- **Connection refused:** confirm that the server terminal is still running and
  that both client and Inspector use port `8020` and path
  `/accu-mcp-server`.
- **404 from Inspector:** select `Streamable HTTP`; do not append `/sse` or
  `/mcp` to the documented URL.
- **AccuWeather 401/403:** replace `ACCUWEATHER_API_KEY` in the root `.env` with
  an active key, then restart the server.
- **Address already in use:** stop the process already listening on port 8020
  before starting another server instance.
- **FastMCP fails while importing Pydantic settings:** recreate the virtual
  environment and reinstall the pinned `ch11/requirements.txt`.

Stop the server with `Ctrl+C` in its terminal.
