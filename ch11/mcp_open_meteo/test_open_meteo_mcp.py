"""End-to-end client for the Open-Meteo FastMCP server."""

import asyncio

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


transport = StreamableHttpTransport(
    url="http://127.0.0.1:8021/weather-mcp-server"
)
client = Client(transport)


async def main() -> None:
    async with client:
        print(f"Client connected: {client.is_connected()}")
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        if not any(tool.name == "get_weather_conditions" for tool in tools):
            raise RuntimeError("get_weather_conditions was not exposed by the server")

        result = await client.call_tool(
            "get_weather_conditions",
            {"location": "Penzance, UK"},
        )
        print(f"Structured result: {result.structured_content}")

    print(f"Client connected: {client.is_connected()}")


if __name__ == "__main__":
    asyncio.run(main())
