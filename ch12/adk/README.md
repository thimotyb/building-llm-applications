# Budget day-trip advisor (ADK + LiteLLM)

This small ADK agent plans an inexpensive one-day trip. DeepSeek V4 or a local
Ollama model handles the conversation and tool decisions through ADK's LiteLLM
connector. DuckDuckGo provides web results with links and snippets, as in the
`ch04` search example.
Open-Meteo supplies a free weather forecast without another API key.

## Setup

Use Python 3.10+ from the repository root:

```bash
python3 -m venv ch12/env_ch12
ch12/env_ch12/bin/python -m pip install -r ch12/adk/requirements.txt
```

The agent loads the existing **root** `.env`. Select one of these values there:

```dotenv
# Choose exactly one; keep the other existing variables in the same root file.
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma4:e2b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Switch `LLM_PROVIDER` back to `deepseek` to use `DEEPSEEK_MODEL` and
`DEEPSEEK_API_KEY` already in that file. Restart `adk run` after changing it.

| `LLM_PROVIDER` | Required configuration | LiteLLM route |
| --- | --- | --- |
| `deepseek` | `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` (V4) | `deepseek/<DEEPSEEK_MODEL>` |
| `ollama` | `OLLAMA_MODEL` (for example `gemma4:e2b`) | `ollama_chat/<OLLAMA_MODEL>` |

`DEEPSEEK_BASE_URL` and `DEEPSEEK_THINKING` are optional for DeepSeek.
`OLLAMA_BASE_URL` defaults to `http://127.0.0.1:11434`; the agent passes it to
LiteLLM and sets `OLLAMA_API_BASE` for LiteLLM's model inspection. For Ollama,
start the local server and pull the selected model before running ADK:

```bash
ollama serve
ollama pull gemma4:e2b
ollama show gemma4:e2b
```

The `ollama show` output should list `tools` under Capabilities. Neither
provider needs a Google API key. This sample uses LiteLLM's Python connector
directly; a LiteLLM Proxy is not required.

## Run

```bash
ch12/env_ch12/bin/adk run ch12/adk
```

Example prompt:

> Da Firenze, sabato prossimo, proponimi una gita economica di un giorno in treno,
> massimo 60 euro a persona. Considera il meteo e suggerisci un piano al coperto
> se piove.

For exact forecast lookup, give the date in `YYYY-MM-DD` if the agent asks.
Prices and opening times found in search snippets are indicative: verify them
at the linked operator or venue before booking. Open-Meteo forecasts cover up
to 16 days ahead; the agent reports when a date is outside that window.

## Sources and architecture

- [ADK LiteLLM connector](https://adk.dev/agents/models/litellm/)
- [ADK Ollama connector and `ollama_chat` guidance](https://adk.dev/agents/models/ollama/)
- [LiteLLM DeepSeek provider](https://docs.litellm.ai/docs/providers/deepseek)
- [Open-Meteo forecast API](https://open-meteo.com/en/docs) and [geocoding API](https://open-meteo.com/en/docs/geocoding-api)

`root_agent` (selected model via LiteLLM) calls two ADK function tools:
`search_day_trip_options` (DuckDuckGo) and `get_weather_forecast` (Open-Meteo).
