# Flight upgrade graph workflow (ADK 2)

This example implements the flight-upgrade workflow illustrated in the
[official ADK graph documentation](https://adk.dev/graphs/). The documentation
contains the design but not its complete implementation, so this project turns
the diagram into a runnable ADK 2 `Workflow`.

The graph combines every node type shown in the figure:

- `[Function] Check Miles`
- `[Human Input] Get Consent`
- `[Tool] Fetch Flight History`
- `[LLM] Analyze History`
- `[Function] Process Upgrade`
- terminal `Deny Upgrade` and `Issue Upgrade` nodes

## Routing policy

The source figure leaves the exact threshold values undefined. This example
makes the boundaries total and testable:

| Miles | Route |
| --- | --- |
| `< 5,000` | Deny the upgrade |
| `5,000` through `20,000` | Ask for consent, fetch history, and ask the LLM to analyse it |
| `> 20,000` | Process the upgrade directly |

The history service is intentionally simulated, while the analysis step uses a
real local Ollama model. No OpenAI or Google API key is required.

## Isolated ADK 2 environment

The sibling `ch12/adk` example currently uses ADK 1.x. Create a separate
environment for this graph example from the repository root:

```bash
python3 -m venv ch12/env_ch12_adk2
ch12/env_ch12_adk2/bin/python -m pip install -r ch12/adk_graph_upgrade/requirements.txt
```

Start Ollama and make sure the course-default model is available:

```bash
ollama serve
ollama pull gemma4:e4b
```

`OLLAMA_MODEL` and `OLLAMA_BASE_URL` can be set in the project-root `.env`.
Their defaults are `gemma4:e4b` and `http://127.0.0.1:11434`.

## Run

Terminal mode:

```bash
ch12/env_ch12_adk2/bin/adk run ch12/adk_graph_upgrade
```

For the middle route, the workflow pauses after the first command and prints a
session ID. Resume that same session with the consent reply:

```bash
ch12/env_ch12_adk2/bin/adk run ch12/adk_graph_upgrade "12000"
ch12/env_ch12_adk2/bin/adk run --session_id <SESSION_ID> ch12/adk_graph_upgrade "yes"
```

Do not add `--in_memory` to this two-command flow: the session must persist so
ADK can resume the human-input interruption.

Browser mode can point directly to this one agent folder, avoiding the sibling
ADK 1.x project:

```bash
ch12/env_ch12_adk2/bin/adk web --no_use_local_storage ch12/adk_graph_upgrade
```

Open <http://127.0.0.1:8000> and try each route with `4000`, `12000`, and
`25000`. The middle route pauses for a `yes` or `no` reply.

## Offline checks

The deterministic threshold logic and graph construction can be checked
without starting Ollama:

```bash
ch12/env_ch12_adk2/bin/python -m unittest ch12.adk_graph_upgrade.test_agent
```

The LLM route itself requires the local Ollama service.
