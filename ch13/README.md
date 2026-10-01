# Chapter 13: ADK agents over A2A

This example separates one agentic application into two independently running
processes:

- `a2a_server` exposes a shipping specialist through the Agent2Agent protocol;
- `a2a_client` discovers that specialist from its Agent Card and delegates work
  through ADK's `RemoteA2aAgent`.

The server owns the shipping tool and its implementation. The client sees only
the remote agent's public A2A contract. Both agents use the course-default local
Ollama runtime, so the local exercise needs no cloud API key.

The A2A protocol and SDK are stable specifications, while ADK 1.39 still marks
its A2A integration classes as experimental. Pin and test the ADK version before
promoting this teaching example to production.

The validated environment used ADK 1.39.1 with `a2a-sdk` 0.3.26; its generated
card advertises A2A protocol version 0.3.0. Check client, server, SDK, and current
specification compatibility together when upgrading either side.

## Architecture

```text
User
  |
  v
ADK shipping_client (port 8000 with adk web)
  |
  | A2A: Agent Card discovery + task/message exchange
  v
ADK shipping_specialist (Uvicorn on port 8001)
  |
  v
estimate_delivery function tool
```

The server is wrapped by `to_a2a()`. ADK creates the A2A request handling,
in-memory task services, Starlette application, and generated Agent Card. The
client reads the card at `/.well-known/agent-card.json` and represents the
remote service as an ADK sub-agent.

## Setup

Run all commands from the repository root:

```bash
python3 -m venv ch13/.venv
ch13/.venv/bin/python -m pip install --upgrade pip
ch13/.venv/bin/python -m pip install -r ch13/requirements.txt
```

Start Ollama and ensure the default model is installed:

```bash
ollama serve
ollama pull gemma4:e4b
ollama show gemma4:e4b
```

The model can be changed in the existing project-root `.env`:

```dotenv
OLLAMA_MODEL=gemma4:e4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

## 1. Start the A2A server

In the first terminal, from the repository root:

```bash
ch13/.venv/bin/uvicorn ch13.a2a_server.agent:a2a_app \
  --host 127.0.0.1 --port 8001
```

Inspect the generated Agent Card:

```bash
curl --fail --silent http://127.0.0.1:8001/.well-known/agent-card.json \
  | python3 -m json.tool
```

`A2A_SERVER_HOST`, `A2A_SERVER_PORT`, and `A2A_SERVER_PROTOCOL` control the URL
advertised in that card. The advertised port must match Uvicorn's actual port.

## 2. Run the consuming client

Keep the server running. In a second terminal:

```bash
ch13/.venv/bin/adk run ch13/a2a_client
```

Example prompt:

> Estimate delivery for a 3 kg parcel to Europe.

For the browser interface, run:

```bash
ch13/.venv/bin/adk web --no_use_local_storage --port 8000 ch13
```

Open <http://127.0.0.1:8000>, select `a2a_client`, and submit the same prompt.
Set `A2A_AGENT_CARD_URL` if the server is hosted at a different address.

## Tests

The fast tests validate the deterministic tool without starting Ollama or the
A2A network path:

```bash
ch13/.venv/bin/python -m pytest ch13/tests
```

For an end-to-end check, start Ollama and both agents, then verify that the
client delegates the request and returns the remote estimate.

## From local development to deployment

The local server is an ASGI application, so it can be containerized for Cloud
Run, GKE, or another platform. In production, replace in-memory state with
durable services where needed, publish the externally reachable HTTPS URL in
the Agent Card, add authentication and authorization, protect secrets, and
collect logs, metrics, traces, and task identifiers.

Google Agent Runtime is the managed option covered by the course module. The
Agent Runtime notebook demonstrates project initialization, staging, deployment,
remote queries, and cleanup. Deployment changes the hosting and operational
services; it does not remove the need for a correct A2A contract.

## Primary references

- <https://a2a-protocol.org/latest/>
- <https://adk.dev/a2a/quickstart-exposing/>
- <https://adk.dev/a2a/quickstart-consuming/>
- <https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime>
