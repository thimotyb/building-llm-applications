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

## Run the A2A server as a Docker image

The image contains the A2A server, while Ollama and the model remain on the
host. With Docker Desktop integrated into this Ubuntu WSL2 distribution,
containers can reach the host through `host.docker.internal`.

First verify the exact model tag exposed by Ollama:

```bash
ollama list
curl --fail http://127.0.0.1:11434/api/tags
```

The Compose configuration defaults to `gemma4:12b`. If `ollama list` reports a
different tag, export it before starting the container, for example:

```bash
export OLLAMA_MODEL=gemma4:12b
```

Ollama must accept connections that originate outside its own loopback
interface. If it currently listens only on `127.0.0.1`, restart it with:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Do not publish port 11434 on an untrusted network. On Windows, also allow the
connection through the firewall only for the Docker/WSL private network.

Build and start the server from the repository root:

```bash
docker compose -f ch13/compose.yaml up --build -d
docker compose -f ch13/compose.yaml logs -f shipping-specialist
```

Verify that the container is healthy and that its Agent Card is reachable from
WSL2:

```bash
docker compose -f ch13/compose.yaml ps
curl --fail --silent http://127.0.0.1:8001/.well-known/agent-card.json \
  | python3 -m json.tool
```

The local client continues to use the default card URL, so it can be run
unchanged in another WSL2 terminal:

```bash
ch13/.venv/bin/adk run ch13/a2a_client
```

To build and run without Compose:

```bash
docker build -f ch13/Dockerfile -t ch13-a2a-shipping-specialist .
docker run --rm --name shipping-specialist \
  --add-host host.docker.internal:host-gateway \
  -p 8001:8001 \
  -e OLLAMA_MODEL=gemma4:12b \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e A2A_SERVER_HOST=127.0.0.1 \
  ch13-a2a-shipping-specialist
```

Stop and remove the Compose container with:

```bash
docker compose -f ch13/compose.yaml down
```

`A2A_SERVER_HOST` is the address advertised in the Agent Card, not Uvicorn's
bind address. Keep `127.0.0.1` when the consumer runs on the same host. For a
remote consumer, set it to the DNS name or IP through which port 8001 is
actually reachable.

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
