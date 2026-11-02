# Chapter 14 – Agentic loop, judges and harness (Workshop 2)

Self-contained building blocks for the Workshop 2 demos. Each file works on its own.

| File | Purpose |
| --- | --- |
| `llm_factory.py` | One LiteLLM-based entry point for Ollama, DeepSeek, Gemini, OpenAI. Worker and judge can use different providers. |
| `judge.py` | Typed judge interface: state + typed questions (`Choice`, `Noul`, `Score`) → typed answers with probabilities. `LLMJudge` (any LiteLLM model), `FakeJudge` (tests), `decide()` (thresholds stay in code). |
| `demo_judge.py` | Same questions on any provider. |
| `jev_judge.py` | Backend for the hosted TypeSafe Jev decision model (`TYPESAFE_API_KEY`, `JEV_MODEL`; closed weights, API only). Input-token billing only. |
| `judge_factory.py` | `make_judge()` picks the backend: `JUDGE_BACKEND=llm` (default) or `jev`; `JUDGE_FALLBACK=llm` falls back to the LLM judge if no key. |
| `harness.py` | Deterministic harness for the payment-retry case: runs the read-only tests on the agent's file, plus static scope checks. It is the hard stop criterion. |
| `agentic_loop.py` | LangGraph loop: generate → harness → judge → decide (repeat / human / stop), budgeted iterations, human-in-the-loop interrupt, live trace and cost. |
| `case_retry/` | Spec (Spec Kit), starter, tests and reference solution of the payment-retry case (copied from the augmented-coding demo). |
| `tests/` | Offline tests (no network, no API key). |

## Setup

```bash
cd ch14
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
```

Keys and model names are read from the repo-root `.env` (same names as the other chapters).

## Choosing the LLM

```bash
LLM_PROVIDER=ollama   OLLAMA_MODEL=gemma4:e4b .venv/bin/python demo_judge.py
LLM_PROVIDER=deepseek .venv/bin/python demo_judge.py
JUDGE_PROVIDER=gemini LLM_PROVIDER=ollama .venv/bin/python demo_judge.py   # separate judge
LLM_FALLBACK_OLLAMA=1 LLM_PROVIDER=deepseek .venv/bin/python demo_judge.py # offline fallback
```

## Notes on probabilities

With `samples=1` the probabilities are self-reported by the model and are often over-confident
(small models answer `1.0` too easily). `--samples N` derives them from vote shares. Always
calibrate the thresholds used by `decide()` on labelled examples before relying on them.

## Agentic loop

```bash
LLM_PROVIDER=deepseek .venv/bin/python agentic_loop.py --human ask
LLM_PROVIDER=ollama OLLAMA_MODEL=gemma4:e4b JUDGE_PROVIDER=deepseek .venv/bin/python agentic_loop.py --human ask
```

The human is asked only when the judge is uncertain (0.3 < p < 0.7) or the iteration budget is exhausted.
A strong worker passes in one iteration; use a small local worker to show the repair loop.

## Jev judge

```bash
JUDGE_BACKEND=jev JEV_MODEL=jev-1.14.0 .venv/bin/python demo_judge.py
LLM_PROVIDER=deepseek JUDGE_BACKEND=jev .venv/bin/python agentic_loop.py --human ask
```

Pin `JEV_MODEL` to a version once thresholds are calibrated (`jev-latest` is an alias that can move).
Do not send personal or private data to the hosted API without approval.

## Planned

Local open decision-model backend, model routing, A2A reviewer. See `notes/TODO-seminar.md` in the course repo.
