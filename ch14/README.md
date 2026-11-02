# Chapter 14 – Agentic loop, judges and harness (Workshop 2)

Self-contained building blocks for the Workshop 2 demos. Each file works on its own.

| File | Purpose |
| --- | --- |
| `llm_factory.py` | One LiteLLM-based entry point for Ollama, DeepSeek, Gemini, OpenAI. Worker and judge can use different providers. |
| `judge.py` | Typed judge interface: state + typed questions (`Choice`, `Noul`, `Score`) → typed answers with probabilities. `LLMJudge` (any LiteLLM model), `FakeJudge` (tests), `decide()` (thresholds stay in code). |
| `demo_judge.py` | Same questions on any provider. |
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

## Planned

Local decision-model backend, hosted decision-model backend (TypeSafe Jev), agentic loop with harness,
model routing, A2A reviewer. See `notes/TODO-seminar.md` in the course repo.
