# Chapter 14 – Agentic loop, judges and harness (Workshop 2)

Self-contained building blocks for the Workshop 2 demos. Each file works on its own.

| File | Purpose |
| --- | --- |
| `llm_factory.py` | One LiteLLM-based entry point for Ollama, DeepSeek, Gemini, OpenAI. Worker and judge can use different providers. |
| `judge.py` | Typed judge interface: state + typed questions (`Choice`, `Noul`, `Score`) → typed answers with probabilities. `LLMJudge` (any LiteLLM model), `FakeJudge` (tests), `decide()` (thresholds stay in code). |
| `demo_judge.py` | Same questions on any provider. |
| `jev_judge.py` | Backend for the hosted TypeSafe Jev decision model (`TYPESAFE_API_KEY`, `JEV_MODEL`; closed weights, API only). Input-token billing only. |
| `judge_factory.py` | `make_judge()` picks the backend: `JUDGE_BACKEND=llm` (default) or `jev`; `JUDGE_FALLBACK=llm` falls back to the LLM judge if no key. |
| `router.py` | Routing: `CascadeJudge` (cheap judge first, unsure questions go to the next tier, still unsure -> `needs_human`) and `escalating_generate` (small worker model until the harness fails, then the large one). `JUDGE_BACKEND=cascade`, `--escalate`. |
| `bench/routing_replay.py` | Replays a cascade on stored benchmark results (no model calls): accuracy, human share, cost and latency per threshold. |
| `harness.py` | Deterministic harness for the payment-retry case: runs the read-only tests on the agent's file, plus static scope checks. It is the hard stop criterion. |
| `demo_live.py` | **Narrated demo of the whole loop** (icons in the log, probability bars, diff of the code, closing table). Offline `replay` mode or `live` mode with real models. |
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

## Narrated demo (`demo_live.py`)

Starting situation: a payment-retry function that makes a single attempt (2 of 3 tests fail). The loop
generates candidates, the harness runs read-only tests and static rules, the judge answers typed questions with
probabilities, and code decides: repeat, ask a person, or accept. Every step is printed with an icon.

```bash
# Offline, scripted, no network and no API key (the safe option in a classroom)
.venv/bin/python demo_live.py --mode replay                  # full story: 4 iterations, every branch
.venv/bin/python demo_live.py --mode replay --story short    # shortest path: 2 iterations
.venv/bin/python demo_live.py --mode replay --delay 0.6      # slower pacing for a projector

# Live with real models (keys in the repo-root .env); the first attempt is a PLANTED flaw, labelled in the log
LLM_PROVIDER=deepseek JUDGE_BACKEND=jev JEV_MODEL=jev-1.14.0 .venv/bin/python demo_live.py --mode live --inject off-by-one,backoff
JUDGE_BACKEND=cascade CASCADE_THRESHOLD=0.7 LLM_PROVIDER=deepseek .venv/bin/python demo_live.py --mode live --human ask
LLM_PROVIDER=ollama OLLAMA_MODEL=gemma4:e4b .venv/bin/python demo_live.py --mode live --inject none   # fully local
```

| Option | Meaning |
| --- | --- |
| `--mode replay\|live` | Scripted offline run, or real models (default: replay) |
| `--story full\|short` | Replay storyline |
| `--inject LIST` | Live: planted first attempts: `off-by-one` (test fails), `backoff` (static rule), `scope-creep` (judge flags), or `none` |
| `--human ask\|approve\|stop` | How the person step is answered (default: approve) |
| `--max-iters N` | Iteration budget (default 5) |
| `--escalate` | Live: switch to `LARGE_PROVIDER`/`LARGE_MODEL` after a failed attempt |
| `--delay S` | Pause between sections, for pacing |
| `--log FILE` | Also write a plain-text log (no colours) |
| `--out FILE` | Save the accepted `payment_retry.py` |
| `--no-color` | Disable ANSI colours (also automatic when output is not a terminal) |

What the log shows, in order: goal and exclusions, the starter against the tests, configuration (worker, judge,
thresholds, budget), then per iteration: feedback received, the candidate with its diff, the harness result per
acceptance scenario, the judge probabilities against the thresholds, the decision, and finally a table of
iterations, cost and time plus the diff from the starter to the accepted code.

Notes: the replay is scripted (costs are zero by construction); in live mode a strong model rarely fails alone,
which is why a flawed first attempt is planted and labelled. A backup log can be saved with `--log`.

## Agentic loop (without the narration)

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

## Routing

```bash
JUDGE_BACKEND=cascade CASCADE_THRESHOLD=0.7 LLM_PROVIDER=deepseek .venv/bin/python demo_judge.py   # Jev, then DeepSeek on unsure questions
LLM_PROVIDER=ollama OLLAMA_MODEL=gemma4:e4b LARGE_PROVIDER=ollama LARGE_MODEL=<gemma 12b tag> .venv/bin/python agentic_loop.py --escalate
.venv/bin/python bench/routing_replay.py --tiers jev-1.14.0 deepseek-v4-flash
```

## Planned

A2A reviewer. See `notes/TODO-seminar.md` in the course repo.

## Traces

`agentic_loop.py --trace FILE` and `demo_live.py --trace FILE` write the trace of a run as JSON Lines: a header record
(`"node": "run"`, with worker, judge, status and total cost) followed by one record per node execution
(`generate`, `harness`, `judge`, `human`), in order. Saved examples are in `traces/`:

- `traces/live-deepseek-jev.jsonl`: real run, DeepSeek as worker and Jev as judge, accepted on the first iteration.
- `traces/replay-full.jsonl`: the scripted four-iteration story (test failure, static rule, judge revise, uncertain judge, person approves).

`bench/cascade_trace.py` runs a real cascade (Jev first, an LLM judge on the unsure questions) on 12 tickets and records every call to every tier:

```bash
JUDGE_PROVIDER=deepseek JEV_MODEL=jev-1.14.0 .venv/bin/python bench/cascade_trace.py --out traces/cascade-live.jsonl [--threshold 0.7] [--tickets 12]
```

- `traces/cascade-live.jsonl`: threshold 0.7 (2 of 36 questions escalated, none to a person).
- `traces/cascade-live-threshold-095.jsonl`: threshold 0.95 (16 escalated, 10 to a person).
