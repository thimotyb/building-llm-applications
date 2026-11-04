"""Run one backend over the labelled tickets and store per-item probabilities.

    python bench/run_bench.py --backend jev     --label jev-1.14.0
    python bench/run_bench.py --backend llm     --label deepseek      (LLM_PROVIDER / JUDGE_PROVIDER from .env)
    python bench/run_bench.py --backend llm     --label gemma-12b     LLM_PROVIDER=ollama OLLAMA_MODEL=...
    OPENDECIDER_MODEL=manjunathshiva/opendecider-small python bench/run_bench.py --backend opendecider --label opendecider-small
    python bench/run_bench.py --backend xgboost --label xgboost-tfidf
Results go to bench/results/<label>.json; build the table with bench/report.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset  # noqa: E402
from judge import Choice, Noul  # noqa: E402

# Where per-backend result files (<label>.json, <label>.partial.json) are written.
RESULTS = Path(__file__).resolve().parent / "results"
# The three questions asked of every ticket. The judges answer all of them in a single call;
# Choice = one option among several (team), Noul = yes/no (urgent, refund).
QUESTIONS = [
    Choice(id="team", prompt=dataset.QUESTION_TEAM, options=dataset.TEAMS, descriptions=dataset.TEAM_DESCRIPTIONS),
    Noul(id="urgent", prompt=dataset.QUESTION_URGENT),
    Noul(id="refund", prompt=dataset.QUESTION_REFUND),
]


def run_judge(judge, items: list[dict], workers: int, label: str = "run", resume: bool = False) -> tuple[list[dict], float, float]:
    """Judge every ticket with live progress; partial results are saved so a long run can be resumed."""
    # The partial file is rewritten after every ticket, so an interrupted run loses nothing.
    partial = RESULTS / f"{label}.partial.json"
    # Finished rows by ticket id; with --resume, start from the successful ones already stored.
    done: dict[int, dict] = {}
    if resume and partial.exists():
        done = {r["id"]: r for r in json.loads(partial.read_text()) if not r.get("error")}
        print(f"[{label}] resuming: {len(done)} tickets already done", flush=True)
    # Only tickets without a successful answer are sent to the judge (failed ones are retried on resume).
    todo = [i for i in items if i["id"] not in done]

    def one(item):
        """Ask the three questions for one ticket; one retry on failure, then record the error."""
        err = ""
        for attempt in range(2):
            try:
                v = judge.judge(item["text"], QUESTIONS)
                # Keep the probability of each outcome; for yes/no questions only P(true).
                # 0.5 = no information, used if the judge did not report the "true" outcome.
                return {
                    "id": item["id"], "latency": v.latency_s, "cost": v.cost_usd, "error": None,
                    "team": v["team"].probabilities, "urgent": v["urgent"].probabilities.get("true", 0.5),
                    "refund": v["refund"].probabilities.get("true", 0.5),
                }
            except Exception as exc:  # keep going; failures are reported, not hidden
                err = f"{type(exc).__name__}: {exc}"[:600]
        return {"id": item["id"], "latency": 0.0, "cost": 0.0, "error": err}

    start = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    errors = 0
    # Tickets are judged in parallel by a thread pool (the calls are network/IO bound);
    # pool.map yields results in ticket order, so progress and the saved file stay ordered.
    with ThreadPoolExecutor(workers) as pool:
        for n, row in enumerate(pool.map(one, todo), 1):
            done[row["id"]] = row
            errors += bool(row["error"])
            # Linear ETA: average time per finished ticket times tickets still to do.
            elapsed = time.perf_counter() - start
            eta = elapsed / n * (len(todo) - n)
            status = f"ERROR {row['error']}" if row["error"] else f"{row['latency']:.1f}s"
            print(f"[{label}] {n}/{len(todo)}  ticket {row['id']}  {status}  elapsed {elapsed:.0f}s  "
                  f"ETA {eta:.0f}s  errors {errors}", flush=True)
            partial.write_text(json.dumps(list(done.values())))
    # Return rows in the original ticket order; the third value (unused cost slot) is always 0.0.
    rows = [done[i["id"]] for i in items]
    return rows, time.perf_counter() - start, 0.0


def run_xgboost(items: list[dict]) -> tuple[list[dict], float, float]:
    """5-fold out-of-fold predictions on TF-IDF features: needs labels for every task."""
    # Imported here so the other backends run without scikit-learn and xgboost installed.
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    texts = [i["text"] for i in items]
    n = len(items)
    # Same row shape as the judges; latency and cost are zero because inference is local and free.
    rows = [{"id": i["id"], "latency": 0.0, "cost": 0.0, "error": None} for i in items]
    start = time.perf_counter()
    # One independent model per question: the loop below trains a separate
    # vectorizer + classifier for "team" (3 classes), "urgent" and "refund" (2 classes each).
    for task in ("team", "urgent", "refund"):
        classes = dataset.TEAMS if task == "team" else [False, True]
        # Target variable: the label stored with each ticket for this question, as a class index.
        y = np.array([classes.index(i[task]) for i in items])
        # Out-of-fold probabilities: row = ticket, column = class. Each ticket is filled exactly once,
        # by the model that did not see it during training.
        oof = np.zeros((n, len(classes)))
        # 5 folds, stratified so every fold keeps the class proportions; fixed seed for reproducibility.
        for train, test in StratifiedKFold(5, shuffle=True, random_state=14).split(texts, y):
            # Features: TF-IDF over words and word pairs. Fitted on the training fold only,
            # so the vocabulary and word weights never leak information from the held-out tickets.
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            x_train = vec.fit_transform([texts[i] for i in train])
            # Gradient-boosted trees: 120 shallow trees (depth 3) added one after another,
            # each correcting the errors of the previous ones, shrunk by the learning rate. No tuning.
            model = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.1, verbosity=0)
            model.fit(x_train, y[train])
            # Predict the held-out tickets with the vocabulary learned on the training fold.
            proba = model.predict_proba(vec.transform([texts[i] for i in test]))
            # model.classes_ lists only the classes seen in training: map columns back to class indices.
            for col, cls in enumerate(model.classes_):
                oof[test, cls] = proba[:, col]
        # Store the result in the shape the judges produce: a dict of team probabilities,
        # or the probability of the positive class for the yes/no questions.
        for row, p in zip(rows, oof):
            row[task] = dict(zip(dataset.TEAMS, map(float, p))) if task == "team" else float(p[1])
    return rows, time.perf_counter() - start, 0.0


def main() -> None:
    """Parse options, run the chosen backend over the tickets, and save the results with metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["jev", "llm", "opendecider", "xgboost"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="reuse tickets already stored in <label>.partial.json")
    parser.add_argument("--finalize", action="store_true",
                        help="turn <label>.partial.json into <label>.json as-is (no model calls), for an interrupted run")
    args = parser.parse_args()

    # --finalize: convert the partial file of an interrupted run into a normal result file,
    # without calling any model; unfinished tickets are simply absent and failed ones count as errors.
    if args.finalize:
        rows = json.loads((RESULTS / f"{args.label}.partial.json").read_text())
        bad = sum(1 for r in rows if r.get("error"))
        meta = {"label": args.label, "backend": args.backend, "n": len(rows),
                "note": f"PARTIAL: {len(rows)} of {len(dataset.load())} tickets, {bad} failed (counted as errors)"}
        (RESULTS / f"{args.label}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
        print(f"{args.label}: finalized {len(rows)} tickets ({bad} errors) -> {args.label}.json")
        return

    # --limit N keeps only the first N tickets (quick tests); 0 means all 120.
    items = dataset.load()[: args.limit or None]
    meta: dict = {"label": args.label, "backend": args.backend, "n": len(items)}
    if args.backend == "xgboost":
        rows, wall, _ = run_xgboost(items)
        meta["note"] = "5-fold CV; trained on ~80% of the labelled tickets, needs labels per task"
    else:
        import os

        # judge_factory reads the backend from the environment and the provider/model from .env.
        os.environ["JUDGE_BACKEND"] = args.backend
        from judge_factory import make_judge

        # --samples N asks the judge N times per question (when the backend supports it).
        judge = make_judge(samples=args.samples)
        # Record which model answered, for the report.
        meta["model"] = getattr(judge, "model_id", None) or getattr(judge, "model", None) or judge.spec.label
        # Local models (opendecider, Ollama) handle one request at a time: no parallelism there.
        workers = 1 if args.backend == "opendecider" or str(meta["model"]).startswith("ollama") else args.workers
        rows, wall, _ = run_judge(judge, items, workers, args.label, args.resume)
    # Total elapsed time of the run, stored next to the per-ticket results.
    meta["wall_seconds"] = round(wall, 2)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.label}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    # Final one-line summary: tickets processed, failures, time and total cost.
    errors = sum(1 for r in rows if r["error"])
    print(f"{args.label}: {len(rows)} items, {errors} errors, {wall:.1f}s, cost=${sum(r['cost'] for r in rows):.4f}")


if __name__ == "__main__":
    main()
