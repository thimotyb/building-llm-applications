"""Run one backend over the labelled tickets and store per-item probabilities.

    python bench/run_bench.py --backend jev     --label jev-1.14.0
    python bench/run_bench.py --backend llm     --label deepseek      (LLM_PROVIDER / JUDGE_PROVIDER from .env)
    python bench/run_bench.py --backend llm     --label gemma-12b     LLM_PROVIDER=ollama OLLAMA_MODEL=...
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

RESULTS = Path(__file__).resolve().parent / "results"
QUESTIONS = [
    Choice(id="team", prompt=dataset.QUESTION_TEAM, options=dataset.TEAMS, descriptions=dataset.TEAM_DESCRIPTIONS),
    Noul(id="urgent", prompt=dataset.QUESTION_URGENT),
    Noul(id="refund", prompt=dataset.QUESTION_REFUND),
]


def run_judge(judge, items: list[dict], workers: int) -> tuple[list[dict], float, float]:
    def one(item):
        for attempt in range(2):
            try:
                v = judge.judge(item["text"], QUESTIONS)
                return {
                    "id": item["id"], "latency": v.latency_s, "cost": v.cost_usd, "error": None,
                    "team": v["team"].probabilities, "urgent": v["urgent"].probabilities.get("true", 0.5),
                    "refund": v["refund"].probabilities.get("true", 0.5),
                }
            except Exception as exc:  # keep going; failures are reported, not hidden
                err = f"{type(exc).__name__}: {exc}"[:200]
        return {"id": item["id"], "latency": 0.0, "cost": 0.0, "error": err}

    start = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, items))
    return rows, time.perf_counter() - start, 0.0


def run_xgboost(items: list[dict]) -> tuple[list[dict], float, float]:
    """5-fold out-of-fold predictions on TF-IDF features: needs labels for every task."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    texts = [i["text"] for i in items]
    n = len(items)
    rows = [{"id": i["id"], "latency": 0.0, "cost": 0.0, "error": None} for i in items]
    start = time.perf_counter()
    for task in ("team", "urgent", "refund"):
        classes = dataset.TEAMS if task == "team" else [False, True]
        y = np.array([classes.index(i[task]) for i in items])
        oof = np.zeros((n, len(classes)))
        for train, test in StratifiedKFold(5, shuffle=True, random_state=14).split(texts, y):
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            x_train = vec.fit_transform([texts[i] for i in train])
            model = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.1, verbosity=0)
            model.fit(x_train, y[train])
            proba = model.predict_proba(vec.transform([texts[i] for i in test]))
            for col, cls in enumerate(model.classes_):
                oof[test, cls] = proba[:, col]
        for row, p in zip(rows, oof):
            row[task] = dict(zip(dataset.TEAMS, map(float, p))) if task == "team" else float(p[1])
    return rows, time.perf_counter() - start, 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["jev", "llm", "xgboost"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    items = dataset.load()[: args.limit or None]
    meta: dict = {"label": args.label, "backend": args.backend, "n": len(items)}
    if args.backend == "xgboost":
        rows, wall, _ = run_xgboost(items)
        meta["note"] = "5-fold CV; trained on ~80% of the labelled tickets, needs labels per task"
    else:
        import os

        os.environ["JUDGE_BACKEND"] = "jev" if args.backend == "jev" else "llm"
        from judge_factory import make_judge

        judge = make_judge(samples=args.samples)
        meta["model"] = getattr(judge, "model", None) or judge.spec.label
        rows, wall, _ = run_judge(judge, items, 1 if args.backend == "llm" and meta["model"].startswith("ollama") else args.workers)
    meta["wall_seconds"] = round(wall, 2)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.label}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    errors = sum(1 for r in rows if r["error"])
    print(f"{args.label}: {len(rows)} items, {errors} errors, {wall:.1f}s, cost=${sum(r['cost'] for r in rows):.4f}")


if __name__ == "__main__":
    main()
