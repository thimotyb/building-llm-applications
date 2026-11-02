"""Show what a judge answers for ONE ticket (debug aid): python bench/probe.py --id 11 [--backend llm]."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset  # noqa: E402
from run_bench import QUESTIONS  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, required=True)
parser.add_argument("--backend", default="llm")
args = parser.parse_args()

os.environ["JUDGE_BACKEND"] = args.backend
import judge as judge_mod  # noqa: E402
from judge_factory import make_judge  # noqa: E402

item = dataset.load()[args.id]
print("TICKET:", item)
j = make_judge()
original = judge_mod.complete


def spy(*a, **k):
    response = original(*a, **k)
    print("\n--- RAW REPLY ---\n", response.choices[0].message.content, "\n-----------------")
    return response


judge_mod.complete = spy
try:
    print(j.judge(item["text"], QUESTIONS))
except Exception as exc:
    print("ERROR:", exc)
