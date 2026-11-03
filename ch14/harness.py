"""Deterministic harness for the payment-retry case.

The agent may only produce `payment_retry.py`. The tests and the spec are
read-only for it: the harness copies the agent's file into a fresh temporary
directory next to the original tests, so the agent cannot "pass" by editing
the tests. The harness result is the hard stop criterion of the loop; the
judge only evaluates what tests cannot (fit to the spec, scope creep).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent / "case_retry"
REQUIRED_NAMES = ("authorize_with_retry", "FakeGateway", "TemporaryGatewayError", "PaymentResult")
FORBIDDEN = {
    r"\btime\.sleep\b": "no sleeping/backoff: it is outside the spec",
    r"\bimport\s+(requests|httpx|urllib)": "no real network calls",
    r"\bsubprocess\b|\bos\.system\b": "no process execution",
}


@dataclass(frozen=True)
class HarnessReport:
    passed: bool
    tests_run: int
    failures: int
    static_violations: list[str]
    output: str  # trimmed unittest output, used as feedback

    @property
    def feedback(self) -> str:
        """Text for the next attempt: static rule violations first, then the test output."""
        if not self.static_violations:
            return self.output
        rules = "\n".join(f"- {v}" for v in self.static_violations)
        return f"Static rule violations (fix these):\n{rules}\n\n{self.output}"

    @property
    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        extra = f", static: {'; '.join(self.static_violations)}" if self.static_violations else ""
        return f"{status} ({self.tests_run} tests, {self.failures} failing{extra})"


def static_checks(code: str) -> list[str]:
    violations = [why for pattern, why in FORBIDDEN.items() if re.search(pattern, code)]
    violations += [f"missing required name {n}" for n in REQUIRED_NAMES if n not in code]
    return violations


def run_harness(code: str, case_dir: Path = CASE_DIR, timeout: int = 30) -> HarnessReport:
    violations = static_checks(code)
    with tempfile.TemporaryDirectory(prefix="harness_") as tmp:
        work = Path(tmp)
        (work / "payment_retry.py").write_text(code, encoding="utf-8")
        shutil.copy(case_dir / "test_payment_retry.py", work / "test_payment_retry.py")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-m", "unittest", "-v"],
                cwd=work, capture_output=True, text=True, timeout=timeout,
            )
            output, returncode = proc.stderr + proc.stdout, proc.returncode
        except subprocess.TimeoutExpired:
            output, returncode = "TIMEOUT: tests did not finish (infinite loop?)", 1

    run = re.search(r"Ran (\d+) test", output)
    fails = len(re.findall(r"^(FAIL|ERROR):", output, flags=re.M))
    return HarnessReport(
        passed=returncode == 0 and not violations,
        tests_run=int(run.group(1)) if run else 0,
        failures=fails,
        static_violations=violations,
        output=output[-1800:],
    )
