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

# Folder with the case: spec, read-only tests, starter and reference solution.
CASE_DIR = Path(__file__).resolve().parent / "case_retry"
# Public names the tests import from the candidate file; a candidate that lacks one is rejected
# by the static checks before the tests even run.
REQUIRED_NAMES = ("authorize_with_retry", "FakeGateway", "TemporaryGatewayError", "PaymentResult")
# Static scope rules: textual patterns for behaviour that the specification excludes.
# Key = regular expression searched in the candidate source, value = reason shown as feedback.
# They are cheap and need no model, but they only match these literal forms.
FORBIDDEN = {
    r"\btime\.sleep\b": "no sleeping/backoff: it is outside the spec",
    r"\bimport\s+(requests|httpx|urllib)": "no real network calls",
    r"\bsubprocess\b|\bos\.system\b": "no process execution",
}


@dataclass(frozen=True)
class HarnessReport:
    """The outcome of one harness run: the stop criterion of the loop and the feedback for the next attempt."""

    passed: bool  # True only if the tests exited with code 0 and no static rule was violated
    tests_run: int  # parsed from the unittest output; 0 if the output has no "Ran N tests" line
    failures: int  # number of FAIL and ERROR entries in the unittest output
    static_violations: list[str]  # reasons of the static rules that matched, plus any missing required name
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
        """One line for logs and tables, e.g. "FAIL (3 tests, 1 failing, static: ...)"."""
        status = "PASS" if self.passed else "FAIL"
        extra = f", static: {'; '.join(self.static_violations)}" if self.static_violations else ""
        return f"{status} ({self.tests_run} tests, {self.failures} failing{extra})"


def static_checks(code: str) -> list[str]:
    """First gate: cheap textual rules on the candidate source. Returns the list of violations (empty = clean)."""
    # One violation for every forbidden pattern found in the source.
    violations = [why for pattern, why in FORBIDDEN.items() if re.search(pattern, code)]
    # One violation for every required public name that does not appear in the source.
    violations += [f"missing required name {n}" for n in REQUIRED_NAMES if n not in code]
    return violations


def run_harness(code: str, case_dir: Path = CASE_DIR, timeout: int = 30) -> HarnessReport:
    """Check one candidate file: static rules first, then the read-only tests in a clean directory."""
    # STEP 1 - static rules. They never stop the run: the tests below still execute, so that the
    # feedback contains both the rule violations and the test output.
    violations = static_checks(code)
    # STEP 2 - a fresh temporary directory per run, deleted when the block ends.
    with tempfile.TemporaryDirectory(prefix="harness_") as tmp:
        work = Path(tmp)
        # The agent's only contribution is this single file.
        (work / "payment_retry.py").write_text(code, encoding="utf-8")
        # The tests are always copied from the original: whatever the agent does to a test file
        # elsewhere cannot reach this copy.
        shutil.copy(case_dir / "test_payment_retry.py", work / "test_payment_retry.py")
        try:
            # STEP 3 - run the tests in a separate interpreter. "-I" is isolated mode: it ignores the
            # user's environment variables and site settings, so the result does not depend on the machine.
            # The timeout stops an infinite loop in the generated code. Note that the tests import the
            # candidate file, so its top-level code runs inside this process, in this directory.
            proc = subprocess.run(
                [sys.executable, "-I", "-m", "unittest", "-v"],
                cwd=work, capture_output=True, text=True, timeout=timeout,
            )
            # unittest writes its report on stderr; keep both streams and the exit code.
            output, returncode = proc.stderr + proc.stdout, proc.returncode
        except subprocess.TimeoutExpired:
            # A timeout counts as a failure and becomes feedback like any other.
            output, returncode = "TIMEOUT: tests did not finish (infinite loop?)", 1

    # STEP 4 - turn the raw output into a report. The test count comes from the "Ran N tests" line.
    run = re.search(r"Ran (\d+) test", output)
    # Failed and errored tests are the lines that start with FAIL: or ERROR:.
    fails = len(re.findall(r"^(FAIL|ERROR):", output, flags=re.M))
    return HarnessReport(
        # The stop criterion: exit code 0 and no static violation. The number of tests run is
        # reported but not required to be greater than zero.
        passed=returncode == 0 and not violations,
        tests_run=int(run.group(1)) if run else 0,
        failures=fails,
        static_violations=violations,
        output=output[-1800:],  # keep the end of the output: the failures are printed last
    )
