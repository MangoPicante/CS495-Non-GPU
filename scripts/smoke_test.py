"""
Smoke test for Phase 4 scripts.

Runs compare_runs.py against the existing results and verifies all expected
output files are produced with non-zero content.  Also checks that
metrics_tracker.py and eval_accuracy.py parse arguments cleanly (full
benchmark/eval runs are not attempted here).

Exit 0 = all checks passed.  Non-zero = at least one failure.

Usage:
    python scripts/smoke_test.py
    make smoke-test
"""

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    suffix = f": {detail}" if detail and not condition else ""
    print(f"  {mark}  {label}{suffix}")
    if not condition:
        _failures.append(label)


def run(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *cmd],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


# ── compare_runs.py ───────────────────────────────────────────────────────────

print("\n=== compare_runs.py ===")

result = run(str(SCRIPTS / "compare_runs.py"))
check("exits 0", result.returncode == 0, result.stderr.strip())

for plot_name in [
    "throughput_comparison.png",
    "memory_comparison.png",
    "bitnet_throughput_configs.png",
    "accuracy_comparison.png",
    "cost_accuracy.png",
]:
    p = RESULTS / "plots" / plot_name
    check(f"plots/{plot_name} non-empty", p.exists() and p.stat().st_size > 0)

csv_path = RESULTS / "comparison_table.csv"
check("comparison_table.csv exists", csv_path.exists())
if csv_path.exists():
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    check("7 data rows (5 FP16 + paper + ours)", len(rows) == 7, f"got {len(rows)}")

    required_cols = {
        "model", "source", "throughput_tokens_s", "peak_rss_mb",
        "cost_per_1k_tokens",
        "arc_easy", "arc_challenge", "winogrande", "hellaswag", "mmlu",
    }
    missing = required_cols - set(rows[0].keys()) if rows else required_cols
    check("all expected columns present", not missing, f"missing {missing}")

    ours = next((r for r in rows if r.get("source") == "ours"), None)
    check("'ours' row present", ours is not None)
    if ours:
        check("'ours' row has throughput",        ours["throughput_tokens_s"] != "")
        check("'ours' row has cost_per_1k_tokens", ours["cost_per_1k_tokens"] != "")
        check("'ours' row has arc_easy",           ours["arc_easy"] != "")

# ── metrics_tracker.py ────────────────────────────────────────────────────────

print("\n=== metrics_tracker.py ===")

result = run(str(SCRIPTS / "metrics_tracker.py"), "--help")
check("--help exits 0", result.returncode == 0, result.stderr.strip())
check("--no-energy flag present", "--no-energy" in result.stdout)

# ── eval_accuracy.py ──────────────────────────────────────────────────────────

print("\n=== eval_accuracy.py ===")

result = run(str(SCRIPTS / "eval_accuracy.py"), "--help")
check("--help exits 0", result.returncode == 0, result.stderr.strip())
check("--task flag present", "--task" in result.stdout)

# ── summary ───────────────────────────────────────────────────────────────────

print()
if _failures:
    print(f"FAILED  ({len(_failures)} check(s) failed):")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
