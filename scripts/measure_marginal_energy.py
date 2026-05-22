"""
Measure CPU idle baseline power and compute marginal J/tok for each bench row.

Resolves the §6.4 "CodeCarbon resolution" threat: the bench CSVs record
*total* system energy (CPU package power × wall time, as estimated by
CodeCarbon on Windows from utilization × TDP), so the headline J/tok
numbers are 100-200× higher than the BitNet paper's *inference-marginal*
J/tok.  Subtracting an idle-baseline × wall_time term per row recovers
an approximate marginal figure.

Method:
  1. Run CodeCarbon's EmissionsTracker for IDLE_DURATION seconds with no
     work beyond a sleep loop.  Compute `idle_power_W = energy_kwh × 3.6e6
     / duration`.
  2. For each row in `results/*_step_metrics.csv`:
       wall_time = (n_prompt + n_gen) / throughput_tokens_s
       idle_J    = idle_power_W × wall_time
       marginal_J = max(0, total_J − idle_J)
       marginal_J/tok = marginal_J / (n_prompt + n_gen)
  3. Print a table of total vs idle vs marginal J/tok per (model, config).

Caveats — see REPORT.md §4.3:
  - "Idle" here means *the host system as measured at script-run time*,
    which on a developer machine includes a browser, IDE, etc. — not a
    true 0%-utilization baseline.  CodeCarbon on Windows further
    estimates CPU power from utilization × TDP rather than reading RAPL.
    The marginal numbers therefore close most of the gap to the paper's
    figures but leave a residual ~3-10× from the unmeasured estimation
    overhead.
  - The idle measurement happens in a separate temporal window from the
    bench runs, so background-load drift between the two contaminates
    the subtraction. Re-run paired (idle + bench) for tighter coupling.

Usage:
    poetry run python scripts/measure_marginal_energy.py [--duration 90]
    make marginal-energy
"""

import argparse
import contextlib
import csv
import io
import logging
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _clear_codecarbon_lock() -> None:
    """Remove any leftover lock from an abnormally terminated prior run."""
    lock = Path(tempfile.gettempdir()) / ".codecarbon.lock"
    if lock.exists():
        try:
            lock.unlink()
        except OSError:
            pass


def measure_idle(duration: int) -> tuple[float, float, float]:
    """Run CodeCarbon for `duration` seconds with no work; return (kwh, co2_kg, watts)."""
    _clear_codecarbon_lock()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        from codecarbon import EmissionsTracker
    logging.getLogger("codecarbon").setLevel(logging.CRITICAL)

    tracker = EmissionsTracker(
        project_name="cpu-idle-baseline",
        output_dir=str(ROOT / "results"),
        log_level="error",
        save_to_file=False,
    )
    print(f"Measuring CPU idle baseline for {duration}s (no inference work)...",
          flush=True)
    tracker.start()
    t0 = time.time()
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("Interrupted; stopping tracker early.", file=sys.stderr)
    elapsed = time.time() - t0

    with contextlib.redirect_stderr(io.StringIO()):
        co2_kg = tracker.stop()
    try:
        kwh = tracker.final_emissions_data.energy_consumed
    except Exception:
        kwh = 0.0
    watts = (kwh * 3.6e6) / elapsed if elapsed and kwh else 0.0
    return kwh, float(co2_kg or 0.0), watts


def print_marginal_table(idle_power_W: float) -> None:
    """Apply idle subtraction to every row of every results/*_step_metrics.csv."""
    bench_csvs = [
        ("BitNet",   ROOT / "results" / "bitnet_step_metrics.csv"),
        ("Qwen Q8",  ROOT / "results" / "qwen_q8_step_metrics.csv"),
        ("Qwen Q4",  ROOT / "results" / "qwen_q4_step_metrics.csv"),
    ]
    print()
    print("=== Marginal J/tok per bench row ===")
    print(f"{'Model':<10} {'Config':<12} {'Tokens':>7} {'Wall':>7} "
          f"{'Total J/tok':>12} {'Idle J/tok':>11} {'Marginal J/tok':>15}")
    print("-" * 80)
    for label, path in bench_csvs:
        if not path.exists():
            print(f"  (missing: {path.name})")
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                try:
                    ekwh = float(row["energy_kwh"])
                    tps = float(row["throughput_tokens_s"])
                    n_p, n_g = int(row["n_prompt"]), int(row["n_gen"])
                except (KeyError, ValueError):
                    continue
                tokens = n_p + n_g
                wall = tokens / tps if tps else 0
                total_J = ekwh * 3.6e6
                idle_J = idle_power_W * wall
                marginal_J = max(0.0, total_J - idle_J)
                cfg = f"({n_p},{n_g})"
                print(f"{label:<10} {cfg:<12} {tokens:>7} {wall:>6.2f}s "
                      f"{total_J/tokens:>12.3f} {idle_J/tokens:>11.3f} "
                      f"{marginal_J/tokens:>15.3f}")


def main():
    p = argparse.ArgumentParser(
        description="Measure CPU idle baseline and compute marginal J/tok.")
    p.add_argument("--duration", type=int, default=90,
                   help="Idle measurement duration in seconds (default: 90)")
    args = p.parse_args()

    kwh, co2_kg, watts = measure_idle(args.duration)
    print(f"  Energy:      {kwh*1000:.4f} Wh ({kwh:.6f} kWh)")
    print(f"  CO2:         {co2_kg*1000:.4f} g")
    print(f"  Idle power:  {watts:.2f} W")

    if watts <= 0:
        print("\nNo idle energy reported; cannot compute marginal numbers.",
              file=sys.stderr)
        sys.exit(1)
    print_marginal_table(watts)


if __name__ == "__main__":
    main()
