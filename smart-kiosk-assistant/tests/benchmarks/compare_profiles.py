#!/usr/bin/env python3
"""Build a cross-profile comparison table from device-combination benchmark runs.

Each ``make benchmark PROFILE=<name>`` leaves a self-contained directory::

    results/<profile>/
      run_manifest.json              devices, models, git SHA, host
      smart_kiosk_v2v_results_*.json per-turn latency + summary stats
      consolidated_metrics.csv       hardware counters (key,value rows)

This script walks those directories and emits one row per profile into
``results/matrix_summary.csv`` — the actual deliverable when comparing
"ASR on CPU vs GPU vs NPU" against "TTS on CPU vs GPU vs NPU".

Design notes
------------
* **p95 is the headline, not the median.** A single bad tail turn is what a
  customer notices; the median hides it. Both are emitted, p95 first.
* **Device columns come from ``run_manifest.json``, not the directory name.**
  A directory can be renamed or copied; the manifest records what actually ran,
  including whether validation passed and whether the tree was dirty.
* **Only the newest ``smart_kiosk_v2v_results_*.json`` per profile is read**,
  so re-running a profile supersedes rather than double-counts.
* **Stdlib only** — no pandas. ``consolidated_metrics.csv`` is a flat two-column
  key/value file, so there is nothing here worth a virtualenv for.

Usage::

    python3 tests/benchmarks/compare_profiles.py
    python3 tests/benchmarks/compare_profiles.py --results-root results \
        --output results/matrix_summary.csv
    python3 tests/benchmarks/compare_profiles.py --sort v2v_p95_ms
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# (csv column, summary key, statistic) — the latency spine of a voice turn.
LATENCY_COLUMNS: list[tuple[str, str, str]] = [
    ("v2v_p95_ms", "voice_to_voice_ms", "p95"),
    ("v2v_p50_ms", "voice_to_voice_ms", "median"),
    ("v2v_mean_ms", "voice_to_voice_ms", "mean"),
    ("v2v_informative_p95_ms", "voice_to_voice_informative_ms", "p95"),
    ("asr_p95_ms", "asr_ms", "p95"),
    ("asr_mean_ms", "asr_ms", "mean"),
    ("agent_ttft_p95_ms", "agent_ttft_ms", "p95"),
    ("agent_ttft_mean_ms", "agent_ttft_ms", "mean"),
    ("tts_p95_ms", "tts_ms", "p95"),
    ("tts_mean_ms", "tts_ms", "mean"),
    ("endpoint_wait_p95_ms", "endpoint_wait_ms", "p95"),
    ("time_to_first_audio_p95_ms", "time_to_first_audio_ms", "p95"),
]

# Hardware KPI name substrings -> csv column. consolidate_multiple_run_of_
# metrics.py emits slightly different labels depending on which collectors
# were active (e.g. "GPU_0 Compute[CCS] Utilization %"), so match by substring
# rather than exact key.
HARDWARE_COLUMNS: list[tuple[str, tuple[str, ...]]] = [
    ("cpu_util_pct", ("cpu utilization",)),
    ("npu_util_pct", ("npu utilization",)),
    ("gpu_util_pct", ("gpu utilization", "compute[ccs] utilization")),
    ("memory_util_pct", ("memory utilization",)),
    ("power_w", ("power draw",)),
    ("mem_bandwidth_mbps", ("memory bandwidth",)),
]


def _stat(summary: dict[str, Any], key: str, stat: str) -> float | str:
    """Pull one statistic out of a v2v summary block.

    Args:
        summary: The ``benchmark_report.summary`` mapping.
        key: Metric name, e.g. ``voice_to_voice_ms``.
        stat: Statistic name, e.g. ``p95``.

    Returns:
        The value, or "" when the metric/statistic is absent.
    """
    block = summary.get(key)
    if not isinstance(block, dict):
        return ""
    val = block.get(stat)
    return val if isinstance(val, (int, float)) else ""


def load_v2v(profile_dir: Path) -> tuple[dict[str, Any], str]:
    """Load the newest v2v result JSON in a profile directory.

    Args:
        profile_dir: Directory for one profile.

    Returns:
        Tuple of (summary mapping, source filename). Both empty when absent.
    """
    candidates = sorted(
        profile_dir.glob("smart_kiosk_v2v_results_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    # Also accept a plain <label>.json written by an explicit --label run.
    if not candidates:
        candidates = sorted(
            (p for p in profile_dir.glob("*.json") if p.name != "run_manifest.json"),
            key=lambda p: p.stat().st_mtime,
        )
    if not candidates:
        return {}, ""

    newest = candidates[-1]
    try:
        data = json.loads(newest.read_text())
    except Exception:
        return {}, newest.name

    report = data.get("benchmark_report") or {}
    summary = report.get("summary") or {}
    return (summary if isinstance(summary, dict) else {}), newest.name


def load_hardware(profile_dir: Path) -> dict[str, float]:
    """Parse ``consolidated_metrics.csv`` (flat key,value rows) into columns."""
    path = profile_dir / "consolidated_metrics.csv"
    if not path.is_file():
        return {}

    raw: list[tuple[str, str]] = []
    try:
        with path.open(newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    raw.append((row[0].strip().lower(), row[1].strip()))
    except Exception:
        return {}

    out: dict[str, float] = {}
    for col, needles in HARDWARE_COLUMNS:
        for key, val in raw:
            if any(n in key for n in needles):
                try:
                    out[col] = float(val)
                except ValueError:
                    continue
                break
    return out


def load_manifest(profile_dir: Path) -> dict[str, Any]:
    """Load ``run_manifest.json``, tolerating absence."""
    path = profile_dir / "run_manifest.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def collect(results_root: Path) -> list[dict[str, Any]]:
    """Build one result row per profile directory under ``results_root``."""
    rows: list[dict[str, Any]] = []

    for d in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if d.name == "warmup":
            continue
        summary, src = load_v2v(d)
        manifest = load_manifest(d)
        if not summary and not manifest:
            continue  # not a profile result directory

        devices = manifest.get("devices", {})
        models = manifest.get("models", {})
        run = manifest.get("run", {})
        vcs = manifest.get("vcs", {})

        row: dict[str, Any] = {
            "profile": manifest.get("profile") or d.name,
            "asr_device": devices.get("asr", ""),
            "tts_device": devices.get("tts", ""),
            "llm_device": devices.get("llm", ""),
            "asr_model": models.get("asr", ""),
            "tts_model": models.get("tts", ""),
            "tts_dtype": models.get("tts_dtype", ""),
            "turns_ok": summary.get("turns_ok", ""),
            "turns_failed": summary.get("turns_failed", ""),
        }
        for col, key, stat in LATENCY_COLUMNS:
            row[col] = _stat(summary, key, stat)
        row.update({c: "" for c, _ in HARDWARE_COLUMNS})
        row.update(load_hardware(d))
        row.update(
            {
                "queue_enabled": run.get("queue_enabled", ""),
                "diarization_enabled": run.get("diarization_enabled", ""),
                "v2v_runs": run.get("v2v_runs", ""),
                "valid": manifest.get("valid", ""),
                "git_sha": vcs.get("git_sha", ""),
                "git_dirty": vcs.get("git_dirty", ""),
                "started_at": run.get("started_at", ""),
                "source_file": src,
            }
        )
        rows.append(row)

    return rows


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--output", default="")
    ap.add_argument(
        "--sort",
        default="v2v_p95_ms",
        help="column to sort by (blank rows last); default v2v_p95_ms",
    )
    args = ap.parse_args()

    root = Path(args.results_root)
    if not root.is_dir():
        print(f"no such results root: {root}", file=sys.stderr)
        return 1

    rows = collect(root)
    if not rows:
        print(
            f"No profile result directories found under {root}/.\n"
            "Run 'make benchmark PROFILE=<name>' or 'make benchmark-matrix' first.",
            file=sys.stderr,
        )
        return 1

    if args.sort and args.sort in rows[0]:
        rows.sort(
            key=lambda r: (
                r.get(args.sort) == "",
                r.get(args.sort) if r.get(args.sort) != "" else 0,
            )
        )

    out = Path(args.output) if args.output else root / "matrix_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Console table — the headline comparison, p95 first.
    print(f"\n{'profile':<24} {'ASR':<5} {'TTS':<5} {'LLM':<5} "
          f"{'v2v p95':>9} {'v2v p50':>9} {'asr p95':>8} {'ttft p95':>9} "
          f"{'tts p95':>8} {'CPU%':>6} {'GPU%':>6} {'NPU%':>6}")
    print("-" * 118)
    for r in rows:
        def f(key: str, nd: int = 1) -> str:
            v = r.get(key, "")
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"

        flag = "" if r.get("valid") in (True, "") else " !"
        print(
            f"{str(r['profile'])[:24]:<24} {str(r['asr_device']):<5} "
            f"{str(r['tts_device']):<5} {str(r['llm_device']):<5} "
            f"{f('v2v_p95_ms'):>9} {f('v2v_p50_ms'):>9} {f('asr_p95_ms'):>8} "
            f"{f('agent_ttft_p95_ms'):>9} {f('tts_p95_ms'):>8} "
            f"{f('cpu_util_pct'):>6} {f('gpu_util_pct'):>6} "
            f"{f('npu_util_pct'):>6}{flag}"
        )
    print(f"\nWrote {out}  ({len(rows)} profile(s))")
    print("Judge on p95 — a single bad tail turn is what a customer notices.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
