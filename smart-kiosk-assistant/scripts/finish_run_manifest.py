"""Mark a run manifest as completed.

Written as a separate step rather than folded into ``write_run_manifest.py``
because the manifest has to exist *before* the measured runs start -- it
records the configuration the stack was brought up with -- while the fact
that the run finished is only known afterwards.

Without this, a benchmark that dies partway leaves a fresh manifest
timestamp sitting next to whatever result files the *previous* run left in
the same directory, and the two read as if they belong together (PR #112
review, item 16).

Usage::

    python3 scripts/finish_run_manifest.py --results-dir results
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path


def finish(results_dir: str) -> int:
    """Flip ``run.status`` to ``completed`` and stamp the finish time.

    Args:
        results_dir: Directory holding ``run_manifest.json``.

    Returns:
        Process exit code. Always 0 -- a missing or unreadable manifest is
        reported but never fails the benchmark, because the measured runs
        themselves have already succeeded by the time this is called.
    """
    path = Path(results_dir) / "run_manifest.json"
    if not path.exists():
        print(f"No manifest at {path}; nothing to finish.", file=sys.stderr)
        return 0

    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read {path}: {exc}", file=sys.stderr)
        return 0

    run = manifest.setdefault("run", {})
    run["status"] = "completed"
    run["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Marked {path} completed")
    return 0


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()
    return finish(args.results_dir)


if __name__ == "__main__":
    sys.exit(main())
