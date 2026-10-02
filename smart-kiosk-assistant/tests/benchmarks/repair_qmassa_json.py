"""Repair truncated qmassa*-tool-generated.json files in place.

qmassa (invoked by performance-tools' collect_gpu.sh, running inside the
metrics-collector container) periodically rewrites a single JSON document
(``{"version": ..., "args": {...}, "states": [ <state>, <state>, ... ]}``) --
each write serializes the whole accumulated state list from scratch, it does
not append. If the container is killed mid-write (SIGKILL after Docker's stop
grace period elapses, or -- observed in practice -- CPU/IO contention while
`docker compose down` tears down the whole Smart Kiosk stack in parallel,
which slows this rewrite down enough to lose the race even with a generous
`stop_grace_period`), the file is left truncated mid-object and
performance-tools' ``parse_qmassa_metrics_to_json.py`` rejects it with
"No valid state data found".

This is a repair pass, not a graceful-shutdown fix: raising
`stop_grace_period` on the metrics-collector service (see docker-compose.yml)
reduces how often this happens, but cannot make a hard kill impossible, so
this script is the safety net that keeps `make consolidate-metrics`/
`make plot-metrics` reliable regardless of exactly when the container died.

Approach: track bracket depth ({/}/[/]) outside of quoted strings, and note
every position where depth returns to exactly 2 (i.e. we are back at the
top level of the "states" array, between one complete state entry and the
next). The last such position is a guaranteed-safe truncation point --
whatever came after it was a partial state entry. Truncate there and append
the closing brackets needed to re-balance the document (closing "states"
array's `]` and the root object's `}`), independent of how deeply nested an
individual state entry happens to be.

Usage::

    python repair_qmassa_json.py --dir ./results
    python repair_qmassa_json.py --dir ./results --keyword qmassa
"""

import argparse
import glob
import json
import os


def is_valid_json(content: str) -> bool:
    try:
        json.loads(content)
        return True
    except json.JSONDecodeError:
        return False


def find_last_safe_truncation(content: str) -> tuple[int | None, list[str] | None]:
    """Return (position, open_bracket_stack) at the last depth==2 boundary.

    ``open_bracket_stack`` is the stack of currently-open brackets ('{'/'[')
    at that exact position, in outer-to-inner order -- needed to close them
    in the right (reverse) order.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    last_safe_pos = None
    last_safe_stack: list[str] | None = None

    for i, ch in enumerate(content):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if len(stack) == 2:
                last_safe_pos = i + 1
                last_safe_stack = list(stack)

    return last_safe_pos, last_safe_stack


def repair_file(path: str) -> str:
    """Repair one qmassa JSON file in place if truncated.

    Returns one of "already_valid", "repaired", "unrepairable" for the
    caller's summary line.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if is_valid_json(content):
        return "already_valid"

    pos, stack = find_last_safe_truncation(content)
    if pos is None or not stack:
        return "unrepairable"

    closers = "".join("}" if c == "{" else "]" for c in reversed(stack))
    repaired = content[:pos] + closers

    if not is_valid_json(repaired):
        return "unrepairable"

    with open(path, "w", encoding="utf-8") as f:
        f.write(repaired)
    return "repaired"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", "-d", default="./results", help="Directory to scan")
    parser.add_argument(
        "--keyword", "-k", default="qmassa", help="Filename prefix to match"
    )
    args = parser.parse_args()

    pattern = os.path.join(args.dir, f"{args.keyword}*-tool-generated.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matching {pattern}; nothing to repair.")
        return 0

    exit_code = 0
    for path in files:
        status = repair_file(path)
        print(f"{os.path.basename(path)}: {status}")
        if status == "unrepairable":
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
