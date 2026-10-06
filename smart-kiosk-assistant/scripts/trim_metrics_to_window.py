"""Trim hardware counters to the window the measured turns ran over.

The metrics-collector container samples CPU, GPU, NPU and memory for the whole
time the stack is up -- image pulls, the health wait, the discarded warmup,
and the teardown. Averaging the raw files therefore describes a stack that was
mostly idle, not the turns being measured (PR #112 review, item 18). On an
88-turn run the collector was alive for ~15 minutes while the measured turns
spanned ~6 of them.

``benchmark_smart_kiosk_v2v.py`` records the measured span in
``results/measured_window.json``. This script narrows each counter file to that
span *before* ``make parse-qmassa-metrics`` and ``make consolidate-metrics``
read them. Every source is timestamped, so the trim is exact rather than
proportional:

* ``cpu_usage.log`` -- sar text, ``HH:MM:SS`` per row
* ``memory_usage.log`` -- repeated ``free`` output, no timestamps at all
* ``npu_usage.csv`` -- ISO-8601 per row
* ``qmassa*-tool-generated.json`` -- per-state offsets in ms from collector start

None of those three stamps is an epoch time, and the first two are written in
the *container's* timezone, which need not match the host's. Rather than guess
at an offset, every format is anchored the same way: the last sample in a file
was written at (approximately) the file's modification time, which pins the
whole series onto the epoch clock without ever naming a timezone. This is why
``make sync-metrics-into-results`` copies with ``cp -p`` -- a copy that reset
the modification times would destroy the anchor.

Each original is kept alongside with a ``.full`` suffix. That suffix is
deliberately chosen so it no longer matches the ``*.log`` / ``*.csv`` /
``*-tool-generated.json`` patterns the downstream tools glob for, which would
otherwise pull the untrimmed data straight back in.

Usage::

    python3 scripts/trim_metrics_to_window.py --results-dir results
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import re
import sys
from pathlib import Path

_SAR_ROW_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\s")


def _is_sar_header(line: str) -> bool:
    """True for sar's repeated column-header line.

    sar prefixes its column header with a timestamp exactly like a data row
    ("03:00:00        CPU     %user ... %idle"), and reprints it periodically.
    Treating one as a sample drops it whenever it falls outside the window,
    which leaves the downstream parser unable to find its columns -- a broken
    parse rather than a skewed average.
    """
    return any(token.startswith("%") for token in line.split())


def cpu_log_span_seconds(path: Path) -> float | None:
    """Seconds between the first and last sample of a sar log.

    Used to date ``memory_usage.log``, which carries no timestamps of its own
    but is written by the same collector over the same lifetime. Must be
    called before the sar log is trimmed.

    Args:
        path: An untrimmed ``cpu_usage.log``.

    Returns:
        The span in seconds, or None when there are too few rows to measure.
    """
    if not path.exists():
        return None
    rows = [
        m
        for l in path.read_text(errors="replace").splitlines()
        if (m := _SAR_ROW_RE.match(l)) and not _is_sar_header(l)
    ]
    if len(rows) < 2:
        return None
    first = _seconds_of_day(*(int(g) for g in rows[0].groups()))
    last = _seconds_of_day(*(int(g) for g in rows[-1].groups()))
    span = last - first
    if span < 0:
        span += 86400
    return float(span)


def trim_memory_log(
    path: Path, start_ms: int, end_ms: int, span_s: float | None
) -> tuple[int, int]:
    """Trim the repeated ``free`` output to the measured window.

    ``memory_usage.log`` is the one counter with no timestamps -- it is just
    ``free`` run on a loop. Its samples are evenly spaced, so each block is
    dated by assuming the last one landed at the file's modification time and
    that the blocks together span the collector's lifetime, taken from
    ``cpu_usage.log``.

    This is the only file here dated by interpolation rather than from its own
    stamps, so it is left untouched when the lifetime is unknown: an untrimmed
    average is a known quantity, a misaligned one is not.

    Args:
        path: ``memory_usage.log``, rewritten in place.
        start_ms: Window start, epoch ms.
        end_ms: Window end, epoch ms.
        span_s: Collector lifetime in seconds, from ``cpu_log_span_seconds``.

    Returns:
        ``(kept, dropped)`` block counts.
    """
    text = path.read_text(errors="replace")
    blocks = re.split(r"(?=^\s*total\s+used\s+free)", text, flags=re.MULTILINE)
    blocks = [b for b in blocks if b.strip()]
    if len(blocks) < 2 or not span_s:
        return len(blocks), 0

    mtime_ms = path.stat().st_mtime * 1000
    interval_ms = (span_s * 1000) / (len(blocks) - 1)

    kept = []
    dropped = 0
    for i, block in enumerate(blocks):
        stamp_ms = mtime_ms - (len(blocks) - 1 - i) * interval_ms
        if start_ms <= stamp_ms <= end_ms:
            kept.append(block)
        else:
            dropped += 1

    if dropped and kept:
        _backup(path)
        path.write_text("".join(kept))
        return len(kept), dropped
    return len(blocks), 0


def _iso_ms(value: str) -> float:
    """Milliseconds for a naive ISO-8601 timestamp, on an arbitrary origin.

    Only ever used for differences between two stamps from the same file, so
    the timezone the value was written in does not matter.
    """
    return datetime.datetime.fromisoformat(value).timestamp() * 1000


def _backup(path: Path) -> None:
    """Preserve the untrimmed original next to the trimmed file."""
    os.replace(path, path.with_suffix(path.suffix + ".full"))


def _seconds_of_day(h: int, m: int, s: int) -> int:
    """Seconds since midnight, used to order sar rows within a run."""
    return h * 3600 + m * 60 + s


def trim_sar_log(path: Path, start_ms: int, end_ms: int) -> tuple[int, int]:
    """Trim a sar text log to the measured window.

    sar rows carry a time of day with no date and no timezone, so they are
    placed on the epoch clock by anchoring the final row to the file's
    modification time and measuring every other row back from it. A run that
    crosses midnight shows up as the time of day jumping forward relative to
    the last row, which is corrected by subtracting a day.

    Header lines (the kernel banner, blank lines, the repeated column header)
    carry no timestamp and are always kept -- the parsers downstream rely on
    them to find the columns.

    Args:
        path: The sar log to trim, rewritten in place.
        start_ms: Window start, epoch ms.
        end_ms: Window end, epoch ms.

    Returns:
        ``(kept, dropped)`` row counts.
    """
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    rows = [
        (i, m)
        for i, l in enumerate(lines)
        if (m := _SAR_ROW_RE.match(l)) and not _is_sar_header(l)
    ]
    if not rows:
        return len(lines), 0

    mtime_ms = path.stat().st_mtime * 1000
    last_sod = _seconds_of_day(*(int(g) for g in rows[-1][1].groups()))

    drop_indices: set[int] = set()
    for idx, match in rows:
        sod = _seconds_of_day(*(int(g) for g in match.groups()))
        delta_s = sod - last_sod
        if delta_s > 0:
            # Later in the day than the final row: the run crossed midnight.
            delta_s -= 86400
        stamp_ms = mtime_ms + delta_s * 1000
        if not (start_ms <= stamp_ms <= end_ms):
            drop_indices.add(idx)

    if not drop_indices:
        return len(lines), 0

    kept = [l for i, l in enumerate(lines) if i not in drop_indices]
    _backup(path)
    path.write_text("".join(kept))
    return len(kept), len(drop_indices)


def trim_npu_csv(path: Path, start_ms: int, end_ms: int) -> tuple[int, int]:
    """Trim the NPU usage CSV to the measured window.

    Args:
        path: ``npu_usage.csv``, rewritten in place.
        start_ms: Window start, epoch ms.
        end_ms: Window end, epoch ms.

    Returns:
        ``(kept, dropped)`` data-row counts, excluding the header.
    """
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0, 0

    header, data = rows[0], rows[1:]

    # The CSV's timestamps are naive local-to-the-container times. Shift the
    # whole series so its final row lands on the file's modification time,
    # which puts it on the same epoch clock as the window without having to
    # know the container's timezone.
    anchor_delta_ms = 0.0
    for row in reversed(data):
        if not row:
            continue
        try:
            anchor_delta_ms = path.stat().st_mtime * 1000 - _iso_ms(row[0])
        except ValueError:
            continue
        break

    kept: list[list[str]] = []
    dropped = 0
    for row in data:
        if not row:
            continue
        try:
            stamp_ms = _iso_ms(row[0]) + anchor_delta_ms
        except ValueError:
            # Unparseable timestamp: keep it rather than silently discarding
            # data this script does not understand.
            kept.append(row)
            continue
        if start_ms <= stamp_ms <= end_ms:
            kept.append(row)
        else:
            dropped += 1

    if dropped:
        _backup(path)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(kept)
    return len(kept), dropped


def trim_qmassa_json(path: Path, start_ms: int, end_ms: int) -> tuple[int, int]:
    """Trim a raw qmassa document to the measured window.

    qmassa stamps each state with milliseconds elapsed since *its own* start,
    not epoch time, so the two clocks have to be aligned. The file is rewritten
    from scratch on every sample, which makes its modification time the instant
    the final state was written -- aligning that with the final state's offset
    recovers the collector's start instant.

    Args:
        path: A ``qmassa*-tool-generated.json`` file, rewritten in place.
        start_ms: Window start, epoch ms.
        end_ms: Window end, epoch ms.

    Returns:
        ``(kept, dropped)`` state counts.
    """
    doc = json.loads(path.read_text())
    states = doc.get("states") or []
    if not states:
        return 0, 0

    def offset_ms(state: dict) -> int | None:
        stamps = state.get("timestamps") or []
        return stamps[-1] if stamps else None

    last_offset = next(
        (offset_ms(st) for st in reversed(states) if offset_ms(st) is not None), None
    )
    if last_offset is None:
        return len(states), 0

    collector_start_ms = path.stat().st_mtime * 1000 - last_offset

    kept = []
    dropped = 0
    for state in states:
        offset = offset_ms(state)
        if offset is None:
            # No usable timestamp -- keep it, for the same reason as above.
            kept.append(state)
            continue
        stamp_ms = collector_start_ms + offset
        if start_ms <= stamp_ms <= end_ms:
            kept.append(state)
        else:
            dropped += 1

    if dropped and kept:
        _backup(path)
        doc["states"] = kept
        path.write_text(json.dumps(doc))
    elif dropped and not kept:
        # Trimming to nothing would leave the parser with an empty document
        # and the run with no GPU figures at all. A wrong-looking average is
        # more useful than a missing one, so leave the file alone and say so.
        print(
            f"  {path.name}: window matched no samples; left untrimmed "
            "(check that the collector and the measured run overlap)",
            file=sys.stderr,
        )
        return len(states), 0
    return len(kept), dropped


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()

    results = Path(args.results_dir)
    window_path = results / "measured_window.json"
    if not window_path.exists():
        # Expected whenever metrics are consolidated outside a benchmark run.
        print(f"No {window_path}; leaving hardware counters untrimmed.")
        return 0

    window = json.loads(window_path.read_text())
    start_ms, end_ms = int(window["start_ms"]), int(window["end_ms"])
    print(
        f"Trimming hardware counters to the measured window "
        f"({window.get('duration_s')}s)..."
    )

    # Measured before cpu_usage.log is trimmed -- memory_usage.log is dated
    # from it.
    span_s = cpu_log_span_seconds(results / "cpu_usage.log")

    cpu = results / "cpu_usage.log"
    if cpu.exists():
        kept, dropped = trim_sar_log(cpu, start_ms, end_ms)
        print(f"  cpu_usage.log: kept {kept}, dropped {dropped}")

    mem = results / "memory_usage.log"
    if mem.exists():
        kept, dropped = trim_memory_log(mem, start_ms, end_ms, span_s)
        print(f"  memory_usage.log: kept {kept} block(s), dropped {dropped}")

    npu = results / "npu_usage.csv"
    if npu.exists():
        kept, dropped = trim_npu_csv(npu, start_ms, end_ms)
        print(f"  npu_usage.csv: kept {kept}, dropped {dropped}")

    for qmassa in glob.glob(str(results / "qmassa*-tool-generated.json")):
        path = Path(qmassa)
        kept, dropped = trim_qmassa_json(path, start_ms, end_ms)
        print(f"  {path.name}: kept {kept}, dropped {dropped}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
