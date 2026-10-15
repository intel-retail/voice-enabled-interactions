"""Tests for scripts/trim_metrics_to_window.py.

Background (PR #112 review, item 18): the metrics-collector samples CPU, GPU,
NPU and memory for the whole time the stack is up -- image pulls, the health
wait, the discarded warmup, teardown -- so averaging the raw counter files
describes a mostly-idle stack rather than the turns being measured.

The thing worth testing is the dating, not the filtering. None of these four
formats records an epoch timestamp, and two of them are written in the
*container's* timezone, which need not match the host's. Every format is
therefore anchored off the file's modification time. A regression here would
not crash: it would quietly keep the wrong samples and report a plausible but
wrong utilisation figure.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from trim_metrics_to_window import (  # noqa: E402
    cpu_log_span_seconds,
    trim_memory_log,
    trim_npu_csv,
    trim_qmassa_json,
    trim_sar_log,
)

# The collector ran for 10 samples; only the last 4 seconds are measured.
_MTIME = 1_700_000_000.0
_WINDOW_START_MS = int((_MTIME - 3) * 1000)
_WINDOW_END_MS = int((_MTIME + 1) * 1000)


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    os.utime(path, (_MTIME, _MTIME))
    return path


@pytest.fixture
def cpu_log(tmp_path: Path) -> Path:
    # Deliberately written in a timezone that is almost certainly NOT the
    # host's: the anchoring must not depend on interpreting these as local
    # times. Rows run 03:00:00..03:00:09, one per second.
    rows = "\n".join(
        f"03:00:{i:02d}        all     10.00      0.00      5.00      0.00      0.00     85.00"
        for i in range(10)
    )
    return _write(
        tmp_path / "cpu_usage.log",
        "Linux 7.0.0 (host) \t10/06/26 \t_x86_64_\t(16 CPU)\n\n"
        "03:00:00        CPU     %user     %nice   %system   %iowait    %steal     %idle\n"
        + rows
        + "\n",
    )


def test_sar_log_keeps_only_the_measured_tail(cpu_log: Path):
    kept, dropped = trim_sar_log(cpu_log, _WINDOW_START_MS, _WINDOW_END_MS)
    assert dropped == 6
    text = cpu_log.read_text()
    # The final four samples are inside the window...
    for i in (6, 7, 8, 9):
        assert f"03:00:{i:02d}        all" in text
    # ...and everything before it is gone.
    for i in (0, 1, 2, 5):
        assert f"03:00:{i:02d}        all" not in text
    assert kept == 3 + 4  # banner + blank + column header + 4 rows


def test_sar_log_preserves_header_lines(cpu_log: Path):
    trim_sar_log(cpu_log, _WINDOW_START_MS, _WINDOW_END_MS)
    text = cpu_log.read_text()
    # Downstream parsers locate the columns from these; dropping them would
    # break parsing rather than merely skew the average.
    assert text.startswith("Linux 7.0.0")
    assert "%idle" in text


def test_sar_log_keeps_untrimmed_original(cpu_log: Path):
    trim_sar_log(cpu_log, _WINDOW_START_MS, _WINDOW_END_MS)
    backup = cpu_log.with_suffix(".log.full")
    assert backup.exists()
    # ".full" must not match the *.log glob the collector tooling uses, or the
    # untrimmed data would be read straight back in alongside the trimmed file.
    assert not backup.name.endswith(".log")
    assert "03:00:00        all" in backup.read_text()


def test_sar_log_untouched_when_window_covers_everything(cpu_log: Path):
    kept, dropped = trim_sar_log(cpu_log, 0, int((_MTIME + 60) * 1000))
    assert dropped == 0
    assert not cpu_log.with_suffix(".log.full").exists()
    assert kept == 13


def test_sar_log_handles_a_run_crossing_midnight(tmp_path: Path):
    # Samples at 23:59:58, 23:59:59, 00:00:00, 00:00:01 -- the last row is the
    # anchor, so the two 23:59:5x rows are the day before, not 24h later.
    rows = ["23:59:58", "23:59:59", "00:00:00", "00:00:01"]
    path = _write(
        tmp_path / "cpu_usage.log",
        "\n".join(f"{t}        all     10.00" for t in rows) + "\n",
    )
    # Window covers the final two seconds only.
    trim_sar_log(path, int((_MTIME - 1) * 1000), int((_MTIME + 1) * 1000))
    text = path.read_text()
    assert "23:59:58" not in text
    assert "00:00:00        all" in text
    assert "00:00:01        all" in text


def test_npu_csv_is_anchored_not_read_as_local_time(tmp_path: Path):
    # Naive timestamps in an arbitrary timezone, one per second. If these were
    # compared against the window as local times they would all fall outside
    # it and the file would be emptied.
    rows = "\n".join(
        f"2026-10-06T03:00:{i:02d}.000000,{i}.00" for i in range(10)
    )
    path = _write(tmp_path / "npu_usage.csv", "timestamp,percent_usage\n" + rows + "\n")

    kept, dropped = trim_npu_csv(path, _WINDOW_START_MS, _WINDOW_END_MS)
    assert (kept, dropped) == (4, 6)
    text = path.read_text()
    assert text.startswith("timestamp,percent_usage")
    assert "03:00:09" in text
    assert "03:00:00" not in text


def test_qmassa_states_are_dated_from_their_own_offsets(tmp_path: Path):
    # qmassa stamps ms since its own start, and rewrites the whole document
    # every sample -- so the final state's offset lands at the mtime.
    states = [
        {"timestamps": [i * 1000], "devs_state": [{"n": i}]} for i in range(10)
    ]
    path = tmp_path / "qmassa0-xe-tool-generated.json"
    _write(path, json.dumps({"version": "1", "args": {}, "states": states}))

    kept, dropped = trim_qmassa_json(path, _WINDOW_START_MS, _WINDOW_END_MS)
    assert (kept, dropped) == (4, 6)
    remaining = json.loads(path.read_text())["states"]
    assert [s["devs_state"][0]["n"] for s in remaining] == [6, 7, 8, 9]


def test_qmassa_left_alone_when_nothing_matches(tmp_path: Path):
    # Trimming to nothing would leave the run with no GPU figures at all. A
    # wrong-looking average is more useful than a missing one.
    states = [{"timestamps": [i * 1000], "devs_state": []} for i in range(5)]
    path = tmp_path / "qmassa0-xe-tool-generated.json"
    _write(path, json.dumps({"states": states}))

    kept, dropped = trim_qmassa_json(path, 0, 1000)
    assert (kept, dropped) == (5, 0)
    assert not path.with_suffix(".json.full").exists()


def test_memory_log_is_trimmed_by_block_position(tmp_path: Path, cpu_log: Path):
    # `free` output has no timestamps at all, so blocks are dated by assuming
    # they are evenly spread over the collector lifetime measured from the
    # sar log.
    block = (
        "               total        used        free      shared  buff/cache   available\n"
        "Mem:        65386900    {used}    15637768      295900    30735636    45487728\n"
        "Swap:        6094844     1991760     4103084\n\n"
    )
    path = _write(
        tmp_path / "memory_usage.log",
        "".join(block.format(used=10000000 + i) for i in range(10)),
    )
    span_s = cpu_log_span_seconds(cpu_log)
    assert span_s == 9.0

    kept, dropped = trim_memory_log(path, _WINDOW_START_MS, _WINDOW_END_MS, span_s)
    assert (kept, dropped) == (4, 6)
    text = path.read_text()
    assert "10000009" in text
    assert "10000000" not in text


def test_memory_log_left_alone_without_a_known_lifetime(tmp_path: Path):
    # This is the one file dated by interpolation rather than from its own
    # stamps. With no lifetime to interpolate over, an untrimmed average is a
    # known quantity and a misaligned one is not.
    path = _write(
        tmp_path / "memory_usage.log",
        "               total        used        free\n"
        "Mem:        65386900    19899172    15637768\n\n" * 4,
    )
    before = path.read_text()
    kept, dropped = trim_memory_log(path, _WINDOW_START_MS, _WINDOW_END_MS, None)
    assert dropped == 0
    assert path.read_text() == before


def test_cpu_log_span_needs_at_least_two_rows(tmp_path: Path):
    path = _write(tmp_path / "cpu_usage.log", "Linux 7.0.0\n\n03:00:00  all  1.0\n")
    assert cpu_log_span_seconds(path) is None
    assert cpu_log_span_seconds(tmp_path / "absent.log") is None
