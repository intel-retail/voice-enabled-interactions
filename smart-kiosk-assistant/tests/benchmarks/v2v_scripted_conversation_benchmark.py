#!/usr/bin/env python3
"""Voice-to-voice latency benchmark using TTS-synthesised scripted conversations.

Why this exists
----------------
``v2v_fixture_benchmark.py`` replays real HUMAN recordings -- the most
faithful source of "what does a real customer's trailing silence look
like", but no recordings are bundled in this repo (kept free of binary
audio) and it requires the caller to supply fixtures via ``--fixture``,
limited to whatever handful of recordings the caller has: one line each,
no multi-turn conversations.

``kiosk-voice-lab-main`` (the reference lab implementation) instead
synthesises its own scripted customer conversations with its TTS engine
(Kokoro) and feeds the resulting audio into the live pipeline exactly like a
real mic would -- see ``pipeline/live.py``'s ``_feed_replay()``. That is the
method this script ports: it lets us author arbitrary multi-turn ordering
conversations (not just single lines) and benchmark them the same way,
directly comparable to the lab's own ``make replay`` / ``make bench``
numbers, without needing a library of real recordings for every scenario.

Method
------
* Each scripted customer line is synthesised with kiosk-core's OWN
  text-to-speech service (``synthesize_prompt_wav``/``resample_wav_to_16k_mono``,
  reused from ``agent_latency_benchmark.py``) -- same TTS backend the kiosk
  itself uses, so the synthesised voice is representative.
* ``lead_pad_seconds`` of silence is prepended and ``trail_pad_seconds`` of
  silence is appended to each line's audio (defaults 0.3s / 2.5s, matching
  ``kiosk-voice-lab-main``'s ``_feed_replay()`` padding) so the endpoint
  detector has real trailing silence to key off, exactly as it would for a
  live customer who has finished talking.
* Each padded line is written to a temporary WAV file and replayed through
  ``v2v_fixture_benchmark.replay_fixture()`` UNMODIFIED -- same
  ``start-stream`` + chunked-push + ``audio/end`` flow the browser UI and the
  human-recording benchmark both use, so results are directly comparable
  across all three benchmarks.
* One session (one kiosk-core "turn") per scripted line, run sequentially --
  kiosk-core's session model is one turn per session (see
  ``BrowserStreamSession``), same limitation ``v2v_fixture_benchmark.py``
  already has when replaying multiple fixtures.

Usage::

    # Default: reads Sample_data/conversation.jsonl (one file, edit it to
    # change the scripted conversation -- no code change needed)
    python tests/benchmarks/v2v_scripted_conversation_benchmark.py --runs 3

    # Point at a different conversation file
    python tests/benchmarks/v2v_scripted_conversation_benchmark.py \\
        --conversation-file Sample_data/another_conversation.jsonl --runs 2

    # Legacy named scripts (kept for backward compatibility) still work too
    python tests/benchmarks/v2v_scripted_conversation_benchmark.py \\
        --script order-multi --runs 2 --label v2v-scripted-multi

    python tests/benchmarks/v2v_scripted_conversation_benchmark.py --list-scripts

Stdlib only (plus reusing sibling benchmark helpers), consistent with the
sibling benchmarks.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
import wave
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parents[1]  # tests/benchmarks/<this file> -> smart-kiosk-assistant/
for _p in (BENCHMARKS_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v2v_fixture_benchmark as v2v  # noqa: E402  (path inserts above must run first)
from agent_latency_benchmark import (  # noqa: E402
    resample_wav_to_16k_mono,
    synthesize_prompt_wav,
)

from kiosk_core import config  # noqa: E402

RESULTS_DIR = v2v.RESULTS_DIR

# ---------------------------------------------------------------------------
# Sample_data conversation file (default conversation source)
# ---------------------------------------------------------------------------
# Rather than hardcoding the scripted conversation in this file (see the
# legacy SCRIPTS dict below), the default source is a single JSONL file under
# Sample_data/. Format matches conversations/*.jsonl (one {"user": ...,
# "assistant": ...} object per line, "assistant" ignored here since this
# benchmark only speaks the customer's lines); to change what gets
# benchmarked, edit Sample_data/conversation.jsonl -- no code change needed.
SAMPLE_DATA_DIR = REPO_ROOT / "Sample_data"
DEFAULT_CONVERSATION_FILE = SAMPLE_DATA_DIR / "conversation.jsonl"


def load_conversation_file(path: Path) -> list[str]:
    """Read the customer's ``user`` lines, in order, from a conversation JSONL file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Conversation file not found: {path}\n"
            f"  Create it (one {{\"user\": \"...\"}} JSON object per line) or pass "
            f"--conversation-file to point at a different one."
        )
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        user = (rec.get("user") or "").strip()
        if user:
            lines.append(user)
    if not lines:
        raise ValueError(f"Conversation file {path} contains no usable \"user\" lines")
    return lines

# ---------------------------------------------------------------------------
# performance-tools vlm_metrics_logger integration
# ---------------------------------------------------------------------------
# Previously this harness emitted a SYNTHESIZED start/end pair per turn itself
# (end=now, start=end-voice_to_voice_ms), because kiosk-core only exposed
# voice_to_voice_ms after the fact over HTTP and the harness had no access to
# the real last-word/first-audio instants.
#
# kiosk-core now emits these natively and unconditionally -- see
# kiosk_core.audio_session._emit_vlm_metrics(), called right after every
# completed turn's TurnTrace is recorded, using the REAL monotonic marks
# (converted to epoch ms via a same-instant anchor pair), not a
# post-hoc-synthesized duration. This harness deliberately no longer emits at
# all: doing so as well would double-count every turn (two start/end pairs
# under two different unique_ids landing in the same
# vlm_application_metrics_*.txt) when collect_vlm_logger_metrics() sums
# transactions across all files in --results-dir. kiosk-core is configured to
# write to the same host ./results directory this harness already uses (see
# docker-compose.yml's CONTAINER_RESULTS_PATH/./results:/app/results mount),
# so no code here has to do anything for those files to appear.

# ---------------------------------------------------------------------------
# Scripted conversations -- QSR ordering domain, real product names from
# configs/ordering/products.yaml so the agent/RAG/ordering tools resolve them
# exactly as they would for a live customer. Modelled on
# kiosk-voice-lab-main's session2..5/demo/drift scripts: short, realistic,
# multi-turn where useful.
#
# IMPORTANT when authoring multi-turn scripts -- ALWAYS NAME THE PRODUCT:
# each line is replayed as its own kiosk-core session with its own freshly
# generated conversation_id (see replay_fixture), so the AGENT carries no
# memory from the previous line. Continuity across turns comes from the CART,
# not the conversation: the benchmark sends no user_id, so every turn runs as
# config.DEFAULT_ORDERING_USER_ID and the ordering repository resolves the
# same active draft order for all of them (repository.py looks the draft up
# by user_id). Adding items and confirming therefore work across lines, but
# anaphora does not -- "add one of those too" has no referent and will not
# resolve. Say "add one Chocolate Brownie".
#
# Multi-turn scripts should also END IN A CONFIRM. Confirming flips the draft
# to 'confirmed', so the next --runs repetition starts from an empty cart. A
# script that never confirms leaves a draft behind that the next run keeps
# adding to, inflating later turns' totals and upsell behaviour.
#
# NOTE: this dict is kept only for backward-compatible --script NAME runs.
# The default conversation source is now Sample_data/conversation.jsonl (see
# load_conversation_file() above) -- edit that file to change what gets
# benchmarked by default, rather than adding entries here.
# ---------------------------------------------------------------------------
SCRIPTS: dict[str, list[str]] = {
    # One turn -- directly comparable to the single-line rec1_16k.wav fixture.
    "order-simple": [
        "Hi, I would like to order one Classic Chicken Burger, please.",
    ],
    # A realistic three-turn order: base item, an addition, then confirm.
    "order-multi": [
        "Hi, can I get one Classic Chicken Burger and a regular Classic French Fries.",
        "Actually, please also add a Pepsi three thirty ml.",
        "Yes, that is everything, please confirm my order.",
    ],
    # A hesitant customer who changes their mind mid-order -- the lab's
    # "drift" scenario, ported: stresses the endpoint detector with
    # mid-thought pauses and self-corrections.
    "order-drift": [
        "Um, I think I want... the Spicy Chicken Crunch Burger.",
        "Wait, actually, can you make that the Double Chicken Tower instead.",
        "And one Chocolate Brownie too, please.",
    ],
    # Full QSR journey: order an item, ASK A PRICE (a knowledge/menu-lookup
    # turn rather than an ordering action), accept the upsell the burger
    # triggers, then confirm. Exercises a different tool path per turn --
    # place_order -> list_products -> place_order -> confirm_active_order --
    # so a regression in any one of them shows up as a failed/slow turn here
    # rather than only in the single-action scripts above.
    #
    # Chocolate Brownie (DESSERT-001) is not an arbitrary dessert: it is a
    # genuine upsell target for the burgers category in
    # configs/ordering/upsell_rules.yaml ("Leave room for dessert..."), so
    # turn 3 accepts something the agent itself would have suggested on
    # turn 1.
    "order-upsell": [
        "Hi, I would like to order one Classic Chicken Burger, please.",
        "How much does the Chocolate Brownie cost?",
        "Okay, please add one Chocolate Brownie to my order.",
        "That is everything, please confirm my order.",
    ],
}

DEFAULT_LEAD_PAD_SECONDS = 0.3
DEFAULT_TRAIL_PAD_SECONDS = 2.5

# --explicit-end-mark trailing pad: just enough real silence for ASR to
# safely capture the last word's decay tail (matches
# config.DEFAULT_ASR_TRIM_DECAY_SECONDS) without giving the VAD/silence-run
# loop time to accumulate anywhere near DEFAULT_ENDPOINT_SHORT_SECONDS or
# silence_timeout_seconds before the explicit /audio/end call arrives. This
# is what makes the mode bypass the endpoint-completeness-shortcut's
# documented 0%-80% run-to-run firing-rate lottery (see
# BaseAudioSession._endpoint_transcript_stable's docstring): rather than
# waiting for the shortcut to notice silence and decide the sentence "reads
# complete", the test harness just tells kiosk-core directly, the same way
# a live customer releasing a push-to-talk button would.
DEFAULT_END_MARK_TRAIL_PAD_SECONDS = 0.15
# 0.15 s (not the original 0.3 s) because in explicit-end-mark mode this pad is
# pure harness overhead charged directly to the measured voice-to-voice number:
# the fixture's audio ends, we push 0.15 s of digital silence, and only then
# call /audio/end. The server never needs it -- it is told the turn is over --
# so every millisecond here is dead time a real push-to-talk customer (who
# releases the button the instant they stop speaking) would never pay.
#
# Why not lower still: 0.15 s matches DEFAULT_ASR_TRIM_DECAY_SECONDS, i.e. the
# tail the ASR path already expects to see after the last word. Validated over
# 8 runs with transcripts inspected -- median v2v ~1030 ms (vs ~1200 ms at
# 0.3 s) with no truncated or altered transcripts.
#
# Prerequisite: the ffmpeg silencedetect fix in v2v_fixture_benchmark.py
# (_SILENCE_MIN_DURATION 0.3 -> 0.1). At the old 0.3 s minimum, a pad shorter
# than 0.3 s was not recognised as silence at all, so the ground-truth
# cross-check silently latched onto an earlier mid-sentence pause and reported
# a bogus ~5000 ms. Do not raise that constant back above this pad.


def synthesize_padded_line_wav(
    text: str,
    tts_model: str,
    tts_language: str,
    lead_pad_seconds: float,
    trail_pad_seconds: float,
) -> tuple[bytes, float]:
    """Synthesise one scripted line and pad it with real leading/trailing silence.

    Mirrors kiosk-voice-lab-main's ``_feed_replay()``: real silence before and
    after the utterance gives the endpoint detector something genuine to key
    off, exactly as a live mic would produce around a spoken sentence.
    Returns (wav_bytes, synth_ms).
    """
    raw_wav, synth_ms = synthesize_prompt_wav(text, tts_model, tts_language)
    wav_16k = resample_wav_to_16k_mono(raw_wav)

    with wave.open(io.BytesIO(wav_16k), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    lead_silence = b"\x00" * int(lead_pad_seconds * sample_rate) * sample_width * channels
    trail_silence = b"\x00" * int(trail_pad_seconds * sample_rate) * sample_width * channels

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(sample_width)
        out.setframerate(sample_rate)
        out.writeframes(lead_silence + frames + trail_silence)
    return buf.getvalue(), synth_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--script",
        choices=sorted(SCRIPTS),
        default=None,
        help=(
            "Legacy named scripted conversation to replay (hardcoded in this file). "
            "If omitted (the default), the conversation is instead read from "
            "--conversation-file."
        ),
    )
    parser.add_argument(
        "--conversation-file",
        type=Path,
        default=DEFAULT_CONVERSATION_FILE,
        help=(
            "JSONL conversation file to read the scripted customer lines from "
            f"(default: {DEFAULT_CONVERSATION_FILE}). Ignored if --script is given."
        ),
    )
    parser.add_argument("--list-scripts", action="store_true", help="Print available scripts and exit")
    parser.add_argument("--label", default=None, help="Name for this run, used in the result filename")
    parser.add_argument("--runs", type=int, default=3, help="Repetitions of the whole script")
    parser.add_argument("--realtime-factor", type=float, default=1.0, help="Playback speed (1.0 = real time)")
    parser.add_argument(
        "--push-chunk-seconds",
        type=float,
        default=0.5,
        help="Size of each WAV chunk pushed to /audio while streaming (default: 0.5s, like a live browser)",
    )
    parser.add_argument(
        "--silence-timeout-seconds",
        type=float,
        default=config.DEFAULT_SILENCE_TIMEOUT_SECONDS,
        help="Overrides the session's silence_timeout_seconds (defaults to kiosk-core's own configured value)",
    )
    parser.add_argument("--session-timeout", type=float, default=60.0, help="Max seconds to wait per turn")
    parser.add_argument("--lead-pad-seconds", type=float, default=DEFAULT_LEAD_PAD_SECONDS)
    parser.add_argument("--trail-pad-seconds", type=float, default=DEFAULT_TRAIL_PAD_SECONDS)
    parser.add_argument(
        "--explicit-end-mark",
        action="store_true",
        help=(
            "Signal end-of-turn explicitly via POST /audio/end right after the "
            "scripted line's own (short) trailing pad, instead of relying on "
            "kiosk-core's silence-timeout/completeness-shortcut endpoint "
            "detector to notice the customer stopped talking. Produces a "
            "clean, endpoint-detector-noise-free voice_to_voice_ms -- comparable "
            "to a live customer releasing a push-to-talk button -- instead of a "
            "number subject to the shortcut's documented 0%%-80%% run-to-run "
            "firing-rate swing (see BaseAudioSession._endpoint_transcript_stable). "
            "Overrides --trail-pad-seconds with --end-mark-trail-pad-seconds."
        ),
    )
    parser.add_argument(
        "--end-mark-trail-pad-seconds",
        type=float,
        default=DEFAULT_END_MARK_TRAIL_PAD_SECONDS,
        help="Trailing pad used when --explicit-end-mark is set (default: 0.15s -- just enough for ASR's decay tail)",
    )
    parser.add_argument("--tts-model", default=config.DEFAULT_TTS_MODEL)
    parser.add_argument("--tts-language", default=config.DEFAULT_TTS_LANGUAGE)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write the JSON report to (created if missing)",
    )
    parser.add_argument(
        "--emit-vlm-metrics",
        action="store_true",
        help=(
            "Deprecated, now a no-op: kiosk-core itself emits every turn's "
            "vlm_application_metrics_*.txt start/end pair natively (see "
            "kiosk_core.audio_session._emit_vlm_metrics). Accepted and ignored "
            "so existing callers (benchmark_smart_kiosk_v2v.py) keep working "
            "unchanged; emitting here too would double-count every turn."
        ),
    )
    args = parser.parse_args(argv)

    if args.list_scripts:
        print(f"Default conversation source: {DEFAULT_CONVERSATION_FILE}")
        print("Legacy named --script values (hardcoded, kept for backward compatibility):")
        for name, lines in SCRIPTS.items():
            print(f"  {name} ({len(lines)} turn{'s' if len(lines) != 1 else ''}):")
            for line in lines:
                print(f"    - {line!r}")
        return 0

    if args.script:
        script_name = args.script
        lines = SCRIPTS[args.script]
    else:
        script_name = args.conversation_file.stem
        try:
            lines = load_conversation_file(args.conversation_file)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(f"[v2v-scripted] {exc}")
            return 1

    label = args.label or f"v2v-scripted-{script_name}"

    print(f"[v2v-scripted] waiting for kiosk-core at {v2v.CORE_BASE_URL} ...")
    v2v.wait_for_core()

    report = v2v.BenchmarkReport(
        label=label,
        started_at=datetime.now(UTC).isoformat(),
        realtime_factor=args.realtime_factor,
        fixtures=[f"{script_name}:line{i + 1}" for i in range(len(lines))],
    )

    with tempfile.TemporaryDirectory(prefix="v2v-scripted-") as tmpdir:
        tmp_path = Path(tmpdir)
        for run in range(1, args.runs + 1):
            for i, line in enumerate(lines, start=1):
                print(f'[v2v-scripted] {script_name} turn {i}/{len(lines)} run {run}/{args.runs}: "{line}"')
                try:
                    trail_pad = (
                        args.end_mark_trail_pad_seconds if args.explicit_end_mark else args.trail_pad_seconds
                    )
                    wav_bytes, synth_ms = synthesize_padded_line_wav(
                        line,
                        args.tts_model,
                        args.tts_language,
                        args.lead_pad_seconds,
                        trail_pad,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[v2v-scripted]   TTS synthesis FAILED: {exc}")
                    turn = v2v.TurnResult(fixture=f"{script_name}:line{i}", run=run, error=f"tts synth failed: {exc}")
                    report.turns.append(turn)
                    continue

                fixture_path = tmp_path / f"{script_name}-line{i}-run{run}.wav"
                fixture_path.write_bytes(wav_bytes)

                turn = v2v.replay_fixture(
                    fixture=fixture_path,
                    run=run,
                    realtime_factor=args.realtime_factor,
                    silence_timeout_seconds=args.silence_timeout_seconds,
                    session_timeout=args.session_timeout,
                    push_chunk_seconds=args.push_chunk_seconds,
                    explicit_end_mark=args.explicit_end_mark,
                )
                turn.fixture = f"{script_name}:line{i}"
                report.turns.append(turn)

                if turn.error:
                    print(f"[v2v-scripted]   FAILED: {turn.error}")
                else:
                    gt = (
                        f"  v2v_ground_truth={turn.voice_to_voice_ground_truth_ms} ms"
                        if turn.voice_to_voice_ground_truth_ms is not None
                        else ""
                    )
                    print(
                        f"[v2v-scripted]   tts_synth={round(synth_ms, 1)} ms  "
                        f"v2v={turn.voice_to_voice_ms} ms  "
                        f"v2v_post_endpoint={turn.voice_to_voice_post_endpoint_ms} ms  "
                        f"endpoint_wait={turn.endpoint_wait_ms} ms  "
                        f"final_flush_wait={turn.final_flush_wait_ms} ms  "
                        f"shortcut_fired={turn.endpoint_shortcut_fired}  "
                        f"ttfa={turn.time_to_first_audio_ms} ms{gt}  "
                        f"transcript={turn.transcript[:80]!r}"
                    )

    report.finished_at = datetime.now(UTC).isoformat()
    report.summary = v2v.build_summary(report.turns)
    v2v.print_report(report)  # already prints the stage breakdown internally

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{label}.json"
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"\n[v2v-scripted] wrote {out_path}")

    return 0 if report.summary.get("turns_failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
