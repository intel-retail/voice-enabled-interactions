#!/usr/bin/env python3
"""Voice-to-voice latency benchmark using real recorded fixtures.

Why this exists
----------------
``agent_latency_benchmark.py`` (tier B) replays TTS-*synthesised* prompts at
``realtime_factor=100`` (as fast as kiosk-core allows). That is the right tool
for isolating LLM/TTS compute cost, but it is the WRONG tool for
voice-to-voice latency: replaying at 100x skips the customer's actual trailing
silence, so the endpoint detector never runs through its real wait.

This script instead:

* Replays real recorded human-speech WAV fixtures — not synthesised audio.
  No fixtures ship in this repo (to keep it free of binary recordings);
  supply your own via one or more ``--fixture`` flags.
* Drives kiosk-core through the same **continuous streaming** session API the
  browser UI itself uses — ``POST /api/v1/sessions/start-stream`` followed by
  repeated ``POST /api/v1/sessions/{id}/audio`` chunk pushes and a final
  ``POST /api/v1/sessions/{id}/audio/end`` — instead of ``start-file``
  (single whole-file upload, paced open-loop on the server). Each fixture is
  sliced into ``--push-chunk-seconds`` WAV chunks and pushed one at a time,
  sleeping between pushes at ``realtime_factor`` speed, exactly like
  ``gradio_app.py``'s ``on_chunk``/``_push``. This exercises the exact same
  ``BrowserStreamSession`` code path (and therefore the exact same VAD/
  endpoint-detection/``_t_last_word`` logic) that live customer traffic runs
  through — ``start-file``'s ``FileAudioSession`` shares that VAD logic too,
  but never round-trips audio over HTTP incrementally the way a live browser
  does, so it under-exercises chunk-arrival timing/jitter.
* Replays at ``realtime_factor=1.0`` (real-time — the default), so the
  customer's actual trailing silence is what trips the endpoint detector,
  exactly as it would live.
* Reads kiosk-core's wall-clock latency fields directly from the turn trace.
  The headline identity is now exact per turn:
  ``voice_to_voice_ms == endpointing_delay_ms + processing_latency_ms``.
  ``voice_to_voice_ms`` stops on the first sound at the speaker, which may be
  the cached "One moment." opener; therefore the report also surfaces
  ``voice_to_voice_answer_ms`` and ``first_audio_was_opener`` so the first
  answer-bearing audio is visible.
* Reports mean/median/p90/**p95** (p95 is the customer-facing target — a
  single bad tail turn is what a customer actually notices; median hides it).
* Writes per-turn latencies and the aggregate summary to a JSON file under
  ``smart-kiosk-assistant/results/``.

Usage::

    python tests/benchmarks/v2v_fixture_benchmark.py --runs 5 --label baseline

    # Point at your own recorded WAV fixtures (repeatable flag)
    python tests/benchmarks/v2v_fixture_benchmark.py \
        --fixture /path/to/order1.wav --fixture /path/to/order2.wav --runs 3

Stdlib only, consistent with the sibling benchmarks.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

CORE_BASE_URL = os.getenv("V2V_CORE_BASE_URL", "http://127.0.0.1:8012")
CONTAINER_ANALYZER_URL = "http://audio-analyzer:8010/v1/audio/transcriptions"
CONTAINER_RAG_URL = "http://rag-service:8020/api/v1/query"
CONTAINER_TTS_URL = "http://text-to-speech:8011/v1/audio/speech"

# Real recorded kiosk utterances -- NOT synthesised. Ground truth "last
# spoken word" is whatever silence the recording itself ends on; replaying at
# realtime_factor=1.0 lets the endpoint detector discover that the same way
# it would for a live customer.
#
# No fixtures are bundled in this repo (kept free of binary audio); you must
# pass at least one via --fixture, e.g.:
#   --fixture /path/to/your_recording.wav
DEFAULT_FIXTURES: list[Path] = []

# Bypass any corporate proxy for localhost calls, consistent with the sibling
# benchmarks (see agent_latency_benchmark.py).
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _request(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    method: str | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with _opener.open(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def http_get_json(url: str, timeout: float = 30.0) -> Any:
    status, body = _request(url, timeout=timeout)
    if status != 200:
        raise RuntimeError(f"GET {url} -> HTTP {status}: {body[:300]!r}")
    return json.loads(body)


def http_post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> Any:
    """POST a JSON body -- used for /api/v1/sessions/start-stream."""
    body = json.dumps(payload).encode()
    status, resp = _request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        method="POST",
        timeout=timeout,
    )
    if status != 200:
        raise RuntimeError(f"POST {url} -> HTTP {status}: {resp[:300]!r}")
    return json.loads(resp)


def http_post_wav_chunk(url: str, wav_bytes: bytes, timeout: float = 30.0) -> bool:
    """POST a raw WAV-container chunk -- used for /api/v1/sessions/{id}/audio.

    Mirrors gradio_app.py's ``_push()``: the whole request body IS the WAV
    file (with its own RIFF header), not a multipart field -- that is what
    ``BrowserStreamSession.push_audio`` expects to ``wave.open()`` per chunk.

    Returns False (instead of raising) on HTTP 409 "Session is not active":
    a fast-firing endpoint shortcut can close the session out from under a
    still-in-progress push loop that is only feeding trailing silence at
    that point (e.g. the scripted-conversation benchmark's padding) -- that
    is a normal race, not a real failure, so the caller should just stop
    pushing rather than fail the turn.
    """
    status, resp = _request(
        url,
        data=wav_bytes,
        headers={"Content-Type": "audio/wav", "Content-Length": str(len(wav_bytes))},
        method="POST",
        timeout=timeout,
    )
    if status == 409:
        return False
    if status != 200:
        raise RuntimeError(f"POST {url} -> HTTP {status}: {resp[:300]!r}")
    return True


def http_post_empty(url: str, timeout: float = 30.0) -> None:
    """POST with no body -- used for /api/v1/sessions/{id}/audio/end."""
    status, resp = _request(url, data=b"", method="POST", timeout=timeout)
    if status != 200:
        raise RuntimeError(f"POST {url} -> HTTP {status}: {resp[:300]!r}")


def iter_wav_chunks(fixture: Path, chunk_seconds: float) -> Iterator[tuple[bytes, float]]:
    """Slice a WAV fixture into self-contained WAV-container chunks.

    Each yielded chunk is a fully valid RIFF/WAV byte string (own header +
    frames) at the source sample rate/width/channels, matching exactly what
    ``BrowserStreamSession.push_audio`` expects per push (it calls
    ``wave.open()`` on every incoming chunk independently -- see
    ``kiosk_core/audio_session.py``). Also yields each chunk's real duration
    in seconds, so the caller can pace pushes at ``realtime_factor`` speed the
    same way ``FileAudioSession`` paces internal frame reads.
    """
    with wave.open(str(fixture), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames_per_chunk = max(1, int(chunk_seconds * sample_rate))

        while True:
            raw = wav_file.readframes(frames_per_chunk)
            if not raw:
                return
            n_frames = len(raw) // (sample_width * channels)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setnchannels(channels)
                out.setsampwidth(sample_width)
                out.setframerate(sample_rate)
                out.writeframes(raw)
            yield buf.getvalue(), n_frames / sample_rate


def poll_session(session_id: str, timeout: float, poll_interval: float = 0.25) -> dict[str, Any]:
    """Poll a kiosk-core session until it leaves the running/stopping state."""
    deadline = time.monotonic() + timeout
    snapshot: dict[str, Any] = {}
    while time.monotonic() < deadline:
        snapshot = http_get_json(f"{CORE_BASE_URL}/api/v1/sessions/{session_id}", timeout=15.0)
        if snapshot.get("status") not in ("created", "running", "stopping"):
            return snapshot
        time.sleep(poll_interval)
    raise TimeoutError(f"session {session_id} did not complete within {timeout}s")


def fetch_pipeline_trace() -> dict[str, Any] | None:
    """Fetch the latest kiosk-core turn trace (unwraps the ``trace`` envelope)."""
    try:
        payload = http_get_json(f"{CORE_BASE_URL}/api/v1/pipeline/latest", timeout=15.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[v2v]   (pipeline trace unavailable: {exc})")
        return None
    if isinstance(payload, dict) and "trace" in payload:
        return payload.get("trace") or None
    return payload


# ---------------------------------------------------------------------------
# Ground-truth end-of-speech offset
# ---------------------------------------------------------------------------
# kiosk-voice-lab's v2v uses a fixture manifest: the exact sample where real
# speech ends, authored once per fixture, independent of any run. A live mic
# has no such anchor (only a detector's guess), but a recorded fixture does —
# ffmpeg's silencedetect finds it directly from the waveform, with zero
# dependency on OUR pipeline's own VAD/endpoint timing. Cached per fixture
# path so repeated runs/turns don't re-invoke ffmpeg.
_TRUE_EOS_CACHE: dict[str, float | None] = {}

# -30dB / 0.3s matches the quiet-room, denoised recordings in tests/ -- loud
# enough to separate real trailing silence from mic noise floor, short enough
# not to swallow a mid-sentence breath pause. Override per-fixture if a
# different recording's noise floor needs a different threshold.
#
# Lowered 0.3 -> 0.1s (2026-09-16): --end-mark-trail-pad-seconds below 0.3s
# (e.g. 0.15-0.18s, used to shave the artificial benchmark-harness pad closer
# to config.DEFAULT_ASR_TRIM_DECAY_SECONDS's 0.15s floor) produced a trailing
# silence run shorter than this constant's old 0.3s minimum, so ffmpeg's
# silencedetect never found a silence_start mark in that trailing pad at all
# and fell back to an earlier, mid-utterance silence mark instead --
# reporting a bogus ~4000ms-inflated voice_to_voice_ground_truth_ms even
# though the real voice_to_voice_ms (server-side VAD-anchored) and the
# transcript were both correct the whole time. TTS-synthesised scripted audio
# (unlike the real tests/fixtures/*.wav recordings this constant also serves)
# has true digital silence with no mic noise floor to fight, so 0.1s is safe
# here without risking a false positive on a mid-sentence breath pause.
_SILENCE_NOISE_DB = "-30dB"
_SILENCE_MIN_DURATION = "0.1"


def true_end_of_speech_seconds(fixture: Path) -> float | None:
    """Return the offset (seconds from file start) where real speech ends.

    Runs ffmpeg's silencedetect once per fixture and takes the LAST
    ``silence_start`` mark before the file ends — i.e. the boundary between
    the customer's last spoken word and the recording's trailing silence.
    Returns ``None`` if ffmpeg is unavailable or no silence was detected
    (caller then falls back to the detector-based voice_to_voice_ms only).
    """
    key = str(fixture.resolve())
    if key in _TRUE_EOS_CACHE:
        return _TRUE_EOS_CACHE[key]

    result: float | None = None
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-i", str(fixture),
                "-af", f"silencedetect=noise={_SILENCE_NOISE_DB}:d={_SILENCE_MIN_DURATION}",
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        starts = [
            float(m.group(1))
            for m in re.finditer(r"silence_start:\s*([0-9.]+)", proc.stderr)
        ]
        # The trailing silence run is the LAST silence_start mark (any earlier
        # ones are pauses mid-utterance, not the end of speech).
        if starts:
            result = starts[-1]
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[v2v]   (ffmpeg silencedetect unavailable for {fixture.name}: {exc})")

    _TRUE_EOS_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    fixture: str
    run: int

    transcript: str = ""
    reply: str = ""
    error: str | None = None

    # Wall-clock spans, never blended with audio-domain diagnostics:
    #   voice_to_voice_ms          : customer's last word -> first sound at speaker
    #   endpointing_delay_ms       : customer's last word -> turn-end decision
    #   processing_latency_ms      : turn-end decision -> first sound at speaker
    #   voice_to_voice_answer_ms   : customer's last word -> first answer audio
    voice_to_voice_ms: float | None = None
    voice_to_voice_answer_ms: float | None = None
    processing_latency_ms: float | None = None
    processing_latency_answer_ms: float | None = None
    endpointing_delay_ms: float | None = None
    first_audio_was_opener: bool = False
    endpoint_silence_run_ms: float | None = None
    # Browser-mic-release turns only: diagnostic sub-component of
    # endpointing_delay_ms on that path.
    post_speech_gap_ms: float | None = None
    # The final chunk's real ASR round-trip -- see
    # pipeline_latency.WallTimes.final_flush_wait_ms.
    final_flush_wait_ms: float | None = None
    turn_total_ms: float | None = None

    # Ground-truth clock (see true_end_of_speech_seconds): last-word -> first
    # audio, anchored to an independent silence-detector pass over the fixture
    # itself rather than this pipeline's own endpoint/VAD timing. This is the
    # number directly comparable to kiosk-voice-lab's fixture-manifest v2v.
    playback_to_first_audio_ms: float | None = None
    playback_to_answer_audio_ms: float | None = None
    playback_to_endpoint_decision_ms: float | None = None
    true_end_of_speech_s: float | None = None
    voice_to_voice_ground_truth_ms: float | None = None
    voice_to_voice_answer_ground_truth_ms: float | None = None
    # The clip-anchored split of voice_to_voice_ground_truth_ms. Both halves
    # are measured on the playback clock and anchored on the fixture's own
    # end-of-speech sample, so unlike the server-side endpointing_delay_ms
    # they contain no dependency on this pipeline's VAD for the START of the
    # span -- which matters because endpointing is one of the things being
    # measured, so anchoring on the endpoint detector would define its own
    # error away. They satisfy, by construction:
    #
    #   voice_to_voice_ground_truth_ms == endpointing_delay_ground_truth_ms
    #                                     + processing_latency_ground_truth_ms
    #
    # which _assert_ground_truth_identity() checks on every turn.
    endpointing_delay_ground_truth_ms: float | None = None
    processing_latency_ground_truth_ms: float | None = None

    # True/False/None -- see pipeline_latency.WallTimes.endpoint_shortcut_fired.
    # Surfaced here to directly answer "is the adaptive completeness shortcut
    # ever firing, or is every turn falling back to the full
    # silence_timeout_seconds wait?" ``endpoint_silence_run_ms`` carries the
    # audio-domain run length when that diagnostic is needed.
    endpoint_shortcut_fired: bool | None = None

    # Per-stage context, useful for explaining a slow/fast run.
    asr_ms: float | None = None
    asr_chunks: int | None = None
    # Continuous-streaming mode only -- see pipeline_latency.AsrSpan. Customer's
    # actual last spoken word -> transcript ready, excluding the deliberate
    # trailing-silence wait (endpointing_delay_ms). This is the real, comparable
    # ASR latency figure -- asr_ms above only covers kiosk-core-triggered
    # flush/commit round trips and drastically under-reports this in
    # streaming mode.
    asr_transcription_latency_ms: float | None = None
    agent_ttft_ms: float | None = None
    agent_total_ms: float | None = None
    # Pure agent/LLM/tool round-trip -- agent_start to end of token stream,
    # BEFORE the TTS-drain wait. agent_total_ms above still includes that
    # drain (existing, intentional whole-orchestration semantics), which
    # makes it look ~2x the real agent cost on turns with a long reply. Use
    # this field to isolate what the LLM+tool call itself actually took.
    agent_stream_ms: float | None = None
    mcp_ms: float | None = None
    mcp_calls: int | None = None
    guard_ms: float | None = None
    template_ms: float | None = None
    tts_ms: float | None = None
    tts_ttfb_ms: float | None = None
    # True when sentence 1 came from the speculative/opener TTS cache, so
    # tts_ttfb_ms is a file copy rather than synthesis. Reported per turn
    # because a run where most turns hit the cache has a TTFB figure that
    # says nothing about the synthesiser.
    tts_first_segment_cached: bool | None = None
    tts_segments: int | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.voice_to_voice_ms is not None


@dataclass
class BenchmarkReport:
    label: str
    started_at: str
    finished_at: str = ""
    realtime_factor: float = 1.0
    fixtures: list[str] = field(default_factory=list)
    turns: list[TurnResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregation -- median AND p95 (p95 is the customer-facing target: a single
# bad tail turn is what a real customer notices, and median hides it).
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float] | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    vals_sorted = sorted(vals)

    def _pct(p: float) -> float:
        idx = max(0, min(len(vals_sorted) - 1, int(round(p * len(vals_sorted))) - 1))
        return round(vals_sorted[idx], 1)

    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 1),
        "median": round(statistics.median(vals), 1),
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
        "p90": _pct(0.90),
        "p95": _pct(0.95),
    }


def build_summary(turns: list[TurnResult]) -> dict[str, Any]:
    ok = [t for t in turns if t.ok]
    failed = [t for t in turns if not t.ok]
    fields = [
        "voice_to_voice_ms",
        "voice_to_voice_answer_ms",
        "processing_latency_ms",
        "processing_latency_answer_ms",
        "endpointing_delay_ms",
        "endpoint_silence_run_ms",
        "post_speech_gap_ms",
        "final_flush_wait_ms",
        "turn_total_ms",
        "asr_ms",
        "asr_transcription_latency_ms",
        "agent_ttft_ms",
        "agent_total_ms",
        "agent_stream_ms",
        "mcp_ms",
        "guard_ms",
        "template_ms",
        "tts_ms",
        "tts_ttfb_ms",
        "playback_to_first_audio_ms",
        "playback_to_answer_audio_ms",
        "playback_to_endpoint_decision_ms",
        "voice_to_voice_ground_truth_ms",
        "voice_to_voice_answer_ground_truth_ms",
        "endpointing_delay_ground_truth_ms",
        "processing_latency_ground_truth_ms",
    ]
    summary: dict[str, Any] = {
        "turns_total": len(turns),
        "turns_ok": len(ok),
        "turns_failed": len(failed),
    }
    for f_ in fields:
        summary[f_] = _stats([getattr(t, f_) for t in ok])
    summary["asr_chunks"] = _stats([t.asr_chunks for t in ok if t.asr_chunks is not None])
    opener_flags = [t.first_audio_was_opener for t in ok]
    summary["first_audio_was_opener_count"] = sum(1 for f_ in opener_flags if f_)
    summary["first_audio_turn_count"] = len(opener_flags)
    # A TTS TTFB of ~1 ms is a cache hit, not a fast synthesiser. Report how
    # many turns it applied to so the TTFB percentiles above can be read
    # correctly instead of looking like a broken metric.
    cached = [
        t.tts_first_segment_cached for t in ok
        if t.tts_first_segment_cached is not None
    ]
    summary["tts_first_segment_cached_count"] = sum(1 for f_ in cached if f_)
    summary["tts_first_segment_turn_count"] = len(cached)
    fired = [t.endpoint_shortcut_fired for t in ok if t.endpoint_shortcut_fired is not None]
    summary["endpoint_shortcut_fired_count"] = sum(1 for f_ in fired if f_)
    summary["endpoint_shortcut_eligible_count"] = len(fired)
    # Turns whose turn-end decision landed BEFORE the clip's real end of
    # speech -- i.e. the customer was cut off mid-utterance. Only detectable
    # with the clip anchor: measured against our own endpoint detector this
    # is 0 by construction, because the detector's decision IS the anchor.
    # A non-zero count means truncated transcripts, so it is surfaced as a
    # correctness counter rather than left to show up as a negative
    # percentile that reads like a measurement bug.
    early = [
        t.endpointing_delay_ground_truth_ms
        for t in ok
        if t.endpointing_delay_ground_truth_ms is not None
    ]
    summary["early_commit_count"] = sum(1 for v in early if v < 0)
    summary["early_commit_eligible_count"] = len(early)
    summary["latency_breakdown"] = build_latency_breakdown(summary)
    summary["kpi_vocabulary"] = build_kpi_vocabulary(summary)
    return summary


# ---------------------------------------------------------------------------
# Canonical KPI vocabulary
# ---------------------------------------------------------------------------
# Restates the numbers above using the shared cross-team terminology so our
# report can be read side by side with other teams' without a translation
# step. Nothing here is newly measured -- every value is already present in
# ``summary``.
#
# The defining identity is:
#
#     voice-to-voice = endpointing delay + processing latency
#
# which holds exactly per turn in kiosk-core (endpointing_delay_ms +
# processing_latency_ms == voice_to_voice_ms on every turn).
#
# If the cached opener is enabled, ``voice_to_voice_ms`` stops on that opener.
# ``voice_to_voice_answer_ms`` is the comparable answer-bearing figure.
# ---------------------------------------------------------------------------

# Customer-experience bands, in ms, keyed by the lower bound of each band.
CX_BANDS: tuple[tuple[float, str], ...] = (
    (0.0, "human gap (natural conversational timing)"),
    (300.0, "kiosk target (typical partner sizing)"),
    (700.0, "reads as hesitation"),
    (1000.0, "reads as a machine"),
    (4000.0, "failure"),
)


def classify_cx_band(value_ms: float | None) -> str | None:
    """Map a voice-to-voice latency onto the shared customer-experience scale."""
    if value_ms is None:
        return None
    label = CX_BANDS[0][1]
    for lower, name in CX_BANDS:
        if value_ms >= lower:
            label = name
    return label


def build_kpi_vocabulary(summary: dict[str, Any]) -> dict[str, Any]:
    def _stat(field_name: str, key: str = "median") -> float | None:
        st = summary.get(field_name)
        return st.get(key) if isinstance(st, dict) else None

    def _prefer(ground_truth_field: str, detector_field: str) -> str:
        """Pick the clip-anchored field when the run has one.

        Benchmarks replay a file, so the exact sample where speech ends is
        knowable independently of this pipeline (see
        true_end_of_speech_seconds). That anchor is strictly better here:
        endpointing is one of the spans being measured, so starting the
        clock at our own endpoint detector would subtract out part of the
        very thing under test. Live-mic turns have no such ground truth and
        fall back to the detector-anchored field.
        """
        return (
            ground_truth_field
            if _stat(ground_truth_field) is not None
            else detector_field
        )

    def _kpi(term: str, source: str, starts: str, stops: str, note: str = "") -> dict[str, Any]:
        return {
            "term": term,
            "source_field": source,
            "anchor": (
                "clip (ffmpeg silencedetect)"
                if source.endswith("_ground_truth_ms")
                else "detector (kiosk-core VAD/endpoint)"
            ),
            "starts": starts,
            "stops": stops,
            "p50_ms": _stat(source),
            "p95_ms": _stat(source, "p95"),
            "note": note,
        }

    v2v_field = _prefer("voice_to_voice_ground_truth_ms", "voice_to_voice_ms")
    v2v_answer_field = _prefer(
        "voice_to_voice_answer_ground_truth_ms", "voice_to_voice_answer_ms"
    )
    endpointing_field = _prefer(
        "endpointing_delay_ground_truth_ms", "endpointing_delay_ms"
    )
    processing_field = _prefer(
        "processing_latency_ground_truth_ms", "processing_latency_ms"
    )

    v2v_p50 = _stat(v2v_field)
    v2v_answer_p50 = _stat(v2v_answer_field)
    return {
        "voice_to_voice_latency": _kpi(
            "Voice-to-voice latency", v2v_field,
            "customer's last word", "first sound at speaker",
            "Includes the cached opener when first_audio_was_opener is true.",
        ),
        "voice_to_voice_answer_latency": _kpi(
            "Voice-to-voice answer latency", v2v_answer_field,
            "customer's last word", "first answer-bearing sound at speaker",
            "Use this as the pipeline figure when the cached opener is enabled.",
        ),
        "endpointing_delay": _kpi(
            "Endpointing delay", endpointing_field,
            "customer's last word", "turn-end decision",
            "Wall-clock customer wait; endpoint_silence_run_ms is only an "
            "audio-domain diagnostic for shortcut/full-timeout behaviour.",
        ),
        "processing_latency": _kpi(
            "Processing latency", processing_field,
            "turn-end decision", "first sound at speaker",
            "Pipeline work after endpointing; may stop on the cached opener.",
        ),
        "processing_latency_answer": _kpi(
            "Processing latency to answer", "processing_latency_answer_ms",
            "turn-end decision", "first answer-bearing sound at speaker",
            "Pipeline work excluding the cached opener shortcut.",
        ),
        "transcription_latency": _kpi(
            "Transcription latency", "asr_transcription_latency_ms",
            "customer's last word", "transcript update covering it",
        ),
        "llm_time_to_first_token": _kpi(
            "LLM TTFT", "agent_ttft_ms",
            "prompt in", "first token out",
        ),
        "tts_time_to_first_byte": _kpi(
            "TTS TTFB", "tts_ttfb_ms",
            "first sentence handed to synthesizer", "first audio byte on disk",
        ),
        "customer_experience_band": {
            "voice_to_voice_p50_ms": v2v_p50,
            "voice_to_voice_answer_p50_ms": v2v_answer_p50,
            "band": classify_cx_band(v2v_p50),
            "scale_ms": [{"from_ms": lo, "label": name} for lo, name in CX_BANDS],
        },
        "identity_check": {
            "expression": "voice_to_voice = endpointing_delay + processing_latency",
            "anchor": (
                "clip (ffmpeg silencedetect)"
                if v2v_field.endswith("_ground_truth_ms")
                else "detector (kiosk-core VAD/endpoint)"
            ),
            # The identity holds exactly PER TURN (and is checked there by
            # _assert_ground_truth_identity). These three are medians, and a
            # median of sums is not the sum of medians, so they are expected
            # to differ by a few percent. Do not "fix" that drift here --
            # treat only the per-turn warning as a real failure.
            "scope": "per-turn exact; medians below will not sum exactly",
            "endpointing_delay_p50_ms": _stat(endpointing_field),
            "processing_latency_p50_ms": _stat(processing_field),
            "voice_to_voice_p50_ms": v2v_p50,
        },
        # Both anchors side by side. The gap is the error the endpoint
        # detector makes relative to the clip's real end of speech -- the
        # exact quantity that anchoring on the detector would have hidden.
        "anchor_cross_check": {
            "clip_anchored_v2v_p50_ms": _stat("voice_to_voice_ground_truth_ms"),
            "detector_anchored_v2v_p50_ms": _stat("voice_to_voice_ms"),
            "shortcut_fired_count": summary.get("endpoint_shortcut_fired_count"),
            "shortcut_eligible_count": summary.get("endpoint_shortcut_eligible_count"),
            "early_commit_count": summary.get("early_commit_count"),
            "early_commit_eligible_count": summary.get("early_commit_eligible_count"),
        },
    }


# ---------------------------------------------------------------------------
# Per-stage breakdown (ASR / LLM / TTS) -- median ms and % of processing
# latency. Percentages are computed against the turn-end-decision -> first
# speaker-audio wall clock, not the full voice-to-voice latency, so they are
# not diluted by endpointing delay.
# ---------------------------------------------------------------------------


def build_latency_breakdown(summary: dict[str, Any]) -> list[dict[str, Any]]:
    def _median(field_name: str) -> float | None:
        st = summary.get(field_name)
        return st["median"] if isinstance(st, dict) else None

    asr_ms = _median("asr_ms")
    asr_chunks = _median("asr_chunks")
    asr_transcription_ms = _median("asr_transcription_latency_ms")
    final_flush_wait_ms = _median("final_flush_wait_ms")
    llm_ttft_ms = _median("agent_ttft_ms")
    llm_total_ms = _median("agent_total_ms")
    tts_total_ms = _median("tts_ms")
    tts_ttfb_ms = _median("tts_ttfb_ms")
    processing_ms = _median("processing_latency_ms")
    processing_answer_ms = _median("processing_latency_answer_ms")
    endpointing_delay_ms = _median("endpointing_delay_ms")
    endpoint_silence_run_ms = _median("endpoint_silence_run_ms")
    v2v_ms = _median("voice_to_voice_ms")
    v2v_answer_ms = _median("voice_to_voice_answer_ms")

    rows: list[dict[str, Any]] = []

    def _row(stage: str, label: str, ms: float | None, base: float | None, extra: dict[str, Any] | None = None) -> None:
        pct = round(100.0 * ms / base, 1) if ms is not None and base else None
        row = {"stage": stage, "label": label, "median_ms": ms, "pct_of_processing": pct}
        if extra:
            row.update(extra)
        rows.append(row)

    # --- Components that gate processing latency.
    _row(
        "llm_ttft", "LLM TTFT (agent-start -> first reply token/sentence)",
        llm_ttft_ms, processing_ms,
    )
    _row(
        "tts_ttfb",
        "TTS TTFB (first sentence -> first audio byte on disk)",
        tts_ttfb_ms, processing_ms,
    )
    _row(
        "processing_total", "Processing latency (endpoint decision -> first sound)",
        processing_ms, processing_ms,
    )
    _row(
        "processing_answer_total", "Processing latency to answer (endpoint decision -> first answer audio)",
        processing_answer_ms, processing_answer_ms,
    )

    # --- Context only: not part of the processing-latency gate above.
    _row(
        "asr", "ASR (all chunks summed -- mostly overlaps endpointing delay)",
        asr_ms, None, extra={"chunks": asr_chunks},
    )
    _row(
        "asr_transcription_latency",
        "Transcription latency (last spoken word -> transcript ready, excludes endpointing delay)",
        asr_transcription_ms, None,
    )
    _row("endpointing_delay", "Endpointing delay (wall-clock last word -> turn-end decision)", endpointing_delay_ms, None)
    _row(
        "endpoint_silence_run",
        "Endpoint silence run (audio-domain diagnostic; shortcut/full-timeout behaviour)",
        endpoint_silence_run_ms, None,
    )
    _row(
        "final_flush_wait",
        "Final-chunk ASR round-trip (flush-queue join after the turn-end decision)",
        final_flush_wait_ms, None,
    )
    _row("llm_total", "LLM full reply (all sentences, for context only)", llm_total_ms, None)
    _row("tts_total", "TTS full reply (all segments, for context only)", tts_total_ms, None)
    _row("voice_to_voice", "Voice-to-voice latency (last word -> first sound)", v2v_ms, None)
    _row("voice_to_voice_answer", "Voice-to-voice answer latency (last word -> first answer audio)", v2v_answer_ms, None)

    return rows


def _print_kpi_vocabulary(kpi: dict[str, Any] | None) -> None:
    """Render the shared-vocabulary KPI block (see build_kpi_vocabulary)."""
    if not kpi:
        return
    order = (
        "voice_to_voice_latency", "voice_to_voice_answer_latency",
        "endpointing_delay", "processing_latency", "processing_latency_answer",
        "transcription_latency", "llm_time_to_first_token", "tts_time_to_first_byte",
    )
    print(f"\n{'-' * 78}\nCANONICAL KPI VOCABULARY\n{'-' * 78}")
    print(f"{'KPI':<34}{'p50 ms':>10}{'p95 ms':>10}")
    for key in order:
        row = kpi.get(key) or {}
        p50 = row.get("p50_ms")
        p95 = row.get("p95_ms")
        print(
            f"{row.get('term', key):<34}"
            f"{('n/a' if p50 is None else f'{p50:,.1f}'):>10}"
            f"{('-' if p95 is None else f'{p95:,.1f}'):>10}"
        )
    ident = kpi.get("identity_check") or {}
    ep, proc, v2v = (
        ident.get("endpointing_delay_p50_ms"),
        ident.get("processing_latency_p50_ms"),
        ident.get("voice_to_voice_p50_ms"),
    )
    if ep is not None and proc is not None and v2v is not None:
        total = ep + proc
        status = "ok" if abs(total - v2v) <= 1.0 else f"MISMATCH (delta={total - v2v:+.1f} ms)"
        print(
            f"\nidentity check: endpointing {ep:,.1f} + processing {proc:,.1f} "
            f"= {total:,.1f} vs voice-to-voice {v2v:,.1f}  -> {status}"
        )
    band = (kpi.get("customer_experience_band") or {}).get("band")
    if band:
        print(f"customer-experience band: {band}")


def print_report(report: BenchmarkReport) -> None:
    print(f"\n{'=' * 78}")
    print(f"V2V FIXTURE BENCHMARK  label={report.label!r}  realtime_factor={report.realtime_factor}")
    print(f"fixtures: {', '.join(report.fixtures)}")
    print(f"{'=' * 78}")
    print(f"{'stage':<28}{'mean':>9}{'median':>9}{'p90':>9}{'p95':>9}{'min':>9}{'max':>9}")
    for key, st in report.summary.items():
        # Stats rows only. Other structured blocks (latency_breakdown,
        # kpi_vocabulary) are rendered separately below.
        if not isinstance(st, dict) or "mean" not in st:
            continue
        print(
            f"{key:<28}{st['mean']:>9,.0f}{st['median']:>9,.0f}{st['p90']:>9,.0f}"
            f"{st['p95']:>9,.0f}{st['min']:>9,.0f}{st['max']:>9,.0f}"
        )
    print(
        f"\nturns: {report.summary.get('turns_ok')} ok / "
        f"{report.summary.get('turns_failed')} failed "
        f"(of {report.summary.get('turns_total')})"
    )
    _print_kpi_vocabulary(report.summary.get("kpi_vocabulary"))
    print(
        f"endpoint completeness shortcut fired: "
        f"{report.summary.get('endpoint_shortcut_fired_count')} / "
        f"{report.summary.get('endpoint_shortcut_eligible_count')} eligible turns "
        f"(the rest fell back to the full silence_timeout_seconds wait)"
    )
    print(
        "\nCustomer-facing number: voice_to_voice_ms p95 = "
        f"{(report.summary.get('voice_to_voice_ms') or {}).get('p95', 'n/a')} ms"
    )
    answer_stats = report.summary.get("voice_to_voice_answer_ms")
    if answer_stats:
        print(
            "Answer-bearing number: voice_to_voice_answer_ms p95 = "
            f"{answer_stats['p95']} ms "
            f"(first_audio_was_opener={report.summary.get('first_audio_was_opener_count')} / "
            f"{report.summary.get('first_audio_turn_count')} turns)"
        )
    processing_stats = report.summary.get("processing_latency_ms")
    if processing_stats:
        print(
            "Processing latency (turn-end decision -> first sound at speaker): "
            f"median = {processing_stats['median']} ms  p95 = {processing_stats['p95']} ms"
        )
    gt_stats = report.summary.get("voice_to_voice_ground_truth_ms")
    if gt_stats:
        print(
            "Ground-truth v2v (independent silence-detector EOS, kiosk-voice-lab "
            f"method) median = {gt_stats['median']} ms  p95 = {gt_stats['p95']} ms"
        )
    print_breakdown(report)


def print_breakdown(report: BenchmarkReport) -> None:
    rows = report.summary.get("latency_breakdown") or []
    if not rows:
        return
    gated = {"llm_ttft", "tts_ttfb", "processing_total", "processing_answer_total"}
    print(f"\n{'-' * 96}")
    print("STAGE BREAKDOWN -- processing latency gates")
    print(f"{'-' * 96}")
    print(f"{'stage':<72}{'median_ms':>12}{'% processing':>12}")
    for row in rows:
        if row["stage"] not in gated:
            continue
        ms = row.get("median_ms")
        pct = row.get("pct_of_processing")
        ms_str = f"{ms:,.0f}" if isinstance(ms, (int, float)) else "n/a"
        pct_str = f"{pct:.0f}%" if isinstance(pct, (int, float)) else "-"
        print(f"{row['label']:<72}{ms_str:>12}{pct_str:>12}")

    print(f"\n{'-' * 96}")
    print("CONTEXT ONLY -- wider windows / endpointing diagnostics")
    print(f"{'-' * 96}")
    for row in rows:
        if row["stage"] in gated:
            continue
        ms = row.get("median_ms")
        ms_str = f"{ms:,.0f}" if isinstance(ms, (int, float)) else "n/a"
        extra = f"  (median {row['chunks']:.0f} ASR chunks/turn)" if row.get("chunks") else ""
        print(f"{row['label']:<72}{ms_str:>12}{extra}")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_fixture(
    fixture: Path,
    run: int,
    realtime_factor: float,
    silence_timeout_seconds: float,
    session_timeout: float,
    push_chunk_seconds: float = 0.5,
    explicit_end_mark: bool = False,
) -> TurnResult:
    result = TurnResult(fixture=fixture.name, run=run)
    conversation_id = f"v2v-bench-{uuid.uuid4().hex[:8]}"

    # Same session parameters start-file used to send, minus the file itself
    # and realtime_factor (which is now purely a client-side push-pacing
    # knob -- see the push loop below -- since start-stream has no server-
    # side playback to pace; it just receives whatever is pushed).
    session_payload = {
        "sample_rate": 16000,
        "chunk_seconds": 4.0,
        "silence_timeout_seconds": silence_timeout_seconds,
        "max_session_seconds": 30.0,
        "silence_threshold": 900,
        "temperature": 0.0,
        "analyzer_url": CONTAINER_ANALYZER_URL,
        "rag_url": CONTAINER_RAG_URL,
        "tts_url": CONTAINER_TTS_URL,
        "conversation_id": conversation_id,
    }

    try:
        started = http_post_json(f"{CORE_BASE_URL}/api/v1/sessions/start-stream", session_payload, timeout=30.0)
        session_id = str(started.get("session_id"))

        # Push the fixture as a live browser would: slice it into
        # self-contained WAV chunks and POST them one at a time, sleeping
        # between pushes so the customer's own trailing silence is what the
        # server-side endpoint detector actually waits through -- not a
        # single upload the server paces internally (start-file's model).
        # realtime_factor=1.0 (default) reproduces true real-time pacing;
        # >1.0 speeds pushes up (useful for fast smoke-testing, at the cost
        # of no longer matching a live customer's cadence).
        #
        # The sleep is deliberately NOT applied after the last chunk. A real
        # browser session (see kiosk-ui's useVoiceSession.stop()) force-
        # flushes whatever audio is buffered and calls /audio/end the
        # instant the customer releases the push-to-talk button -- it never
        # waits out the remainder of a chunkSeconds interval first. Sleeping
        # after the final chunk here would instead charge up to a full
        # push_chunk_seconds (0.5s default) of pure harness dead time to
        # every turn that reaches this loop's end (i.e. every turn where the
        # explicit end mark -- not the server's own silence-timeout shortcut
        # -- is what closes the turn), inflating post_speech_gap_ms with
        # nothing a live customer would ever experience. Requires
        # look-ahead (buffering one chunk) since iter_wav_chunks is a plain
        # generator with no "is this the last one" signal of its own.
        still_active = True
        chunk_iter = iter_wav_chunks(fixture, push_chunk_seconds)
        pending = next(chunk_iter, None)
        while pending is not None:
            wav_chunk, chunk_duration_s = pending
            pending = next(chunk_iter, None)
            still_active = http_post_wav_chunk(f"{CORE_BASE_URL}/api/v1/sessions/{session_id}/audio", wav_chunk, timeout=30.0)
            if not still_active:
                # Endpoint already fired and closed the session -- the rest
                # of this fixture is just trailing silence padding anyway,
                # so there is nothing left worth pushing.
                break
            if pending is not None and realtime_factor > 0:
                time.sleep(chunk_duration_s / realtime_factor)

        # explicit_end_mark: tell kiosk-core the turn is over the instant
        # every real-audio chunk has been pushed, the same way a live
        # customer releasing a push-to-talk button would -- instead of
        # relying on the server's own silence-timeout/completeness-shortcut
        # endpoint detector to notice the trailing silence and decide the
        # sentence "reads complete". BaseAudioSession.signal_end() (invoked
        # by POST /audio/end) stamps the voice-to-voice anchor and cuts off
        # the frame supply immediately, bypassing the
        # silence_run_seconds >= ... shortcut-firing race entirely (see
        # BaseAudioSession._endpoint_transcript_stable's docstring for that
        # race's documented 0%-80% run-to-run firing-rate swing). Only fires
        # if the session is still open -- if the server's own detector
        # already closed it (still_active is False above), there is nothing
        # left to signal.
        if still_active:
            http_post_empty(f"{CORE_BASE_URL}/api/v1/sessions/{session_id}/audio/end", timeout=30.0)
        elif explicit_end_mark:
            # The server's own detector beat the explicit signal to it (fired
            # during the short end-mark trailing pad) -- rare, but tell the
            # caller so a suspiciously-fast/slow turn isn't mistaken for a
            # clean explicit-end-mark measurement.
            print(
                f"[v2v] session={session_id} explicit_end_mark requested but the endpoint "
                "detector already closed the session first -- this turn's timing still "
                "reflects the silence-timeout/shortcut path, not the explicit signal."
            )

        snapshot = poll_session(session_id, timeout=session_timeout)
        result.transcript = snapshot.get("transcript", "") or ""
        result.reply = snapshot.get("response", "") or ""
        if snapshot.get("error"):
            result.error = str(snapshot["error"])
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        return result

    trace = fetch_pipeline_trace() or {}
    # /api/v1/pipeline/latest is a GLOBAL "most recent turn" endpoint, not a
    # per-session one, so it can hand back a PREVIOUS turn's numbers. That is
    # not hypothetical: when a turn produces an empty transcript kiosk-core
    # "stays silent" and never records a trace at all, so the fetch silently
    # returns the last successful turn's trace -- and a failed turn is then
    # reported with a completely plausible-looking latency (observed: an
    # empty-transcript turn reporting 768.3ms copied from the turn before it).
    #
    # The trace echoes the conversation_id this replay sent in its
    # start-stream payload, so it can be verified. A mismatch means kiosk-core
    # produced no trace for THIS turn; failing loudly is the only safe
    # behaviour, since the alternative is a benchmark that quietly invents
    # numbers.
    trace_conversation_id = trace.get("conversation_id")
    if trace and trace_conversation_id != conversation_id:
        result.error = (
            f"stale pipeline trace: /api/v1/pipeline/latest returned "
            f"conversation_id={trace_conversation_id!r}, expected {conversation_id!r} "
            f"-- kiosk-core recorded no trace for this turn "
            f"(transcript={result.transcript[:40]!r})"
        )
        return result

    wall = trace.get("wall", {}) or {}
    asr = trace.get("asr", {}) or {}
    agent = trace.get("agent", {}) or {}
    tts = trace.get("tts", {}) or {}

    result.voice_to_voice_ms = wall.get("voice_to_voice_ms")
    result.voice_to_voice_answer_ms = wall.get("voice_to_voice_answer_ms")
    result.processing_latency_ms = wall.get("processing_latency_ms")
    result.processing_latency_answer_ms = wall.get("processing_latency_answer_ms")
    result.endpointing_delay_ms = wall.get("endpointing_delay_ms")
    result.first_audio_was_opener = bool(wall.get("first_audio_was_opener", False))
    result.endpoint_silence_run_ms = wall.get("endpoint_silence_run_ms")
    result.post_speech_gap_ms = wall.get("post_speech_gap_ms")
    result.final_flush_wait_ms = wall.get("final_flush_wait_ms")
    result.turn_total_ms = wall.get("turn_total_ms")
    result.asr_ms = asr.get("ms")
    result.asr_chunks = asr.get("chunks")
    result.asr_transcription_latency_ms = asr.get("transcription_latency_ms")
    result.agent_ttft_ms = agent.get("ttft_ms")
    result.agent_total_ms = agent.get("total_ms")
    result.agent_stream_ms = agent.get("stream_ms")
    mcp = agent.get("mcp", {}) or {}
    guard = agent.get("guard", {}) or {}
    template = agent.get("template", {}) or {}
    result.mcp_ms = mcp.get("ms")
    result.mcp_calls = mcp.get("calls")
    result.guard_ms = guard.get("ms")
    result.template_ms = template.get("ms")
    result.tts_ms = tts.get("ms")
    result.tts_ttfb_ms = tts.get("ttfb_ms")
    result.tts_first_segment_cached = tts.get("first_segment_cached")
    result.tts_segments = tts.get("segments")
    result.endpoint_shortcut_fired = wall.get("endpoint_shortcut_fired")

    # Ground-truth v2v: playback_to_first_audio_ms (server, from the instant
    # the fixture started streaming) minus the fixture's own true-speech-end
    # offset (ffmpeg, independent of this pipeline's VAD/endpoint timing).
    result.playback_to_first_audio_ms = wall.get("playback_to_first_audio_ms")
    result.playback_to_answer_audio_ms = wall.get("playback_to_answer_audio_ms")
    result.playback_to_endpoint_decision_ms = wall.get("playback_to_endpoint_decision_ms")
    result.true_end_of_speech_s = true_end_of_speech_seconds(fixture)
    true_eos_ms = (
        result.true_end_of_speech_s * 1000
        if result.true_end_of_speech_s is not None
        else None
    )
    if result.playback_to_first_audio_ms is not None and true_eos_ms is not None:
        result.voice_to_voice_ground_truth_ms = round(
            result.playback_to_first_audio_ms - true_eos_ms, 1
        )
    if result.playback_to_answer_audio_ms is not None and true_eos_ms is not None:
        result.voice_to_voice_answer_ground_truth_ms = round(
            result.playback_to_answer_audio_ms - true_eos_ms, 1
        )
    # Split the clip-anchored total into the two spans the customer actually
    # experiences. Both subtractions stay on the playback clock.
    if result.playback_to_endpoint_decision_ms is not None:
        if true_eos_ms is not None:
            result.endpointing_delay_ground_truth_ms = round(
                result.playback_to_endpoint_decision_ms - true_eos_ms, 1
            )
        if result.playback_to_first_audio_ms is not None:
            result.processing_latency_ground_truth_ms = round(
                result.playback_to_first_audio_ms
                - result.playback_to_endpoint_decision_ms,
                1,
            )
    _assert_ground_truth_identity(result)
    return result


# Rounding each span independently can leave at most 0.1 ms per term, so a
# 1 ms window is comfortably tight enough to catch a real anchoring mistake
# (which would be off by the length of an utterance, not a rounding step).
_GROUND_TRUTH_IDENTITY_TOLERANCE_MS = 1.0


def _assert_ground_truth_identity(result: "TurnResult") -> None:
    """Warn if the clip-anchored spans stop reconciling.

    ``voice_to_voice`` must equal ``endpointing + processing`` by
    construction -- all three are differences of instants on the same
    playback clock. If that ever stops holding, one of the three is being
    anchored on a different clock (the exact class of bug that made
    voice-to-voice start at the customer's first word), so it is worth
    saying so loudly rather than publishing a number that does not add up.

    Warns rather than raises: a broken invariant should not destroy an
    otherwise complete benchmark run, and the warning names the turn.
    """
    total = result.voice_to_voice_ground_truth_ms
    endpointing = result.endpointing_delay_ground_truth_ms
    processing = result.processing_latency_ground_truth_ms
    if total is None or endpointing is None or processing is None:
        return
    drift = abs(total - (endpointing + processing))
    if drift > _GROUND_TRUTH_IDENTITY_TOLERANCE_MS:
        print(
            f"[v2v]   WARNING: ground-truth spans do not reconcile "
            f"(v2v={total} != endpointing={endpointing} + "
            f"processing={processing}, drift={drift:.1f}ms). "
            f"One of the three is anchored on a different clock."
        )


def wait_for_core(timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _ = _request(f"{CORE_BASE_URL}/health", timeout=5.0)
            if status == 200:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0)
    raise RuntimeError(f"kiosk-core not healthy at {CORE_BASE_URL} after {timeout}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="v2v-run", help="Name for this run, used in the result filename")
    parser.add_argument("--runs", type=int, default=3, help="Repetitions per fixture")
    parser.add_argument(
        "--fixture",
        action="append",
        dest="fixtures",
        required=True,
        help="Path to a WAV fixture (repeatable, required -- no fixtures are bundled in this repo).",
    )
    parser.add_argument("--realtime-factor", type=float, default=1.0, help="Playback speed (1.0 = real time)")
    parser.add_argument(
        "--push-chunk-seconds",
        type=float,
        default=0.5,
        help="Size of each WAV chunk pushed to /audio while streaming (default: 0.5s, like a live browser)",
    )
    # 1.1s matches config.DEFAULT_SILENCE_TIMEOUT_SECONDS (kiosk-voice-lab-main
    # parity, lowered from 1.5s). Kept overridable so older/1.5s comparisons
    # are still possible via --silence-timeout-seconds 1.5.
    parser.add_argument("--silence-timeout-seconds", type=float, default=1.1)
    parser.add_argument("--session-timeout", type=float, default=60.0, help="Max seconds to wait per turn")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write the JSON report to (created if missing)",
    )
    args = parser.parse_args(argv)

    fixtures = [Path(p) for p in args.fixtures] if args.fixtures else DEFAULT_FIXTURES
    missing = [str(p) for p in fixtures if not p.is_file()]
    if missing:
        print(f"[v2v] ERROR: fixture(s) not found: {missing}", file=sys.stderr)
        return 1

    print(f"[v2v] waiting for kiosk-core at {CORE_BASE_URL} ...")
    wait_for_core()

    report = BenchmarkReport(
        label=args.label,
        started_at=datetime.now(UTC).isoformat(),
        realtime_factor=args.realtime_factor,
        fixtures=[p.name for p in fixtures],
    )

    for fixture in fixtures:
        for run in range(1, args.runs + 1):
            print(f"[v2v] {fixture.name} run {run}/{args.runs} (realtime_factor={args.realtime_factor}) ...")
            turn = replay_fixture(
                fixture=fixture,
                run=run,
                realtime_factor=args.realtime_factor,
                silence_timeout_seconds=args.silence_timeout_seconds,
                session_timeout=args.session_timeout,
                push_chunk_seconds=args.push_chunk_seconds,
            )
            report.turns.append(turn)
            if turn.error:
                print(f"[v2v]   FAILED: {turn.error}")
            else:
                gt = (
                    f"  v2v_ground_truth={turn.voice_to_voice_ground_truth_ms} ms"
                    if turn.voice_to_voice_ground_truth_ms is not None
                    else ""
                )
                print(
                    f"[v2v]   v2v={turn.voice_to_voice_ms} ms  "
                    f"v2v_answer={turn.voice_to_voice_answer_ms} ms  "
                    f"processing={turn.processing_latency_ms} ms  "
                    f"endpointing={turn.endpointing_delay_ms} ms  "
                    f"opener={turn.first_audio_was_opener}  "
                    f"final_flush_wait={turn.final_flush_wait_ms} ms  "
                    f"shortcut_fired={turn.endpoint_shortcut_fired}  "
                    f"tts_ttfb={turn.tts_ttfb_ms} ms{gt}  "
                    f"transcript={turn.transcript[:80]!r}"
                )

    report.finished_at = datetime.now(UTC).isoformat()
    report.summary = build_summary(report.turns)
    print_report(report)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"\n[v2v] wrote {out_path}")

    return 0 if report.summary.get("turns_failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
