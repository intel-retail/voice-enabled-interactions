#!/usr/bin/env python3
"""Voice-to-voice latency benchmark using real recorded fixtures.

Why this exists
----------------
``agent_latency_benchmark.py`` (tier B) and ``conversation_replay_benchmark.py``
both replay TTS-*synthesised* prompts at ``realtime_factor=100`` (as fast as
kiosk-core allows). That is the right tool for isolating LLM/TTS compute cost,
but it is the WRONG tool for voice-to-voice: replaying at 100x skips the
customer's actual trailing silence, so the endpoint detector never runs
through its real wait, and neither script even reads ``voice_to_voice_ms`` or
``endpoint_wait_ms`` out of the trace.

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
* Reads ALL of kiosk-core's wall-clock fields from the turn trace, not just
  ``time_to_first_audio_ms``: ``voice_to_voice_ms``,
  ``voice_to_voice_informative_ms`` and ``endpoint_wait_ms`` alongside
  ``turn_total_ms``/``time_to_first_audio_ms``.
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

    # Two clocks, never blended -- see docs discussion:
    #   voice_to_voice_ms   : customer's last word -> first sound (opener or reply)
    #   endpoint_wait_ms    : how long the endpoint sat in trailing silence
    #   time_to_first_audio_ms : endpoint DECISION -> first sound (compute-only)
    voice_to_voice_ms: float | None = None
    voice_to_voice_informative_ms: float | None = None
    # v2v with the mandatory trailing-silence wait subtracted back out (see
    # WallTimes.voice_to_voice_post_endpoint_ms) -- the number to compare
    # against the lab's "pipeline" clock, since that wait is a design choice,
    # not compute cost.
    voice_to_voice_post_endpoint_ms: float | None = None
    endpoint_wait_ms: float | None = None
    # Browser-mic-release turns only (--explicit-end-mark, which is what
    # `make benchmark` passes by default): no silence-timeout endpoint fires,
    # so endpoint_wait_ms is None and THIS is the real "customer's last word
    # -> turn-end decision" gap instead -- see
    # pipeline_latency.WallTimes.post_speech_gap_ms. Without this, the
    # "Endpointing delay" KPI silently reads null for every explicit-end-mark
    # run even though a real (smaller) endpointing delay occurred.
    post_speech_gap_ms: float | None = None
    # The final chunk's real ASR round-trip -- see
    # pipeline_latency.WallTimes.final_flush_wait_ms. Previously invisible:
    # voice_to_voice_ms could exceed endpoint_wait_ms + time_to_first_audio_ms
    # by however long this blocked, with no field explaining the gap.
    final_flush_wait_ms: float | None = None
    time_to_first_audio_ms: float | None = None
    turn_total_ms: float | None = None

    # Ground-truth clock (see true_end_of_speech_seconds): last-word -> first
    # audio, anchored to an independent silence-detector pass over the fixture
    # itself rather than this pipeline's own endpoint/VAD timing. This is the
    # number directly comparable to kiosk-voice-lab's fixture-manifest v2v.
    playback_to_first_audio_ms: float | None = None
    true_end_of_speech_s: float | None = None
    voice_to_voice_ground_truth_ms: float | None = None



    # True/False/None -- see pipeline_latency.WallTimes.endpoint_shortcut_fired.
    # Surfaced here to directly answer "is the adaptive completeness shortcut
    # ever firing, or is every turn falling back to the full
    # silence_timeout_seconds wait?" without inferring it from endpoint_wait_ms
    # clustering near one value or the other.
    endpoint_shortcut_fired: bool | None = None

    # Per-stage context, useful for explaining a slow/fast run.
    asr_ms: float | None = None
    asr_chunks: int | None = None
    # Continuous-streaming mode only -- see pipeline_latency.AsrSpan. Customer's
    # actual last spoken word -> transcript ready, excluding the deliberate
    # trailing-silence wait (endpoint_wait_ms). This is the real, comparable
    # ASR latency figure -- asr_ms above only covers kiosk-core-triggered
    # flush/commit round trips and drastically under-reports this in
    # streaming mode.
    asr_last_word_to_transcript_ms: float | None = None
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
        "voice_to_voice_informative_ms",
        "voice_to_voice_post_endpoint_ms",
        "endpoint_wait_ms",
        "post_speech_gap_ms",
        "final_flush_wait_ms",
        "time_to_first_audio_ms",
        "turn_total_ms",
        "asr_ms",
        "asr_last_word_to_transcript_ms",
        "agent_ttft_ms",
        "agent_total_ms",
        "agent_stream_ms",
        "mcp_ms",
        "guard_ms",
        "template_ms",
        "tts_ms",
        "playback_to_first_audio_ms",
        "voice_to_voice_ground_truth_ms",
    ]
    summary: dict[str, Any] = {
        "turns_total": len(turns),
        "turns_ok": len(ok),
        "turns_failed": len(failed),
    }
    for f_ in fields:
        summary[f_] = _stats([getattr(t, f_) for t in ok])
    summary["asr_chunks"] = _stats([t.asr_chunks for t in ok if t.asr_chunks is not None])
    fired = [t.endpoint_shortcut_fired for t in ok if t.endpoint_shortcut_fired is not None]
    summary["endpoint_shortcut_fired_count"] = sum(1 for f_ in fired if f_)
    summary["endpoint_shortcut_eligible_count"] = len(fired)
    summary["latency_breakdown"] = build_latency_breakdown(summary)
    summary["kpi_vocabulary"] = build_kpi_vocabulary(summary)
    return summary


# ---------------------------------------------------------------------------
# Canonical KPI vocabulary
# ---------------------------------------------------------------------------
# Restates the numbers above using the shared cross-team terminology so our
# report can be read side by side with other teams' without a translation
# step. Nothing here is newly measured -- every value is an alias or a simple
# derivation of a field already in ``summary``. The internal names are kept
# as-is so existing tooling keeps working.
#
# The defining identity is:
#
#     voice-to-voice = endpointing delay + processing latency
#
# which holds EXACTLY per turn in our pipeline (verified: endpoint_wait_ms +
# voice_to_voice_post_endpoint_ms == voice_to_voice_ms on every turn).
#
# Two traps this encodes deliberately:
#
#   * "processing latency" is measured turn-end-decision -> first sound, i.e.
#     voice_to_voice_post_endpoint_ms. It is NOT time_to_first_audio_ms.
#     Those differ because work is started speculatively DURING the silence
#     wait, so ttfa overlaps the endpointing window -- on shortcut turns ttfa
#     is ~195ms while processing latency is ~3ms, because the audio was
#     already rendered by the time the turn-end decision fired.
#
#   * "transcription latency" is reported PER CALL (asr_ms is the sum over
#     all chunks in the turn), because that is the per-call convention the
#     <200ms target is expressed against.
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

    def _kpi(term: str, source: str, starts: str, stops: str, note: str = "") -> dict[str, Any]:
        return {
            "term": term,
            "source_field": source,
            "starts": starts,
            "stops": stops,
            "p50_ms": _stat(source),
            "p95_ms": _stat(source, "p95"),
            "note": note,
        }

    # Transcription latency is quoted per ASR call, not per turn.
    asr_total = _stat("asr_ms")
    asr_chunks = _stat("asr_chunks")
    per_call = round(asr_total / asr_chunks, 1) if asr_total and asr_chunks else None

    # TTS: the opener is served from a pre-rendered cache, so time-to-first-byte
    # on the voice-to-voice path is a file copy, not synthesis. Both are
    # reported -- the cache hit is what the customer experiences and therefore
    # what belongs inside voice-to-voice; the synthesis cost is disclosed
    # separately so the number is not mistaken for a like-for-like TTS
    # benchmark against a system that synthesises its opener live.
    tts_ttfb = None
    for row in summary.get("latency_breakdown") or []:
        if row.get("stage") == "tts_first_segment":
            tts_ttfb = row.get("median_ms")
            break

    # Endpointing delay has two possible source fields depending on how the
    # turn ended: endpoint_wait_ms (silence-timeout detector) is null when the
    # harness runs with --explicit-end-mark (the mode `make benchmark` uses by
    # default), which instead ends turns via mic-release / an explicit signal.
    # post_speech_gap_ms is the real equivalent delay on that path (customer's
    # last word -> turn-end signal). Fall back to it so this KPI isn't
    # silently null on every explicit-end-mark run.
    endpoint_wait_p50 = _stat("endpoint_wait_ms")
    if endpoint_wait_p50 is not None:
        endpointing_delay = _kpi(
            "Endpointing delay", "endpoint_wait_ms",
            "customer's last word", "turn-end decision",
            "Detector behaviour, not compute (silence-timeout path).",
        )
    else:
        endpointing_delay = _kpi(
            "Endpointing delay", "post_speech_gap_ms",
            "customer's last word", "turn-end signal",
            "Detector behaviour, not compute (explicit-end-mark / mic-release "
            "path -- endpoint_wait_ms is null in this mode, see "
            "pipeline_latency.WallTimes.post_speech_gap_ms).",
        )

    v2v_p50 = _stat("voice_to_voice_ms")
    return {
        "voice_to_voice_latency": _kpi(
            "Voice-to-voice latency", "voice_to_voice_ms",
            "customer's last word", "first sound at speaker",
            "Primary customer-facing KPI. Independently cross-checked by "
            "voice_to_voice_ground_truth_ms (ffmpeg silencedetect).",
        ),
        "endpointing_delay": endpointing_delay,
        "processing_latency": _kpi(
            "Processing latency", "voice_to_voice_post_endpoint_ms",
            "turn-end decision", "first sound at speaker",
            "Everything after the turn-end decision. Not time_to_first_audio_ms "
            "-- see module comment.",
        ),
        "transcription_latency": {
            "term": "Transcription latency", "source_field": "asr_ms / asr_chunks",
            "starts": "speech in", "stops": "transcript out",
            "p50_ms": per_call, "p95_ms": None,
            "note": f"Per ASR call ({asr_chunks} calls/turn median). Target <200ms.",
        },
        "llm_time_to_first_token": _kpi(
            "LLM time to first token (TTFT)", "agent_ttft_ms",
            "prompt in", "first token out",
        ),
        "tts_time_to_first_byte": {
            "term": "TTS time to first byte (TTFB)", "source_field": "latency_breakdown.tts_first_segment",
            "starts": "text in", "stops": "first audio byte",
            "p50_ms": tts_ttfb, "p95_ms": None,
            "note": "Opener served from a pre-rendered cache by design, so this "
                    "is a file copy. Live synthesis cost is reported separately "
                    "as tts_synthesis_ms and is NOT on the voice-to-voice path.",
        },
        "customer_experience_band": {
            "voice_to_voice_p50_ms": v2v_p50,
            "band": classify_cx_band(v2v_p50),
            "scale_ms": [{"from_ms": lo, "label": name} for lo, name in CX_BANDS],
        },
        "identity_check": {
            "expression": "voice_to_voice = endpointing_delay + processing_latency",
            "endpointing_delay_p50_ms": _stat("endpoint_wait_ms"),
            "processing_latency_p50_ms": _stat("voice_to_voice_post_endpoint_ms"),
            "voice_to_voice_p50_ms": v2v_p50,
        },
    }


# ---------------------------------------------------------------------------
# Per-stage breakdown (ASR / LLM / TTS) -- median ms and % of the compute-only
# clock (``time_to_first_audio_ms``), the same clock the "1,135 ms" style
# stage tables use. Percentages are computed against the compute total, NOT
# against full voice_to_voice_ms, so the split isn't diluted by the trailing
# silence wait (which is reported separately and is not "compute").
# ---------------------------------------------------------------------------


def build_latency_breakdown(summary: dict[str, Any]) -> list[dict[str, Any]]:
    def _median(field_name: str) -> float | None:
        st = summary.get(field_name)
        return st["median"] if isinstance(st, dict) else None

    # NOTE on why this doesn't just split time_to_first_audio_ms three ways
    # into (asr_ms, agent_total_ms, tts_ms) verbatim: those three raw fields
    # are each a DIFFERENT, WIDER window than the compute-only clock --
    # asr_ms sums every chunk transcribed all session (most of it happens
    # *during* the endpoint silence wait, i.e. BEFORE t0, so it mostly does
    # not cost any of the compute-only clock); agent_total_ms and tts_ms are
    # the FULL reply (every sentence/segment), not just the first one that
    # gates first-audio. Naively summing them overshoots time_to_first_audio_ms
    # by 2-3x (verified empirically). The two components that actually gate
    # time_to_first_audio_ms are:
    #   agent_ttft_ms  : agent-start -> first reply token/sentence ready
    #   (implied) TTS  : time_to_first_audio_ms - agent_ttft_ms, i.e. whatever
    #                    is left over to synthesize + write that first segment
    asr_ms = _median("asr_ms")
    asr_chunks = _median("asr_chunks")
    asr_last_word_ms = _median("asr_last_word_to_transcript_ms")
    final_flush_wait_ms = _median("final_flush_wait_ms")
    llm_ttft_ms = _median("agent_ttft_ms")
    llm_total_ms = _median("agent_total_ms")
    tts_total_ms = _median("tts_ms")
    compute_ms = _median("time_to_first_audio_ms")
    endpoint_wait_ms = _median("endpoint_wait_ms")
    v2v_ms = _median("voice_to_voice_ms")
    v2v_post_endpoint_ms = _median("voice_to_voice_post_endpoint_ms")

    tts_first_segment_ms = (
        round(compute_ms - llm_ttft_ms, 1)
        if compute_ms is not None and llm_ttft_ms is not None
        else None
    )

    rows: list[dict[str, Any]] = []

    def _row(stage: str, label: str, ms: float | None, base: float | None, extra: dict[str, Any] | None = None) -> None:
        pct = round(100.0 * ms / base, 1) if ms is not None and base else None
        row = {"stage": stage, "label": label, "median_ms": ms, "pct_of_compute": pct}
        if extra:
            row.update(extra)
        rows.append(row)

    # --- Components that gate the compute-only clock (time_to_first_audio_ms)
    _row(
        "llm_ttft", "LLM (agent-start -> first reply token/sentence)",
        llm_ttft_ms, compute_ms,
    )
    _row(
        "tts_first_segment",
        "TTS (implied: first segment synth + WAV write = compute - LLM ttft)",
        tts_first_segment_ms, compute_ms,
    )
    _row(
        "compute_total", "Compute total (endpoint decision -> first audio)",
        compute_ms, compute_ms,
    )

    # --- Context only: NOT part of the compute-only clock above.
    _row(
        "asr", "ASR (all chunks summed -- mostly overlaps the silence wait, not the compute clock)",
        asr_ms, None, extra={"chunks": asr_chunks},
    )
    _row(
        "asr_last_word",
        "ASR real latency (continuous-streaming mode: last spoken word -> transcript ready, excludes silence wait)",
        asr_last_word_ms, None,
    )
    _row("endpoint_wait", "Endpoint silence wait (detector behaviour, not pipeline compute)", endpoint_wait_ms, None)
    _row(
        "final_flush_wait",
        "Final-chunk ASR round-trip (flush-queue join, blocks turn start -- not silence wait, not compute)",
        final_flush_wait_ms, None,
    )
    _row("llm_total", "LLM full reply (all sentences, for context only)", llm_total_ms, None)
    _row("tts_total", "TTS full reply (all segments, for context only)", tts_total_ms, None)
    _row("voice_to_voice", "Voice-to-voice total (last word -> first audio)", v2v_ms, None)
    _row(
        "voice_to_voice_post_endpoint",
        "Voice-to-voice EXCLUDING the deliberate endpoint silence wait (final_flush_wait + ttfa; the number comparable to the lab's pipeline-only clock)",
        v2v_post_endpoint_ms, None,
    )

    return rows


def _print_kpi_vocabulary(kpi: dict[str, Any] | None) -> None:
    """Render the shared-vocabulary KPI block (see build_kpi_vocabulary)."""
    if not kpi:
        return
    order = (
        "voice_to_voice_latency", "endpointing_delay", "processing_latency",
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
    if ep is None:
        print(
            "\nidentity check: n/a -- endpoint detection was bypassed "
            "(--explicit-end-mark), so endpointing delay was never measured. "
            "Do NOT quote this run's voice-to-voice as a customer-facing figure."
        )
    else:
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
    post_ep_stats = report.summary.get("voice_to_voice_post_endpoint_ms")
    if post_ep_stats:
        print(
            "Pipeline-only number (EXCLUDES the deliberate endpoint silence "
            f"wait, comparable to the lab's clock): median = {post_ep_stats['median']} ms  "
            f"p95 = {post_ep_stats['p95']} ms"
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
    gated = {"llm_ttft", "tts_first_segment", "compute_total"}
    print(f"\n{'-' * 96}")
    print("STAGE BREAKDOWN -- components that gate time_to_first_audio_ms (compute-only clock)")
    print(f"{'-' * 96}")
    print(f"{'stage':<72}{'median_ms':>12}{'% compute':>12}")
    for row in rows:
        if row["stage"] not in gated:
            continue
        ms = row.get("median_ms")
        pct = row.get("pct_of_compute")
        ms_str = f"{ms:,.0f}" if isinstance(ms, (int, float)) else "n/a"
        pct_str = f"{pct:.0f}%" if isinstance(pct, (int, float)) else "-"
        print(f"{row['label']:<72}{ms_str:>12}{pct_str:>12}")

    print(f"\n{'-' * 96}")
    print("CONTEXT ONLY -- NOT counted in the compute-only clock above (wider windows / detector time)")
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
    result.voice_to_voice_informative_ms = wall.get("voice_to_voice_informative_ms")
    result.voice_to_voice_post_endpoint_ms = wall.get("voice_to_voice_post_endpoint_ms")
    result.endpoint_wait_ms = wall.get("endpoint_wait_ms")
    result.post_speech_gap_ms = wall.get("post_speech_gap_ms")
    result.final_flush_wait_ms = wall.get("final_flush_wait_ms")
    result.time_to_first_audio_ms = wall.get("time_to_first_audio_ms")
    result.turn_total_ms = wall.get("turn_total_ms")
    result.asr_ms = asr.get("ms")
    result.asr_chunks = asr.get("chunks")
    result.asr_last_word_to_transcript_ms = asr.get("last_word_to_transcript_ms")
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
    result.tts_segments = tts.get("segments")
    result.endpoint_shortcut_fired = wall.get("endpoint_shortcut_fired")

    # Ground-truth v2v: playback_to_first_audio_ms (server, from the instant
    # the fixture started streaming) minus the fixture's own true-speech-end
    # offset (ffmpeg, independent of this pipeline's VAD/endpoint timing).
    result.playback_to_first_audio_ms = wall.get("playback_to_first_audio_ms")
    result.true_end_of_speech_s = true_end_of_speech_seconds(fixture)
    if result.playback_to_first_audio_ms is not None and result.true_end_of_speech_s is not None:
        result.voice_to_voice_ground_truth_ms = round(
            result.playback_to_first_audio_ms - result.true_end_of_speech_s * 1000, 1
        )
    return result


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
                    f"v2v_post_endpoint={turn.voice_to_voice_post_endpoint_ms} ms  "
                    f"endpoint_wait={turn.endpoint_wait_ms} ms  "
                    f"final_flush_wait={turn.final_flush_wait_ms} ms  "
                    f"shortcut_fired={turn.endpoint_shortcut_fired}  "
                    f"ttfa={turn.time_to_first_audio_ms} ms{gt}  "
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
