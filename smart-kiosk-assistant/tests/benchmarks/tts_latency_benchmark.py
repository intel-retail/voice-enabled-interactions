#!/usr/bin/env python3
"""Text-to-speech latency benchmark for the Smart AI Kiosk.

The end-to-end ``agent_latency_benchmark.py`` reports a single ``trace_tts``
number taken from ``GET /api/v1/pipeline/latest``. That number is a *span*
(first sentence queued → last segment written) and hides where the cost
actually is. This harness isolates the text-to-speech microservice itself.

Three tiers are measured:

Tier A — RAW SYNTHESIS (``POST text-to-speech:8011/v1/audio/speech``)
    One request per prompt, exactly the payload
    ``kiosk_core.tts_client.TtsClient`` sends. Prompts are bucketed by length
    (short / medium / long) because SpeechT5 and Qwen3-TTS are both
    autoregressive — latency scales with generated audio duration, not with
    request count. Reports wall latency, decoded audio duration and the real
    time factor ``RTF = latency / audio_duration`` (RTF < 1.0 means the model
    generates faster than playback).

Tier B — KIOSK SENTENCE PIPELINE
    ``kiosk_core.audio_session._tts_worker`` splits the agent reply into
    sentences and synthesises them one at a time on a single worker thread.
    This tier replays a realistic multi-sentence agent reply the same way and
    reports **TTFA** (time-to-first-audio — what the customer actually waits
    for) versus total synthesis time. TTFA is the metric to optimise; total is
    what ``trace_tts`` shows today.

Tier C — CONCURRENCY CEILING
    ``BaseTTSService`` serialises inference behind a per-model
    ``threading.Lock``, so the service is single-flight regardless of the
    FastAPI threadpool. This tier fires N identical requests in parallel and
    reports the queueing penalty (p95 latency vs. the Tier-A warm baseline).

Results are written to JSON so runs can be diffed before/after an optimisation.

Usage::

    # Baseline against an already-running text-to-speech container
    python tests/benchmarks/tts_latency_benchmark.py --label tts-baseline

    # Bring the service up first, then measure
    python tests/benchmarks/tts_latency_benchmark.py --label tts-baseline --up

    # Re-measure after a config change and diff against the baseline
    python tests/benchmarks/tts_latency_benchmark.py \
        --label tts-qwen-gpu-fp16 \
        --compare tests/benchmarks/results/tts-baseline.json

    # Single tier, fast loop
    python tests/benchmarks/tts_latency_benchmark.py --tier A --runs 5

Stdlib only — no third-party dependencies, so it runs on the host without a
virtualenv. All HTTP bypasses the corporate proxy (same as ``make test``).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# tests/benchmarks/<this file> → parents[2] is the smart-kiosk-assistant dir,
# which is where the Makefile and docker-compose.yml live.
STACK_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

TTS_HEALTH_URL = "http://127.0.0.1:8011/health"
TTS_SPEECH_URL = "http://127.0.0.1:8011/v1/audio/speech"
TTS_MODEL_INFO_URL = "http://127.0.0.1:8011/v1/model-info"
TTS_PERF_URL = "http://127.0.0.1:8011/v1/performance"

# The kiosk always sends this model id; the service ignores it and uses the
# model pinned in configs/text-to-speech/config.yaml (see the TTS design doc).
DEFAULT_TTS_MODEL = "qwen-tts"
DEFAULT_TTS_LANGUAGE = "English"

# Tier-A corpus. Real kiosk agent replies, bucketed by length. Sentence counts
# and phrasing match what rag-service actually returns for ordering turns.
PROMPTS: list[tuple[str, str]] = [
    ("short", "Sure."),
    ("short", "Added to your order."),
    ("medium", "I have added one Classic Chicken Burger to your order."),
    (
        "medium",
        "Your order total is one hundred and sixty nine rupees. "
        "Would you like anything else?",
    ),
    (
        "long",
        "I have added one Classic Chicken Burger to your order. "
        "Would you like to add a medium Cold Coffee for fifty nine rupees? "
        "Your order total is currently one hundred and sixty nine rupees.",
    ),
]

# Tier-B corpus. One realistic multi-sentence agent reply, split the same way
# kiosk_core.audio_session does before handing sentences to the TTS worker.
KIOSK_REPLY = (
    "I have added one Classic Chicken Burger to your order. "
    "Would you like to add a Cold Coffee for fifty nine rupees? "
    "Your order total is one hundred and sixty nine rupees."
)

# Mirrors the sentence splitting in kiosk_core.audio_session._tts_worker's
# producer so Tier B queues the same units the kiosk queues.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib, proxy-bypassing)
# ---------------------------------------------------------------------------

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    method: str | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes]:
    """Issue an HTTP request bypassing any configured proxy."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with _opener.open(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def http_get_json(url: str, timeout: float = 30.0) -> Any:
    status, body = _request(url, timeout=timeout)
    if status != 200:
        raise RuntimeError(f"GET {url} → HTTP {status}: {body[:300]!r}")
    return json.loads(body)


def synthesize(
    text: str,
    *,
    model: str = DEFAULT_TTS_MODEL,
    language: str | None = DEFAULT_TTS_LANGUAGE,
    voice: str | None = None,
    instructions: str | None = None,
    timeout: float = 300.0,
) -> tuple[float, bytes]:
    """POST /v1/audio/speech and return (latency_ms, wav_bytes).

    The payload is byte-for-byte what ``TtsClient.synthesize_to_file`` sends so
    the measurement includes the same server-side validation and WAV encoding
    the kiosk pays for.
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": text,
        "response_format": "wav",
    }
    if voice:
        payload["voice"] = voice
    if language:
        payload["language"] = language
    if instructions:
        payload["instructions"] = instructions

    body = json.dumps(payload).encode()
    started = time.perf_counter()
    status, resp = _request(
        TTS_SPEECH_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    if status != 200:
        raise RuntimeError(f"TTS synthesis failed HTTP {status}: {resp[:300]!r}")
    if not resp.startswith(b"RIFF"):
        raise RuntimeError(f"TTS returned non-WAV payload ({len(resp)} bytes)")
    return latency_ms, resp


# ---------------------------------------------------------------------------
# WAV inspection (stdlib struct parsing — avoids a soundfile dependency)
# ---------------------------------------------------------------------------


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Return the playback duration of a PCM WAV payload in seconds."""
    if len(wav_bytes) < 12 or not wav_bytes.startswith(b"RIFF"):
        raise ValueError("not a RIFF/WAV payload")

    offset = 12
    sample_rate = 0
    channels = 1
    bits_per_sample = 16
    data_bytes = 0

    while offset + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[offset : offset + 4]
        (chunk_size,) = struct.unpack_from("<I", wav_bytes, offset + 4)
        payload_at = offset + 8

        if chunk_id == b"fmt " and chunk_size >= 16:
            (_fmt, channels, sample_rate, _byte_rate, _align, bits_per_sample) = (
                struct.unpack_from("<HHIIHH", wav_bytes, payload_at)
            )
        elif chunk_id == b"data":
            data_bytes = min(chunk_size, len(wav_bytes) - payload_at)

        offset = payload_at + chunk_size + (chunk_size % 2)

    if not sample_rate or not data_bytes:
        raise ValueError("WAV missing fmt/data chunk")

    bytes_per_frame = max(1, channels * (bits_per_sample // 8))
    return data_bytes / bytes_per_frame / sample_rate


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class SynthResult:
    """One Tier-A synthesis request."""

    run: int
    bucket: str
    text: str
    chars: int
    latency_ms: float = 0.0
    audio_seconds: float = 0.0
    rtf: float = 0.0                 # latency / audio duration; < 1.0 = faster than realtime
    wav_bytes: int = 0
    error: str | None = None


@dataclass
class SentencePipelineResult:
    """One Tier-B kiosk-shaped multi-sentence reply."""

    run: int
    sentences: int = 0
    ttfa_ms: float = 0.0             # time to first audio segment — the felt latency
    total_ms: float = 0.0            # all sentences synthesised serially
    audio_seconds: float = 0.0
    per_sentence_ms: list[float] = field(default_factory=list)
    error: str | None = None


@dataclass
class ConcurrencyResult:
    """One Tier-C parallel request."""

    run: int
    concurrency: int
    latency_ms: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Stack control
# ---------------------------------------------------------------------------


def compose_up_tts() -> None:
    """Start only the text-to-speech service (its models live in a named volume)."""
    print("→ starting text-to-speech (docker compose up -d text-to-speech) ...")
    subprocess.run(
        ["docker", "compose", "up", "-d", "text-to-speech"],
        cwd=STACK_DIR,
        check=True,
    )


def wait_for_health(timeout_seconds: float = 600.0) -> float:
    """Block until /health returns ok. Returns seconds waited.

    Model export + warmup on first boot can take minutes (see the 240s
    healthcheck start_period in docker-compose.yml).
    """
    started = time.perf_counter()
    deadline = started + timeout_seconds
    while time.perf_counter() < deadline:
        try:
            status, body = _request(TTS_HEALTH_URL, timeout=5.0)
            if status == 200 and b"ok" in body:
                waited = time.perf_counter() - started
                print(f"✓ text-to-speech healthy after {waited:.1f}s")
                return waited
        except Exception:
            pass
        time.sleep(3.0)
    raise RuntimeError(f"text-to-speech not healthy within {timeout_seconds:.0f}s")


def read_model_info() -> dict[str, Any]:
    """Snapshot the served model config so results are self-describing."""
    try:
        return http_get_json(TTS_MODEL_INFO_URL, timeout=10.0)
    except Exception as exc:  # informational only — never fail the run
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tier A — raw synthesis
# ---------------------------------------------------------------------------


def run_tier_a(
    runs: int,
    model: str,
    language: str,
    voice: str | None,
    instructions: str | None,
) -> tuple[list[SynthResult], SynthResult | None]:
    """Measure per-prompt synthesis latency. Returns (warm results, cold result).

    The first request after boot pays OpenVINO's lazy first-inference cost even
    though the service warms up at startup, so it is recorded separately rather
    than polluting the warm percentiles.
    """
    print(f"\n=== Tier A — raw synthesis ({runs} run(s) × {len(PROMPTS)} prompt(s)) ===")

    cold: SynthResult | None = None
    try:
        bucket, text = PROMPTS[0]
        latency_ms, wav = synthesize(
            text, model=model, language=language, voice=voice, instructions=instructions
        )
        duration = wav_duration_seconds(wav)
        cold = SynthResult(
            run=0,
            bucket="cold",
            text=text,
            chars=len(text),
            latency_ms=round(latency_ms, 1),
            audio_seconds=round(duration, 3),
            rtf=round(latency_ms / 1000.0 / duration, 3) if duration else 0.0,
            wav_bytes=len(wav),
        )
        print(f"  cold  {latency_ms:8.1f} ms  ({duration:.2f}s audio)  {text[:40]!r}")
    except Exception as exc:
        cold = SynthResult(run=0, bucket="cold", text=PROMPTS[0][1], chars=0, error=str(exc))
        print(f"  cold  FAILED: {exc}")

    results: list[SynthResult] = []
    for run in range(1, runs + 1):
        for bucket, text in PROMPTS:
            try:
                latency_ms, wav = synthesize(
                    text,
                    model=model,
                    language=language,
                    voice=voice,
                    instructions=instructions,
                )
                duration = wav_duration_seconds(wav)
                results.append(
                    SynthResult(
                        run=run,
                        bucket=bucket,
                        text=text,
                        chars=len(text),
                        latency_ms=round(latency_ms, 1),
                        audio_seconds=round(duration, 3),
                        rtf=round(latency_ms / 1000.0 / duration, 3) if duration else 0.0,
                        wav_bytes=len(wav),
                    )
                )
                print(
                    f"  run {run} {bucket:<7} {latency_ms:8.1f} ms  "
                    f"({duration:5.2f}s audio, rtf {results[-1].rtf:.2f})  {len(text):4d} chars"
                )
            except Exception as exc:
                results.append(
                    SynthResult(run=run, bucket=bucket, text=text, chars=len(text), error=str(exc))
                )
                print(f"  run {run} {bucket:<7} FAILED: {exc}")

    return results, cold


# ---------------------------------------------------------------------------
# Tier B — kiosk sentence pipeline
# ---------------------------------------------------------------------------


def split_sentences(reply: str) -> list[str]:
    """Split an agent reply into the units the kiosk TTS worker synthesises."""
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(reply.strip()) if part.strip()]


def run_tier_b(
    runs: int,
    model: str,
    language: str,
    voice: str | None,
    instructions: str | None,
    reply: str,
) -> list[SentencePipelineResult]:
    """Replay a multi-sentence agent reply sentence-by-sentence (serial, like the worker)."""
    sentences = split_sentences(reply)
    print(f"\n=== Tier B — kiosk sentence pipeline ({len(sentences)} sentences × {runs} run(s)) ===")

    results: list[SentencePipelineResult] = []
    for run in range(1, runs + 1):
        per_sentence: list[float] = []
        audio_total = 0.0
        ttfa = 0.0
        started = time.perf_counter()
        try:
            for index, sentence in enumerate(sentences):
                latency_ms, wav = synthesize(
                    sentence,
                    model=model,
                    language=language,
                    voice=voice,
                    instructions=instructions,
                )
                per_sentence.append(round(latency_ms, 1))
                audio_total += wav_duration_seconds(wav)
                if index == 0:
                    ttfa = (time.perf_counter() - started) * 1000.0
            total_ms = (time.perf_counter() - started) * 1000.0
            results.append(
                SentencePipelineResult(
                    run=run,
                    sentences=len(sentences),
                    ttfa_ms=round(ttfa, 1),
                    total_ms=round(total_ms, 1),
                    audio_seconds=round(audio_total, 3),
                    per_sentence_ms=per_sentence,
                )
            )
            print(
                f"  run {run}  ttfa {ttfa:8.1f} ms   total {total_ms:8.1f} ms   "
                f"({audio_total:.2f}s audio)  per-sentence {per_sentence}"
            )
        except Exception as exc:
            results.append(SentencePipelineResult(run=run, error=str(exc)))
            print(f"  run {run}  FAILED: {exc}")

    return results


# ---------------------------------------------------------------------------
# Tier C — concurrency ceiling
# ---------------------------------------------------------------------------


def run_tier_c(
    concurrency: int,
    model: str,
    language: str,
    voice: str | None,
    instructions: str | None,
) -> list[ConcurrencyResult]:
    """Fire N simultaneous requests to expose the per-model inference lock."""
    text = PROMPTS[2][1]  # a medium prompt — representative kiosk reply length
    print(f"\n=== Tier C — concurrency ceiling ({concurrency} parallel requests) ===")

    def one(index: int) -> ConcurrencyResult:
        try:
            latency_ms, _ = synthesize(
                text, model=model, language=language, voice=voice, instructions=instructions
            )
            return ConcurrencyResult(run=index, concurrency=concurrency, latency_ms=round(latency_ms, 1))
        except Exception as exc:
            return ConcurrencyResult(run=index, concurrency=concurrency, error=str(exc))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(1, concurrency + 1)))

    for result in results:
        if result.error:
            print(f"  req {result.run}  FAILED: {result.error}")
        else:
            print(f"  req {result.run}  {result.latency_ms:8.1f} ms")
    return results


# ---------------------------------------------------------------------------
# Statistics & reporting
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float] | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    ordered = sorted(clean)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 1),
        "p50": round(statistics.median(ordered), 1),
        "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
        "max": round(ordered[-1], 1),
        "mean": round(statistics.fmean(ordered), 1),
    }


def summarize(
    tier_a: list[SynthResult],
    cold: SynthResult | None,
    tier_b: list[SentencePipelineResult],
    tier_c: list[ConcurrencyResult],
) -> dict[str, Any]:
    ok_a = [r for r in tier_a if not r.error]
    ok_b = [r for r in tier_b if not r.error]
    ok_c = [r for r in tier_c if not r.error]

    by_bucket: dict[str, Any] = {}
    for bucket in ("short", "medium", "long"):
        rows = [r for r in ok_a if r.bucket == bucket]
        if rows:
            by_bucket[bucket] = {
                "latency_ms": _stats([r.latency_ms for r in rows]),
                "audio_seconds": round(statistics.fmean([r.audio_seconds for r in rows]), 3),
                "rtf": round(statistics.fmean([r.rtf for r in rows]), 3),
                "chars": round(statistics.fmean([float(r.chars) for r in rows]), 1),
            }

    return {
        "tier_a": {
            "cold_ms": cold.latency_ms if cold and not cold.error else None,
            "latency_ms": _stats([r.latency_ms for r in ok_a]),
            "rtf": _stats([r.rtf for r in ok_a]),
            "by_bucket": by_bucket,
            "errors": [r.error for r in tier_a if r.error],
        },
        "tier_b": {
            "ttfa_ms": _stats([r.ttfa_ms for r in ok_b]),
            "total_ms": _stats([r.total_ms for r in ok_b]),
            "sentences": ok_b[0].sentences if ok_b else None,
            "errors": [r.error for r in tier_b if r.error],
        },
        "tier_c": {
            "concurrency": tier_c[0].concurrency if tier_c else None,
            "latency_ms": _stats([r.latency_ms for r in ok_c]),
            "errors": [r.error for r in tier_c if r.error],
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("TTS LATENCY SUMMARY")
    print("=" * 72)

    a = summary["tier_a"]
    if a["latency_ms"]:
        s = a["latency_ms"]
        print(f"Tier A  raw synthesis   p50 {s['p50']:8.1f} ms   p95 {s['p95']:8.1f} ms   max {s['max']:8.1f} ms")
    if a["cold_ms"]:
        print(f"        cold first call {a['cold_ms']:8.1f} ms")
    if a["rtf"]:
        print(f"        real time factor p50 {a['rtf']['p50']:.2f}  (<1.0 = faster than playback)")
    for bucket, row in a["by_bucket"].items():
        print(
            f"        {bucket:<7} {row['latency_ms']['p50']:8.1f} ms p50  "
            f"| {row['audio_seconds']:5.2f}s audio | rtf {row['rtf']:.2f} | {row['chars']:.0f} chars"
        )

    b = summary["tier_b"]
    if b["ttfa_ms"]:
        print(
            f"Tier B  sentence pipe   TTFA p50 {b['ttfa_ms']['p50']:8.1f} ms   "
            f"total p50 {b['total_ms']['p50']:8.1f} ms   ({b['sentences']} sentences)"
        )

    c = summary["tier_c"]
    if c["latency_ms"]:
        print(
            f"Tier C  concurrency {c['concurrency']}   p50 {c['latency_ms']['p50']:8.1f} ms   "
            f"p95 {c['latency_ms']['p95']:8.1f} ms   max {c['latency_ms']['max']:8.1f} ms"
        )
    print("=" * 72)


def print_comparison(baseline: dict[str, Any], current: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"COMPARISON vs {baseline.get('label', 'baseline')}")
    print("=" * 72)

    paths = [
        ("tier_a", "latency_ms", "raw synthesis p50"),
        ("tier_b", "ttfa_ms", "sentence TTFA p50"),
        ("tier_b", "total_ms", "sentence total p50"),
        ("tier_c", "latency_ms", "concurrent p95"),
    ]
    for tier, key, label in paths:
        base = (baseline.get("summary", {}).get(tier, {}) or {}).get(key)
        curr = (current.get(tier, {}) or {}).get(key)
        if not base or not curr:
            continue
        stat = "p95" if "p95" in label else "p50"
        before, after = base[stat], curr[stat]
        delta = after - before
        pct = (delta / before * 100.0) if before else 0.0
        arrow = "▼" if delta < 0 else "▲"
        print(f"  {label:<22} {before:8.1f} → {after:8.1f} ms  {arrow} {abs(pct):5.1f}%")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="tts-run", help="result file name (results/<label>.json)")
    p.add_argument("--tier", choices=["A", "B", "C", "all"], default="all")
    p.add_argument("--runs", type=int, default=3, help="repetitions for tiers A and B")
    p.add_argument("--concurrency", type=int, default=4, help="parallel requests for tier C")
    p.add_argument("--up", action="store_true", help="docker compose up -d text-to-speech first")
    p.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    p.add_argument("--tts-language", default=DEFAULT_TTS_LANGUAGE)
    p.add_argument("--tts-voice", default=None, help="override the configured speaker")
    p.add_argument("--tts-instructions", default=None, help="voice_design instructions (Qwen only)")
    p.add_argument("--reply", default=KIOSK_REPLY, help="tier B multi-sentence agent reply")
    p.add_argument("--compare", default=None, help="path to a previous results JSON to diff against")
    p.add_argument("--no-save", action="store_true", help="do not write a results JSON")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.up:
        compose_up_tts()
    wait_for_health()

    model_info = read_model_info()
    if isinstance(model_info, dict) and "error" not in model_info:
        print(
            f"→ serving: {model_info.get('model')} "
            f"[{model_info.get('runtime', '?')}/{model_info.get('device', '?')}/"
            f"{model_info.get('dtype', '?')}] speakers={model_info.get('supported_speakers')}"
        )

    tier_a: list[SynthResult] = []
    cold: SynthResult | None = None
    tier_b: list[SentencePipelineResult] = []
    tier_c: list[ConcurrencyResult] = []

    if args.tier in ("A", "all"):
        tier_a, cold = run_tier_a(
            args.runs, args.tts_model, args.tts_language, args.tts_voice, args.tts_instructions
        )
    if args.tier in ("B", "all"):
        tier_b = run_tier_b(
            args.runs,
            args.tts_model,
            args.tts_language,
            args.tts_voice,
            args.tts_instructions,
            args.reply,
        )
    if args.tier in ("C", "all"):
        tier_c = run_tier_c(
            args.concurrency, args.tts_model, args.tts_language, args.tts_voice, args.tts_instructions
        )

    summary = summarize(tier_a, cold, tier_b, tier_c)
    print_summary(summary)

    payload = {
        "label": args.label,
        "timestamp": datetime.now(UTC).isoformat(),
        "model_info": model_info,
        "request": {
            "model": args.tts_model,
            "language": args.tts_language,
            "voice": args.tts_voice,
            "instructions": args.tts_instructions,
        },
        "summary": summary,
        "tier_a": [asdict(r) for r in tier_a],
        "tier_a_cold": asdict(cold) if cold else None,
        "tier_b": [asdict(r) for r in tier_b],
        "tier_c": [asdict(r) for r in tier_c],
    }

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text())
            print_comparison(baseline, summary)
        except Exception as exc:
            print(f"! could not compare against {args.compare}: {exc}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{args.label}.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"\n→ results written to {out}")

    failed = any(r.error for r in tier_a) or any(r.error for r in tier_b)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
