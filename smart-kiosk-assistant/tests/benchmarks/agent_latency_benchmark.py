#!/usr/bin/env python3
"""End-to-end agent latency benchmark for the Smart AI Kiosk.

Brings up the full stack (``make up``), waits for every service to become
healthy, then drives an ordering turn through the kiosk and records latency.

Two measurement tiers are captured:

Tier A — AGENT TURN (the number we optimise)
    ``POST rag-service:8020/api/v1/agent/chat``
    Exactly the payload ``kiosk_core.agent_client.AgentClient`` sends. This
    isolates the agentic cost: LLM call 1 (tool selection) + MCP tool exec +
    LLM call 2 (reply composition). Google ADK always makes two LLM calls per
    tool-calling turn, so this is where the ~9 s lives.

Tier B — VOICE E2E VIA KIOSK-CORE (what the customer feels)
    text-to-speech synthesises the prompt to a WAV, the WAV is pushed to
    ``POST kiosk-core:8012/api/v1/sessions/start-file``, and the session is
    polled to completion. The per-stage breakdown (ASR / agent TTFT / TTS /
    wall) is then read from ``GET kiosk-core:8012/api/v1/pipeline/latest``.

Results are written to JSON so runs can be diffed before/after an optimisation.

Usage::

    # Baseline (brings the stack up, runs both tiers, saves JSON)
    python tests/benchmarks/agent_latency_benchmark.py --label baseline

    # Re-measure after tuning, against an already-running stack
    python tests/benchmarks/agent_latency_benchmark.py \
        --label prefix-cache-int4 --skip-up \
        --compare tests/benchmarks/results/baseline.json

    # Agent tier only (fast iteration loop)
    python tests/benchmarks/agent_latency_benchmark.py --tier A --runs 5 --skip-up

Stdlib only — no third-party dependencies, so it runs on the host without a
virtualenv. All HTTP bypasses the corporate proxy (same as ``make test``).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
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

DEFAULT_PROMPT = "I would like to order one classic chicken burger"

# Host-side endpoints (published ports from docker-compose.yml)
OVMS_MODELS_URL = "http://127.0.0.1:8000/v3/models"
ANALYZER_HEALTH_URL = "http://127.0.0.1:8010/health"
TTS_HEALTH_URL = "http://127.0.0.1:8011/health"
TTS_SPEECH_URL = "http://127.0.0.1:8011/v1/audio/speech"
RAG_HEALTH_URL = "http://127.0.0.1:8020/health"
RAG_AGENT_URL = "http://127.0.0.1:8020/api/v1/agent/chat"
CORE_HEALTH_URL = "http://127.0.0.1:8012/health"
CORE_BASE_URL = "http://127.0.0.1:8012"

# In-container service URLs. The /api/v1/sessions/start-file form defaults are
# hardcoded to 127.0.0.1, which is wrong inside the kiosk-core container — so
# every URL is passed explicitly on the multipart form.
CONTAINER_ANALYZER_URL = "http://audio-analyzer:8010/v1/audio/transcriptions"
CONTAINER_RAG_URL = "http://rag-service:8020/api/v1/query"
CONTAINER_TTS_URL = "http://text-to-speech:8011/v1/audio/speech"

HEALTH_CHECKS: list[tuple[str, str, str]] = [
    # (service, url, substring expected in the body)
    ("ovms-llm", OVMS_MODELS_URL, "model"),
    ("audio-analyzer", ANALYZER_HEALTH_URL, "ok"),
    ("text-to-speech", TTS_HEALTH_URL, "ok"),
    ("rag-service", RAG_HEALTH_URL, "ok"),
    ("kiosk-core", CORE_HEALTH_URL, "ok"),
]


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


def http_post_json(url: str, payload: dict[str, Any], timeout: float = 300.0) -> Any:
    body = json.dumps(payload).encode()
    status, resp = _request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
        timeout=timeout,
    )
    if status != 200:
        raise RuntimeError(f"POST {url} → HTTP {status}: {resp[:300]!r}")
    return json.loads(resp)


def http_post_multipart(
    url: str,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str = "audio/wav",
    timeout: float = 300.0,
) -> Any:
    """POST a multipart/form-data body (hand-rolled — stdlib has no helper)."""
    boundary = f"----kioskbench{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode()
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)

    status, resp = _request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
        timeout=timeout,
    )
    if status != 200:
        raise RuntimeError(f"POST {url} → HTTP {status}: {resp[:300]!r}")
    return json.loads(resp)


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class AgentTurnResult:
    """One Tier-A agent turn."""

    run: int
    latency_ms: float
    tool_calls: list[str] = field(default_factory=list)
    reply: str = ""
    error: str | None = None
    session_id: str = ""
    # Derived from rag-service logs — the ADK two-call breakdown.
    llm1_ms: float | None = None   # tool-selection call
    tool_ms: float | None = None   # MCP round trip
    llm2_ms: float | None = None   # reply-composition call
    tool_name: str | None = None


@dataclass
class VoiceTurnResult:
    """One Tier-B full voice turn through kiosk-core."""

    run: int
    tts_synth_ms: float = 0.0        # harness-side prompt synthesis (not kiosk latency)
    session_wall_ms: float = 0.0     # upload → session status == completed
    transcript: str = ""
    reply: str = ""
    # Extracted from GET /api/v1/pipeline/latest
    trace_wall_total_ms: float | None = None
    trace_time_to_first_audio_ms: float | None = None
    trace_asr_ms: float | None = None
    trace_agent_ttft_ms: float | None = None
    trace_agent_total_ms: float | None = None
    trace_tts_ms: float | None = None
    error: str | None = None


@dataclass
class BenchmarkReport:
    label: str
    prompt: str
    started_at: str
    finished_at: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    agent_runs: list[AgentTurnResult] = field(default_factory=list)
    voice_runs: list[VoiceTurnResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stack lifecycle
# ---------------------------------------------------------------------------


def stack_up(timeout: float) -> None:
    """Run ``make up`` from the smart-kiosk-assistant directory."""
    workdir = STACK_DIR
    print(f"[stack] make up  (cwd={workdir})")
    proc = subprocess.run(
        ["make", "up"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"`make up` failed with exit code {proc.returncode}")
    print("[stack] make up completed")


def wait_for_health(timeout: float, poll_interval: float = 5.0) -> dict[str, float]:
    """Block until every service reports healthy. Returns time-to-healthy per service."""
    print(f"[stack] waiting for services to become healthy (timeout={timeout:.0f}s) …")
    deadline = time.monotonic() + timeout
    start = time.monotonic()
    ready: dict[str, float] = {}

    while time.monotonic() < deadline:
        for name, url, expect in HEALTH_CHECKS:
            if name in ready:
                continue
            try:
                status, body = _request(url, timeout=5.0)
                if status == 200 and expect in body.decode(errors="ignore").lower():
                    ready[name] = round(time.monotonic() - start, 1)
                    print(f"[stack]   ✓ {name} healthy after {ready[name]}s")
            except Exception:
                pass
        if len(ready) == len(HEALTH_CHECKS):
            print("[stack] all services healthy ✓")
            return ready
        pending = [n for n, _, _ in HEALTH_CHECKS if n not in ready]
        print(f"[stack]   … still waiting on: {', '.join(pending)}")
        time.sleep(poll_interval)

    pending = [n for n, _, _ in HEALTH_CHECKS if n not in ready]
    raise TimeoutError(f"Services never became healthy: {', '.join(pending)}")


def collect_stack_config() -> dict[str, Any]:
    """Capture what is actually being served, so results are attributable."""
    cfg: dict[str, Any] = {}
    try:
        models = http_get_json(OVMS_MODELS_URL, timeout=10.0)
        cfg["ovms_models"] = [
            m.get("id") for m in models.get("data", []) if isinstance(m, dict)
        ] or models
    except Exception as exc:
        cfg["ovms_models"] = f"unavailable: {exc}"

    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{json .Config.Cmd}}", "ovms-llm"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            cfg["ovms_cmd"] = json.loads(out.stdout.strip())
    except Exception:
        pass

    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.Config.Image}}", "ovms-llm"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            cfg["ovms_image"] = out.stdout.strip()
    except Exception:
        pass

    return cfg


# ---------------------------------------------------------------------------
# Tier A — agent turn
# ---------------------------------------------------------------------------


def run_agent_tier(
    prompt: str,
    runs: int,
    warmup: int,
    user_id: str,
    fresh_session: bool,
) -> list[AgentTurnResult]:
    """Drive ``POST /api/v1/agent/chat`` and time the full agent round-trip."""
    print(f"\n[tier-A] agent turn × {runs} (warmup={warmup}) → {RAG_AGENT_URL}")
    results: list[AgentTurnResult] = []
    shared_session = f"bench-{uuid.uuid4().hex[:8]}"

    total = warmup + runs
    for i in range(total):
        is_warmup = i < warmup
        session_id = f"bench-{uuid.uuid4().hex[:8]}" if fresh_session else shared_session
        payload = {
            "transcription": prompt,
            "user_id": user_id,
            "session_id": session_id,
            "history": [],
        }

        t0 = time.perf_counter()
        try:
            data = http_post_json(RAG_AGENT_URL, payload, timeout=300.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            reply = data.get("reply", "")
            tool_calls = data.get("tool_calls", []) or []
            err = None
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            reply, tool_calls, err = "", [], str(exc)

        tag = "warmup" if is_warmup else f"run {i - warmup + 1}/{runs}"
        status = f"ERROR {err}" if err else f"tools={tool_calls or '[]'}"
        print(f"[tier-A]   {tag:>12}: {elapsed_ms:8.0f} ms  {status}")
        if reply:
            print(f"[tier-A]                 reply: {reply[:150]!r}")

        if not is_warmup:
            results.append(
                AgentTurnResult(
                    run=i - warmup + 1,
                    latency_ms=round(elapsed_ms, 1),
                    tool_calls=list(tool_calls),
                    reply=reply,
                    error=err,
                    session_id=session_id,
                )
            )

    enrich_agent_results(results)
    for r in results:
        if r.llm1_ms is not None:
            print(
                f"[tier-A]   split run {r.run}: LLM#1={r.llm1_ms:.0f} ms  "
                f"tool({r.tool_name})={r.tool_ms:.0f} ms  LLM#2={r.llm2_ms:.0f} ms"
            )

    return results


# ---------------------------------------------------------------------------
# Log-derived ADK two-call breakdown
# ---------------------------------------------------------------------------

_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
_MCP_CALL_RE = re.compile(r"\[MCP\] Calling tool=(\S+) on server=")
_MCP_DONE_RE = re.compile(r"\[MCP\] Tool=(\S+) result=")
_AGENT_RESP_RE = re.compile(r"\[AGENT←OVMS\] Response received \| session=(\S+) latency_ms=(\d+)")


def _parse_log_ts(line: str) -> float | None:
    m = _LOG_TS_RE.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()


def derive_llm_split(container: str = "rag-service", tail: int = 4000) -> dict[str, dict[str, Any]]:
    """Reconstruct the per-session LLM#1 / tool / LLM#2 split from rag-service logs.

    Google ADK issues two LLM calls per tool-calling turn. The agent logs give
    us three anchors per turn — MCP call start, MCP result, and the final
    ``[AGENT←OVMS] Response received ... latency_ms=`` — which is enough to
    attribute the total to each phase. This is what tells us whether an
    optimisation helped prefill (LLM#1) or decode (LLM#2).

    Returns:
        Mapping of session_id → {llm1_ms, tool_ms, llm2_ms, total_ms, tool_name}.
    """
    try:
        out = subprocess.run(
            ["docker", "logs", "--tail", str(tail), container],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:
        print(f"[split] could not read {container} logs: {exc}")
        return {}
    if out.returncode != 0:
        print(f"[split] docker logs {container} failed: {out.stderr[:200]}")
        return {}

    lines = (out.stdout + out.stderr).splitlines()
    split: dict[str, dict[str, Any]] = {}
    pending_call: tuple[float, str] | None = None
    pending_done: float | None = None

    for line in lines:
        ts = _parse_log_ts(line)
        if ts is None:
            continue
        m_call = _MCP_CALL_RE.search(line)
        if m_call:
            pending_call = (ts, m_call.group(1))
            pending_done = None
            continue
        if _MCP_DONE_RE.search(line):
            pending_done = ts
            continue
        m_resp = _AGENT_RESP_RE.search(line)
        if m_resp:
            session_id, total_ms = m_resp.group(1), float(m_resp.group(2))
            turn_start = ts - (total_ms / 1000.0)
            entry: dict[str, Any] = {"total_ms": total_ms}
            if pending_call and pending_call[0] >= turn_start:
                call_ts, tool_name = pending_call
                entry["tool_name"] = tool_name
                entry["llm1_ms"] = round((call_ts - turn_start) * 1000, 1)
                done_ts = pending_done if (pending_done and pending_done >= call_ts) else call_ts
                entry["tool_ms"] = round((done_ts - call_ts) * 1000, 1)
                entry["llm2_ms"] = round((ts - done_ts) * 1000, 1)
            split[session_id] = entry
            pending_call, pending_done = None, None

    return split


def enrich_agent_results(results: list[AgentTurnResult]) -> None:
    """Attach the log-derived LLM#1/tool/LLM#2 split onto Tier-A results."""
    split = derive_llm_split()
    if not split:
        return
    for r in results:
        entry = split.get(r.session_id)
        if not entry:
            continue
        r.llm1_ms = entry.get("llm1_ms")
        r.tool_ms = entry.get("tool_ms")
        r.llm2_ms = entry.get("llm2_ms")
        r.tool_name = entry.get("tool_name")


# ---------------------------------------------------------------------------
# Tier B — full voice E2E through kiosk-core
# ---------------------------------------------------------------------------


def synthesize_prompt_wav(prompt: str, tts_model: str, tts_language: str) -> tuple[bytes, float]:
    """Use the kiosk's own TTS service to render the prompt to a WAV."""
    payload: dict[str, Any] = {
        "model": tts_model,
        "input": prompt,
        "response_format": "wav",
    }
    if tts_language:
        payload["language"] = tts_language

    t0 = time.perf_counter()
    status, body = _request(
        TTS_SPEECH_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
        timeout=300.0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if status != 200:
        raise RuntimeError(f"TTS synthesis failed HTTP {status}: {body[:300]!r}")
    if not body.startswith(b"RIFF"):
        raise RuntimeError(f"TTS returned non-WAV payload ({len(body)} bytes)")
    return body, elapsed_ms


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
    """Fetch the latest kiosk-core turn trace.

    The endpoint wraps the record as ``{"trace": {...}}``; unwrap it so callers
    always receive the trace body itself.
    """
    try:
        payload = http_get_json(f"{CORE_BASE_URL}/api/v1/pipeline/latest", timeout=15.0)
    except Exception as exc:
        print(f"[tier-B]   (pipeline trace unavailable: {exc})")
        return None
    if isinstance(payload, dict) and "trace" in payload:
        return payload.get("trace") or None
    return payload


def run_voice_tier(
    prompt: str,
    runs: int,
    tts_model: str,
    tts_language: str,
    session_timeout: float,
    max_session_seconds: float,
) -> list[VoiceTurnResult]:
    """Synthesise the prompt, push it through kiosk-core, and read the trace."""
    print(f"\n[tier-B] voice E2E × {runs} → {CORE_BASE_URL}/api/v1/sessions/start-file")
    results: list[VoiceTurnResult] = []

    try:
        wav_bytes, synth_ms = synthesize_prompt_wav(prompt, tts_model, tts_language)
        print(f"[tier-B]   prompt synthesised: {len(wav_bytes)} bytes in {synth_ms:.0f} ms")
    except Exception as exc:
        print(f"[tier-B]   FAILED to synthesise prompt: {exc}")
        return [VoiceTurnResult(run=1, error=f"tts synthesis failed: {exc}")]

    for i in range(1, runs + 1):
        conversation_id = f"bench-voice-{uuid.uuid4().hex[:8]}"
        fields = {
            "sample_rate": "16000",
            "chunk_seconds": "4.0",
            "silence_timeout_seconds": "1.5",
            "max_session_seconds": str(max_session_seconds),
            "silence_threshold": "900",
            "temperature": "0.0",
            "realtime_factor": "100.0",  # replay the file as fast as allowed
            "analyzer_url": CONTAINER_ANALYZER_URL,
            "rag_url": CONTAINER_RAG_URL,
            "tts_url": CONTAINER_TTS_URL,
            "tts_model": tts_model,
            "tts_language": tts_language,
        }

        t0 = time.perf_counter()
        try:
            started = http_post_multipart(
                f"{CORE_BASE_URL}/api/v1/sessions/start-file",
                fields=fields,
                file_field="file",
                filename="prompt.wav",
                file_bytes=wav_bytes,
                timeout=300.0,
            )
            session_id = str(started.get("session_id"))
            snapshot = poll_session(session_id, timeout=session_timeout)
            wall_ms = (time.perf_counter() - t0) * 1000

            trace = fetch_pipeline_trace() or {}
            agent = trace.get("agent", {}) or {}
            result = VoiceTurnResult(
                run=i,
                tts_synth_ms=round(synth_ms, 1),
                session_wall_ms=round(wall_ms, 1),
                transcript=snapshot.get("transcript", ""),
                reply=snapshot.get("response", ""),
                trace_wall_total_ms=(trace.get("wall", {}) or {}).get("turn_total_ms"),
                trace_time_to_first_audio_ms=(trace.get("wall", {}) or {}).get("time_to_first_audio_ms"),
                trace_asr_ms=(trace.get("asr", {}) or {}).get("ms"),
                trace_agent_ttft_ms=agent.get("ttft_ms"),
                trace_agent_total_ms=agent.get("total_ms"),
                trace_tts_ms=(trace.get("tts", {}) or {}).get("ms"),
                error=snapshot.get("error"),
            )
            print(
                f"[tier-B]   run {i}/{runs}: wall={wall_ms:.0f} ms  "
                f"status={snapshot.get('status')}  "
                f"agent_total={result.trace_agent_total_ms}  "
                f"asr={result.trace_asr_ms}"
            )
            print(f"[tier-B]                transcript: {result.transcript[:120]!r}")
            print(f"[tier-B]                reply     : {result.reply[:150]!r}")
        except Exception as exc:
            wall_ms = (time.perf_counter() - t0) * 1000
            print(f"[tier-B]   run {i}/{runs}: FAILED after {wall_ms:.0f} ms — {exc}")
            result = VoiceTurnResult(run=i, session_wall_ms=round(wall_ms, 1), error=str(exc))

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "count": len(values),
        "min_ms": round(min(values), 1),
        "mean_ms": round(statistics.fmean(values), 1),
        "median_ms": round(statistics.median(values), 1),
        "max_ms": round(max(values), 1),
    }


def build_summary(report: BenchmarkReport) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    ok_agent = [r.latency_ms for r in report.agent_runs if not r.error]
    if ok_agent:
        summary["agent_turn"] = _stats(ok_agent)
        tool_sets = [tuple(r.tool_calls) for r in report.agent_runs if not r.error]
        summary["agent_tool_calls"] = [list(t) for t in set(tool_sets)]
        for attr, key in (("llm1_ms", "llm1"), ("tool_ms", "tool"), ("llm2_ms", "llm2")):
            vals = [
                getattr(r, attr) for r in report.agent_runs
                if not r.error and getattr(r, attr) is not None
            ]
            if vals:
                summary.setdefault("agent_split", {})[key] = _stats(vals)
        tools_used = {r.tool_name for r in report.agent_runs if r.tool_name}
        if tools_used:
            summary["agent_tool_invoked"] = sorted(tools_used)

    ok_voice = [r for r in report.voice_runs if not r.error]
    if ok_voice:
        summary["voice_e2e"] = {
            "session_wall": _stats([r.session_wall_ms for r in ok_voice]),
            "trace_turn_total": _stats(
                [r.trace_wall_total_ms for r in ok_voice if r.trace_wall_total_ms]
            ),
            "trace_time_to_first_audio": _stats(
                [r.trace_time_to_first_audio_ms for r in ok_voice if r.trace_time_to_first_audio_ms]
            ),
            "trace_asr": _stats([r.trace_asr_ms for r in ok_voice if r.trace_asr_ms]),
            "trace_agent_total": _stats(
                [r.trace_agent_total_ms for r in ok_voice if r.trace_agent_total_ms]
            ),
            "trace_tts": _stats([r.trace_tts_ms for r in ok_voice if r.trace_tts_ms]),
        }

    errors = [r.error for r in report.agent_runs if r.error]
    errors += [r.error for r in report.voice_runs if r.error]
    if errors:
        summary["errors"] = errors
    return summary


def print_summary(report: BenchmarkReport) -> None:
    print("\n" + "=" * 74)
    print(f"BENCHMARK SUMMARY — label={report.label!r}")
    print("=" * 74)
    print(f"prompt: {report.prompt!r}")
    if report.config.get("ovms_models"):
        print(f"ovms model(s): {report.config['ovms_models']}")
    if report.config.get("ovms_image"):
        print(f"ovms image   : {report.config['ovms_image']}")

    agent = report.summary.get("agent_turn")
    if agent:
        print("\nTier A — agent turn (LLM#1 + tool + LLM#2)")
        print(f"  runs   : {agent['count']}")
        print(f"  min    : {agent['min_ms']:>9.0f} ms")
        print(f"  median : {agent['median_ms']:>9.0f} ms")
        print(f"  mean   : {agent['mean_ms']:>9.0f} ms")
        print(f"  max    : {agent['max_ms']:>9.0f} ms")
        print(f"  tools  : {report.summary.get('agent_tool_invoked') or report.summary.get('agent_tool_calls')}")

        split = report.summary.get("agent_split") or {}
        if split:
            print("\n  ADK two-call breakdown (median):")
            for key, label in (("llm1", "LLM#1 tool selection"), ("tool", "MCP tool exec"), ("llm2", "LLM#2 reply")):
                st = split.get(key) or {}
                if st:
                    pct = (st["median_ms"] / agent["median_ms"]) * 100
                    print(f"    {label:<22}: {st['median_ms']:>8.0f} ms  ({pct:4.1f}%)")

    voice = report.summary.get("voice_e2e")
    if voice:
        print("\nTier B — voice E2E through kiosk-core")
        for key, label in (
            ("session_wall", "session wall (upload→done)"),
            ("trace_turn_total", "trace turn_total"),
            ("trace_time_to_first_audio", "trace time_to_first_audio"),
            ("trace_asr", "trace asr"),
            ("trace_agent_total", "trace agent_total"),
            ("trace_tts", "trace tts"),
        ):
            st = voice.get(key) or {}
            if st:
                print(f"  {label:<28}: median {st['median_ms']:>8.0f} ms   mean {st['mean_ms']:>8.0f} ms")

    if report.summary.get("errors"):
        print("\nERRORS:")
        for err in report.summary["errors"]:
            print(f"  - {err}")
    print("=" * 74)


def print_comparison(baseline_path: Path, current: BenchmarkReport) -> None:
    try:
        base = json.loads(baseline_path.read_text())
    except Exception as exc:
        print(f"\n[compare] could not read baseline {baseline_path}: {exc}")
        return

    print("\n" + "=" * 74)
    print(f"COMPARISON — baseline={base.get('label')!r} vs current={current.label!r}")
    print("=" * 74)

    def _row(name: str, before: float | None, after: float | None) -> None:
        if not before or not after:
            return
        delta = after - before
        pct = (delta / before) * 100 if before else 0.0
        arrow = "▼" if delta < 0 else "▲"
        print(f"  {name:<30} {before:>8.0f} → {after:>8.0f} ms   {arrow} {abs(pct):5.1f}%")

    b_agent = (base.get("summary", {}).get("agent_turn") or {}).get("median_ms")
    c_agent = (current.summary.get("agent_turn") or {}).get("median_ms")
    _row("Tier A agent turn (median)", b_agent, c_agent)

    b_split = base.get("summary", {}).get("agent_split") or {}
    c_split = current.summary.get("agent_split") or {}
    for key, label in (
        ("llm1", "  ├─ LLM#1 tool selection"),
        ("tool", "  ├─ MCP tool exec"),
        ("llm2", "  └─ LLM#2 reply"),
    ):
        _row(label, (b_split.get(key) or {}).get("median_ms"), (c_split.get(key) or {}).get("median_ms"))

    b_voice = base.get("summary", {}).get("voice_e2e") or {}
    c_voice = current.summary.get("voice_e2e") or {}
    for key, label in (
        ("trace_turn_total", "Tier B turn_total (median)"),
        ("trace_time_to_first_audio", "Tier B first audio (median)"),
        ("trace_agent_total", "Tier B agent_total (median)"),
        ("trace_asr", "Tier B asr (median)"),
    ):
        _row(label, (b_voice.get(key) or {}).get("median_ms"), (c_voice.get(key) or {}).get("median_ms"))
    print("=" * 74)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="baseline", help="Name for this measurement run")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Utterance to send")
    p.add_argument("--tier", choices=["A", "B", "AB"], default="AB", help="Which tiers to run")
    p.add_argument("--runs", type=int, default=3, help="Measured runs per tier")
    p.add_argument("--warmup", type=int, default=1, help="Unmeasured warmup turns (Tier A)")
    p.add_argument("--user-id", default="kiosk-user", help="Ordering user id")
    p.add_argument(
        "--reuse-session",
        action="store_true",
        help="Reuse one agent session across Tier-A runs (exercises prefix-cache reuse)",
    )
    p.add_argument("--skip-up", action="store_true", help="Assume the stack is already running")
    p.add_argument("--up-timeout", type=float, default=900.0, help="Timeout for `make up`")
    p.add_argument("--health-timeout", type=float, default=1200.0, help="Timeout waiting for health")
    p.add_argument("--session-timeout", type=float, default=180.0, help="Timeout per voice session")
    p.add_argument("--max-session-seconds", type=float, default=25.0, help="kiosk-core max_session_seconds")
    p.add_argument("--tts-model", default="qwen-tts")
    p.add_argument("--tts-language", default="English")
    p.add_argument("--compare", type=Path, default=None, help="Baseline JSON to diff against")
    p.add_argument("--out", type=Path, default=None, help="Output JSON path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    report = BenchmarkReport(
        label=args.label,
        prompt=args.prompt,
        started_at=datetime.now(UTC).isoformat(),
    )

    if not args.skip_up:
        stack_up(args.up_timeout)
    else:
        print("[stack] --skip-up: using the already-running stack")

    try:
        wait_for_health(args.health_timeout)
    except TimeoutError as exc:
        print(f"[stack] {exc}")
        return 2

    report.config = collect_stack_config()
    report.config.update(
        {
            "runs": args.runs,
            "warmup": args.warmup,
            "reuse_session": args.reuse_session,
            "tier": args.tier,
        }
    )

    if "A" in args.tier:
        report.agent_runs = run_agent_tier(
            prompt=args.prompt,
            runs=args.runs,
            warmup=args.warmup,
            user_id=args.user_id,
            fresh_session=not args.reuse_session,
        )

    if "B" in args.tier:
        report.voice_runs = run_voice_tier(
            prompt=args.prompt,
            runs=args.runs,
            tts_model=args.tts_model,
            tts_language=args.tts_language,
            session_timeout=args.session_timeout,
            max_session_seconds=args.max_session_seconds,
        )

    report.finished_at = datetime.now(UTC).isoformat()
    report.summary = build_summary(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (RESULTS_DIR / f"{args.label}.json")
    out_path.write_text(json.dumps(report.to_dict(), indent=2))

    print_summary(report)
    print(f"\n[out] results written to {out_path}")

    if args.compare:
        print_comparison(args.compare, report)

    has_data = bool(report.summary.get("agent_turn") or report.summary.get("voice_e2e"))
    return 0 if has_data else 1


if __name__ == "__main__":
    raise SystemExit(main())
