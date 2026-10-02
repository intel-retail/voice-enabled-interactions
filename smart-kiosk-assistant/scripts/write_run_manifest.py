#!/usr/bin/env python3
"""Write a benchmark run manifest capturing exactly what produced a result set.

A latency number found months later is only useful if it can be traced back to
the configuration that produced it: which devices, which models, which image
tag, which commit, on what hardware. ``make benchmark`` calls this before the
measured runs so every ``results/<profile>/`` directory is self-describing.

It builds on ``validate_device_profile.py --json`` (reusing its profile
resolution, device summary and host detection rather than re-deriving them) and
adds run/VCS/host facts on top.

Usage::

    python3 scripts/write_run_manifest.py --results-dir results/asr-npu_tts-cpu \
        --profile asr-npu_tts-cpu --release-tag 2026.1.0 --v2v-runs 3

Stdlib only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_device_profile import (  # noqa: E402
    detect_gpu,
    detect_npu,
    resolve_profile,
    validate,
    _dev,
)


def _sh(cmd: str) -> str:
    """Run a shell command, returning stripped stdout or "" on any failure."""
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def build_manifest(
    profile: str | None,
    release_tag: str,
    v2v_runs: str,
    v2v_warmup_runs: str,
    queue_enabled: str,
    results_dir: str,
    metrics_dir: str,
) -> dict:
    """Assemble the manifest dictionary.

    Args:
        profile: Profile name, or None/"" when running off the bare ``.env``.
        release_tag: Image tag the stack runs at.
        v2v_runs: Measured run count.
        v2v_warmup_runs: Discarded warm-up run count.
        queue_enabled: Whether queue-service was running.
        results_dir: Destination results directory.
        metrics_dir: Hardware counter directory.

    Returns:
        A JSON-serialisable manifest.
    """
    source, env = resolve_profile(profile or None, None)
    rep = validate(env, check_hardware=True)

    asr_dev = _dev(env, "ASR_DEVICE", "CPU")
    return {
        "profile": env.get("PROFILE_NAME", "") or (profile or ""),
        "profile_source": source,
        "valid": rep.ok,
        "validation_errors": rep.errors,
        "validation_warnings": rep.warnings,
        "devices": {
            "asr": asr_dev,
            "asr_preview": _dev(env, "ASR_PREVIEW_DEVICE", asr_dev),
            "tts": _dev(env, "TTS_DEVICE", "CPU"),
            "llm": _dev(env, "TARGET_DEVICE", "GPU"),
            "diarization": _dev(env, "DIARIZATION_DEVICE", "CPU"),
            "rag_embedding": _dev(env, "RAG_EMBEDDING_DEVICE", "GPU"),
            "rag_reranker": _dev(env, "RAG_RERANKER_DEVICE", "GPU"),
        },
        "models": {
            "asr": env.get("ASR_MODEL", "distil-whisper/distil-small.en"),
            "tts": env.get("TTS_MODEL", "kokoro"),
            "tts_runtime": env.get("TTS_RUNTIME", "kokoro"),
            "tts_dtype": env.get("TTS_DTYPE", "int8"),
            "tts_speaker": env.get("TTS_SPEAKER", "am_michael"),
            "llm": env.get("OVMS_MODEL_NAME", "OpenVINO/Qwen3-4B-int8-ov"),
        },
        "run": {
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "release_tag": release_tag,
            "v2v_runs": v2v_runs,
            "v2v_warmup_runs": v2v_warmup_runs,
            "queue_enabled": queue_enabled,
            "diarization_enabled": env.get("KIOSK_CORE_DIARIZATION_ENABLED", "true"),
            "results_dir": results_dir,
            "metrics_dir": metrics_dir,
        },
        "vcs": {
            "git_sha": _sh("git rev-parse --short HEAD"),
            "git_branch": _sh("git rev-parse --abbrev-ref HEAD"),
            "git_dirty": bool(_sh("git status --porcelain")),
        },
        "host": {
            "kernel": _sh("uname -r"),
            "cpu_model": _sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-"),
            "cpu_cores": _sh("nproc"),
            "gpu": detect_gpu()[1],
            "npu": detect_npu()[1],
        },
    }


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--metrics-dir", default="")
    ap.add_argument("--profile", default="")
    ap.add_argument("--release-tag", default="")
    ap.add_argument("--v2v-runs", default="")
    ap.add_argument("--v2v-warmup-runs", default="")
    ap.add_argument("--queue-enabled", default="")
    args = ap.parse_args()

    manifest = build_manifest(
        profile=args.profile,
        release_tag=args.release_tag,
        v2v_runs=args.v2v_runs,
        v2v_warmup_runs=args.v2v_warmup_runs,
        queue_enabled=args.queue_enabled,
        results_dir=args.results_dir,
        metrics_dir=args.metrics_dir,
    )

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
