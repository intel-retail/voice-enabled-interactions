#!/usr/bin/env python3
"""Pre-flight validation for a benchmark device profile.

Why this exists
---------------
A device-combination benchmark run costs 20+ minutes of wall clock. Two classes
of mistake are worth catching in the first two seconds instead:

1. **Impossible combinations** — the engine physically cannot run on the
   requested device (Kokoro on GPU, ``distil-small.en`` on NPU, diarization
   anywhere but CPU). These fail at model load, deep into bring-up.

2. **Silently mislabelled combinations** — far more dangerous. If
   ``ACCEL_MOUNT_PATH`` is left at its portable default of ``/dev/null``, the
   container has no NPU node, OpenVINO quietly falls back to CPU, and the run
   completes successfully producing entirely plausible numbers filed under an
   ``asr-npu`` label. Nothing downstream can detect this. A wrong benchmark
   that looks right is worse than one that crashes, so availability of every
   requested accelerator is checked against the *host* before bring-up.

Exit codes:
    0 — valid (warnings may still have been printed)
    1 — invalid; the run must not proceed

Usage::

    python3 scripts/validate_device_profile.py --profile asr-npu_tts-cpu
    python3 scripts/validate_device_profile.py --env-file configs/benchmark-profiles/x.env
    python3 scripts/validate_device_profile.py --profile x --json
    python3 scripts/validate_device_profile.py --profile x --no-hardware-check

Stdlib only — runs on the host without a virtualenv, like the other benchmark
tooling.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

PROFILE_DIR = Path("configs/benchmark-profiles")

VALID_DEVICES = {"CPU", "GPU", "NPU"}

# (name-substring, runtime) -> set of devices the implementation supports.
TTS_ENGINES: dict[tuple[str, str], set[str]] = {
    ("kokoro", "kokoro"): {"CPU"},
    ("speecht5", "openvino"): {"CPU", "GPU", "NPU"},
    ("speecht5", "pytorch"): {"CPU"},
    ("qwen", "openvino"): {"CPU", "GPU"},
    ("qwen", "pytorch"): {"CPU"},
}

TTS_SPEAKERS: dict[str, set[str]] = {
    "speecht5": {
        "Ryan", "Miles", "Aaron", "Nora", "Elena", "Kabir", "Angus",
        "bdl", "jmk", "rms", "clb", "slt", "ksp", "awb",
    },
    "kokoro": {
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice",
        "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable",
        "bm_george", "bm_lewis",
    },
}

TTS_VARIANTS: dict[str, set[str]] = {
    "speecht5": {"default"},
    "qwen": {"custom_voice", "voice_design"},
    "kokoro": {"default", "custom_voice"},
}

ASR_PROVIDER_DEVICES: dict[str, set[str]] = {
    "openvino": {"CPU", "GPU", "NPU"},
    "openai": {"CPU"},
    "whispercpp": {"CPU"},
}


class Report:
    """Collects errors and warnings for a single profile validation."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, rule: str, msg: str) -> None:
        """Record a fatal rule violation."""
        self.errors.append(f"[{rule}] {msg}")

    def warn(self, rule: str, msg: str) -> None:
        """Record a non-fatal advisory."""
        self.warnings.append(f"[{rule}] {msg}")

    @property
    def ok(self) -> bool:
        """True when no fatal violations were recorded."""
        return not self.errors


# --------------------------------------------------------------------------
# env file loading
# --------------------------------------------------------------------------

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE env file, ignoring comments and blank lines.

    Args:
        path: Path to the env file.

    Returns:
        Mapping of variable name to value, with surrounding quotes stripped.
    """
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val and val[0] not in "\"'":
            val = val.split(" #", 1)[0].strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out


def resolve_profile(
    profile: str | None, env_file: str | None
) -> tuple[str, dict[str, str]]:
    """Resolve the effective profile env, layering base ``.env`` underneath.

    Args:
        profile: Profile name under ``configs/benchmark-profiles/``.
        env_file: Explicit env file path, taking precedence over ``profile``.

    Returns:
        Tuple of (source description, merged environment mapping).
    """
    base: dict[str, str] = {}
    if Path(".env").is_file():
        base = load_env_file(Path(".env"))

    if env_file:
        path = Path(env_file)
    elif profile:
        path = PROFILE_DIR / f"{profile}.env"
    else:
        return ("<.env>", base)

    if not path.is_file():
        available = sorted(p.stem for p in PROFILE_DIR.glob("*.env"))
        raise SystemExit(
            f"profile not found: {path}\n"
            f"available: {', '.join(available) or '(none)'}"
        )

    merged = dict(base)
    merged.update(load_env_file(path))
    return (str(path), merged)


# --------------------------------------------------------------------------
# host capability detection
# --------------------------------------------------------------------------

def detect_gpu() -> tuple[bool, str]:
    """Detect a DRM render node on the host.

    Returns:
        Tuple of (present, human-readable detail).
    """
    nodes = sorted(glob.glob("/dev/dri/renderD*"))
    if nodes:
        return True, f"render node(s): {', '.join(nodes)}"
    return False, "no /dev/dri/renderD* render node found"


def detect_npu() -> tuple[bool, str]:
    """Detect an Intel NPU accel node on the host.

    Returns:
        Tuple of (present, human-readable detail).
    """
    nodes = sorted(glob.glob("/dev/accel/accel*"))
    if nodes:
        return True, f"accel node(s): {', '.join(nodes)}"
    return False, (
        "no /dev/accel/accel* node found (is the intel_vpu/NPU driver loaded?)"
    )


def check_npu_wiring(env: dict[str, str], rep: Report) -> None:
    """Verify ``ACCEL_MOUNT_PATH`` actually points at a usable NPU node.

    This is the rule that prevents a silently-CPU run being filed as an NPU
    result: compose mounts ``${ACCEL_MOUNT_PATH:-/dev/null}`` into the
    container, so the portable default makes the accelerator invisible without
    any error surfacing.

    Args:
        env: Resolved profile environment.
        rep: Report to append findings to.
    """
    mount = env.get("ACCEL_MOUNT_PATH", "").strip()
    if not mount:
        rep.error(
            "npu-mount",
            "an NPU device is requested but ACCEL_MOUNT_PATH is unset. "
            "Compose then mounts /dev/null as the accel node, OpenVINO falls "
            "back to CPU silently, and the run is mislabelled. Set "
            "ACCEL_MOUNT_PATH=/dev/accel/accel0 in .env.",
        )
        return
    if mount == "/dev/null":
        rep.error(
            "npu-mount",
            "an NPU device is requested but ACCEL_MOUNT_PATH=/dev/null "
            "(the portable 'NPU disabled' default). OpenVINO would silently "
            "fall back to CPU and the results would be mislabelled. Point it "
            "at the real node, e.g. /dev/accel/accel0.",
        )
        return
    if not Path(mount).exists():
        rep.error(
            "npu-mount", f"ACCEL_MOUNT_PATH={mount} does not exist on this host."
        )


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------

def _dev(env: dict[str, str], key: str, default: str) -> str:
    """Read a device variable, normalised to upper case."""
    return (env.get(key) or default).strip().upper()


def _engine_key(model: str, runtime: str) -> tuple[str, str] | None:
    """Resolve a (model, runtime) pair to a known TTS engine key."""
    m = model.lower()
    r = runtime.lower()
    for name_sub, rt in TTS_ENGINES:
        if name_sub in m and rt == r:
            return (name_sub, rt)
    return None


def validate(env: dict[str, str], check_hardware: bool = True) -> Report:
    """Apply every profile rule and return the collected report.

    Args:
        env: Resolved profile environment.
        check_hardware: Whether to probe the host for GPU/NPU nodes.

    Returns:
        A populated :class:`Report`.
    """
    rep = Report()

    asr_dev = _dev(env, "ASR_DEVICE", "CPU")
    asr_prev = _dev(env, "ASR_PREVIEW_DEVICE", asr_dev)
    asr_model = (env.get("ASR_MODEL") or "distil-whisper/distil-small.en").strip()
    asr_provider = (env.get("ASR_PROVIDER") or "openvino").strip().lower()
    diar_dev = _dev(env, "DIARIZATION_DEVICE", "CPU")

    tts_dev = _dev(env, "TTS_DEVICE", "CPU")
    tts_model = (env.get("TTS_MODEL") or "kokoro").strip()
    tts_runtime = (env.get("TTS_RUNTIME") or "kokoro").strip().lower()
    tts_dtype = (env.get("TTS_DTYPE") or "int8").strip().lower()
    tts_speaker = (env.get("TTS_SPEAKER") or "am_michael").strip()
    tts_variant = (env.get("TTS_MODEL_VARIANT") or "custom_voice").strip().lower()

    llm_dev = _dev(env, "TARGET_DEVICE", "GPU")
    emb_dev = _dev(env, "RAG_EMBEDDING_DEVICE", "GPU")
    rrk_dev = _dev(env, "RAG_RERANKER_DEVICE", "GPU")

    # -- device string sanity --------------------------------------------
    for label, value in (
        ("ASR_DEVICE", asr_dev),
        ("ASR_PREVIEW_DEVICE", asr_prev),
        ("DIARIZATION_DEVICE", diar_dev),
        ("TTS_DEVICE", tts_dev),
        ("TARGET_DEVICE", llm_dev),
        ("RAG_EMBEDDING_DEVICE", emb_dev),
        ("RAG_RERANKER_DEVICE", rrk_dev),
    ):
        base = value.split(".", 1)[0]
        if base not in VALID_DEVICES:
            # A .env file is not a shell: docker-compose does not expand
            # ${OTHER_VAR} on the right-hand side of a .env assignment, it
            # passes the literal text through. Call that out explicitly --
            # the generic message below sent a reviewer hunting for a bad
            # device name when the real problem was the syntax.
            if "${" in value:
                rep.error(
                    "device-string",
                    f"{label}={value} contains an unexpanded shell "
                    "substitution. A .env file is not a shell -- write a "
                    "literal CPU/GPU/NPU, or leave the variable unset to "
                    "inherit the fallback defined in docker-compose.yml.",
                )
            else:
                rep.error("device-string", f"{label}={value} is not one of CPU/GPU/NPU")

    # -- ASR --------------------------------------------------------------
    allowed = ASR_PROVIDER_DEVICES.get(asr_provider)
    if allowed is None:
        rep.error("asr-provider", f"unknown ASR provider '{asr_provider}'")
    else:
        for label, value in (
            ("ASR_DEVICE", asr_dev),
            ("ASR_PREVIEW_DEVICE", asr_prev),
        ):
            if value in VALID_DEVICES and value not in allowed:
                rep.error(
                    "asr-provider-device",
                    f"{label}={value} but provider '{asr_provider}' supports "
                    f"only {'/'.join(sorted(allowed))}.",
                )

    if "distil" in asr_model.lower():
        for label, value in (
            ("ASR_DEVICE", asr_dev),
            ("ASR_PREVIEW_DEVICE", asr_prev),
        ):
            if value == "NPU":
                rep.error(
                    "asr-npu-corruption",
                    f"{label}=NPU with ASR_MODEL={asr_model}. distil-small.en "
                    "corrupts on this NPU + openvino_genai stack (output goes "
                    "stale/frozen, repeating the first call's text; confirmed "
                    "twice independently). Use ASR_MODEL=whisper-base for the "
                    "NPU leg.",
                )

    # OpenVINO NPU only supports static-shape models. Whisper's IR is dynamic,
    # so GenAI's NPUW plugin pattern-matches attention blocks to make it static
    # -- a heuristic that succeeds for tiny/base and fails for small and up with
    # "Check '!self_attn_nodes.empty()' failed" at compile time.
    _NPU_SAFE_ASR = ("whisper-tiny", "whisper-base")
    for label, value in (("ASR_DEVICE", asr_dev), ("ASR_PREVIEW_DEVICE", asr_prev)):
        if value == "NPU" and not any(
            asr_model.lower().endswith(m) or asr_model.lower() == m
            for m in _NPU_SAFE_ASR
        ):
            rep.error(
                "asr-npu-compile",
                f"{label}=NPU with ASR_MODEL={asr_model}. OpenVINO NPU only "
                "supports static shapes; the NPUW attention-block heuristic "
                "succeeds only for whisper-tiny/whisper-base. Larger Whisper "
                "checkpoints fail to compile with "
                "\"Check '!self_attn_nodes.empty()' failed\". Use "
                "ASR_MODEL=whisper-base.",
            )

    # -- diarization -------------------------------------------------------
    if diar_dev != "CPU":
        enabled = (env.get("KIOSK_CORE_DIARIZATION_ENABLED") or "true").lower()
        msg = (
            f"DIARIZATION_DEVICE={diar_dev}. The diarization component loads via "
            "torch.device(<value>); PyTorch has no 'gpu' device string (Intel "
            "GPU needs 'xpu', which this component does not wire). Only CPU is "
            "valid."
        )
        if enabled in ("false", "0", "no"):
            rep.warn(
                "diarization-device",
                msg + " Diarization is disabled, so this is inert.",
            )
        else:
            rep.error("diarization-device", msg)

    # -- TTS engine consistency -------------------------------------------
    key = _engine_key(tts_model, tts_runtime)
    if key is None:
        rep.error(
            "tts-engine",
            f"TTS_MODEL={tts_model} + TTS_RUNTIME={tts_runtime} is not a known "
            "pair. Valid: (kokoro, kokoro), (microsoft/speecht5_tts, openvino), "
            "(Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice, openvino).",
        )
    else:
        engine = key[0]
        supported = TTS_ENGINES[key]
        if tts_dev in VALID_DEVICES and tts_dev not in supported:
            detail = ""
            if engine == "kokoro":
                detail = (
                    " kokoro-onnx uses the CPU execution provider and is not "
                    "wired for GPU/NPU in kokoro_tts.py. For a TTS device "
                    "comparison use TTS_MODEL=microsoft/speecht5_tts with "
                    "TTS_RUNTIME=openvino."
                )
            rep.error(
                "tts-engine-device",
                f"TTS_DEVICE={tts_dev} but engine '{engine}' "
                f"({tts_runtime} runtime) supports only "
                f"{'/'.join(sorted(supported))}.{detail}",
            )

        speakers = TTS_SPEAKERS.get(engine)
        if speakers and tts_speaker not in speakers:
            rep.error(
                "tts-speaker",
                f"TTS_SPEAKER={tts_speaker} is not a valid voice for engine "
                f"'{engine}'. Voice vocabularies do not overlap between "
                "engines, so every synthesis request would fail validation. "
                f"Valid examples: {', '.join(sorted(speakers)[:6])}.",
            )

        variants = TTS_VARIANTS.get(engine)
        if variants and tts_variant not in variants:
            rep.error(
                "tts-variant",
                f"TTS_MODEL_VARIANT={tts_variant} is not valid for engine "
                f"'{engine}'. Valid: {'/'.join(sorted(variants))}.",
            )

    # -- TTS dtype ---------------------------------------------------------
    if tts_dev == "GPU" and tts_dtype == "int8":
        rep.error(
            "tts-gpu-dtype",
            "TTS_DEVICE=GPU with TTS_DTYPE=int8. Measured on this iGPU: 281s "
            "first call, 8.7-8.8s steady-state per synthesis, versus 0.92-0.95s "
            "on CPU — int8 falls back to software/emulated ops. Set "
            "TTS_DTYPE=fp16 for any GPU profile.",
        )
    if tts_dev == "GPU" and tts_dtype == "int4":
        rep.error(
            "tts-gpu-dtype", "int4 on this iGPU produces audible noise. Use fp16."
        )
    if tts_dtype not in ("int8", "int4", "fp16", "fp32"):
        rep.error("tts-dtype", f"TTS_DTYPE={tts_dtype} is not int8/int4/fp16/fp32")

    # -- hardware availability ---------------------------------------------
    requested = {
        "ASR_DEVICE": asr_dev,
        "ASR_PREVIEW_DEVICE": asr_prev,
        "TTS_DEVICE": tts_dev,
        "TARGET_DEVICE": llm_dev,
        "RAG_EMBEDDING_DEVICE": emb_dev,
        "RAG_RERANKER_DEVICE": rrk_dev,
    }
    wants_npu = [k for k, v in requested.items() if v == "NPU"]
    wants_gpu = [k for k, v in requested.items() if v.startswith("GPU")]

    if check_hardware:
        if wants_npu:
            present, detail = detect_npu()
            if not present:
                rep.error(
                    "npu-available",
                    f"{', '.join(wants_npu)} request NPU but {detail}. "
                    "Benchmarking NPU on a host without one silently measures "
                    "CPU.",
                )
            check_npu_wiring(env, rep)
        if wants_gpu:
            present, detail = detect_gpu()
            if not present:
                rep.error(
                    "gpu-available",
                    f"{', '.join(wants_gpu)} request GPU but {detail}.",
                )
    elif wants_npu or wants_gpu:
        rep.warn(
            "hardware-check",
            "host accelerator checks skipped (--no-hardware-check); a missing "
            "device will silently fall back to CPU.",
        )

    # -- benchmarking hygiene ----------------------------------------------
    if (env.get("QUEUE") or "true").lower() in ("true", "1", "yes"):
        rep.warn(
            "queue-contention",
            "QUEUE is enabled. queue-service (YOLO) burns ~650-750% CPU "
            "continuously and contends with TTS, the voice pipeline's only "
            "CPU-bound stage. Set QUEUE=false for latency benchmarking.",
        )
    if not (env.get("PROFILE_NAME") or "").strip():
        rep.warn(
            "profile-name",
            "PROFILE_NAME is not set; results will not be namespaced.",
        )

    return rep


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--profile", help="profile name under configs/benchmark-profiles/")
    ap.add_argument("--env-file", help="explicit path to a profile env file")
    ap.add_argument(
        "--no-hardware-check",
        action="store_true",
        help="skip host GPU/NPU availability detection (CI/dry-run use only)",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    source, env = resolve_profile(args.profile, args.env_file)
    rep = validate(env, check_hardware=not args.no_hardware_check)

    asr_dev = _dev(env, "ASR_DEVICE", "CPU")
    summary = {
        "profile": env.get("PROFILE_NAME", ""),
        "source": source,
        "devices": {
            "asr": asr_dev,
            "asr_preview": _dev(env, "ASR_PREVIEW_DEVICE", asr_dev),
            "tts": _dev(env, "TTS_DEVICE", "CPU"),
            "llm": _dev(env, "TARGET_DEVICE", "GPU"),
            "diarization": _dev(env, "DIARIZATION_DEVICE", "CPU"),
        },
        "models": {
            "asr": env.get("ASR_MODEL", "distil-whisper/distil-small.en"),
            "tts": env.get("TTS_MODEL", "kokoro"),
            "tts_runtime": env.get("TTS_RUNTIME", "kokoro"),
            "tts_dtype": env.get("TTS_DTYPE", "int8"),
        },
        "host": {"gpu": detect_gpu()[1], "npu": detect_npu()[1]},
        "errors": rep.errors,
        "warnings": rep.warnings,
        "ok": rep.ok,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0 if rep.ok else 1

    d = summary["devices"]
    m = summary["models"]
    print(f"Profile : {summary['profile'] or '(unnamed)'}  ({source})")
    print(
        f"Devices : ASR={d['asr']} (preview {d['asr_preview']})  "
        f"TTS={d['tts']}  LLM={d['llm']}  diarization={d['diarization']}"
    )
    print(
        f"Models  : ASR={m['asr']}  "
        f"TTS={m['tts']}/{m['tts_runtime']}/{m['tts_dtype']}"
    )

    for w in rep.warnings:
        print(f"  WARN  {w}")
    for e in rep.errors:
        print(f"  ERROR {e}")

    if rep.ok:
        print("OK — profile is valid.")
        return 0
    print(f"INVALID — {len(rep.errors)} error(s). Run aborted.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
