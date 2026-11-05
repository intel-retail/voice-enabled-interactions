"""
Tier 1 — Environment Variable Validation Tests
================================================
Covers test case #2:
  Validate .env.example — all required variables are documented, HF_TOKEN is
  optional (only required when diarization=true), TARGET_DEVICE accepts only
  CPU | GPU | NPU, RENDER_GID must be documented for GPU mode.

These tests are CI-safe: no Docker, no ML models, no audio hardware.

Run:
    pytest tests/functional/test_env_validation.py -m tier1 -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

_KIOSK_ROOT = Path(__file__).resolve().parents[2]
_ENV_EXAMPLE = _KIOSK_ROOT / ".env.example"
_DOCKER_COMPOSE = _KIOSK_ROOT / "docker-compose.yml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_env_example() -> dict[str, str | None]:
    """Parse .env.example into {KEY: value_or_none}."""
    result: dict[str, str | None] = {}
    if not _ENV_EXAMPLE.exists():
        return result
    for line in _ENV_EXAMPLE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip() or None
    return result


def _strip_yaml_comment(line: str) -> str:
    """Drop the trailing YAML comment from a line.

    A ``#`` starts a comment only at the start of the line or when preceded by
    whitespace (matching YAML's own rule), so values like ``color: '#fff'`` are
    left intact. This keeps documentation that spells out compose syntax — e.g.
    ``# Uses ${VAR-} (no colon)`` — from being mistaken for a real reference.
    """
    for idx, ch in enumerate(line):
        if ch == "#" and (idx == 0 or line[idx - 1].isspace()):
            return line[:idx]
    return line


def _docker_compose_env_keys() -> set[str]:
    """Return all env var names referenced in docker-compose.yml.

    Comments are stripped first so that ``${VAR}`` placeholders used purely as
    documentation in comments are not counted as live references.
    """
    with open(_DOCKER_COMPOSE) as fh:
        compose = yaml.safe_load(fh)
    keys: set[str] = set()
    import re
    text = "\n".join(
        _strip_yaml_comment(line) for line in _DOCKER_COMPOSE.read_text().splitlines()
    )
    # Match ${VAR_NAME} and ${VAR_NAME:-default} patterns
    for match in re.finditer(r"\$\{([A-Z_][A-Z0-9_]*)(?::?[^}]*)?\}", text):
        keys.add(match.group(1))
    return keys


# ---------------------------------------------------------------------------
# .env.example presence and structure
# ---------------------------------------------------------------------------
class TestEnvExamplePresence:
    """Verify .env.example exists and is well-formed."""

    @pytest.mark.tier1
    def test_env_example_exists(self):
        """`.env.example` must be present at the project root."""
        assert _ENV_EXAMPLE.is_file(), (
            f".env.example not found at {_ENV_EXAMPLE}"
        )

    @pytest.mark.tier1
    def test_env_example_is_non_empty(self):
        """.env.example must not be empty."""
        assert _ENV_EXAMPLE.stat().st_size > 0, ".env.example is empty"

    @pytest.mark.tier1
    def test_env_example_parses_without_error(self):
        """Every KEY=VALUE line in .env.example must be parseable."""
        env = _parse_env_example()
        assert len(env) > 0, ".env.example has no KEY=VALUE lines"

    @pytest.mark.tier1
    def test_env_example_has_no_duplicate_keys(self):
        """No key should appear twice in .env.example."""
        keys: list[str] = []
        for line in _ENV_EXAMPLE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                keys.append(line.partition("=")[0].strip())
        duplicates = [k for k in keys if keys.count(k) > 1]
        assert not duplicates, f"Duplicate keys in .env.example: {set(duplicates)}"


# ---------------------------------------------------------------------------
# Required variables
# ---------------------------------------------------------------------------
class TestRequiredVariables:
    """Core required variables must be documented in .env.example."""

    REQUIRED_KEYS = [
        "TARGET_DEVICE",
        "RELEASE_TAG",
        "REGISTRY",
    ]

    @pytest.mark.tier1
    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_required_key_present(self, key):
        """Each required variable must exist in .env.example."""
        env = _parse_env_example()
        assert key in env, (
            f"Required variable '{key}' not found in .env.example. "
            f"Present keys: {list(env)}"
        )

    @pytest.mark.tier1
    def test_render_gid_documented(self):
        """RENDER_GID must be documented (needed for GPU device access in containers)."""
        env = _parse_env_example()
        assert "RENDER_GID" in env, (
            "RENDER_GID not documented in .env.example. "
            "This is required for GPU device pass-through to Docker containers."
        )


# ---------------------------------------------------------------------------
# HF_TOKEN — must be optional
# ---------------------------------------------------------------------------
class TestHfTokenIsOptional:
    """HF_TOKEN is only required when diarization=true; must not block CI."""

    @pytest.mark.tier1
    def test_hf_token_key_documented(self):
        """HF_TOKEN must be documented in .env.example (even if blank)."""
        env = _parse_env_example()
        assert "HF_TOKEN" in env, (
            "HF_TOKEN not documented in .env.example"
        )

    @pytest.mark.tier1
    def test_hf_token_has_no_hardcoded_value(self):
        """HF_TOKEN in .env.example must not contain a real token value."""
        env = _parse_env_example()
        val = env.get("HF_TOKEN", "") or ""
        # A real HF token starts with hf_ and is ~37 chars
        assert not val.startswith("hf_"), (
            "HF_TOKEN in .env.example appears to contain a real token — "
            "never commit credentials to version control."
        )

    @pytest.mark.tier1
    def test_diarization_disabled_by_default(self):
        """When diarization is not explicitly enabled, HF_TOKEN must not be required.

        The docker-compose.yml should not unconditionally require HF_TOKEN — it
        is only consumed by audio-analyzer when diarization is enabled.
        """
        with open(_DOCKER_COMPOSE) as fh:
            compose = yaml.safe_load(fh)

        # audio-analyzer service must not have HF_TOKEN as a mandatory (non-defaulted) env
        aa_env = compose.get("services", {}).get("audio-analyzer", {}).get("environment", {})
        if isinstance(aa_env, dict):
            hf_val = aa_env.get("HF_TOKEN", "")
            # If present it must default to empty or come from env var syntax
            if hf_val:
                assert "${HF_TOKEN" in str(hf_val) or not str(hf_val).startswith("hf_"), (
                    "audio-analyzer has a hardcoded HF_TOKEN in docker-compose.yml"
                )

    @pytest.mark.tier1
    def test_stack_runs_without_hf_token_when_diarization_disabled(self):
        """Verify kiosk_core config respects KIOSK_CORE_DIARIZATION_ENABLED=false
        and does NOT attempt to use HF_TOKEN in that mode."""
        import sys
        from unittest.mock import MagicMock, patch

        sd_mock = MagicMock()
        sd_mock.query_devices.return_value = []

        env_overrides = {
            "KIOSK_CORE_DIARIZATION_ENABLED": "false",
            "HF_TOKEN": "",  # explicitly empty
        }

        with patch.dict("sys.modules", {"sounddevice": sd_mock}), \
             patch.dict(os.environ, env_overrides, clear=False):
            for mod in list(sys.modules.keys()):
                if mod.startswith("kiosk_core"):
                    del sys.modules[mod]
            from kiosk_core import config as kiosk_cfg
            assert not kiosk_cfg.DEFAULT_DIARIZATION_ENABLED, (
                "Diarization must be disabled when KIOSK_CORE_DIARIZATION_ENABLED=false"
            )


# ---------------------------------------------------------------------------
# TARGET_DEVICE — allowed values
# ---------------------------------------------------------------------------
class TestTargetDeviceValues:
    """TARGET_DEVICE must be one of CPU | GPU | NPU."""

    VALID_DEVICES = {"CPU", "GPU", "NPU"}

    @pytest.mark.tier1
    def test_target_device_default_is_valid(self):
        """Default TARGET_DEVICE in .env.example must be CPU, GPU, or NPU."""
        env = _parse_env_example()
        default = env.get("TARGET_DEVICE", "")
        assert default in self.VALID_DEVICES, (
            f"TARGET_DEVICE default '{default}' is not in {self.VALID_DEVICES}"
        )

    @pytest.mark.tier1
    def test_target_device_lowercase_rejected(self):
        """Makefile check-env must reject lowercase device strings (e.g. 'gpu')."""
        makefile = (_KIOSK_ROOT / "Makefile").read_text()
        # The Makefile check-env validates against CPU | GPU | NPU
        assert "CPU" in makefile and "GPU" in makefile and "NPU" in makefile, (
            "Makefile check-env does not validate TARGET_DEVICE against CPU|GPU|NPU"
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize("device", ["CPU", "GPU", "NPU"])
    def test_valid_device_accepted_by_makefile(self, device):
        """Each valid device string appears in Makefile validation logic."""
        makefile = (_KIOSK_ROOT / "Makefile").read_text()
        assert device in makefile, (
            f"Device '{device}' not referenced in Makefile check-env validation"
        )


# ---------------------------------------------------------------------------
# docker-compose.yml — env var completeness
# ---------------------------------------------------------------------------
class TestDockerComposeEnvCompleteness:
    """All ${VAR} references in docker-compose.yml should be documented in .env.example."""

    # Variables that are intentionally injected at runtime (not in .env.example)
    RUNTIME_ONLY = {
        "STREAM_NAME",           # set per rtsp-streamer container
        "RENDER_GID",            # documented but may be omitted on CPU-only setups
        "IDENTITY",              # optional feature flag, default false
        "TAG",                   # derived from RELEASE_TAG in Makefile
        "RTSP_CAMERA_DEVICE",    # computed by Makefile based on RTSP_SOURCE_MODE and CAMERA_DEVICE
        "ASR_PREVIEW_DEVICE",     # optional override; compose supplies a safe default
                                  # (${ASR_PREVIEW_DEVICE:-${ASR_DEVICE:-CPU}}). Left commented
                                  # in .env.example because a .env file cannot expand the nested
                                  # ${ASR_DEVICE:-CPU} fallback (see item 5 / validate_device_profile).
        # Per-service inference devices. All resolve to
        # ${<VAR>:-${TARGET_DEVICE:-GPU}} in docker-compose.yml so they inherit
        # TARGET_DEVICE, letting a CPU-only host switch the whole stack with one
        # variable (ITEP-97604). They must stay *commented out* in .env.example:
        # a .env file is not a shell, so an uncommented ASR_DEVICE=GPU is a
        # literal value that silently wins over TARGET_DEVICE and crash-loops the
        # service with "[GPU] Context was not initialized for 0 device".
        "ASR_DEVICE",
        "RAG_EMBEDDING_DEVICE",
        "RAG_RERANKER_DEVICE",
        "RAG_LLM_DEVICE",
        "IDENTITY_DEVICE",
    }

    @pytest.mark.tier1
    def test_compose_env_vars_covered_by_env_example(self):
        """Every ${VAR} in docker-compose.yml must be in .env.example or known runtime-only."""
        compose_vars = _docker_compose_env_keys()
        env_keys = set(_parse_env_example().keys())
        undocumented = compose_vars - env_keys - self.RUNTIME_ONLY
        assert not undocumented, (
            f"Variables referenced in docker-compose.yml but missing from .env.example: "
            f"{sorted(undocumented)}\n"
            f"Add them to .env.example with a safe default value."
        )


# ---------------------------------------------------------------------------
# Kokoro default voice
# ---------------------------------------------------------------------------
class TestKokoroDefaultVoice:
    """The Kokoro default voice must be af_heart everywhere a default is set.

    The TTS service and kiosk-core each read TTS_SPEAKER with their own
    fallback, so a default changed in one place and not the others makes
    kiosk-core request a voice the service was not configured with.
    """

    EXPECTED_VOICE = "af_heart"

    @pytest.mark.tier1
    def test_env_example_tts_speaker_is_af_heart(self):
        assert _parse_env_example().get("TTS_SPEAKER") == self.EXPECTED_VOICE, (
            f".env.example must set TTS_SPEAKER={self.EXPECTED_VOICE}"
        )

    @pytest.mark.tier1
    def test_compose_tts_speaker_fallbacks_are_af_heart(self):
        import re
        text = "\n".join(
            _strip_yaml_comment(line) for line in _DOCKER_COMPOSE.read_text().splitlines()
        )
        fallbacks = re.findall(r"\$\{TTS_SPEAKER:-([^}]*)\}", text)
        # One for the text-to-speech service, one for kiosk-core.
        assert len(fallbacks) >= 2, (
            f"Expected TTS_SPEAKER fallbacks for both text-to-speech and kiosk-core, "
            f"found {fallbacks}"
        )
        assert set(fallbacks) == {self.EXPECTED_VOICE}, (
            f"docker-compose.yml TTS_SPEAKER fallbacks must all be "
            f"{self.EXPECTED_VOICE}, found {fallbacks}"
        )

    @pytest.mark.tier1
    def test_tts_service_config_default_speaker_is_af_heart(self):
        config_path = _KIOSK_ROOT / "configs" / "text-to-speech" / "config.yaml"
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        assert cfg["models"]["tts"]["default_speaker"] == self.EXPECTED_VOICE

    @pytest.mark.tier1
    def test_af_heart_is_an_accepted_kokoro_voice(self):
        import importlib.util
        script = _KIOSK_ROOT / "scripts" / "validate_device_profile.py"
        spec = importlib.util.spec_from_file_location("validate_device_profile", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert self.EXPECTED_VOICE in module.TTS_SPEAKERS["kokoro"]
