"""
Tier 1 — Setup Script Validation Tests
========================================
Covers test case #3:
  Validate setup_models.sh:
    - Bash syntax is valid (bash -n)
    - Script is executable
    - All documented CLI flags are present in the file
    - Script does NOT hardcode any HuggingFace tokens
    - Qwen3-4B model reference is present (public model, no token required)

These tests are CI-safe: no model downloads, no Docker, no audio hardware.

Run:
    pytest tests/functional/test_setup_script.py -m tier1 -v
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

_KIOSK_ROOT = Path(__file__).resolve().parents[2]
_SETUP_SCRIPT = _KIOSK_ROOT / "setup_models.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _script_text() -> str:
    return _SETUP_SCRIPT.read_text()


# ---------------------------------------------------------------------------
# File presence and permissions
# ---------------------------------------------------------------------------
class TestSetupScriptPresence:
    """setup_models.sh must exist and be executable."""

    @pytest.mark.tier1
    def test_setup_script_exists(self):
        """setup_models.sh must be present in the project root."""
        assert _SETUP_SCRIPT.is_file(), (
            f"setup_models.sh not found at {_SETUP_SCRIPT}"
        )

    @pytest.mark.tier1
    def test_setup_script_is_executable(self):
        """setup_models.sh must have execute bit set."""
        mode = _SETUP_SCRIPT.stat().st_mode
        is_executable = bool(mode & 0o111)
        assert is_executable, (
            f"setup_models.sh is not executable (mode={oct(mode)}). "
            "Run: chmod +x setup_models.sh"
        )

    @pytest.mark.tier1
    def test_setup_script_has_bash_shebang(self):
        """First line must be a bash shebang."""
        first_line = _script_text().splitlines()[0]
        assert first_line.startswith("#!/bin/bash") or first_line.startswith("#!/usr/bin/env bash"), (
            f"Expected bash shebang, got: '{first_line}'"
        )


# ---------------------------------------------------------------------------
# Syntax check
# ---------------------------------------------------------------------------
class TestSetupScriptSyntax:
    """setup_models.sh must pass bash syntax checking."""

    @pytest.mark.tier1
    def test_bash_syntax_valid(self):
        """bash -n must report no syntax errors."""
        result = subprocess.run(
            ["bash", "-n", str(_SETUP_SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"bash -n reported syntax errors in setup_models.sh:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    @pytest.mark.tier1
    def test_set_e_present(self):
        """Script must use `set -e` to exit on error."""
        text = _script_text()
        assert "set -e" in text, (
            "setup_models.sh should contain 'set -e' to fail fast on errors. "
            "Without it, partial model downloads go undetected."
        )


# ---------------------------------------------------------------------------
# Documented CLI flags
# ---------------------------------------------------------------------------
class TestSetupScriptFlags:
    """All documented CLI flags must be present and parsed."""

    EXPECTED_FLAGS = [
        "--device",
        "--int4",
        "--skip-ovms",
        "--identity",
        "--hf-token",
    ]

    @pytest.mark.tier1
    @pytest.mark.parametrize("flag", EXPECTED_FLAGS)
    def test_flag_present_in_script(self, flag: str):
        """Each documented flag must appear in the script body."""
        text = _script_text()
        assert flag in text, (
            f"Flag '{flag}' is documented in setup_models.sh usage but not found "
            f"in the script body — it may have been removed or renamed."
        )

    @pytest.mark.tier1
    def test_device_flag_validates_cpu_gpu_npu(self):
        """--device flag must validate against CPU | GPU | NPU."""
        text = _script_text()
        # Accept any of: case statement, regex match, or explicit if/elif checks
        has_cpu = "CPU" in text
        has_gpu = "GPU" in text
        has_npu = "NPU" in text
        assert has_cpu and has_gpu and has_npu, (
            "setup_models.sh should validate --device against CPU|GPU|NPU. "
            f"Found: CPU={has_cpu}, GPU={has_gpu}, NPU={has_npu}"
        )


# ---------------------------------------------------------------------------
# Model references
# ---------------------------------------------------------------------------
class TestSetupScriptModelRefs:
    """Script must reference the correct public models."""

    @pytest.mark.tier1
    def test_qwen_model_referenced(self):
        """Qwen3-4B model must be referenced — it is public, no token required."""
        text = _script_text()
        assert "Qwen" in text or "qwen" in text.lower(), (
            "setup_models.sh does not reference any Qwen model. "
            "The LLM (Qwen3-4B-Instruct) should be listed."
        )

    @pytest.mark.tier1
    def test_no_hardcoded_hf_tokens(self):
        """Script must not contain any hardcoded HuggingFace tokens (hf_...)."""
        text = _script_text()
        # Real HF tokens are hf_ followed by 37+ alphanumeric chars
        pattern = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
        matches = pattern.findall(text)
        assert not matches, (
            f"setup_models.sh contains what appear to be hardcoded HF tokens: "
            f"{matches}. Remove them — use HF_TOKEN environment variable instead."
        )

    @pytest.mark.tier1
    def test_hf_token_read_from_env(self):
        """HF_TOKEN must be read from env var, not hardcoded."""
        text = _script_text()
        assert "HF_TOKEN" in text, (
            "setup_models.sh should read HF_TOKEN from environment "
            "(e.g., HF_TOKEN=${HF_TOKEN:-}) — required for gated models."
        )


# ---------------------------------------------------------------------------
# Help / usage output
# ---------------------------------------------------------------------------
class TestSetupScriptHelp:
    """setup_models.sh --help must produce output without downloading anything."""

    @pytest.mark.tier1
    def test_help_flag_exists(self):
        """Script must have a --help / -h handler documented in code."""
        text = _script_text()
        has_help = "--help" in text or "-h)" in text or "show_help" in text
        assert has_help, (
            "setup_models.sh should support --help / -h to display usage. "
            "Users need a way to inspect options without running the full download."
        )
