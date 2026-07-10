"""
Functional test configuration for Smart Kiosk Assistant.

Tier markers
------------
  @pytest.mark.tier1  — CI-safe: no Docker, no ML models, no audio hardware.
                        Runs on ubuntu-latest with only requirements.txt + libportaudio2.
  @pytest.mark.tier2  — Docker-required: builds and runs the kiosk-core container.
                        Skipped automatically when Docker daemon is unavailable.
  @pytest.mark.tier3  — Full-stack: all 5 services + real ML models.
                        Self-hosted / manual gate only.

Zephyr test run coverage
-------------------------
  ITEP-C5759  "KIOSK 2026.1 Rel"  (24 test cases, NEX-T24243 – NEX-T24266)
"""
import csv
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

# Pre-import pydantic submodules so they are in sys.modules BEFORE any
# patch.dict("sys.modules", ...) block runs.  patch.dict snapshots sys.modules
# on entry and restores it on exit — if pydantic.root_model is first imported
# *inside* a patch.dict block it gets evicted on exit, causing a
# KeyError: 'pydantic.root_model' on the next attempt to create a generic
# submodel of RootModel (e.g. mcp.types.JSONRPCMessage).
import pydantic.root_model  # noqa: F401

import pytest

_CSV_PATH = Path(__file__).resolve().parent / "test_results.csv"
_csv_results: list[dict] = []


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "tier1: CI-safe functional tests — no Docker, no ML models, no audio hardware",
    )
    config.addinivalue_line(
        "markers",
        "tier2: Docker-required tests — build and run kiosk-core container",
    )
    config.addinivalue_line(
        "markers",
        "tier3: Full-stack tests — requires all 5 services with real ML models",
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the smart-kiosk-assistant directory."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="function")
def kiosk_app():
    """
    FastAPI TestClient for kiosk-core with sounddevice mocked.

    sounddevice needs PortAudio at import time; we mock it so that
    Tier 1 tests never require a real audio device or ALSA stack.
    The mock returns one fake input device to satisfy device-listing tests.
    """
    fake_device = {
        "name": "Mock Microphone",
        "max_input_channels": 2,
        "default_samplerate": 16000,
    }

    sd_mock = MagicMock()
    sd_mock.query_devices.return_value = [fake_device]

    with patch.dict("sys.modules", {"sounddevice": sd_mock}):
        # Import inside patch so that audio_session.py picks up the mock
        from fastapi.testclient import TestClient
        import importlib
        import sys

        # Remove any previously cached imports that loaded real sounddevice
        for mod in list(sys.modules.keys()):
            if mod.startswith("kiosk_core") or mod == "main":
                del sys.modules[mod]

        import main  # noqa: PLC0415  — intentional deferred import

        yield TestClient(main.app)

        # Clean up cached modules so other test sessions start fresh
        for mod in list(sys.modules.keys()):
            if mod.startswith("kiosk_core") or mod == "main":
                del sys.modules[mod]


# ---------------------------------------------------------------------------
# CSV result reporter
# ---------------------------------------------------------------------------
def _humanize(nodeid: str) -> str:
    parts = nodeid.split("::")
    parts[0] = parts[0].replace(".py", "").replace("/", " > ").replace("\\", " > ")
    label = " > ".join(parts)
    label = re.sub(r"\btest_", "", label)
    return label.replace("_", " ")


def pytest_runtest_logreport(report):
    if report.skipped:
        if not any(r["nodeid"] == report.nodeid for r in _csv_results):
            _csv_results.append(
                {"nodeid": report.nodeid, "description": _humanize(report.nodeid), "status": "SKIP"}
            )
        return
    if report.when != "call":
        return
    _csv_results.append(
        {
            "nodeid": report.nodeid,
            "description": _humanize(report.nodeid),
            "status": "PASS" if report.passed else "FAIL",
        }
    )


def pytest_sessionfinish(session, exitstatus):
    if not _csv_results:
        return
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["description", "status"])
        writer.writeheader()
        writer.writerows(
            {"description": r["description"], "status": r["status"]} for r in _csv_results
        )
    print(f"\n📄 CSV report → {_CSV_PATH}")
