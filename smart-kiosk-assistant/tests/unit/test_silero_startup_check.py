"""kiosk-core refuses to start when Silero VAD is enabled and its model won't load.

The model file is not in git; setup_models.sh downloads it and the image bakes
it in. Sessions used to fall back to the RMS VAD on their own when it was
missing, so a kiosk built without it ran degraded and nothing failed. The
startup check turns that into one error before the service accepts traffic.

The per-session fallback for unsupported sample rates is separate and stays;
see test_vad_backend_trace.py.
"""
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")

from kiosk_core.silero_vad import (  # noqa: E402
    SileroModelUnavailableError,
    require_silero_vad_model,
)

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "kiosk_core" / "models" / "silero_vad.onnx"


class TestRequireSileroVadModel:
    def test_enabled_and_missing_refuses(self, tmp_path):
        missing = tmp_path / "silero_vad.onnx"

        with pytest.raises(SileroModelUnavailableError) as excinfo:
            require_silero_vad_model(True, missing)

        message = str(excinfo.value)
        assert "does not exist" in message
        assert str(missing) in message
        assert "./setup_models.sh" in message
        assert "KIOSK_CORE_SILERO_VAD_ENABLED=false" in message

    def test_enabled_and_unloadable_refuses(self, tmp_path):
        """A truncated or wrong file (e.g. a proxy's HTML page) must not pass."""
        broken = tmp_path / "silero_vad.onnx"
        broken.write_bytes(b"<html>proxy login</html>")

        with pytest.raises(SileroModelUnavailableError, match="failed to load"):
            require_silero_vad_model(True, broken)

    def test_enabled_and_present_starts(self):
        assert MODEL_PATH.is_file(), f"run ./setup_models.sh --silero-only first ({MODEL_PATH})"

        require_silero_vad_model(True, MODEL_PATH)

    def test_disabled_and_missing_starts(self, tmp_path):
        """RMS chosen on purpose: no model needed, nothing to check."""
        require_silero_vad_model(False, tmp_path / "silero_vad.onnx")


class TestStartupHook:
    """The check runs in kiosk-core's lifespan, so the app itself won't start."""

    def test_app_startup_fails_when_model_missing(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        import main
        from kiosk_core import config

        monkeypatch.setattr(config, "KIOSK_CORE_SILERO_VAD_ENABLED", True)
        monkeypatch.setattr(
            config, "DEFAULT_SILERO_VAD_MODEL_PATH", str(tmp_path / "silero_vad.onnx")
        )

        with pytest.raises(SileroModelUnavailableError):
            with TestClient(main.app):
                pass
