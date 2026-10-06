"""Tests that the turn trace names the voice-activity detector that ran.

Silero is the default but falls back to the rate-agnostic RMS VAD silently
whenever the ONNX model or onnxruntime is unavailable, or the session's
sample rate is one Silero does not support. That fallback changes endpointing
behaviour, so a trace that does not name the detector is not comparable
between runs -- CI's Tier 1 tests, for instance, run before the model
download and therefore exercise RMS.
"""
from unittest.mock import MagicMock

from kiosk_core import config
from kiosk_core.audio_session import BaseAudioSession


class TestVadBackendIsRecorded:
    """The turn trace must name the detector that actually ran."""

    def test_wall_times_defaults_to_unknown(self):
        from kiosk_core.pipeline_latency import WallTimes

        assert WallTimes().vad_backend is None

    def test_fallback_to_rms_is_recorded(self, monkeypatch):
        """A missing model file must leave "rms" in the trace, not "silero"."""
        monkeypatch.setattr(config, "KIOSK_CORE_SILERO_VAD_ENABLED", True)
        monkeypatch.setattr(
            config, "DEFAULT_SILERO_VAD_MODEL_PATH", "/nonexistent/silero.onnx"
        )
        session = _bare_session()

        session._init_vad_backend()

        assert session._vad_backend == "rms"
        assert session._silero_vad is None

    def test_unsupported_sample_rate_falls_back_to_rms(self, monkeypatch):
        """24kHz browser audio is the real-world case for this branch."""
        monkeypatch.setattr(config, "KIOSK_CORE_SILERO_VAD_ENABLED", True)

        class _RateRejectingVAD:
            def __init__(self, *args, **kwargs):
                raise ValueError("unsupported sample rate 24000")

        import kiosk_core.silero_vad as silero_module

        monkeypatch.setattr(silero_module, "SileroVAD", _RateRejectingVAD)
        session = _bare_session(sample_rate=24000)

        session._init_vad_backend()

        assert session._vad_backend == "rms"
        assert session._silero_vad is None

    def test_disabled_silero_is_recorded_as_rms(self, monkeypatch):
        monkeypatch.setattr(config, "KIOSK_CORE_SILERO_VAD_ENABLED", False)
        session = _bare_session()

        session._init_vad_backend()

        assert session._vad_backend == "rms"

    def test_successful_init_is_recorded_as_silero(self, monkeypatch):
        monkeypatch.setattr(config, "KIOSK_CORE_SILERO_VAD_ENABLED", True)

        import kiosk_core.silero_vad as silero_module

        monkeypatch.setattr(silero_module, "SileroVAD", lambda *a, **k: object())
        session = _bare_session()

        session._init_vad_backend()

        assert session._vad_backend == "silero"
        assert session._silero_vad is not None


def _bare_session(sample_rate: int = 16000) -> BaseAudioSession:
    """A session carrying only what ``_init_vad_backend`` reads.

    ``BaseAudioSession.__init__`` needs an audio device and a full client
    set, none of which the detector choice depends on.
    """
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = "test-session"
    session.request = MagicMock(sample_rate=sample_rate)
    return session
