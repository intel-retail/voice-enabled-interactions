"""Tests for the opener's failure and post-processing edge cases.

A latency optimisation that quietly degrades is worse than one that is off,
because nothing in the logs or the turn trace says it stopped working.

* A failed synthesis used to be cached as ``None`` forever, so one bad TTS
  round-trip -- typically at startup, before text-to-speech is warm --
  disabled the opener for the whole life of the process.
* The opener is published as a plain file copy, so it was the only segment in
  a turn that never received the trim and gain every other segment gets.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kiosk_core import config
from kiosk_core import audio_session as audio_session_module
from kiosk_core.audio_session import BaseAudioSession, _render_opener


@pytest.fixture(autouse=True)
def _isolate_opener_cache(tmp_path, monkeypatch):
    """Give every test an empty opener cache and its own cache directory.

    The cache is module-level and keyed only by (text, model, voice,
    language, instructions), so without this a rendered opener — or a
    remembered failure — would leak between tests.
    """
    monkeypatch.setattr(audio_session_module, "_opener_cache", {})
    monkeypatch.setattr(audio_session_module, "_opener_failed_at", {})
    monkeypatch.setattr(config, "DEFAULT_OPENER_CACHE_DIR", str(tmp_path))
    yield


def _args(tts_client):
    return (tts_client, "One moment.", "kokoro", "am_michael", "en", None)


def _writing_client(payload: bytes = b"RIFF0000WAVEfmt "):
    """A TTS client whose synthesize_to_file actually creates the file."""
    client = MagicMock()

    def _write(text, output_path, **kwargs):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    client.synthesize_to_file.side_effect = _write
    return client


class TestOpenerFailureIsRetried:
    """A transient TTS failure must not disable the opener permanently."""

    def test_failure_is_not_retried_during_the_cooldown(self, monkeypatch):
        monkeypatch.setattr(config, "DEFAULT_OPENER_RETRY_SECONDS", 60.0)
        client = MagicMock()
        client.synthesize_to_file.side_effect = RuntimeError("tts down")

        assert _render_opener(*_args(client)) is None
        assert _render_opener(*_args(client)) is None

        # Second call served from the remembered failure, not a new round-trip.
        assert client.synthesize_to_file.call_count == 1

    def test_failure_is_retried_once_the_cooldown_elapses(self, monkeypatch):
        monkeypatch.setattr(config, "DEFAULT_OPENER_RETRY_SECONDS", 60.0)
        failing = MagicMock()
        failing.synthesize_to_file.side_effect = RuntimeError("tts down")

        assert _render_opener(*_args(failing)) is None

        # Pretend the cooldown has passed.
        key = ("One moment.", "kokoro", "am_michael", "en", None)
        audio_session_module._opener_failed_at[key] -= 61.0

        recovered = _writing_client()
        path = _render_opener(*_args(recovered))

        assert path is not None and path.exists()
        assert recovered.synthesize_to_file.call_count == 1

    def test_a_successful_render_clears_the_remembered_failure(self, monkeypatch):
        monkeypatch.setattr(config, "DEFAULT_OPENER_RETRY_SECONDS", 0.0)
        failing = MagicMock()
        failing.synthesize_to_file.side_effect = RuntimeError("tts down")
        assert _render_opener(*_args(failing)) is None

        recovered = _writing_client()
        assert _render_opener(*_args(recovered)) is not None

        key = ("One moment.", "kokoro", "am_michael", "en", None)
        assert key not in audio_session_module._opener_failed_at


class TestOpenerPostProcessing:
    """The opener must get the same trim and gain as every other segment."""

    def test_postprocess_runs_on_a_freshly_synthesised_opener(self):
        client = _writing_client()
        seen: list[Path] = []

        path = _render_opener(*_args(client), postprocess=seen.append)

        assert path is not None
        assert seen == [path]

    def test_postprocess_does_not_run_again_on_the_cached_file(self):
        """Re-applying gain to an already-normalised file would clip it."""
        client = _writing_client()
        seen: list[Path] = []

        first = _render_opener(*_args(client), postprocess=seen.append)
        # Drop the in-memory cache only: the WAV stays on disk, which is the
        # state a process restart leaves behind.
        audio_session_module._opener_cache.clear()
        second = _render_opener(*_args(client), postprocess=seen.append)

        assert first == second
        assert len(seen) == 1
        assert client.synthesize_to_file.call_count == 1

    def test_postprocess_failure_is_treated_as_a_failed_render(self):
        """A half-processed opener must never be published."""
        client = _writing_client()

        def _explode(path: Path) -> None:
            raise RuntimeError("ffmpeg missing")

        assert _render_opener(*_args(client), postprocess=_explode) is None

    def test_a_failed_render_leaves_nothing_on_disk(self, tmp_path):
        """Otherwise the next process serves the half-processed file."""
        client = _writing_client()

        def _explode(path: Path) -> None:
            raise RuntimeError("ffmpeg missing")

        _render_opener(*_args(client), postprocess=_explode)

        assert list(tmp_path.glob("opener_*.wav")) == []

    def test_session_postprocess_applies_trim_then_gain(self, monkeypatch):
        """``_postprocess_opener`` is the hook that wires both steps in."""
        session = BaseAudioSession.__new__(BaseAudioSession)
        calls: list[str] = []
        monkeypatch.setattr(
            BaseAudioSession,
            "_trim_tts_segment",
            lambda self, path, text: calls.append("trim"),
        )
        monkeypatch.setattr(
            BaseAudioSession,
            "_apply_tts_gain",
            lambda self, path: calls.append("gain"),
        )

        session._postprocess_opener(Path("opener.wav"))

        assert calls == ["trim", "gain"]


