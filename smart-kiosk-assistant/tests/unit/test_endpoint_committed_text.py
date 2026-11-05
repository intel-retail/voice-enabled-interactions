"""The endpoint shortcut reading the adaptive-pause commit (streaming mode).

In streaming mode the adaptive-pause flush commits the utterance, which
clears the preview. The completeness check in ``_process_frame_stream`` then
reads ``RealtimeAnalyzerClient.committed_text_since(<last speech frame>)``
instead, and that text skips the stability window because a commit that
covers the last speech frame will not change.

These tests pin the session side of that:

* the committed text is only read after the adaptive-pause flush, and with
  the last speech frame as the reference
* committed text skips the stability window
* no committed text means no early commit (fail closed)
* a non-empty preview takes precedence over the committed text

The harness mirrors tests/unit/test_last_word_anchor.py, with a fake
realtime client so streaming is on.
"""

from __future__ import annotations

import sys
import threading
import types
from collections import deque
from pathlib import Path
from queue import Queue

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kiosk_core import config  # noqa: E402
from kiosk_core.audio_session import BaseAudioSession  # noqa: E402

SAMPLE_RATE = 16000
FRAME_DURATION_SECONDS = 0.02
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_SECONDS)
SEED_THRESHOLD = 0.02
SILENCE_TIMEOUT_SECONDS = 1.0
SHORT_SECONDS = 0.2
SPEECH_FRAMES = 30  # 0.6 s: enough audio for the adaptive flush's 0.5 s minimum


class _FakeRealtimeClient:
    """Streaming is live, previews are empty, the commit text is scripted."""

    def __init__(self, committed_text: str, preview_text: str = ""):
        self.committed_text = committed_text
        self.preview_text = preview_text
        self.committed_text_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return True

    def send_frame(self, pcm_bytes: bytes) -> None:
        pass

    def preview(self) -> None:
        pass

    def latest_preview_text(self) -> str:
        return self.preview_text

    def committed_text_since(self, reference_ts):
        self.committed_text_calls.append(reference_ts)
        return self.committed_text


def _speech_frame() -> np.ndarray:
    return np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)


def _silent_frame() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.float32)


def _make_session(fake_client: _FakeRealtimeClient, adaptive_pause: float) -> BaseAudioSession:
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = "test-session"
    session.agent_session_id = "test-conversation"
    session.request = types.SimpleNamespace(
        chunk_seconds=10.0,
        adaptive_flush_pause_seconds=adaptive_pause,
        silence_timeout_seconds=SILENCE_TIMEOUT_SECONDS,
        sample_rate=SAMPLE_RATE,
        max_session_seconds=30.0,
        silence_threshold=SEED_THRESHOLD,
    )
    session._stop_event = threading.Event()
    session._vad_threshold = float(SEED_THRESHOLD)
    session._noise_floor = None
    session._vad_calibrating = False
    session._vad_calibration_rms = []
    session._silero_vad = None
    session._speech_started = False
    session._t_capture_start = None
    session._preroll_frames = deque(maxlen=1)
    session._captured_samples = 0
    session._chunk_has_speech = False
    session._final_flush_skipped = False
    session._frame_duration_seconds = FRAME_DURATION_SECONDS
    session._lock = threading.Lock()
    session.realtime_client = fake_client
    session._streaming_active = True
    session._last_preview_ping_at = 0.0
    session._unconfirmed_speech_pending = False
    session._endpoint_stable_transcript = None
    session._endpoint_stable_since = None
    session.transcript_parts = []
    session.end_reason = None
    session._endpoint_wait_seconds = None
    session._endpoint_shortcut_fired = None
    session._t_last_speech_frame = None
    session._t_endpoint_decision = None
    session._t_last_word = None
    session._t_final_flush_start = None
    session._flush_queue = Queue()
    return session


def _drain_in_background(session: BaseAudioSession) -> threading.Thread:
    def _drain() -> None:
        while True:
            item = session._flush_queue.get()
            session._flush_queue.task_done()
            if item is None:
                return

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    return thread


class _FrameClock:
    """A fake monotonic clock that advances one frame per consumed frame."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def feed(self, frames):
        for index, frame in enumerate(frames):
            self.now = index * FRAME_DURATION_SECONDS
            yield frame


def _run(session: BaseAudioSession, monkeypatch) -> None:
    # The stability window is longer than the whole silence timeout here, so
    # only the committed-text path can end the turn before the timeout.
    monkeypatch.setattr(config, "DEFAULT_ENDPOINT_COMPLETE_ENABLED", True)
    monkeypatch.setattr(config, "DEFAULT_ENDPOINT_SHORT_SECONDS", SHORT_SECONDS)
    monkeypatch.setattr(config, "DEFAULT_ENDPOINT_MIN_WORDS", 3)
    monkeypatch.setattr(config, "DEFAULT_ENDPOINT_STABLE_SECONDS", 5.0)
    clock = _FrameClock()
    monkeypatch.setattr("kiosk_core.audio_session.time.monotonic", clock)
    drain = _drain_in_background(session)
    frames = [_speech_frame() for _ in range(SPEECH_FRAMES)]
    frames += [_silent_frame() for _ in range(int(2 * SILENCE_TIMEOUT_SECONDS / FRAME_DURATION_SECONDS))]
    session._process_frame_stream(clock.feed(frames))
    session._flush_queue.put(None)
    drain.join(timeout=5)


class TestCommittedTextEndsTheTurn:
    def test_committed_text_skips_the_stability_window(self, monkeypatch):
        fake = _FakeRealtimeClient("Please add one chocolate brownie to my order.")
        session = _make_session(fake, adaptive_pause=0.1)
        _run(session, monkeypatch)

        assert session._endpoint_shortcut_fired is True
        assert session._endpoint_wait_seconds == pytest.approx(SHORT_SECONDS, abs=FRAME_DURATION_SECONDS)

    def test_reference_is_the_last_speech_frame(self, monkeypatch):
        fake = _FakeRealtimeClient("Please add one chocolate brownie to my order.")
        session = _make_session(fake, adaptive_pause=0.1)
        _run(session, monkeypatch)

        expected = (SPEECH_FRAMES - 1) * FRAME_DURATION_SECONDS
        assert fake.committed_text_calls
        assert all(ts == pytest.approx(expected) for ts in fake.committed_text_calls)


class TestCommittedTextGate:
    def test_not_read_before_the_adaptive_flush(self, monkeypatch):
        # No adaptive flush this turn (pause longer than the timeout): the
        # last commit cannot cover this utterance, so it must not be read.
        fake = _FakeRealtimeClient("Please add one chocolate brownie to my order.")
        session = _make_session(fake, adaptive_pause=10.0)
        _run(session, monkeypatch)

        assert fake.committed_text_calls == []
        assert session._endpoint_shortcut_fired is False

    def test_no_committed_text_keeps_the_full_timeout(self, monkeypatch):
        fake = _FakeRealtimeClient("")
        session = _make_session(fake, adaptive_pause=0.1)
        _run(session, monkeypatch)

        assert fake.committed_text_calls  # it was asked, and had nothing
        assert session._endpoint_shortcut_fired is False
        assert session._endpoint_wait_seconds == pytest.approx(SILENCE_TIMEOUT_SECONDS, abs=FRAME_DURATION_SECONDS)

    def test_preview_takes_precedence_over_committed_text(self, monkeypatch):
        # Audio after the adaptive flush shows up in the preview. The commit
        # is older than that audio, so it must not be read.
        fake = _FakeRealtimeClient(
            "Please add one chocolate brownie to my order.", preview_text="and a coffee"
        )
        session = _make_session(fake, adaptive_pause=0.1)
        _run(session, monkeypatch)

        assert fake.committed_text_calls == []
        assert session._endpoint_shortcut_fired is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
