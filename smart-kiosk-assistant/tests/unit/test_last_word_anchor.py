"""Regression tests for the voice-to-voice anchor (`_t_last_speech_frame`).

The voice-to-voice clock starts at the customer's LAST word. The instant is
stamped in ``_process_frame_stream`` every time a frame is classified as
speech, so only the most recent one survives when the endpoint fires.

That stamp used to live inside the ``if self._streaming_active:`` branch for
every frame except the very first one. Continuous streaming
(``KIOSK_CORE_ANALYZER_STREAMING_ENABLED``) is off on the default path, so in
the shipped configuration nothing after the first speech frame ever updated
it: every turn was anchored on the customer's FIRST word and
``voice_to_voice_ms`` was inflated by the entire length of the utterance.

These tests pin the anchor to the last speech frame with streaming OFF.
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

from kiosk_core.audio_session import BaseAudioSession  # noqa: E402

SAMPLE_RATE = 16000
FRAME_DURATION_SECONDS = 0.02
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_SECONDS)
SEED_THRESHOLD = 0.02
SILENCE_TIMEOUT_SECONDS = 0.2


def _speech_frame() -> np.ndarray:
    """A frame loud enough to clear the RMS VAD threshold."""
    return np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)


def _silent_frame() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.float32)


def _make_session() -> BaseAudioSession:
    """Minimal harness around _process_frame_stream, streaming OFF.

    Mirrors tests/unit/test_skip_empty_final_flush.py: __init__ is bypassed
    via __new__ because it would start real threads and open real clients.
    """
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = "test-session"
    session.agent_session_id = "test-conversation"
    session.request = types.SimpleNamespace(
        chunk_seconds=10.0,
        adaptive_flush_pause_seconds=10.0,  # never adaptive-flush in these tests
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
    # The whole point: continuous streaming is OFF, as it is by default.
    session.realtime_client = None
    session._streaming_active = False
    session._unconfirmed_speech_pending = False
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
    """A fake monotonic clock that only advances when a frame is consumed.

    Every ``time.monotonic()`` call inside the frame loop returns the instant
    of the frame currently being processed, so an assertion can name the exact
    expected anchor value (``frame_index * FRAME_DURATION_SECONDS``) instead of
    an inequality that a buggy implementation can still satisfy by accident.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def feed(self, frames: list[np.ndarray]):
        for index, frame in enumerate(frames):
            self.now = index * FRAME_DURATION_SECONDS
            yield frame


class TestLastWordAnchorWithStreamingOff:
    def test_anchor_tracks_the_last_speech_frame_not_the_first(self, monkeypatch):
        """A long utterance must anchor on its end, not its start.

        Without the fix the stamp is written once (first speech frame) and
        never again with streaming off, so the recorded anchor sits a whole
        utterance earlier than the truth -- 0.0s instead of 0.08s here, which
        is exactly the inflation the reviewer measured in ``drift_ms``.
        """
        session = _make_session()
        drain = _drain_in_background(session)
        clock = _FrameClock()
        monkeypatch.setattr("kiosk_core.audio_session.time.monotonic", clock)

        speech_frames = 5
        frames = [_speech_frame() for _ in range(speech_frames)]
        frames += [_silent_frame() for _ in range(40)]
        session._process_frame_stream(clock.feed(frames))
        session._flush_queue.put(None)
        drain.join(timeout=5)

        expected = (speech_frames - 1) * FRAME_DURATION_SECONDS
        assert session._t_last_speech_frame == pytest.approx(expected), (
            f"anchor is {session._t_last_speech_frame}, expected the LAST speech "
            f"frame at {expected}s -- a value of 0.0 means the stamp is still "
            "gated on continuous streaming being active"
        )

    def test_anchor_is_stamped_at_all_with_streaming_off(self, monkeypatch):
        """Sanity: a streaming-off session still produces an observed anchor."""
        session = _make_session()
        drain = _drain_in_background(session)
        clock = _FrameClock()
        monkeypatch.setattr("kiosk_core.audio_session.time.monotonic", clock)

        frames = [_speech_frame() for _ in range(3)] + [_silent_frame() for _ in range(40)]
        session._process_frame_stream(clock.feed(frames))
        session._flush_queue.put(None)
        drain.join(timeout=5)

        assert session._t_last_speech_frame is not None
        # _log_last_word_spoken prefers the observed stamp over the derived one.
        assert session._t_last_word == session._t_last_speech_frame

    def test_turn_end_decision_is_stamped_on_the_silence_path(self, monkeypatch):
        """endpointing_delay_ms / processing_latency_ms need this stamp."""
        session = _make_session()
        drain = _drain_in_background(session)
        clock = _FrameClock()
        monkeypatch.setattr("kiosk_core.audio_session.time.monotonic", clock)

        frames = [_speech_frame() for _ in range(3)] + [_silent_frame() for _ in range(40)]
        session._process_frame_stream(clock.feed(frames))
        session._flush_queue.put(None)
        drain.join(timeout=5)

        assert session._t_endpoint_decision is not None
        assert session._t_last_word is not None
        assert session._t_endpoint_decision >= session._t_last_word, (
            "the turn-end decision cannot predate the customer's last word"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
