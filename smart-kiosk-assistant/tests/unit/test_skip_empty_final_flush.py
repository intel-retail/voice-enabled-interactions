"""Unit tests for skipping the empty final tail-chunk ASR flush.

Tier 2 roadmap item #4 ("cumulative-snapshot ASR"), adapted to this
codebase's actual bottleneck: by endpoint, the adaptive-pause flush has
almost always already sent every real word, so the final "is_final=True"
tail chunk enqueued in ``_process_frame_stream`` is typically pure trailing
silence. See ``config.DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED`` for the full
rationale.

These tests drive ``BaseAudioSession._process_frame_stream`` directly with a
synthetic frame stream (no audio hardware, no network) and inspect what gets
enqueued on ``_flush_queue``. A background thread stands in for the real
ASR flush worker, immediately marking every item done, so
``_flush_queue.join()`` at the end of the method does not block.
"""
import sys
import threading
import types
from collections import deque
from queue import Queue
from unittest.mock import MagicMock

import numpy as np
import pytest

# `kiosk_core.audio_session` imports sounddevice, which needs PortAudio at
# import time — unavailable outside the container. Mirrors the mocking done
# in tests/functional/conftest.py. Must run before the kiosk_core import.
sys.modules.setdefault("sounddevice", MagicMock())

from kiosk_core import config  # noqa: E402
from kiosk_core.audio_session import BaseAudioSession  # noqa: E402

FRAME_DURATION_SECONDS = 0.1
ADAPTIVE_FLUSH_PAUSE_SECONDS = 0.3
SILENCE_TIMEOUT_SECONDS = 0.6
SPEECH_RMS = 5000.0
SILENCE_RMS = 0.0
SEED_THRESHOLD = 900


def _make_session() -> BaseAudioSession:
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = "test-session"
    session.agent_session_id = "test-conversation"
    session.request = types.SimpleNamespace(
        chunk_seconds=10.0,
        adaptive_flush_pause_seconds=ADAPTIVE_FLUSH_PAUSE_SECONDS,
        silence_timeout_seconds=SILENCE_TIMEOUT_SECONDS,
        sample_rate=16000,
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
    # _process_frame_stream also touches these -- __init__ normally sets
    # them (kiosk_core/audio_session.py, BaseAudioSession.__init__), but
    # this harness bypasses __init__ via __new__(). Continuous-streaming is
    # off in these tests (no realtime_client), so _streaming_active must
    # resolve to False without raising through its is_alive() check.
    session._lock = threading.Lock()
    session.realtime_client = None
    session._streaming_active = False
    session._unconfirmed_speech_pending = False
    session.transcript_parts = []
    session.end_reason = None
    session._endpoint_wait_seconds = None
    session._flush_queue = Queue()
    return session


def _drain_flush_queue_in_background(session: BaseAudioSession) -> list:
    """Stand in for _flush_worker: record each item, then task_done().

    Also mirrors _flush_worker's real post-ASR-success side effect of
    clearing _unconfirmed_speech_pending (kiosk_core/audio_session.py, the
    `self._unconfirmed_speech_pending = False` line right after a chunk's
    transcript is appended) -- without it, a chunk carrying real speech is
    enqueued+drained here but never "confirmed", so the final-flush skip
    logic (which requires it False) can never fire in this synchronous fake,
    even though the real worker would have cleared it.
    """
    items = []

    def _drain():
        while True:
            item = session._flush_queue.get()
            items.append(item)
            if item is not None:
                frames, _is_final = item
                if any(np.any(f) for f in frames):
                    session._unconfirmed_speech_pending = False
            session._flush_queue.task_done()
            if item is None:
                break

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    return items, thread


def _frame(rms: float) -> np.ndarray:
    samples = int(16000 * FRAME_DURATION_SECONDS)
    return np.full(samples, rms, dtype=np.float32)


def _run(session: BaseAudioSession) -> list:
    items, thread = _drain_flush_queue_in_background(session)
    frames = (
        [_frame(SPEECH_RMS)] * 6  # 0.6s of speech — clears the adaptive
        # flush's 0.5s minimum chunk-duration guard.
        + [_frame(SILENCE_RMS)] * 3  # 0.3s silence -> triggers adaptive flush
        + [_frame(SILENCE_RMS)] * 3  # another 0.3s silence -> endpoint fires
    )
    session._process_frame_stream(iter(frames))
    thread.join(timeout=2)
    return items


class TestSkipEmptyFinalFlush:
    @pytest.fixture(autouse=True)
    def _isolate_from_endpoint_completeness(self, monkeypatch):
        # See the identical fixture in test_asr_trim_trailing_silence.py:
        # these tests use short silence_timeout_seconds values that can
        # overlap config.DEFAULT_ENDPOINT_SHORT_SECONDS's default and let the
        # unrelated completeness shortcut fire, changing frame-count timing.
        monkeypatch.setattr(config, "DEFAULT_ENDPOINT_COMPLETE_ENABLED", False)
        # Same reasoning for config.DEFAULT_PREVIEW_FLUSH_INTERVAL_SECONDS
        # (0.4s default): it can now fire mid-speech within these tests'
        # short continuous-speech runs, splitting one chunk into two.
        monkeypatch.setattr(config, "DEFAULT_PREVIEW_FLUSH_ENABLED", False)

    def test_disabled_still_enqueues_silent_final_flush(self, monkeypatch):
        """Default behavior (flag off): the final tail chunk is always
        enqueued, even when it is pure trailing silence."""
        monkeypatch.setattr(config, "DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED", False)
        session = _make_session()

        items = _run(session)

        final_items = [item for item in items if item is not None and item[1] is True]
        assert len(final_items) == 1, "expected exactly one is_final=True flush"
        assert session._final_flush_skipped is False

    def test_enabled_skips_silent_final_flush(self, monkeypatch):
        """Flag on: no unflushed speech since the last flush -> the final
        tail chunk is never enqueued."""
        monkeypatch.setattr(config, "DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED", True)
        session = _make_session()

        items = _run(session)

        final_items = [item for item in items if item is not None and item[1] is True]
        assert final_items == [], "silent final tail chunk should have been skipped"
        assert session._final_flush_skipped is True

    def test_enabled_still_flushes_genuine_trailing_speech(self, monkeypatch):
        """Flag on, but real trailing speech arrived after the adaptive
        flush (too brief to itself clear the adaptive flush's 0.5s minimum
        chunk-duration guard before the endpoint fires) -> the final flush
        must still run so that speech is not silently dropped.

        Uses a tighter silence_timeout than the other tests specifically so
        the second adaptive-flush opportunity cannot land before endpoint —
        this is the scenario the "no unflushed speech" guard exists for.
        """
        monkeypatch.setattr(config, "DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED", True)
        session = _make_session()
        session.request.silence_timeout_seconds = 0.4
        items, thread = _drain_flush_queue_in_background(session)
        frames = (
            [_frame(SPEECH_RMS)] * 6
            + [_frame(SILENCE_RMS)] * 3  # adaptive flush fires here (0.3s)
            + [_frame(SPEECH_RMS)] * 1  # short trailing speech: too brief to
            # itself clear the adaptive flush's 0.5s duration guard again
            + [_frame(SILENCE_RMS)] * 4  # reaches the 0.4s endpoint timeout
            # before a second adaptive flush's duration guard can pass
        )
        session._process_frame_stream(iter(frames))
        thread.join(timeout=2)

        final_items = [item for item in items if item is not None and item[1] is True]
        assert len(final_items) == 1, "trailing speech must still get a final flush"
        assert session._final_flush_skipped is False

    def test_no_speech_at_all_never_enqueues_anything(self, monkeypatch):
        """No speech detected in the whole turn: nothing should be flushed,
        flag on or off, and the skip counter must not fire (there was never
        a turn to skip)."""
        for flag in (False, True):
            monkeypatch.setattr(config, "DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED", flag)
            session = _make_session()
            items, thread = _drain_flush_queue_in_background(session)
            frames = [_frame(SILENCE_RMS)] * 10
            session._process_frame_stream(iter(frames))
            thread.join(timeout=2)

            real_items = [item for item in items if item is not None]
            assert real_items == []
            assert session._final_flush_skipped is False
