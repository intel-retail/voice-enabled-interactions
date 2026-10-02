"""Unit tests for ASR trailing-silence trim (kiosk-voice-lab-main parity).

Whisper hallucinates a sentence-completing word/phrase when fed audio that
ends in "dangling speech + silence" — measured artifact: a spurious
trailing "Good." after a real order line, which breaks EXACT-STRING
speculative-draft matching (the final transcript rarely matches byte-for-byte
any earlier preview snapshot). kiosk-voice-lab-main avoids this by trimming
the audio actually SENT TO ASR down to "last speech sample + a short decay
tail" instead of sending everything accumulated since speech stopped. See
``config.DEFAULT_ASR_TRIM_TRAILING_SILENCE_ENABLED``/``BaseAudioSession.
_trim_trailing_silence`` for the full rationale.

These tests drive ``BaseAudioSession._process_frame_stream`` directly with a
synthetic frame stream (no audio hardware, no network) and inspect exactly
what gets enqueued on ``_flush_queue`` — same harness pattern as
tests/unit/test_skip_empty_final_flush.py.
"""
import sys
import threading
import types
from collections import deque
from queue import Queue
from unittest.mock import MagicMock

import pytest

import numpy as np

# `kiosk_core.audio_session` imports sounddevice, which needs PortAudio at
# import time — unavailable outside the container. Mirrors the mocking done
# in tests/functional/conftest.py. Must run before the kiosk_core import.
sys.modules.setdefault("sounddevice", MagicMock())

from kiosk_core import config  # noqa: E402
from kiosk_core.audio_session import BaseAudioSession  # noqa: E402

FRAME_DURATION_SECONDS = 0.1
ADAPTIVE_FLUSH_PAUSE_SECONDS = 0.3
SILENCE_TIMEOUT_SECONDS = 0.6
SAMPLE_RATE = 16000
SPEECH_RMS = 5000.0
SILENCE_RMS = 0.0
SEED_THRESHOLD = 900
DECAY_SECONDS = 0.15


def _make_session() -> BaseAudioSession:
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = "test-session"
    session.agent_session_id = "test-conversation"
    session.request = types.SimpleNamespace(
        chunk_seconds=10.0,
        adaptive_flush_pause_seconds=ADAPTIVE_FLUSH_PAUSE_SECONDS,
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
    # _process_frame_stream also touches these -- __init__ normally sets
    # them (kiosk_core/audio_session.py, BaseAudioSession.__init__), but
    # this harness bypasses __init__ via __new__(). Continuous-streaming is
    # off in these tests (no realtime_client), so _streaming_active must
    # resolve to False without raising through its is_alive() check.
    session._lock = threading.Lock()
    session.realtime_client = None
    session._streaming_active = False
    session.transcript_parts = []
    session.end_reason = None
    session._endpoint_wait_seconds = None
    session._flush_queue = Queue()
    return session


def _drain_flush_queue_in_background(session: BaseAudioSession) -> tuple:
    """Stand in for _flush_worker: record each item, then task_done()."""
    items = []

    def _drain():
        while True:
            item = session._flush_queue.get()
            items.append(item)
            session._flush_queue.task_done()
            if item is None:
                break

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    return items, thread


def _frame(rms: float) -> np.ndarray:
    samples = int(SAMPLE_RATE * FRAME_DURATION_SECONDS)
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


class TestAsrTrimTrailingSilence:
    @pytest.fixture(autouse=True)
    def _isolate_from_endpoint_completeness(self, monkeypatch):
        # These tests drive silence_timeout_seconds down to 0.4-0.6s to keep
        # frame counts small, which is short enough to overlap
        # config.DEFAULT_ENDPOINT_SHORT_SECONDS (0.15s default) and let the
        # unrelated sentence-completeness shortcut fire mid-scenario, changing
        # exactly when the endpoint commits. That feature is exercised in
        # test_endpoint_completeness.py; disable it here so trim/flush timing
        # stays deterministic regardless of its default value.
        monkeypatch.setattr(config, "DEFAULT_ENDPOINT_COMPLETE_ENABLED", False)
        # Same reasoning for the preview flush: config.DEFAULT_PREVIEW_
        # FLUSH_INTERVAL_SECONDS (0.4s default) can now fire mid-speech within
        # these tests' short (0.6s) continuous-speech runs, splitting one
        # adaptive-pause chunk into two. That mechanism is exercised
        # elsewhere; disable it here too so these tests only see the
        # adaptive-pause flush they are named for.
        monkeypatch.setattr(config, "DEFAULT_PREVIEW_FLUSH_ENABLED", False)

    def test_disabled_sends_full_untrimmed_buffer(self, monkeypatch):
        """Default behavior (flag off): the adaptive-pause flush sends every
        accumulated frame, trailing silence included."""
        monkeypatch.setattr(config, "DEFAULT_ASR_TRIM_TRAILING_SILENCE_ENABLED", False)
        session = _make_session()

        items = _run(session)

        adaptive_items = [item for item in items if item is not None and item[1] is False]
        assert len(adaptive_items) == 1
        frames, _is_final = adaptive_items[0]
        # 6 speech + 3 silence frames = 9 frames, none dropped.
        assert len(frames) == 9

    def test_enabled_trims_trailing_silence_to_decay_tail(self, monkeypatch):
        """Flag on: the adaptive-pause flush keeps only DECAY_SECONDS worth
        of trailing silence, dropping the rest before ASR ever sees it."""
        monkeypatch.setattr(config, "DEFAULT_ASR_TRIM_TRAILING_SILENCE_ENABLED", True)
        monkeypatch.setattr(config, "DEFAULT_ASR_TRIM_DECAY_SECONDS", DECAY_SECONDS)
        session = _make_session()

        items = _run(session)

        adaptive_items = [item for item in items if item is not None and item[1] is False]
        assert len(adaptive_items) == 1
        frames, _is_final = adaptive_items[0]
        # 6 speech frames kept in full, plus a 0.15s decay tail out of the
        # 0.3s of accumulated trailing silence -> 1-2 silence frames kept
        # (0.15s / 0.1s per frame, rounded), the other 1-2 dropped.
        keep_silence_frames = round(DECAY_SECONDS / FRAME_DURATION_SECONDS)
        assert len(frames) == 6 + keep_silence_frames
        assert len(frames) < 9, "trim must actually shrink the buffer vs. the untrimmed case"

    def test_enabled_does_not_touch_endpoint_timing(self, monkeypatch):
        """Trimming only changes what is SENT TO ASR — silence_run_seconds
        and the endpoint/silence-timeout decision must fire identically to
        the untrimmed case."""
        monkeypatch.setattr(config, "DEFAULT_ASR_TRIM_TRAILING_SILENCE_ENABLED", True)
        monkeypatch.setattr(config, "DEFAULT_ASR_TRIM_DECAY_SECONDS", DECAY_SECONDS)
        trimmed_session = _make_session()
        _run(trimmed_session)

        monkeypatch.setattr(config, "DEFAULT_ASR_TRIM_TRAILING_SILENCE_ENABLED", False)
        untrimmed_session = _make_session()
        _run(untrimmed_session)

        assert trimmed_session._endpoint_wait_seconds == untrimmed_session._endpoint_wait_seconds
        assert trimmed_session.end_reason == untrimmed_session.end_reason

    def test_no_trailing_silence_yet_is_a_no_op(self, monkeypatch):
        """Mid-speech (silence_run_seconds within the decay tail already):
        nothing to trim, buffer passed through unchanged."""
        monkeypatch.setattr(config, "DEFAULT_ASR_TRIM_TRAILING_SILENCE_ENABLED", True)
        monkeypatch.setattr(config, "DEFAULT_ASR_TRIM_DECAY_SECONDS", DECAY_SECONDS)
        session = _make_session()

        frames = [_frame(SPEECH_RMS)] * 3
        result = session._trim_trailing_silence(frames, silence_run_seconds=0.0)

        assert result is frames or result == frames
        assert len(result) == 3
