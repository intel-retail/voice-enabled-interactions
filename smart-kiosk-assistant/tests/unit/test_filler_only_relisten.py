"""Unit tests for re-listening after a filler-only turn.

Bug: the trailing-silence endpoint can commit right after "Uh" while the
customer pauses to think. ASR then returns only "Uh...", _flush_chunk drops
it as filler, and the session used to end with an empty transcript and no
reply. Every later audio chunk for that session got 409 Conflict, so the
rest of the order was lost. See config.DEFAULT_FILLER_RELISTEN_ENABLED.

These tests drive ``BaseAudioSession._process_frame_stream`` with a
synthetic frame stream and a fake flush worker (same harness shape as
test_skip_empty_final_flush.py), plus ``_flush_chunk`` directly against a
fake realtime analyzer client. No audio hardware, no network.
"""
import sys
import threading
import types
from collections import deque
from queue import Queue
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("sounddevice", MagicMock())

from kiosk_core import config  # noqa: E402
from kiosk_core.audio_session import BaseAudioSession  # noqa: E402

FRAME_DURATION_SECONDS = 0.1
ADAPTIVE_FLUSH_PAUSE_SECONDS = 0.3
SILENCE_TIMEOUT_SECONDS = 0.6
SEED_THRESHOLD = 900
FILLER_RMS = 5000.0  # frames the fake ASR hears as "Uh..."
ORDER_RMS = 7000.0  # frames the fake ASR hears as "Two Pepsis."
SILENCE_RMS = 0.0

FILLER = [FILLER_RMS] * 3
ORDER = [ORDER_RMS] * 6
# Long enough for the silence endpoint to fire (0.6 s), not long enough for
# a second endpoint inside the same window.
ENDPOINT_SILENCE = [SILENCE_RMS] * 6


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
        language="en",
        temperature=0.0,
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
    session.realtime_client = None
    session._streaming_active = False
    session._unconfirmed_speech_pending = False
    session.transcript_parts = []
    session.end_reason = None
    session._endpoint_wait_seconds = None
    session._endpoint_shortcut_fired = None
    session._endpoint_stable_transcript = None
    session._endpoint_stable_since = None
    session._t_last_speech_frame = None
    session._t_last_word = None
    session._t_endpoint_decision = None
    session._rejected_speech_chunks = 0
    session._filler_dropped = False
    session._relisten_at = None
    session._stale_snapshot_text = None
    session._flush_queue = Queue()
    return session


def _fake_asr_worker(session: BaseAudioSession, stale_filler: bool = False):
    """Stand in for _flush_worker + _flush_chunk.

    A chunk holding ORDER frames transcribes to "Two Pepsis.". A chunk
    holding FILLER frames transcribes to "Uh...", which the real
    _flush_chunk drops and flags via _filler_dropped. With stale_filler, a
    final flush that carries no order speech returns the old "Uh..." again,
    as a stale realtime snapshot could.
    """
    items = []

    def _drain():
        while True:
            item = session._flush_queue.get()
            items.append(item)
            if item is not None:
                frames, is_final = item
                levels = {float(np.max(f)) for f in frames} if frames else set()
                if ORDER_RMS in levels:
                    session.transcript_parts.append("Two Pepsis.")
                    session._unconfirmed_speech_pending = False
                elif FILLER_RMS in levels or (stale_filler and is_final):
                    session._filler_dropped = True
            session._flush_queue.task_done()
            if item is None:
                break

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    return items, thread


def _frame(rms: float) -> np.ndarray:
    return np.full(int(16000 * FRAME_DURATION_SECONDS), rms, dtype=np.float32)


class _CountingFrames:
    """Iterator over frames that records how many were consumed."""

    def __init__(self, levels):
        self._frames = [_frame(level) for level in levels]
        self.consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.consumed >= len(self._frames):
            raise StopIteration
        frame = self._frames[self.consumed]
        self.consumed += 1
        return frame


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Same isolation as test_skip_empty_final_flush.py: keep the unrelated
    # completeness shortcut and preview flush out of the frame-count timing.
    monkeypatch.setattr(config, "DEFAULT_ENDPOINT_COMPLETE_ENABLED", False)
    monkeypatch.setattr(config, "DEFAULT_PREVIEW_FLUSH_ENABLED", False)
    monkeypatch.setattr(config, "DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED", False)
    monkeypatch.setattr(config, "DEFAULT_FILLER_RELISTEN_ENABLED", True)


def _run(session, levels, stale_filler=False):
    items, thread = _fake_asr_worker(session, stale_filler=stale_filler)
    frames = _CountingFrames(levels)
    result = session._process_frame_stream(frames)
    thread.join(timeout=2)
    assert not thread.is_alive(), "flush worker never got its stop sentinel"
    return result, items, frames


class TestFillerOnlyRelisten:
    def test_words_after_filler_become_the_turn(self):
        """J17 shape: "Uh", a pause long enough to endpoint, then the order.
        The session keeps listening and the order is the transcript."""
        session = _make_session()

        (status, _reason), items, frames = _run(
            session, FILLER + ENDPOINT_SILENCE + ORDER + ENDPOINT_SILENCE
        )

        assert status == "completed"
        assert session.transcript_parts == ["Two Pepsis."]
        assert frames.consumed == len(FILLER + ENDPOINT_SILENCE + ORDER + ENDPOINT_SILENCE)
        assert items.count(None) == 1, "worker must be stopped exactly once"

    def test_resumed_turn_is_timed_from_the_new_words(self):
        """The re-listen clears the filler turn's endpoint stamps, so
        voice-to-voice is measured from the order's last word."""
        session = _make_session()

        _run(session, FILLER + ENDPOINT_SILENCE + ORDER + ENDPOINT_SILENCE)

        assert session._relisten_at is not None
        assert session._t_last_word is not None
        assert session._t_last_word > session._relisten_at
        assert session._t_endpoint_decision > session._relisten_at

    def test_silence_after_filler_still_ends_the_session(self):
        """No new speech after the filler: the session ends at the next
        silence endpoint even if ASR keeps handing back the stale filler,
        rather than re-listening until max_session_seconds."""
        session = _make_session()
        levels = FILLER + ENDPOINT_SILENCE + ENDPOINT_SILENCE + [SILENCE_RMS] * 20

        (status, _reason), items, frames = _run(session, levels, stale_filler=True)

        assert status == "completed"
        assert session.transcript_parts == []
        assert frames.consumed == len(FILLER + ENDPOINT_SILENCE + ENDPOINT_SILENCE)
        assert items.count(None) == 1

    def test_max_duration_still_ends_a_resumed_turn(self):
        session = _make_session()
        session.request.max_session_seconds = 1.5
        levels = FILLER + ENDPOINT_SILENCE + [FILLER_RMS] * 30

        (status, reason), _items, frames = _run(session, levels)

        assert status == "completed"
        assert reason == "max_duration_reached"
        assert frames.consumed < len(levels)

    def test_flag_controls_the_relisten(self, monkeypatch):
        """KIOSK_CORE_FILLER_RELISTEN_ENABLED=false restores the old ending:
        the session stops at the filler and the order is never heard."""
        levels = FILLER + ENDPOINT_SILENCE + ORDER + ENDPOINT_SILENCE
        on = _make_session()
        _run(on, levels)
        assert on.transcript_parts == ["Two Pepsis."]

        monkeypatch.setattr(config, "DEFAULT_FILLER_RELISTEN_ENABLED", False)
        off = _make_session()
        _status, _items, frames = _run(off, levels)
        assert off.transcript_parts == []
        assert frames.consumed == len(FILLER + ENDPOINT_SILENCE)


class TestShouldRelisten:
    @pytest.fixture
    def session(self):
        session = _make_session()
        session._filler_dropped = True
        return session

    def test_filler_only_endpoint_relistens(self, session):
        assert session._should_relisten_after_filler(True, "completed") is True

    def test_not_after_stop_or_new_session(self, session):
        session._stop_event.set()
        assert session._should_relisten_after_filler(True, "completed") is False

    def test_not_without_silence_endpoint(self, session):
        # End-of-stream signal, max duration, or a stop: no endpoint commit.
        assert session._should_relisten_after_filler(False, "completed") is False

    def test_not_when_real_words_were_heard(self, session):
        session.transcript_parts = ["Two Pepsis."]
        assert session._should_relisten_after_filler(True, "completed") is False

    def test_not_on_failure(self, session):
        assert session._should_relisten_after_filler(True, "failed") is False

    def test_not_after_speaker_rejection(self, session):
        session._rejected_speech_chunks = 1
        assert session._should_relisten_after_filler(True, "completed") is False


    def test_reset_clears_the_filler_turn_end_stamps(self, session):
        """A mic-release signal_end() after a re-listen only stamps the last
        word when _t_last_word is None, so the filler's stamp must not
        survive into the resumed turn."""
        session._t_last_word = 1.0
        session._t_endpoint_decision = 1.0
        session._unconfirmed_speech_pending = True

        session._reset_for_relisten()

        assert session._t_last_word is None
        assert session._t_endpoint_decision is None
        assert session._unconfirmed_speech_pending is False
        assert session._filler_dropped is False
        assert session._relisten_at is not None


class _FakeRealtimeClient:
    def __init__(self, text: str):
        self.text = text
        self.commits = []

    def is_alive(self) -> bool:
        return True

    def latest_snapshot(self) -> dict:
        return {"text": self.text, "segments": [], "_analyzer_session_id": "a"}

    def commit(self, wait: bool = False, timeout: float = 0.0) -> None:
        self.commits.append(wait)


def _streaming_session(cached_text: str) -> BaseAudioSession:
    session = _make_session()
    session.realtime_client = _FakeRealtimeClient(cached_text)
    session._streaming_active = True
    session._analyzer_session_id = "a"
    session._asr_ms_total = 0.0
    session._asr_chunks = 0
    session._last_analyzer_segment_end = 0.0
    session._last_cumulative_flat_text = ""
    return session


class TestFlushChunkFiller:
    @pytest.fixture(autouse=True)
    def _no_diarization(self, monkeypatch):
        monkeypatch.setattr(config, "DEFAULT_DIARIZATION_ENABLED", False)

    def test_dropped_filler_is_flagged(self):
        session = _streaming_session("Uh...")

        session._flush_chunk([], is_final=True)

        assert session.transcript_parts == []
        assert session._filler_dropped is True

    def test_stale_filler_snapshot_is_not_reused(self):
        """After a re-listen, a final flush with no new frames must commit and
        wait for the resumed turn's result, not reuse the old snapshot. Once
        a newer snapshot has landed, the cached-snapshot shortcut applies."""
        stale = _streaming_session("Uh...")
        stale._stale_snapshot_text = "Uh..."
        stale._flush_chunk([], is_final=True)
        assert stale.realtime_client.commits == [True]

        fresh = _streaming_session("Uh... Two Pepsis.")
        fresh._stale_snapshot_text = "Uh..."
        fresh._flush_chunk([], is_final=True)
        assert fresh.realtime_client.commits == []
        assert fresh.transcript_parts == ["Uh... Two Pepsis."]
