"""Tests for the honest "sound at the speaker" voice-to-voice anchor.

The hands-free WebSocket uplink feeds raw PCM via ``push_pcm`` and reports the
browser's first-answer playback via ``mark_speaker_playback``. The reported
instant becomes ``voice_to_voice_speaker_ms`` either directly in the turn trace
or, when the report lands after the trace was recorded, by amending the stored
trace in place.
"""
from queue import Queue
from unittest.mock import MagicMock

import numpy as np

from kiosk_core.audio_session import BaseAudioSession, BrowserStreamSession
from kiosk_core.pipeline_latency import (
    PipelineLatencyStore,
    TurnTrace,
    WallTimes,
)


class TestWallTimesSpeakerField:
    def test_defaults_to_none(self):
        assert WallTimes().voice_to_voice_speaker_ms is None


class TestAmendSpeakerV2v:
    def test_amends_matching_trace(self):
        store = PipelineLatencyStore()
        store.record(_trace("s1"))

        assert store.amend_speaker_v2v("s1", 742.0) is True
        assert store.latest()["wall"]["voice_to_voice_speaker_ms"] == 742.0

    def test_returns_false_when_no_match(self):
        store = PipelineLatencyStore()
        store.record(_trace("s1"))

        assert store.amend_speaker_v2v("other", 10.0) is False


class TestMarkSpeakerPlayback:
    def test_stamps_once_and_is_idempotent(self):
        session = _bare_session()

        session.mark_speaker_playback()
        first = session._t_speaker_playback
        session.mark_speaker_playback()

        assert first is not None
        assert session._t_speaker_playback == first

    def test_late_report_amends_recorded_trace(self, monkeypatch):
        import kiosk_core.audio_session as audio_module

        store = PipelineLatencyStore()
        store.record(_trace("test-session"))
        monkeypatch.setattr(audio_module, "pipeline_store", store)

        session = _bare_session()
        session._turn_trace_recorded = True
        session._t_last_word = 100.0

        import time

        monkeypatch.setattr(time, "monotonic", lambda: 100.5)
        session.mark_speaker_playback()

        assert store.latest()["wall"]["voice_to_voice_speaker_ms"] == 500.0


class TestPushPcm:
    def test_raw_pcm_is_framed_into_the_queue(self):
        session = BrowserStreamSession.__new__(BrowserStreamSession)
        session.session_id = "s"
        session.request = MagicMock(sample_rate=16000)
        session._frame_samples = 4
        session._push_queue = Queue()
        session._t_playback_start = None

        pcm = np.arange(10, dtype=np.int16).tobytes()
        session.push_pcm(pcm)

        frames = []
        while not session._push_queue.empty():
            frames.append(session._push_queue.get())

        assert [len(f) for f in frames] == [4, 4, 2]
        assert session._t_playback_start is not None


def _bare_session() -> BaseAudioSession:
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = "test-session"
    session.request = MagicMock(sample_rate=16000)
    session._t_speaker_playback = None
    session._turn_trace_recorded = False
    session._t_last_word = None
    return session


def _trace(turn_id: str) -> TurnTrace:
    return TurnTrace(
        turn_id=turn_id,
        conversation_id=turn_id,
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        wall=WallTimes(),
    )
