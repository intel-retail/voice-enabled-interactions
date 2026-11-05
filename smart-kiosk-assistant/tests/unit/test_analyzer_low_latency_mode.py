"""The session side of audio-analyzer's realtime low-latency mode.

With the mode in effect, the analyzer answers the final commit from its last
full preview, so kiosk-core sends no intermediate (adaptive-pause or
chunk-cap) commit: one would land within a few ms and clear the preview text
the endpoint check is still reading. That change must only happen once the
analyzer has confirmed the mode. An analyzer without it (the pinned
edge-ai-libraries build) ignores the request, and the kiosk must then keep
today's commits, or it would lose both the adaptive-pause commit and the
full preview. See config.DEFAULT_ANALYZER_LOW_LATENCY_ENABLED.

These tests drive ``BaseAudioSession._flush_chunk`` and
``_stream_frame`` against a fake realtime client. No audio hardware, no
network.
"""
import sys
import threading
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("sounddevice", MagicMock())

from kiosk_core import config  # noqa: E402
from kiosk_core.audio_session import BaseAudioSession  # noqa: E402

SAMPLE_RATE = 16000


class _FakeRealtimeClient:
    """Streaming is live; records commits and frames."""

    def __init__(self, mode_confirmed: bool, cached_text: str = "One classic burger."):
        self.mode_confirmed = mode_confirmed
        self.cached_text = cached_text
        self.commits: list[bool] = []
        self.frames: list[dict] = []

    def is_alive(self) -> bool:
        return True

    def low_latency_active(self) -> bool:
        return self.mode_confirmed

    def latest_snapshot(self) -> dict:
        return {"text": self.cached_text, "segments": [], "_analyzer_session_id": "a"}

    def commit(self, wait: bool = False, timeout: float = 0.0) -> None:
        self.commits.append(wait)


class _StrictFrameClient(_FakeRealtimeClient):
    """send_frame with the pre-existing signature (no is_speech)."""

    def send_frame(self, pcm_bytes: bytes) -> None:
        self.frames.append({"pcm": pcm_bytes})


class _VadFrameClient(_FakeRealtimeClient):
    def send_frame(self, pcm_bytes: bytes, is_speech: bool = False) -> None:
        self.frames.append({"pcm": pcm_bytes, "is_speech": is_speech})


def _session(client: _FakeRealtimeClient) -> BaseAudioSession:
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = "test-session"
    session.agent_session_id = "test-conversation"
    session.request = types.SimpleNamespace(
        sample_rate=SAMPLE_RATE,
        language="en",
        temperature=0.0,
    )
    session.realtime_client = client
    session._streaming_active = True
    session._analyzer_session_id = "a"
    session._asr_ms_total = 0.0
    session._asr_chunks = 0
    session._last_analyzer_segment_end = 0.0
    session._last_cumulative_flat_text = ""
    session.transcript_parts = []
    session._lock = threading.Lock()
    session._unconfirmed_speech_pending = True
    return session


def _speech_chunk() -> list[np.ndarray]:
    return [np.full(320, 4000, dtype=np.int16) for _ in range(25)]  # 0.5 s


@pytest.fixture
def mode_requested(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_ANALYZER_LOW_LATENCY_ENABLED", True)
    monkeypatch.setattr(config, "DEFAULT_DIARIZATION_ENABLED", False)


class TestIntermediateCommit:
    def test_confirmed_mode_sends_no_intermediate_commit(self, mode_requested):
        # The cache still holds an earlier result (the mode was confirmed
        # after the turn's first commit, or a filler turn's snapshot after a
        # re-listen). Nothing was transcribed for this flush, so none of it
        # may reach the transcript and no session state may move.
        client = _FakeRealtimeClient(mode_confirmed=True, cached_text="Uh...")
        session = _session(client)

        session._flush_chunk(_speech_chunk(), is_final=False)

        assert client.commits == []
        assert session.transcript_parts == []
        assert session._unconfirmed_speech_pending is True
        assert session._last_analyzer_segment_end == 0.0
        assert session._last_cumulative_flat_text == ""
        assert session._asr_chunks == 0

    def test_unconfirmed_mode_commits_as_before(self, mode_requested):
        # Flag on, but the analyzer never echoed the mode (older build):
        # the adaptive-pause commit must still go out.
        client = _FakeRealtimeClient(mode_confirmed=False)
        session = _session(client)

        session._flush_chunk(_speech_chunk(), is_final=False)

        assert client.commits == [False]

    def test_final_flush_still_commits_and_waits(self, mode_requested):
        client = _FakeRealtimeClient(mode_confirmed=True)
        session = _session(client)

        session._flush_chunk(_speech_chunk(), is_final=True)

        assert client.commits == [True]

    def test_final_flush_without_frames_commits_despite_a_cached_result(self, mode_requested):
        # In low-latency mode the adaptive-pause flush sent no commit, so the
        # final ([], True) flush is the turn's only commit. It must not take
        # the cached-snapshot shortcut, or the turn's audio is never
        # transcribed.
        client = _FakeRealtimeClient(mode_confirmed=True, cached_text="Uh...")
        session = _session(client)

        session._flush_chunk([], is_final=True)

        assert client.commits == [True]

    def test_mode_is_not_used_with_diarization_on(self, monkeypatch):
        # The analyzer applies the mode only with diarization off; with it
        # on, skipping the commit would lose the adaptive-pause commit for
        # nothing.
        monkeypatch.setattr(config, "DEFAULT_ANALYZER_LOW_LATENCY_ENABLED", True)
        monkeypatch.setattr(config, "DEFAULT_DIARIZATION_ENABLED", True)
        monkeypatch.setattr(config, "DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED", False)
        monkeypatch.setattr(config, "DEFAULT_DIARIZATION_ENROLLMENT_PRIMING_ENABLED", False)
        client = _FakeRealtimeClient(mode_confirmed=True)
        session = _session(client)

        session._flush_chunk(_speech_chunk(), is_final=False)

        assert client.commits == [False]


class TestStreamFrame:
    def test_vad_verdict_goes_with_the_frame_when_requested(self, mode_requested):
        client = _VadFrameClient(mode_confirmed=False)
        session = _session(client)

        session._stream_frame(np.zeros(320, dtype=np.int16), is_speech=True)
        session._stream_frame(np.zeros(320, dtype=np.int16), is_speech=False)

        assert [f["is_speech"] for f in client.frames] == [True, False]

    def test_mode_off_keeps_the_old_send_frame_call(self, monkeypatch):
        monkeypatch.setattr(config, "DEFAULT_ANALYZER_LOW_LATENCY_ENABLED", False)
        client = _StrictFrameClient(mode_confirmed=False)
        session = _session(client)

        session._stream_frame(np.zeros(320, dtype=np.int16), is_speech=True)

        assert len(client.frames) == 1
