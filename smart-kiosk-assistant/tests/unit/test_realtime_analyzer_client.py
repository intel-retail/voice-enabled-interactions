"""Tests for RealtimeAnalyzerClient (kiosk_core/realtime_analyzer_client.py).

Background (PR #112 review, item 25): the realtime WS ASR path shipped
without tests. The socket itself is not what is worth testing -- the
interesting logic is what the client does with events once they arrive, and
every one of those rules exists because it was got wrong once and showed up
as a bad latency figure or a frozen transcript:

* out-of-order events must never overwrite a newer snapshot
* a real commit invalidates any preview taken before it
* "transcript ready" is the FIRST qualifying landing, not the most recent
* at most one preview may be in flight, so a slow analyzer cannot build a
  backlog of ever-more-expensive re-transcriptions

None of that needs a server, so these tests drive the event handlers
directly. The client is constructed without ``start()``, which is what opens
the connection.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from kiosk_core.realtime_analyzer_client import (
    RealtimeAnalyzerClient,
    derive_realtime_ws_url,
)


@pytest.fixture
def client():
    """A client that has never connected.

    ``__init__`` only builds the loop and the (unstarted) thread; ``start()``
    is what dials the socket.
    """
    c = RealtimeAnalyzerClient(
        ws_url="ws://audio-analyzer:8010/v1/realtime",
        session_id="sess-1",
        speaker_scope_id=None,
        diarization=False,
        language=None,
    )
    real_loop = c._loop
    yield c
    # Held separately: tests that simulate a live socket swap _loop out for a
    # fake, and the real one still needs closing.
    real_loop.close()


class TestDeriveRealtimeWsUrl:
    def test_http_becomes_ws_and_path_is_replaced(self):
        assert derive_realtime_ws_url(
            "http://audio-analyzer:8010/v1/audio/transcriptions"
        ) == "ws://audio-analyzer:8010/v1/realtime"

    def test_https_becomes_wss(self):
        # Plain ws against a TLS analyzer would fail to connect, and the
        # session would silently fall back to the slower file-POST path.
        assert derive_realtime_ws_url(
            "https://analyzer.example.com/v1/audio/transcriptions"
        ) == "wss://analyzer.example.com/v1/realtime"

    def test_existing_query_and_port_handling(self):
        assert derive_realtime_ws_url(
            "http://127.0.0.1:8010/v1/audio/transcriptions?model=x"
        ) == "ws://127.0.0.1:8010/v1/realtime"


class TestCompletedEvents:
    def test_latest_snapshot_tracks_the_newest_commit(self, client):
        client._apply_completed_event(
            {"sequence": 1, "transcript": "one burger", "segments": [{"t": 1}]}
        )
        snapshot = client.latest_snapshot()
        assert snapshot["text"] == "one burger"
        assert snapshot["segments"] == [{"t": 1}]

    def test_out_of_order_commit_is_ignored(self, client):
        client._apply_completed_event({"sequence": 5, "transcript": "newer"})
        client._apply_completed_event({"sequence": 2, "transcript": "older"})
        # A late-arriving earlier sequence must not resurrect a stale
        # transcript over the one already shown to the customer.
        assert client.latest_snapshot()["text"] == "newer"

    def test_repeated_sequence_is_ignored(self, client):
        client._apply_completed_event({"sequence": 3, "transcript": "first"})
        client._apply_completed_event({"sequence": 3, "transcript": "duplicate"})
        assert client.latest_snapshot()["text"] == "first"

    def test_commit_discards_an_earlier_preview(self, client):
        client._apply_preview_event({"sequence": 1, "transcript": "one bur"})
        assert client.latest_preview_text() == "one bur"
        client._apply_completed_event({"sequence": 1, "transcript": "one burger"})
        # The analyzer's buffer was cleared by the commit, so the preview no
        # longer describes anything that is still buffered. Leaving it in
        # place would let the endpoint-completeness check see text that has
        # already been committed and count it twice.
        assert client.latest_preview_text() == ""

    def test_missing_fields_do_not_raise(self, client):
        client._apply_completed_event({"sequence": 1})
        snapshot = client.latest_snapshot()
        assert snapshot["text"] == ""
        assert snapshot["segments"] == []


class TestPreviewEvents:
    def test_fresh_preview_updates_text(self, client):
        client._apply_preview_event({"sequence": 1, "transcript": "I would like"})
        assert client.latest_preview_text() == "I would like"

    def test_stale_preview_is_ignored(self, client):
        client._apply_preview_event({"sequence": 4, "transcript": "newer"})
        client._apply_preview_event({"sequence": 2, "transcript": "older"})
        assert client.latest_preview_text() == "newer"

    def test_response_clears_the_in_flight_slot(self, client):
        client._preview_in_flight = True
        client._apply_preview_event({"sequence": 1, "transcript": "hi"})
        assert client._preview_in_flight is False

    def test_a_stale_response_still_frees_the_slot(self, client):
        # Whether or not the response was useful, it is what the single
        # in-flight request was waiting on. Leaving the flag set here would
        # block previews for the rest of the session.
        client._apply_preview_event({"sequence": 9, "transcript": "newer"})
        client._preview_in_flight = True
        client._apply_preview_event({"sequence": 1, "transcript": "older"})
        assert client._preview_in_flight is False


class TestContentLandings:
    def test_empty_transcripts_are_not_recorded_as_landings(self, client):
        # Pure trailing silence transcribes to "", which tells us nothing new
        # was heard. Recording it would make "last word -> transcript ready"
        # look later than it was.
        client._apply_preview_event({"sequence": 1, "transcript": "   "})
        client._apply_completed_event({"sequence": 1, "transcript": ""})
        assert client._content_landings == []
        assert client.first_content_landed_at_or_after(0.0) is None

    def test_returns_the_earliest_landing_at_or_after_the_reference(self, client):
        reference = time.monotonic()
        client._content_landings = [
            reference - 5.0,   # before the customer's last word
            reference + 0.20,  # the preview that actually captured the text
            reference + 0.90,  # the commit's redundant re-confirmation
        ]
        got = client.first_content_landed_at_or_after(reference)
        # Taking the latest here would inflate transcription latency by the
        # 700ms the redundant confirmation took to arrive.
        assert got == pytest.approx(reference + 0.20)

    def test_falls_back_to_the_latest_landing_when_none_qualify(self, client):
        # Reference timestamps are backdated by silence_run_seconds and can
        # land before every recorded landing. Callers still need a number.
        reference = time.monotonic() + 100
        client._content_landings = [reference - 10.0, reference - 5.0]
        assert client.first_content_landed_at_or_after(reference) == pytest.approx(
            reference - 5.0
        )


class _FakeQueue:
    """Stands in for the asyncio send queue, recording what was enqueued."""

    def __init__(self):
        self.items: list[str] = []

    def put_nowait(self, item):
        self.items.append(item)


class _FakeLoop:
    """Runs call_soon_threadsafe inline, so sends are observable."""

    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


def _attach_fake_socket(client) -> _FakeQueue:
    """Make the client believe its WS loop is up and running."""
    queue = _FakeQueue()
    client._send_queue = queue
    client._loop = _FakeLoop()
    return queue


class TestPreviewCoalescing:
    def test_preview_is_a_noop_before_the_socket_exists(self, client):
        # _send_queue is only created once the WS loop is running. This must
        # not raise on a session that fell back to the file-POST path.
        assert client._send_queue is None
        client.preview()
        assert client._preview_in_flight is False

    def test_second_preview_marks_a_refresh_instead_of_queueing(self, client):
        queue = _attach_fake_socket(client)
        client._preview_in_flight = True
        client._pending_request_sent_at = time.monotonic()

        client.preview()

        # Each preview re-transcribes the whole buffer, so a second in-flight
        # request would compound under contention rather than overtake.
        assert client._preview_refresh_pending is True
        assert queue.items == []

    def test_a_wedged_in_flight_marker_does_not_block_previews_forever(self, client):
        queue = _attach_fake_socket(client)
        client._preview_in_flight = True
        # Older than the staleness cutoff: the response was never delivered.
        client._pending_request_sent_at = time.monotonic() - 30.0

        client.preview()

        # Falls through and sends, rather than refusing previews for the rest
        # of the session.
        assert len(queue.items) == 1
        assert "input_audio_buffer.preview" in queue.items[0]
        assert client._preview_refresh_pending is False

    def test_first_preview_sends_and_claims_the_slot(self, client):
        queue = _attach_fake_socket(client)

        client.preview()

        assert len(queue.items) == 1
        assert client._preview_in_flight is True

    def test_pending_refresh_fires_once_the_response_lands(self, client):
        queue = _attach_fake_socket(client)
        client.preview()
        client.preview()  # coalesced into a pending refresh
        assert len(queue.items) == 1

        client._apply_preview_event({"sequence": 1, "transcript": "half a sen"})

        # The coalesced ask is re-fired against the freshest buffer rather
        # than being dropped on the floor.
        assert len(queue.items) == 2
        assert client._preview_refresh_pending is False


# ── analyzer low-latency mode ──────────────────────────────────────────────

FRAME_BYTES = 640  # 20 ms of 16 kHz int16 PCM


@pytest.fixture
def ll_client():
    """A client that asks for the analyzer's low-latency mode."""
    c = RealtimeAnalyzerClient(
        ws_url="ws://audio-analyzer:8010/v1/realtime",
        session_id="sess-ll",
        speaker_scope_id=None,
        diarization=False,
        language=None,
        low_latency=True,
        sample_rate=16000,
    )
    real_loop = c._loop
    yield c
    real_loop.close()


def _session_event(event_type: str, low_latency=None) -> dict:
    """A transcription_session.* event as audio-analyzer sends it.

    ``low_latency=None`` leaves the field out, as an analyzer without the
    mode does.
    """
    session = {"id": "sess-ll", "turn_detection": None}
    if low_latency is not None:
        session["low_latency"] = low_latency
    return {"type": event_type, "session": session}


def _speech_then_silence(client, speech_frames: int, silence_frames: int) -> None:
    for _ in range(speech_frames):
        client.send_frame(b"\x01" * FRAME_BYTES, is_speech=True)
    for _ in range(silence_frames):
        client.send_frame(b"\x00" * FRAME_BYTES, is_speech=False)


class _FakeWebSocket:
    """Records what the client sends; replays scripted server events."""

    def __init__(self, events: list[dict]):
        self.sent: list[dict] = []
        self._events = [json.dumps(e) for e in events]

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        await asyncio.sleep(0)
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def _run_main(client, monkeypatch, events: list[dict]) -> _FakeWebSocket:
    """Run the client's connect-and-negotiate coroutine against a fake socket."""
    ws = _FakeWebSocket(events)

    class _Connect:
        async def __aenter__(self):
            return ws

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        "kiosk_core.realtime_analyzer_client.websockets.connect",
        lambda *args, **kwargs: _Connect(),
    )

    async def _drive():
        client._send_queue = asyncio.Queue()
        await client._main()

    asyncio.run(_drive())
    return ws


class TestLowLatencyNegotiation:
    """The mode is asked for in its own session.update and is only in effect
    once the analyzer echoes it. An analyzer without the mode (the pinned
    edge-ai-libraries build) ignores the field and never echoes it."""

    def test_request_goes_in_its_own_session_update(self, ll_client, monkeypatch):
        # One update with only the new field: an analyzer that rejects it
        # still applies turn_detection=None from the first one.
        ws = _run_main(ll_client, monkeypatch, [_session_event("transcription_session.created")])
        updates = [m for m in ws.sent if m["type"] == "session.update"]
        assert len(updates) == 2
        assert "low_latency" not in updates[0]["session"]
        assert updates[0]["session"]["turn_detection"] is None
        assert updates[1]["session"] == {"low_latency": {"enabled": True}}

    def test_analyzer_without_the_mode_leaves_it_off(self, ll_client, monkeypatch):
        _run_main(ll_client, monkeypatch, [
            _session_event("transcription_session.created"),
            _session_event("transcription_session.updated"),
            _session_event("transcription_session.updated"),
        ])
        assert ll_client.low_latency_active() is False

    def test_rejected_request_leaves_it_off(self, ll_client, monkeypatch):
        _run_main(ll_client, monkeypatch, [
            _session_event("transcription_session.created"),
            _session_event("transcription_session.updated"),
            {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
        ])
        assert ll_client.low_latency_active() is False

    def test_echoed_mode_turns_it_on(self, ll_client, monkeypatch):
        _run_main(ll_client, monkeypatch, [
            _session_event("transcription_session.created", {"enabled": False}),
            _session_event("transcription_session.updated", {"enabled": False}),
            _session_event("transcription_session.updated", {"enabled": True}),
        ])
        assert ll_client.low_latency_active() is True

    def test_echo_with_mode_disabled_turns_it_off(self, ll_client):
        ll_client._apply_session_event(_session_event("transcription_session.updated", {"enabled": True}))
        ll_client._apply_session_event(_session_event("transcription_session.updated", {"enabled": False}))
        assert ll_client.low_latency_active() is False


class TestLowLatencyMessages:
    """While the mode is in effect, previews and commits carry the
    trailing-silence hint, and a preview in silence asks for the whole
    buffer. Without the echo the client sends exactly what it sent before."""

    def test_unconfirmed_mode_sends_plain_preview_and_commit(self, ll_client):
        queue = _attach_fake_socket(ll_client)
        _speech_then_silence(ll_client, speech_frames=10, silence_frames=5)
        ll_client.preview()
        ll_client.commit()
        sent = [json.loads(item) for item in queue.items if "append" not in item]
        assert sent == [
            {"type": "input_audio_buffer.preview"},
            {"type": "input_audio_buffer.commit"},
        ]

    def test_preview_in_silence_asks_for_the_whole_buffer(self, ll_client):
        ll_client._apply_session_event(_session_event("transcription_session.updated", {"enabled": True}))
        queue = _attach_fake_socket(ll_client)
        _speech_then_silence(ll_client, speech_frames=10, silence_frames=5)
        ll_client.preview()
        preview = json.loads(queue.items[-1])
        assert preview["type"] == "input_audio_buffer.preview"
        assert preview["full"] is True
        assert preview["trailing_silence_ms"] == pytest.approx(100.0)

    def test_preview_during_speech_keeps_the_trailing_window(self, ll_client):
        ll_client._apply_session_event(_session_event("transcription_session.updated", {"enabled": True}))
        queue = _attach_fake_socket(ll_client)
        _speech_then_silence(ll_client, speech_frames=10, silence_frames=0)
        ll_client.preview()
        preview = json.loads(queue.items[-1])
        assert "full" not in preview
        assert preview["trailing_silence_ms"] == 0.0

    def test_commit_carries_the_hint_and_resets_the_count(self, ll_client):
        ll_client._apply_session_event(_session_event("transcription_session.updated", {"enabled": True}))
        queue = _attach_fake_socket(ll_client)
        _speech_then_silence(ll_client, speech_frames=10, silence_frames=25)
        ll_client.commit()
        commit = json.loads(queue.items[-1])
        assert commit["type"] == "input_audio_buffer.commit"
        assert commit["trailing_silence_ms"] == pytest.approx(500.0)
        assert "full" not in commit
        # The analyzer cleared its buffer, so the count starts again from 0
        # and the earlier speech no longer counts: silence alone is not a
        # finished utterance worth a full preview.
        _speech_then_silence(ll_client, speech_frames=0, silence_frames=10)
        ll_client.preview()
        preview = json.loads(queue.items[-1])
        assert preview["trailing_silence_ms"] == pytest.approx(200.0)
        assert "full" not in preview
