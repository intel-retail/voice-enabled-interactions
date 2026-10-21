"""Regression test for the hands-free WebSocket late-frame handling.

In hands-free mode the browser keeps streaming mic frames until its poll sees
the turn has ended. The backend endpoint detector can close the turn while
those last frames are still in flight, so a late binary frame reaches
``service.push_pcm`` *after* the session completed and raises ``ValueError``.

The browser sends its ``playback_started`` control message only *after* those
trailing frames (its poll sees the turn end once the first audio segment is
ready). The original handler sent ``turn_ended`` and ``break``-ed out of the
receive loop on that ValueError, so the later ``playback_started`` was never
read and ``voice_to_voice_speaker_ms`` was lost — which bit single-TTS-segment
replies, where the first segment and completion land in the same poll (Josh
review, 38/70 one-segment turns lost the stamp).

The fix (``main.py`` ``stream_session_ws``) is to ``continue`` instead of
``break`` on a late frame, and to announce ``turn_ended`` only once. This test
drives the handler with a fake socket that reproduces the exact sequence —
good frame, two late frames, then ``playback_started`` — and asserts the
playback report still reaches the service and ``turn_ended`` is sent once.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import main


class _FakeWebSocket:
    """Minimal async WebSocket double scripted with a fixed receive sequence.

    ``receive_json`` yields the opening ``start`` message; ``receive`` then
    replays ``frames`` one per call (Starlette-style event dicts), ending with
    a ``websocket.disconnect`` so the handler's loop exits cleanly.
    """

    def __init__(self, start: dict, frames: list[dict]) -> None:
        self._start = start
        self._frames = list(frames)
        self.sent: list[dict] = []
        self.closed = False

    async def accept(self) -> None:
        return None

    async def receive_json(self) -> dict:
        return self._start

    async def receive(self) -> dict:
        if self._frames:
            return self._frames.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_json(self, obj: dict) -> None:
        self.sent.append(obj)

    async def close(self, *args, **kwargs) -> None:
        self.closed = True


def _run_handler(monkeypatch, frames: list[dict]):
    """Drive ``stream_session_ws`` with a stubbed service and fake socket."""
    monkeypatch.setattr(main.cfg, "DEFAULT_HANDSFREE_WS_ENABLED", True, raising=False)

    service = MagicMock()
    service.start_stream_session.return_value = {"session_id": "sess-1"}

    def _push(session_id: str, data: bytes) -> None:
        # Frames tagged "late" arrive after the turn completed — the real
        # SessionService raises ValueError for a push onto a finished session.
        if data.startswith(b"late"):
            raise ValueError("stream session already completed")

    service.push_pcm.side_effect = _push
    monkeypatch.setattr(main, "service", service)

    # Avoid touching the ordering DB and the real request model.
    async def _noop_clear(history=None) -> None:
        return None

    monkeypatch.setattr(main, "_clear_stale_cart_for_new_session", _noop_clear)
    monkeypatch.setattr(
        main, "SessionStartRequest", lambda **kw: SimpleNamespace(history=None, **kw)
    )

    ws = _FakeWebSocket(start={"type": "start"}, frames=frames)
    asyncio.run(main.stream_session_ws(ws))
    return service, ws


def test_late_frame_does_not_drop_playback_started(monkeypatch):
    frames = [
        {"type": "websocket.receive", "bytes": b"good-frame"},
        {"type": "websocket.receive", "bytes": b"late-frame-1"},
        {"type": "websocket.receive", "bytes": b"late-frame-2"},
        {"type": "websocket.receive", "text": json.dumps({"type": "playback_started"})},
    ]
    service, ws = _run_handler(monkeypatch, frames)

    # The core regression: the handler kept reading past the late frames, so
    # the browser's playback report still reached the service and the honest
    # voice_to_voice_speaker_ms can be stamped.
    service.mark_playback_started.assert_called_once_with("sess-1")

    # turn_ended is announced exactly once despite two late frames.
    turn_ended = [m for m in ws.sent if m.get("type") == "turn_ended"]
    assert len(turn_ended) == 1, ws.sent
    assert turn_ended[0]["session_id"] == "sess-1"


def test_normal_turn_without_late_frames_reports_playback(monkeypatch):
    # No late frames: playback_started must still reach the service and no
    # turn_ended is emitted.
    frames = [
        {"type": "websocket.receive", "bytes": b"good-frame"},
        {"type": "websocket.receive", "text": json.dumps({"type": "playback_started"})},
        {"type": "websocket.receive", "text": json.dumps({"type": "end"})},
    ]
    service, ws = _run_handler(monkeypatch, frames)

    service.mark_playback_started.assert_called_once_with("sess-1")
    # signal_stream_end fires for the explicit "end" event and again in the
    # handler's finally cleanup (idempotent in the real service), so assert it
    # ran for this session rather than pinning an exact count.
    service.signal_stream_end.assert_any_call("sess-1")
    assert not [m for m in ws.sent if m.get("type") == "turn_ended"]
