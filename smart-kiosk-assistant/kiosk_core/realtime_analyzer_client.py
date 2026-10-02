"""WebSocket client for audio-analyzer's continuous ASR endpoint (/v1/realtime).

Kept intentionally simple for this demo app: one background thread runs a
small asyncio loop for the lifetime of one audio session. PCM frames are
sent continuously as they're captured (``send_frame``); ``commit()`` only
draws an utterance boundary on the analyzer side -- it never blocks the
caller (the frame-capture loop) waiting for a transcription round trip.

A background receive loop applies each
``conversation.item.input_audio_transcription.completed`` event to an
in-memory "latest snapshot", guarded by the server's own monotonically
increasing ``sequence`` number so a late/out-of-order event can never
clobber a newer snapshot.

No queues, retry frameworks, or external infra: an ``asyncio.Queue`` for
outbound frames and a couple of plain fields under a ``threading.Lock`` are
all the state this needs.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from urllib.parse import urlencode, urlparse, urlunparse

import websockets

logger = logging.getLogger(__name__)


def derive_realtime_ws_url(analyzer_url: str) -> str:
    """Turn the configured file-POST analyzer URL into the realtime WS URL.

    e.g. http://audio-analyzer:8010/v1/audio/transcriptions
      -> ws://audio-analyzer:8010/v1/realtime
    """
    parsed = urlparse(analyzer_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, "/v1/realtime", "", "", ""))


class RealtimeAnalyzerClient:
    """One persistent /v1/realtime WebSocket for a single audio session."""

    def __init__(
        self,
        ws_url: str,
        session_id: str,
        speaker_scope_id: str | None,
        diarization: bool | None,
        language: str | None,
    ):
        self._session_id = session_id
        self._speaker_scope_id = speaker_scope_id
        self._diarization = diarization
        self._language = language

        query = urlencode({"session_id": session_id, "language": language or ""})
        self._connect_url = f"{ws_url}?{query}"

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name=f"realtime-analyzer-{session_id}", daemon=True
        )
        self._send_queue: asyncio.Queue | None = None

        self._lock = threading.Lock()
        self._latest_sequence = -1
        self._latest_text = ""
        self._latest_segments: list[dict] = []
        self._update_event = threading.Event()

        # Preview state: updated by non-destructive `input_audio_buffer.preview`
        # pings. Reset to "" whenever a real commit lands (the analyzer's
        # buffer was just cleared, so any prior preview no longer applies to
        # what's now buffered). Tracked with its own sequence counter, kept
        # independent of the real-commit sequence above.
        self._preview_sequence = -1
        self._preview_text = ""

        # Every wall-clock (monotonic) timestamp at which a NON-EMPTY
        # transcript update landed, from either a preview or a real commit.
        # Kept as a list (not just "the latest") because the final commit's
        # own redundant re-confirmation (same text, landing ~200-700ms after
        # the preview that actually captured it -- verified via direct
        # session logs) would otherwise silently overwrite a much earlier,
        # correct landing and inflate "last word -> transcript ready" by
        # however long that redundant confirmation took to arrive. Callers
        # want the FIRST landing at/after the customer's last word, not
        # whatever happened to land most recently. See
        # first_content_landed_at_or_after() below.
        self._content_landings: list[float] = []

        # Diagnostic-only: wall-clock instant the most recent preview()/
        # commit() request was actually enqueued for send, used to log the
        # round-trip time once a matching (non-empty) transcript lands.
        # Single-flight only (good enough since previews are gated to fire
        # once per silence run and commits are infrequent) -- not meant to
        # correlate concurrent in-flight requests precisely.
        self._pending_request_sent_at: float | None = None

        # preview() in-flight coalescing. Previously preview() fired purely
        # on elapsed wall-clock time (cadence tick or VAD silence edge),
        # regardless of whether the analyzer had even answered the LAST
        # preview yet. Each preview re-transcribes the entire buffered
        # utterance from scratch, so under any slowdown (cold model/GPU
        # warm-up, or GPU contention with the LLM/TTS) round trips take
        # longer than the firing cadence -- requests then queue up
        # unboundedly on the single WS connection, each one re-decoding an
        # ever-larger buffer, compounding into multi-second pileups
        # (measured: 390ms -> 2,100ms across one 8-request backlog).
        # Fix: at most one preview request in flight at a time. A preview()
        # call that arrives while one is still outstanding does not enqueue
        # a second request -- it just marks "send another as soon as the
        # current one lands", which the receiver checks and re-fires
        # immediately. This guarantees the analyzer is always working on the
        # MOST RECENT ask, never queued behind stale ones.
        self._preview_in_flight = False
        self._preview_refresh_pending = False

        self._ready_event = threading.Event()
        self._connect_error: Exception | None = None
        self._closed = False
        # Set when the socket dies AFTER a successful connect (as opposed to
        # a failed initial connect, which raises from start() instead). The
        # caller only sees this via is_alive() -- previously nothing checked
        # for a mid-session drop, so send_frame/preview/commit degraded to
        # silent no-ops (RuntimeError on a closed loop, swallowed) and the
        # session was stuck replaying a stale/frozen transcript for its
        # remaining turns with no fallback to the file-based analyzer.
        self._died = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        """False once the socket has connected and then died or been closed.

        Callers (BaseAudioSession) should fall back to the file-based
        AnalyzerClient for the rest of the session once this goes False,
        the same way a failed initial connect is already handled.
        """
        return not self._closed and not self._died

    def start(self, timeout: float) -> None:
        """Connect and negotiate the session.

        Raises on failure/timeout -- the caller (BaseAudioSession) is
        expected to catch this and fall back to the file-based analyzer
        client for this session rather than failing the turn.
        """
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout):
            self.close()
            raise TimeoutError("timed out connecting to audio-analyzer /v1/realtime")
        if self._connect_error is not None:
            raise self._connect_error

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._send_queue = asyncio.Queue()
        try:
            self._loop.run_until_complete(self._main())
        except Exception as exc:  # noqa: BLE001 - surfaced to start() via _connect_error
            if not self._ready_event.is_set():
                self._connect_error = exc
                self._ready_event.set()
            else:
                self._died = True
                logger.warning(
                    "session=%s | realtime analyzer socket ended: %s", self._session_id, exc
                )
        else:
            # _main() can also return normally when the server closes the
            # socket cleanly (no exception) -- e.g. the FIRST_COMPLETED wait
            # unblocks because the receiver's `async for raw in ws` loop
            # simply ended. Only treat that as a live-session drop (and mark
            # the client dead so callers fall back) if it wasn't triggered
            # by our own close().
            if not self._closed:
                self._died = True
                logger.warning(
                    "session=%s | realtime analyzer socket closed by peer", self._session_id
                )
        finally:
            self._loop.close()

    async def _main(self) -> None:
        async with websockets.connect(self._connect_url, open_timeout=5.0) as ws:
            # kiosk-core's own VAD/endpoint logic is the single source of
            # truth for turn boundaries -- disable the analyzer's own
            # server-side VAD auto-commit so only our explicit commits fire.
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "turn_detection": None,
                    "speaker_scope_id": self._speaker_scope_id,
                    "diarization": self._diarization,
                    "input_audio_transcription": {"language": self._language},
                },
            }))
            self._ready_event.set()

            sender = asyncio.ensure_future(self._sender_loop(ws))
            receiver = asyncio.ensure_future(self._receiver_loop(ws))
            _done, pending = await asyncio.wait(
                {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()

    async def _sender_loop(self, ws) -> None:
        while True:
            item = await self._send_queue.get()
            if item is None:  # stop sentinel
                return
            await ws.send(item)

    async def _receiver_loop(self, ws) -> None:
        async for raw in ws:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "conversation.item.input_audio_transcription.completed":
                self._apply_completed_event(event)
            elif event_type == "conversation.item.input_audio_transcription.preview":
                self._apply_preview_event(event)
            elif event_type == "error":
                logger.warning(
                    "session=%s | realtime analyzer error: %s", self._session_id, event.get("error")
                )

    def _apply_completed_event(self, event: dict) -> None:
        sequence = event.get("sequence", 0)
        with self._lock:
            if sequence <= self._latest_sequence:
                # Stale/out-of-order result -- never let it overwrite a
                # newer snapshot.
                return
            self._latest_sequence = sequence
            self._latest_text = event.get("transcript", "") or ""
            self._latest_segments = event.get("segments") or []
            # The analyzer's buffer was just cleared by this real commit, so
            # any preview taken before it no longer describes what's
            # currently buffered -- discard it.
            self._preview_sequence = -1
            self._preview_text = ""
            if self._latest_text.strip():
                self._content_landings.append(time.monotonic())
        self._update_event.set()
        self._log_round_trip("commit")

    def _apply_preview_event(self, event: dict) -> None:
        sequence = event.get("sequence", 0)
        with self._lock:
            is_fresh = sequence > self._preview_sequence
            if is_fresh:
                self._preview_sequence = sequence
                self._preview_text = event.get("transcript", "") or ""
                if self._preview_text.strip():
                    # Record this landing -- used for the "customer's last
                    # word -> transcript ready" figure below. Only recorded
                    # when the transcript is non-empty: an empty preview/
                    # commit (pure trailing silence) tells us nothing new
                    # was transcribed, so it must not look like a fresh,
                    # later "transcript ready" moment.
                    self._content_landings.append(time.monotonic())
            # This response -- fresh or stale -- is what the ONE in-flight
            # preview request was waiting on, so the slot is free again
            # regardless. If another preview() call arrived while this one
            # was outstanding, fire it now with the freshest possible
            # buffer instead of leaving it dropped on the floor.
            self._preview_in_flight = False
            needs_refresh = self._preview_refresh_pending
            self._preview_refresh_pending = False
        self._log_round_trip("preview")
        if needs_refresh:
            self.preview()

    def _log_round_trip(self, kind: str) -> None:
        """Diagnostic-only: log wall-clock ms between the request being
        enqueued and this response landing, to separate "analyzer inference
        is slow" from "queueing/network/WS overhead is slow" when chasing
        asr_last_word_to_transcript_ms.
        """
        sent_at = self._pending_request_sent_at
        if sent_at is None:
            return
        round_trip_ms = (time.monotonic() - sent_at) * 1000
        logger.info(
            "session=%s | realtime analyzer %s round-trip=%.1fms",
            self._session_id, kind, round_trip_ms,
        )

    # ── public, thread-safe API used by BaseAudioSession ────────────────

    def send_frame(self, pcm_bytes: bytes) -> None:
        """Enqueue a frame for continuous streaming.

        Never blocks the caller (the frame-capture loop) beyond a cheap
        queue put -- the actual network send happens on the background loop.
        """
        if self._closed or self._send_queue is None:
            return
        payload = json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm_bytes).decode("ascii"),
        })
        try:
            self._loop.call_soon_threadsafe(self._send_queue.put_nowait, payload)
        except RuntimeError:
            pass  # loop already stopped/closed

    def preview(self) -> None:
        """Ask the analyzer to transcribe everything buffered since the last
        real commit, WITHOUT clearing it -- always fire-and-forget, never
        blocks the caller. Gives full-context rolling transcript snapshots
        during speech (see ``latest_preview_text``) without ever touching
        the persisted/official transcript that real commits build.

        At most one preview request is ever in flight (see
        ``_preview_in_flight``'s docstring in __init__): a call that arrives
        while one is still outstanding just marks a refresh as pending
        instead of enqueuing a second request, so slow/contended periods
        can never build an unbounded backlog of stale, ever-more-expensive
        re-transcriptions. A generous staleness cutoff guards against a
        permanently wedged flag if a response is ever silently dropped
        (e.g. the receiver task died) -- treats that as "not in flight"
        rather than blocking previews for the rest of the session.
        """
        if self._closed or self._send_queue is None:
            return
        with self._lock:
            if self._preview_in_flight:
                sent_at = self._pending_request_sent_at
                if sent_at is not None and (time.monotonic() - sent_at) < 5.0:
                    self._preview_refresh_pending = True
                    return
                # Stale in-flight marker (response never arrived) -- fall
                # through and send a fresh request rather than wedging.
            self._preview_in_flight = True
        self._pending_request_sent_at = time.monotonic()
        try:
            self._loop.call_soon_threadsafe(
                self._send_queue.put_nowait, json.dumps({"type": "input_audio_buffer.preview"})
            )
        except RuntimeError:
            with self._lock:
                self._preview_in_flight = False

    def commit(self, wait: bool = False, timeout: float = 0.0) -> None:
        """Signal an utterance boundary.

        Fire-and-forget unless ``wait`` is set (used only for the FINAL
        commit of a turn), in which case this blocks up to ``timeout``
        seconds for a newer transcript snapshot to land, then returns
        regardless. Callers must read ``latest_snapshot()`` either way --
        this call never hangs the turn.
        """
        if self._closed or self._send_queue is None:
            return
        baseline = self._latest_sequence
        self._update_event.clear()
        self._pending_request_sent_at = time.monotonic()
        try:
            self._loop.call_soon_threadsafe(
                self._send_queue.put_nowait, json.dumps({"type": "input_audio_buffer.commit"})
            )
        except RuntimeError:
            return
        if not wait:
            return

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return  # timed out -- caller falls back to latest_snapshot()
            if self._update_event.wait(timeout=remaining):
                with self._lock:
                    if self._latest_sequence > baseline:
                        return
                self._update_event.clear()
            else:
                return

    def first_content_landed_at_or_after(self, reference_ts: float) -> float | None:
        """Earliest non-empty-transcript landing at/after ``reference_ts``.

        ``reference_ts`` is the caller's own last-word timestamp. Deliberately
        the EARLIEST qualifying landing, not the latest: any *later* preview
        or the final commit's own redundant re-confirmation of the same text
        can land hundreds of ms afterward and would otherwise silently
        inflate "last word -> transcript ready" by however long that
        redundant, later call took -- verified via direct session logs (a
        commit's confirmation landed 700ms+ after the preview that had
        already captured the exact same, correct text). Falls back to the
        latest known landing if nothing qualifies at/after the reference
        (e.g. clock skew from silence_run_seconds backdating), so callers
        still get a number instead of None.
        """
        with self._lock:
            candidates = sorted(ts for ts in self._content_landings if ts >= reference_ts)
            if candidates:
                return candidates[0]
            return self._content_landings[-1] if self._content_landings else None

    def latest_preview_text(self) -> str:
        """Best-effort transcript of the in-progress (uncommitted) speech.

        For the endpoint-completeness check ONLY -- never fed into the
        official transcript, which is built exclusively from real commits.
        """
        with self._lock:
            return self._preview_text

    def latest_snapshot(self) -> dict:
        """Latest cumulative transcript text/segments for this session."""
        with self._lock:
            return {
                "text": self._latest_text,
                "segments": list(self._latest_segments),
                "_analyzer_session_id": self._session_id,
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._send_queue is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._send_queue.put_nowait, None)
            except RuntimeError:
                pass
        self._thread.join(timeout=3.0)
