"""Event delivery fan-out for the Smart Kiosk QSR MCP surface.

Self-contained reimplementation of ``mcp_service_sdk.delivery`` (the only
piece of the SDK ``mcp_service.py``'s ``emit`` depended on beyond the log
itself). ``emit`` always saves to the durable log first; this module then
pushes to whichever sinks are enabled — none (default), the generic
``MCP_WEBHOOK_URL`` callback, and/or the QSR Agent autonomy inbox
(``kiosk_core/qsr_mcp/autonomy.py``) — with retries. Failed callbacks are
retried; nothing is dropped silently.

A sink may raise ``PermanentDeliveryError`` to stop its own retry loop early
when the remote rejected the payload itself (e.g. HTTP 400) and resending the
identical body cannot help.
"""
from __future__ import annotations

import http.client
import logging
import queue
import threading
import time
from typing import Callable, Protocol

from kiosk_core.qsr_mcp.autonomy import PermanentDeliveryError, _post_json
from kiosk_core.qsr_mcp.event_log import EventEnvelope

logger = logging.getLogger(__name__)


class Sink(Protocol):
    """A delivery target. ``push`` returns ``True`` when no retry is needed.

    A sink may also expose ``accepts(event) -> bool`` to filter events it does
    not care about. ``Delivery`` evaluates it exactly once per event, before
    the retry loop, so a sink may keep state there (e.g. change detection)
    without a retry being mistaken for a fresh observation.
    """

    name: str

    def push(self, event: EventEnvelope) -> bool:
        ...


class WebhookSink:
    """POSTs the event envelope to a callback URL (stdlib, no extra deps).

    Shares ``autonomy._post_json`` with :class:`AutonomyEventSink` so both
    sinks behave identically on the wire: redirects are not followed (a 3xx
    would silently re-POST the payload to another host) and a 4xx is
    permanent rather than retried.
    """

    name = "webhook"

    def __init__(
        self,
        url: str,
        timeout_s: float = 5.0,
        post_json: Callable[[str, bytes, dict[str, str], float], int] | None = None,
    ) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._post_json = post_json or _post_json

    def push(self, event: EventEnvelope) -> bool:
        data = event.to_json().encode("utf-8")
        try:
            status = self._post_json(
                self._url, data, {"Content-Type": "application/json"}, self._timeout_s
            )
        except (OSError, http.client.HTTPException) as error:
            # Same catch as AutonomyEventSink: _post_json can raise an
            # http.client.HTTPException (IncompleteRead, BadStatusLine) or a
            # bare OSError, neither of which is a URLError. Letting one escape
            # would abort Delivery.dispatch and skip every later sink for this
            # event.
            logger.warning(
                "[webhook] event %s delivery failed: %s", event.ref_id, error
            )
            return False
        if 200 <= status < 300:
            return True
        # 429 is excluded: it means "slow down", not "this payload is wrong",
        # so it must use the retry budget rather than being dropped.
        if 400 <= status < 500 and status != 429:
            raise PermanentDeliveryError(
                f"webhook rejected event {event.ref_id}: HTTP {status}"
            )
        logger.warning(
            "[webhook] event %s not accepted: HTTP %s", event.ref_id, status
        )
        return False


class Delivery:
    """Dispatches an event to every configured sink, with bounded retries.

    Args:
        webhook_url: Generic callback URL; adds a ``WebhookSink`` when set.
        sinks: Additional pre-built sinks (e.g. ``AutonomyEventSink``).
        max_retries: Attempts per sink before giving up.
        backoff_s: Base delay for exponential backoff between attempts.
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        sinks: tuple[Sink, ...] | list[Sink] | None = None,
        max_retries: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        resolved: list[Sink] = []
        if webhook_url:
            resolved.append(WebhookSink(webhook_url))
        if sinks:
            resolved.extend(sinks)
        self._sinks: tuple[Sink, ...] = tuple(resolved)
        self._max_retries = max_retries
        self._backoff_s = backoff_s

    @property
    def sinks(self) -> tuple[Sink, ...]:
        return self._sinks

    def dispatch(self, event: EventEnvelope) -> bool:
        """Deliver to every enabled sink, if any.

        Returns:
            ``True`` when every sink succeeded — including the no-sink case,
            matching the old ``DisabledSink`` behaviour. One sink's failure
            never prevents the others from being attempted.
        """
        if not self._sinks:
            return True
        results = [self._push_with_retry(sink, event) for sink in self._sinks]
        return all(results)

    def _push_with_retry(self, sink: Sink, event: EventEnvelope) -> bool:
        accepts = getattr(sink, "accepts", None)
        # Evaluated once, outside the loop: a sink's filter may be stateful.
        if accepts is not None and not accepts(event):
            return True
        for attempt in range(self._max_retries):
            try:
                if sink.push(event):
                    return True
            except PermanentDeliveryError:
                # Remote rejected the payload itself; resending is pointless.
                return False
            if attempt < self._max_retries - 1:
                time.sleep(self._backoff_s * (2**attempt))
        return False


class BackgroundDispatcher:
    """Runs a ``Delivery`` on one daemon worker thread, preserving order.

    ``emit`` must never block the queue poller or a ``set_board_mode`` tool
    call on a slow remote, but a thread per event would let a retrying older
    event arrive *after* a newer one — leaving the agent acting on a stale
    queue depth. A single FIFO worker keeps emission order intact and bounds
    thread churn to one thread.

    Args:
        delivery: The configured ``Delivery`` to run.
        name: Worker thread name.
        max_queue: Bound on pending events so an unreachable remote cannot
            grow memory without limit. Newest events are dropped when full.
    """

    def __init__(self, delivery: Delivery, name: str, max_queue: int = 1000) -> None:
        self._delivery = delivery
        self._name = name
        self._queue: "queue.Queue[EventEnvelope | None]" = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, event: EventEnvelope) -> bool:
        """Enqueue an event for delivery. Returns ``False`` if it was dropped."""
        self._ensure_worker()
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            logger.warning(
                "[%s] delivery backlog full (%d); dropping event %s",
                self._name,
                self._queue.maxsize,
                event.ref_id,
            )
            return False

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, daemon=True, name=self._name
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            if event is None:
                return
            try:
                self._delivery.dispatch(event)
            except Exception:
                logger.exception("[%s] delivery raised unexpectedly", self._name)
            finally:
                self._queue.task_done()

    def join(self, timeout: float | None = None) -> None:
        """Block until the backlog drains. Intended for tests."""
        self._queue.join()

    def stop(self, timeout: float | None = 2.0) -> None:
        """Shut the worker down, draining whatever is already queued.

        Sends the ``None`` sentinel the run loop already recognises. Safe to
        call when no worker was ever started: ``_ensure_worker`` is lazy, so
        there may be nothing to stop.

        Args:
            timeout: Seconds to wait for the worker to exit. The thread is a
                daemon, so a hung remote cannot block process exit regardless.
        """
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is None or not thread.is_alive():
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Backlog is full of events we are about to discard anyway; the
            # daemon thread dies with the process.
            return
        thread.join(timeout)
