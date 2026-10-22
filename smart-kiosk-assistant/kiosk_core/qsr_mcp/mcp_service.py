"""MCP integration for kiosk-core — the Central QSR Agent's Kiosk service
(qsr-design §4/§7; retail-use-cases#100/#103).

Built directly on ``fastmcp`` (same library already used by
``kiosk_core/ordering/mcp_server.py``), not ``mcp_service_sdk``: this module
owns its own durable event log (``event_log.py``), delivery fan-out
(``delivery.py``), and action policy gate (``policy.py``) instead of
depending on the external SDK package for that scaffolding. Kiosk is a
**sensor + actuator**:

  * Sensor: queue depth (via the standalone queue-service) and order
    activity (via the existing ``kiosk_core.ordering`` DB).
  * Actuator: ``set_board_mode`` — simplify/expand the on-screen menu board,
    the concrete action behind qsr-design's autonomous action #6 ("Kiosk
    queue deep/short -> Kiosk board simplify").

Read tools answer the qsr-design §7-A queries: "How deep is kiosk 4's
queue?", "Is kiosk 4 always slower?", "What's on the board now?". Queue
samples are recorded to the durable log by a background poller
(``start_queue_poller`` in ``mcp_server.py``) so trend/history queries work
across restarts.

Events are additionally *pushed* to the Central QSR Agent's
``POST /autonomy/events`` inbox when ``KIOSK_CORE_QSR_AUTONOMY_EVENTS_URL`` is
configured (see ``autonomy.py``), so Hermes can propose an action without
polling this service. Delivery is off by default.

The durable event log's on-disk format (SQLite table/columns, JSONL record
shape) is unchanged from the previous ``mcp_service_sdk``-backed
implementation — see ``event_log.py`` — so an existing, populated
``qsr_mcp_events.db`` keeps working without any migration step.
"""
from __future__ import annotations

import calendar
import logging
import os
import sqlite3
import time
from typing import Any, Optional

from fastmcp import FastMCP

from kiosk_core import config as cfg
from kiosk_core.queue_client import QueueClient
from kiosk_core.qsr_mcp import board_state
from kiosk_core.qsr_mcp.autonomy import AutonomyEventSink
from kiosk_core.qsr_mcp.delivery import BackgroundDispatcher, Delivery
from kiosk_core.qsr_mcp.event_log import build_log, new_event
from kiosk_core.qsr_mcp.policy import GateLevel, PolicyGate

logger = logging.getLogger(__name__)

MCP_SERVICE_ENABLED = cfg.QSR_MCP_ENABLED
STORE_ID = cfg.QSR_MCP_STORE_ID
MCP_TRANSPORT = cfg.QSR_MCP_TRANSPORT
MCP_HOST = cfg.QSR_MCP_HOST
MCP_PORT = cfg.QSR_MCP_PORT
SERVICE_NAME = "smart_kiosk"

_EVENT_TYPES: dict[str, dict[str, str]] = {
    "queue_depth_sample": {"count": "int", "nearby": "int", "status": "str"},
    "board_mode_changed": {"mode": "str", "reason": "str|None"},
}

# -- plumbing: durable log, delivery, policy gate (self-contained, no SDK) --

_log = build_log(
    backend=cfg.QSR_MCP_LOG_BACKEND if MCP_SERVICE_ENABLED else "memory",
    path=cfg.QSR_MCP_LOG_PATH,
    service=SERVICE_NAME,
)
_delivery = Delivery(webhook_url=cfg.QSR_MCP_WEBHOOK_URL)
# Outbound autonomy push to the Central QSR Agent; built only when the URL is
# configured, so the default stack performs no outbound delivery at all.
_autonomy_delivery = (
    Delivery(
        sinks=[
            AutonomyEventSink(
                cfg.QSR_AUTONOMY_EVENTS_URL,
                store_id=cfg.QSR_AUTONOMY_STORE_ID,
                token=cfg.QSR_AUTONOMY_EVENTS_TOKEN,
                event_types=cfg.QSR_AUTONOMY_EVENT_TYPES,
                queue_min_delta=cfg.QSR_AUTONOMY_QUEUE_MIN_DELTA,
                timeout_s=cfg.QSR_AUTONOMY_TIMEOUT_SECONDS,
            )
        ]
    )
    if cfg.QSR_AUTONOMY_EVENTS_URL and MCP_SERVICE_ENABLED
    else None
)

# One FIFO worker per destination: emit() stays non-blocking, while delivery
# order (and therefore the agent's view of "latest queue depth") is preserved.
_dispatchers = tuple(
    BackgroundDispatcher(delivery, name)
    for delivery, name in (
        (_delivery, "qsr-webhook-delivery"),
        (_autonomy_delivery, "qsr-autonomy-delivery"),
    )
    if delivery is not None and delivery.sinks
)

_policy = PolicyGate()
_subscriptions: list[dict[str, str]] = []

board_state.configure(cfg.QSR_MCP_BOARD_STATE_PATH)
_queue_client = QueueClient() if cfg.QUEUE_SERVICE_ENABLED else None

mcp = FastMCP(SERVICE_NAME)

# Bookkeeping for `describe` — independent of fastmcp's own type-hint-based
# schema introspection (which still drives the real MCP `tools/list`
# inputSchema for every tool below, unchanged from before this migration).
_READ_TOOLS: dict[str, dict[str, Any]] = {}
_ACT_TOOLS: dict[str, dict[str, Any]] = {}


def emit(event_type: str, payload: dict[str, Any], ref_id: str | None = None) -> None:
    """Durably log one event, then fan out to the configured sinks (if any).

    Emit-to-log-first, then dispatch — same ordering as the old SDK's
    ``ServiceServer.emit``. Dispatch is handed to a background FIFO worker per
    destination, so neither the queue poller nor a ``set_board_mode`` tool call
    blocks on a slow or unreachable remote: the durable log is the
    authoritative record and delivery is best-effort.
    """
    event = new_event(event_type, SERVICE_NAME, STORE_ID, payload, ref_id)
    _log.append(event)
    for dispatcher in _dispatchers:
        dispatcher.submit(event)


def emit_queue_sample(snapshot: dict[str, Any]) -> None:
    """Durably log one queue-depth sample. Best-effort: never raises."""
    if not MCP_SERVICE_ENABLED:
        return
    try:
        emit(
            "queue_depth_sample",
            {
                "count": snapshot.get("count"),
                "nearby": snapshot.get("nearby"),
                "status": snapshot.get("status"),
            },
        )
    except Exception:
        logger.exception("[QSR-MCP] Failed to emit queue_depth_sample")


# ---------------------------------------------------------------------------
# describe / subscribe — same contract shape as the previous SDK-provided
# tools, now implemented directly.
# ---------------------------------------------------------------------------


@mcp.tool(name="describe", description="Describe this service's contract.")
def describe() -> dict[str, Any]:
    """Self-description rich enough for a coding agent to use unassisted."""
    return {
        "service": SERVICE_NAME,
        "store_id": STORE_ID,
        "event_types": _EVENT_TYPES,
        "read_tools": {
            name: {"description": meta["description"], "schema": meta["schema"]}
            for name, meta in _READ_TOOLS.items()
        },
        "act_tools": {
            name: {
                "description": meta["description"],
                "schema": meta["schema"],
                "gate": meta["gate"],
            }
            for name, meta in _ACT_TOOLS.items()
        },
    }


@mcp.tool(name="subscribe", description="Subscribe to an event type.")
def subscribe(event_type: str, condition: str, callback_url: str) -> dict[str, Any]:
    _subscriptions.append(
        {"event_type": event_type, "condition": condition, "callback_url": callback_url}
    )
    return {"subscribed": event_type, "condition": condition}


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

_LOG_READ_PAGE = 5000


def _read_entire_log() -> list:
    """Read the full durable log, oldest -> newest (both durable log
    backends have a gapless, contiguous ``seq``, so unfiltered paging never
    skips or re-reads rows regardless of how large the log grows)."""
    events: list = []
    since_seq = 0
    while True:
        page = _log.read(since_seq=since_seq, limit=_LOG_READ_PAGE)
        if not page:
            break
        events.extend(page)
        since_seq += len(page)
        if len(page) < _LOG_READ_PAGE:
            break
    return events


def _day_bounds_ms(period: str) -> tuple[int, int]:
    """Return ``[start_ms, end_ms)`` epoch-millisecond bounds for a period.

    ``period`` is one of ``"today"``, ``"yesterday"``, ``"all"``, or an
    explicit ``YYYY-MM-DD`` date (UTC calendar day).
    """
    now = time.time()
    day_s = 86400
    today_start = int(now // day_s) * day_s
    if period == "all":
        return 0, int(now * 1000) + 1
    if period == "today":
        start = today_start
    elif period == "yesterday":
        start = today_start - day_s
    else:
        struct = time.strptime(period, "%Y-%m-%d")
        start = calendar.timegm(struct)
    return start * 1000, (start + day_s) * 1000


def _register_read_tool(name: str, description: str, schema: dict[str, str]):
    """Decorator: register a fastmcp tool and its `describe` bookkeeping."""

    def deco(fn):
        _READ_TOOLS[name] = {"description": description, "schema": schema}
        return mcp.tool(name=name, description=description)(fn)

    return deco


@_register_read_tool(
    "get_queue_depth",
    description="Current kiosk queue depth (people waiting) and status (LOW/MEDIUM/HIGH).",
    schema={},
)
def get_queue_depth() -> dict[str, Any]:
    if _queue_client is None:
        return {"available": False, "reason": "queue-service disabled"}
    snapshot = _queue_client.get_queue_snapshot()
    if snapshot is None:
        return {"available": False, "reason": "queue-service unreachable"}
    return {"available": True, **snapshot}


@_register_read_tool(
    "get_queue_history",
    description=(
        "Queue-depth trend for a period (today|yesterday|all|YYYY-MM-DD): "
        "sample count, average, and max depth — answers 'is the queue always "
        "deep at this time?'."
    ),
    schema={"period": "str (default 'today')"},
)
def get_queue_history(period: str = "today") -> dict[str, Any]:
    start_ms, end_ms = _day_bounds_ms(period)
    samples = [
        e
        for e in _read_entire_log()
        if e.event_type == "queue_depth_sample" and start_ms <= e.ts_ms < end_ms
    ]
    counts = [e.payload.get("count", 0) for e in samples]
    return {
        "period": period,
        "samples": len(counts),
        "avg_count": round(sum(counts) / len(counts), 2) if counts else 0.0,
        "max_count": max(counts) if counts else 0,
    }


@_register_read_tool(
    "get_order_stats",
    description=(
        "Order activity for a period (today|yesterday|all|YYYY-MM-DD): "
        "confirmed order count, revenue, and top items."
    ),
    schema={"period": "str (default 'today')", "top_n": "int (default 5)"},
)
def get_order_stats(period: str = "today", top_n: int = 5) -> dict[str, Any]:
    date_filter, params = _order_date_filter(period)
    # Raw sqlite3 (sync) rather than the ordering module's aiosqlite
    # repository: this MCP server runs on its own thread/loop, and a
    # short-lived, read-only sync connection is the simplest way to answer a
    # reporting query without mixing async and sync tool-call contexts. WAL
    # mode (set by kiosk_core.ordering.db.init_db) allows this to read
    # concurrently with the async ordering writers.
    if not os.path.exists(cfg.KIOSK_DB_PATH):
        return {"period": period, "orders": 0, "revenue": 0.0, "top_items": []}
    conn = sqlite3.connect(cfg.KIOSK_DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        orders_row = conn.execute(
            f"SELECT COUNT(*) AS n, COALESCE(SUM(total), 0) AS revenue "
            f"FROM orders WHERE status = 'confirmed' {date_filter}",
            params,
        ).fetchone()
        top_rows = conn.execute(
            f"SELECT oi.product_id, p.name, SUM(oi.quantity) AS qty "
            f"FROM order_items oi "
            f"JOIN orders o ON o.order_id = oi.order_id "
            f"LEFT JOIN products p ON p.product_id = oi.product_id "
            f"WHERE o.status = 'confirmed' {date_filter} "
            f"GROUP BY oi.product_id ORDER BY qty DESC LIMIT ?",
            (*params, top_n),
        ).fetchall()
    finally:
        conn.close()
    return {
        "period": period,
        "orders": orders_row["n"],
        "revenue": round(orders_row["revenue"], 2),
        "top_items": [
            {"product_id": r["product_id"], "name": r["name"], "quantity": r["qty"]}
            for r in top_rows
        ],
    }


def _order_date_filter(period: str) -> tuple[str, tuple]:
    """Build a ``(sql_fragment, params)`` pair filtering ``orders.created_at``
    (a ``CURRENT_TIMESTAMP`` UTC text column) to the given period."""
    if period == "all":
        return "", ()
    if period == "today":
        return "AND date(created_at) = date('now')", ()
    if period == "yesterday":
        return "AND date(created_at) = date('now', '-1 day')", ()
    # explicit YYYY-MM-DD
    time.strptime(period, "%Y-%m-%d")  # validate, raises ValueError if malformed
    return "AND date(created_at) = date(?)", (period,)


@_register_read_tool(
    "get_board_state",
    description="Current kiosk menu-board mode (full|simplified) and when/why it last changed.",
    schema={},
)
def get_board_state() -> dict[str, Any]:
    return board_state.get()


# ---------------------------------------------------------------------------
# Act tool — the only actuator this service exposes
# ---------------------------------------------------------------------------

_SET_BOARD_MODE_DESCRIPTION = (
    "Simplify or restore the kiosk's on-screen menu board "
    "(qsr-design action #6: deep queue -> simplify board)."
)
_SET_BOARD_MODE_SCHEMA = {"mode": "str ('full'|'simplified')", "reason": "str|None"}

_policy.register(
    "set_board_mode", GateLevel.AUTOMATIC, max_calls=20, per_seconds=60.0
)
_ACT_TOOLS["set_board_mode"] = {
    "description": _SET_BOARD_MODE_DESCRIPTION,
    "schema": _SET_BOARD_MODE_SCHEMA,
    "gate": GateLevel.AUTOMATIC.value,
}


def _set_board_mode_impl(mode: str, reason: Optional[str] = None) -> dict[str, Any]:
    snapshot = board_state.set_mode(mode, reason)
    if MCP_SERVICE_ENABLED:
        try:
            emit("board_mode_changed", {"mode": mode, "reason": reason})
        except Exception:
            logger.exception("[QSR-MCP] Failed to emit board_mode_changed")
    return snapshot


@mcp.tool(
    name="set_board_mode",
    description=f"[gate={GateLevel.AUTOMATIC.value}] {_SET_BOARD_MODE_DESCRIPTION}",
)
def set_board_mode(mode: str, reason: Optional[str] = None) -> dict[str, Any]:
    """Run ``set_board_mode`` THROUGH the policy gate — the only path to
    acting, matching the old SDK's ``ServiceServer.call_action`` wire
    contract: ``{"executed", "level", "result"}`` on success or
    ``{"executed": False, "level", "reason"}`` when gated/rate-limited."""
    decision = _policy.evaluate("set_board_mode", {"mode": mode, "reason": reason})
    if not decision.allowed:
        return {
            "executed": False,
            "level": decision.level.value,
            "reason": decision.reason,
        }
    result = _set_board_mode_impl(mode, reason)
    return {"executed": True, "level": decision.level.value, "result": result}
