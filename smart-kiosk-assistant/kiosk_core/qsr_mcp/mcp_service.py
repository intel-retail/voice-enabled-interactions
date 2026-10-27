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
import threading
import time
import uuid
from typing import Any, Optional

from fastmcp import FastMCP

from kiosk_core import config as cfg
from kiosk_core.queue_client import QueueClient
from kiosk_core.qsr_mcp import board_state
from kiosk_core.qsr_mcp.autonomy import AutonomyEventSink
from kiosk_core.qsr_mcp.delivery import BackgroundDispatcher, Delivery
from kiosk_core.qsr_mcp import menu, menu_requests
from kiosk_core.qsr_mcp.event_log import (
    DurableLog,
    build_log,
    new_event,
    summarise_numeric,
)
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
    "menu_item_added": {"product_id": "str", "name": "str", "category": "str", "price": "float"},
    "menu_item_updated": {"product_id": "str", "changed": "dict", "reason": "str|None"},
    "menu_change_requested": {"request_id": "str", "action": "str", "reason": "str|None"},
    "menu_change_resolved": {
        "request_id": "str",
        "action": "str",
        "outcome": "str",
        "status": "str",
        "note": "str|None",
    },
}

# -- plumbing: durable log, delivery, policy gate (self-contained, no SDK) --
#
# These are built by `init()` rather than at import time. Importing this
# module must stay side-effect free: it is imported by the test suite, by
# `main.py` before the app decides whether the feature is enabled, and by
# tooling that only wants `describe`. Building them eagerly would open a
# SQLite connection, create `qsr_mcp_events.db` on disk and construct an
# httpx connection pool as a side effect of an `import` statement.
#
# `init()` is idempotent and is called explicitly from the app's startup
# (`main.py` / `run_mcp_server`); every runtime entry point also calls
# `_ensure_init()` so a direct tool call still works without it.

_log: DurableLog | None = None
_delivery: Delivery | None = None
_autonomy_delivery: Delivery | None = None
_dispatchers: tuple[BackgroundDispatcher, ...] = ()
_queue_client: QueueClient | None = None
_initialised = False
_init_lock = threading.Lock()


def init(force: bool = False) -> None:
    """Build the service's runtime state. Idempotent.

    Reads the configuration it owns — log backend and path, delivery and
    autonomy sinks, queue client, persistence paths and policy gate levels —
    *now* rather than at import time, so tests (and a reloaded ``config``
    module) see the values they set.

    Not covered: ``MCP_SERVICE_ENABLED``, ``STORE_ID``, ``MCP_HOST``,
    ``MCP_PORT`` and ``MCP_TRANSPORT`` are module constants bound at import,
    because ``mcp_server`` imports them by value. Changing those still
    requires a module reload.

    Args:
        force: Rebuild even if already initialised, releasing the previous
            log connection and delivery workers first. Used by tests.
    """
    global _log, _delivery, _autonomy_delivery, _dispatchers
    global _queue_client, _initialised

    with _init_lock:
        if _initialised and not force:
            return
        if _initialised:
            _release_runtime_state()

        _log = build_log(
            backend=cfg.QSR_MCP_LOG_BACKEND if MCP_SERVICE_ENABLED else "memory",
            path=cfg.QSR_MCP_LOG_PATH,
            service=SERVICE_NAME,
        )
        _delivery = Delivery(webhook_url=cfg.QSR_MCP_WEBHOOK_URL)
        # Outbound autonomy push to the Central QSR Agent; built only when the
        # URL is configured, so the default stack performs no outbound
        # delivery at all.
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

        # One FIFO worker per destination: emit() stays non-blocking, while
        # delivery order (and therefore the agent's view of "latest queue
        # depth") is preserved.
        _dispatchers = tuple(
            BackgroundDispatcher(delivery, name)
            for delivery, name in (
                (_delivery, "qsr-webhook-delivery"),
                (_autonomy_delivery, "qsr-autonomy-delivery"),
            )
            if delivery is not None and delivery.sinks
        )

        board_state.configure(cfg.QSR_MCP_BOARD_STATE_PATH)
        menu_requests.configure(cfg.QSR_MENU_REQUESTS_PATH)
        _queue_client = QueueClient() if cfg.QUEUE_SERVICE_ENABLED else None
        _register_policies()

        _initialised = True


def _release_runtime_state() -> None:
    """Close the resources a previous :func:`init` opened.

    Rebuilding without this leaks an open SQLite connection, an httpx
    connection pool and a live worker thread per rebuild. Harmless once, but
    it accumulates across a test suite that rebuilds per test.
    """
    for dispatcher in _dispatchers:
        try:
            dispatcher.stop()
        except Exception:  # pragma: no cover - best-effort teardown
            logger.exception("[QSR-MCP] Failed to stop delivery worker")
    if _queue_client is not None:
        try:
            _queue_client.close()
        except Exception:  # pragma: no cover - best-effort teardown
            logger.exception("[QSR-MCP] Failed to close queue client")
    if _log is not None:
        close = getattr(_log, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # pragma: no cover - best-effort teardown
                logger.exception("[QSR-MCP] Failed to close event log")


def _ensure_init() -> None:
    """Initialise on first use, so a direct tool call works standalone."""
    if not _initialised:
        init()


def get_queue_client() -> QueueClient | None:
    """Return the shared queue-service client, or ``None`` when disabled.

    An accessor rather than a module global: importing the value directly
    would capture ``None`` from before :func:`init` ran.
    """
    _ensure_init()
    return _queue_client


_policy = PolicyGate()
_subscriptions: list[dict[str, str]] = []

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
    _ensure_init()
    assert _log is not None  # set by init()
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


@mcp.tool(
    name="subscribe",
    description=(
        "Register interest in an event type. NOT IMPLEMENTED: this kiosk "
        "does not call back to subscribers. Events are delivered by the "
        "kiosk's own autonomy push (KIOSK_CORE_QSR_AUTONOMY_EVENTS_URL) "
        "instead; the agent should rely on that or poll the read tools."
    ),
)
def subscribe(event_type: str, condition: str, callback_url: str) -> dict[str, Any]:
    """Acknowledge a subscription request without promising delivery.

    The tool is kept so the service matches the platform-wide MCP contract,
    but nothing in this service reads the registry, so reporting success
    would leave the agent waiting for callbacks that never arrive. The
    request is recorded for debugging and the no-op is stated explicitly.
    """
    _subscriptions.append(
        {"event_type": event_type, "condition": condition, "callback_url": callback_url}
    )
    logger.info(
        "[QSR-MCP] subscribe(%s) recorded but not implemented; no callback "
        "will be sent to %s",
        event_type,
        callback_url,
    )
    return {
        "subscribed": False,
        "implemented": False,
        "event_type": event_type,
        "condition": condition,
        "reason": (
            "This service does not deliver subscription callbacks. Events are "
            "pushed to the agent's /autonomy/events inbox when "
            "KIOSK_CORE_QSR_AUTONOMY_EVENTS_URL is configured; otherwise poll "
            "the read tools."
        ),
    }


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

# Upper bound for get_order_stats' top_n (see the clamp there).
_MAX_TOP_N = 50


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
    client = get_queue_client()
    if client is None:
        return {"available": False, "reason": "queue-service disabled"}
    snapshot = client.get_queue_snapshot()
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
    _ensure_init()
    assert _log is not None  # set by init()
    # Count, average and max are computed by the storage backend, so even an
    # unbounded query (period="all") costs one row rather than loading every
    # sample ever recorded onto the heap to reduce it to three numbers.
    summary = summarise_numeric(
        _log,
        "count",
        event_type="queue_depth_sample",
        since_ts_ms=start_ms,
        until_ts_ms=end_ms,
    )
    return {
        "period": period,
        "samples": summary.count,
        "avg_count": summary.average,
        "max_count": summary.maximum,
    }


@_register_read_tool(
    "get_order_stats",
    description=(
        "Order activity for a period (today|yesterday|all|YYYY-MM-DD): "
        "confirmed order count, revenue, and top items."
    ),
    schema={"period": "str (default 'today')", "top_n": "int (default 5, 1-50)"},
)
def get_order_stats(period: str = "today", top_n: int = 5) -> dict[str, Any]:
    # The agent chooses top_n, so clamp rather than trust it: an unbounded
    # (or negative) value would turn a small reporting query into a full
    # order_items scan serialised back over MCP.
    top_n = max(1, min(int(top_n), _MAX_TOP_N))
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


@_register_read_tool(
    "get_menu_requests",
    description=(
        "Menu changes awaiting operator approval, newest last. Use this to "
        "check whether a previously requested change has been actioned."
    ),
    schema={"status": "str ('pending'|'executing'|'approved'|'rejected'|'all')"},
)
def get_menu_requests(status: str = "pending") -> dict[str, Any]:
    requested = None if status == "all" else status
    response: dict[str, Any] = {
        "status": status,
        "requests": menu_requests.list_requests(requested),
    }
    # Surfaced unconditionally: a request left mid-execution by a crash is
    # invisible under any single status filter an operator would think to
    # ask for, and it is the one state that needs a human to check whether
    # the write actually landed.
    incomplete = menu_requests.stale_executing()
    if incomplete:
        response["incomplete"] = incomplete
        response["incomplete_note"] = (
            "claimed but never completed — execution was interrupted; verify "
            "whether the change was applied before approving again"
        )
    return response


# ---------------------------------------------------------------------------
# Act tool — the only actuator this service exposes
# ---------------------------------------------------------------------------

_SET_BOARD_MODE_DESCRIPTION = (
    "Simplify or restore the kiosk's on-screen menu board "
    "(qsr-design action #6: deep queue -> simplify board)."
)
_SET_BOARD_MODE_SCHEMA = {"mode": "str ('full'|'simplified')", "reason": "str|None"}

_SET_BOARD_MODE_GATE = GateLevel.AUTOMATIC
_ACT_TOOLS["set_board_mode"] = {
    "description": _SET_BOARD_MODE_DESCRIPTION,
    "schema": _SET_BOARD_MODE_SCHEMA,
    "gate": _SET_BOARD_MODE_GATE.value,
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
    _ensure_init()
    decision = _policy.evaluate("set_board_mode", {"mode": mode, "reason": reason})
    if not decision.allowed:
        return {
            "executed": False,
            "level": decision.level.value,
            "reason": decision.reason,
        }
    result = _set_board_mode_impl(mode, reason)
    return {"executed": True, "level": decision.level.value, "result": result}


# ---------------------------------------------------------------------------
# Menu act tools — add_menu_item / update_menu_item
# ---------------------------------------------------------------------------

def _gate_level(raw: str, fallback: GateLevel) -> GateLevel:
    """Parse a configured gate level, falling back on an unknown value."""
    try:
        return GateLevel(raw)
    except ValueError:
        logger.warning(
            "[QSR-MCP] Unknown gate level %r; falling back to %s", raw, fallback.value
        )
        return fallback


_ADD_MENU_ITEM_DESCRIPTION = (
    "Add a new item to the kiosk menu. Fails if the product_id already "
    "exists — use update_menu_item to change an existing item. May require "
    "operator approval; call describe() for the current gate level."
)
_ADD_MENU_ITEM_SCHEMA = {
    "product_id": "str",
    "name": "str",
    "category": "str",
    "price": "float",
    "is_veg": "bool",
    "is_bestseller": "bool",
    "reason": "str|None",
}

_UPDATE_MENU_ITEM_DESCRIPTION = (
    "Update an existing menu item's price, availability or details. Set "
    "is_available=false to take an item off the menu (it is then hidden from "
    "the menu and refused at order time). May require operator approval; "
    "call describe() for the current gate level."
)
_UPDATE_MENU_ITEM_SCHEMA = {
    "product_id": "str",
    "price": "float|None",
    "is_available": "bool|None",
    "name": "str|None",
    "category": "str|None",
    "is_bestseller": "bool|None",
    "is_veg": "bool|None",
    "reason": "str|None",
}

_ADD_GATE = _gate_level(cfg.QSR_MENU_ADD_GATE, GateLevel.NEEDS_APPROVAL)
_UPDATE_GATE = _gate_level(cfg.QSR_MENU_UPDATE_GATE, GateLevel.AUTOMATIC)


def _register_policies() -> None:
    """(Re)build the policy gate from current configuration.

    Called by :func:`init`, not at import, so a changed gate level is picked
    up — and so ``init(force=True)`` resets the rate-limit counters with it.
    The ``_ACT_TOOLS`` gate values that ``describe`` reports are refreshed
    here too, or they would advertise the import-time level.
    """
    global _policy, _ADD_GATE, _UPDATE_GATE

    _ADD_GATE = _gate_level(cfg.QSR_MENU_ADD_GATE, GateLevel.NEEDS_APPROVAL)
    _UPDATE_GATE = _gate_level(cfg.QSR_MENU_UPDATE_GATE, GateLevel.AUTOMATIC)

    gate = PolicyGate()
    gate.register(
        "set_board_mode", _SET_BOARD_MODE_GATE, max_calls=20, per_seconds=60.0
    )
    gate.register(
        "add_menu_item", _ADD_GATE,
        max_calls=cfg.QSR_MENU_MAX_CALLS, per_seconds=cfg.QSR_MENU_PER_SECONDS,
    )
    gate.register(
        "update_menu_item", _UPDATE_GATE,
        max_calls=cfg.QSR_MENU_MAX_CALLS, per_seconds=cfg.QSR_MENU_PER_SECONDS,
    )
    _policy = gate

    _ACT_TOOLS["set_board_mode"]["gate"] = _SET_BOARD_MODE_GATE.value
    _ACT_TOOLS["add_menu_item"]["gate"] = _ADD_GATE.value
    _ACT_TOOLS["update_menu_item"]["gate"] = _UPDATE_GATE.value

_ACT_TOOLS["add_menu_item"] = {
    "description": _ADD_MENU_ITEM_DESCRIPTION,
    "schema": _ADD_MENU_ITEM_SCHEMA,
    "gate": _ADD_GATE.value,
}
_ACT_TOOLS["update_menu_item"] = {
    "description": _UPDATE_MENU_ITEM_DESCRIPTION,
    "schema": _UPDATE_MENU_ITEM_SCHEMA,
    "gate": _UPDATE_GATE.value,
}


def _add_menu_item_impl(payload: dict[str, Any], reason: Optional[str] = None) -> dict[str, Any]:
    """Create the product and emit ``menu_item_added``."""
    product = menu.add_item(**payload)
    if MCP_SERVICE_ENABLED:
        try:
            emit(
                "menu_item_added",
                {
                    "product_id": product["product_id"],
                    "name": product["name"],
                    "category": product["category"],
                    "price": product["price"],
                    "reason": reason,
                },
                # Unique per emission for the same reason as the update path:
                # an id that repeats would be swallowed by the log's ref_id
                # idempotency.
                ref_id=f"{product['product_id']}:{uuid.uuid4().hex}",
            )
        except Exception:
            logger.exception("[QSR-MCP] Failed to emit menu_item_added")
    return product


def approve_menu_request(request_id: str) -> dict[str, Any]:
    """Execute a pending menu request after operator approval.

    Called by the operator REST endpoint, not by the agent — self-approval
    would defeat the gate. The request is claimed *before* the write so two
    concurrent approvals cannot both execute it.

    If the write then fails, the claim is not silently consumed: a permanent
    refusal (``menu.MenuError`` — invalid price, duplicate id) rejects the
    request for good, while any other error hands it back to pending so the
    operator can approve it again.

    The claim moves the request to ``executing``, not ``approved``: the
    latter is written only once the result is recorded, so a crash mid-write
    leaves visible evidence rather than an ``approved`` request that never
    ran.

    Raises:
        menu.MenuError: If the request is unknown, already resolved, or the
            underlying write is rejected.
    """
    request = menu_requests.get(request_id)
    if request is None:
        raise menu.MenuError(f"unknown request: {request_id}")
    if request["status"] != menu_requests.STATUS_PENDING:
        raise menu.MenuError(f"request already {request['status']}: {request_id}")

    claimed = menu_requests.claim(request_id)
    if claimed is None:
        raise menu.MenuError(f"request already resolved: {request_id}")

    try:
        if request["action"] == "add_menu_item":
            result = _add_menu_item_impl(request["payload"], request.get("reason"))
        else:
            result = _update_menu_item_impl(
                request["payload"], request.get("reason")
            )
    except menu.MenuError as exc:
        # Permanent, caller-fixable refusal (invalid price, duplicate id).
        # Retrying cannot help, so consume the approval and record why.
        if menu_requests.fail(request_id, str(exc)) is None:
            logger.warning(
                "[QSR-MENU] Could not record failure of %s: it was resolved "
                "concurrently (now %s)",
                request_id,
                _current_status(request_id),
            )
        raise
    except Exception:
        # Anything else is treated as transient (locked database, I/O error):
        # hand the claim back to pending so the operator can approve again
        # rather than silently consuming their approval.
        if menu_requests.release(request_id) is None:
            logger.warning(
                "[QSR-MENU] Could not release %s after a transient failure: "
                "it was resolved concurrently (now %s)",
                request_id,
                _current_status(request_id),
            )
        raise

    completed = menu_requests.complete(request_id, result)
    if completed is None:
        # The write happened, but the claim was resolved underneath us — an
        # operator used resolve-incomplete on it. Reporting a plain success
        # would leave the caller believing the record matches the catalogue
        # when it does not.
        status = _current_status(request_id)
        logger.warning(
            "[QSR-MENU] %s executed but its result could not be recorded: the "
            "request was resolved concurrently (now %s)",
            request_id,
            status,
        )
        return {
            "request_id": request_id,
            "executed": True,
            "result": result,
            "conflict": (
                f"the change was applied, but the request is now {status} and "
                "does not carry this result"
            ),
        }
    return {"request_id": request_id, "executed": True, "result": result}


def _current_status(request_id: str) -> str:
    """Best-effort status of a request, for a conflict message."""
    record = menu_requests.get(request_id)
    return "gone" if record is None else record["status"]


def resolve_incomplete_request(
    request_id: str,
    outcome: str,
    note: Optional[str] = None,
    force: bool = False,
) -> Optional[dict[str, Any]]:
    """Close out an abandoned execution and record that an operator did so.

    Wraps :func:`menu_requests.resolve_incomplete` so the decision reaches
    the durable log and the agent. This matters most for ``applied``: the
    catalogue changed, but the interrupted run never emitted its
    ``menu_item_added`` / ``menu_item_updated`` event, so without this the
    audit trail and the agent would never learn of a change that really
    happened.

    The emitted event is ``menu_change_resolved``, not the original
    add/update event: what the operator is attesting to is the *outcome of a
    request*, and the field-level diff an ``menu_item_updated`` carries is
    not knowable after the fact.

    Args:
        request_id: The abandoned request.
        outcome: One of ``menu_requests.RESOLVE_OUTCOMES``.
        note: Operator note recorded on the request and the event.
        force: Resolve even if the claim still looks live.

    Returns:
        The updated request, or ``None`` if it was not resolvable.

    Raises:
        ValueError: If ``outcome`` is not recognised.
    """
    resolved = menu_requests.resolve_incomplete(request_id, outcome, note, force=force)
    if resolved is None:
        return None
    if MCP_SERVICE_ENABLED:
        try:
            emit(
                "menu_change_resolved",
                {
                    "request_id": request_id,
                    "action": resolved["action"],
                    "payload": resolved["payload"],
                    "outcome": outcome,
                    "status": resolved["status"],
                    "note": note,
                },
                # Unique per resolution: the log drops a repeated ref_id, and
                # a request can legitimately be resolved more than once if an
                # earlier `not_applied` sent it back to pending.
                ref_id=f"{request_id}:resolve:{uuid.uuid4().hex}",
            )
        except Exception:
            logger.exception("[QSR-MENU] Failed to emit menu_change_resolved")
    return resolved


def _update_menu_item_impl(payload: dict[str, Any], reason: Optional[str] = None) -> dict[str, Any]:
    """Apply the update and emit ``menu_item_updated``."""
    product_id = payload["product_id"]
    fields = {k: v for k, v in payload.items() if k != "product_id"}
    outcome = menu.update_item(product_id, **fields)
    if MCP_SERVICE_ENABLED:
        try:
            emit(
                "menu_item_updated",
                {
                    "product_id": product_id,
                    "changed": outcome["changed"],
                    "reason": reason,
                },
                # A product can be updated many times, so the product id alone
                # is not a unique event identity: the log is idempotent on
                # ref_id and would silently drop every later change.
                ref_id=f"{product_id}:{uuid.uuid4().hex}",
            )
        except Exception:
            logger.exception("[QSR-MCP] Failed to emit menu_item_updated")
    return outcome


def _gated_menu_call(
    action: str, payload: dict[str, Any], reason: Optional[str], impl
) -> dict[str, Any]:
    """Run a menu act tool through the policy gate.

    A ``needs_approval`` refusal is not a dead end here: the request is
    recorded durably and returned with its ``request_id`` so an operator can
    approve it later, instead of the agent's intent being dropped.
    """
    _ensure_init()
    decision = _policy.evaluate(action, {**payload, "reason": reason})
    if not decision.allowed:
        response: dict[str, Any] = {
            "executed": False,
            "level": decision.level.value,
            "reason": decision.reason,
        }
        if decision.level is GateLevel.NEEDS_APPROVAL:
            request = menu_requests.create(action, payload, reason)
            response["request_id"] = request["request_id"]
            response["status"] = "pending_approval"
            if MCP_SERVICE_ENABLED:
                try:
                    emit(
                        "menu_change_requested",
                        {
                            "request_id": request["request_id"],
                            "action": action,
                            "payload": payload,
                            "reason": reason,
                        },
                        ref_id=request["request_id"],
                    )
                except Exception:
                    logger.exception("[QSR-MCP] Failed to emit menu_change_requested")
        return response
    try:
        result = impl(payload, reason)
    except menu.MenuError as exc:
        # A caller-fixable rejection (bad price, duplicate id) is a normal
        # tool outcome, not a transport error — report it so the agent can
        # correct itself rather than retrying an invalid call.
        return {"executed": False, "level": decision.level.value, "error": str(exc)}
    return {"executed": True, "level": decision.level.value, "result": result}


@mcp.tool(
    # No "[gate=...]" here, unlike set_board_mode: that tool's level is the
    # literal AUTOMATIC and can never drift, whereas this one is
    # configurable. A tool description is fixed when the module is imported,
    # so baking the level in would make tools/list disagree with describe()
    # after init() rebuilt the gates. describe() is the live answer.
    name="add_menu_item",
    description=_ADD_MENU_ITEM_DESCRIPTION,
)
def add_menu_item(
    product_id: str,
    name: str,
    category: str,
    price: float,
    is_veg: bool = True,
    is_bestseller: bool = False,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    """Add a new item to the kiosk menu, through the policy gate."""
    payload = {
        "product_id": product_id,
        "name": name,
        "category": category,
        "price": price,
        "is_veg": is_veg,
        "is_bestseller": is_bestseller,
    }
    return _gated_menu_call("add_menu_item", payload, reason, _add_menu_item_impl)


@mcp.tool(
    # Configurable gate — see add_menu_item above for why the level is not
    # baked into this description.
    name="update_menu_item",
    description=_UPDATE_MENU_ITEM_DESCRIPTION,
)
def update_menu_item(
    product_id: str,
    price: Optional[float] = None,
    is_available: Optional[bool] = None,
    name: Optional[str] = None,
    category: Optional[str] = None,
    is_bestseller: Optional[bool] = None,
    is_veg: Optional[bool] = None,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    """Update an existing menu item, through the policy gate."""
    payload: dict[str, Any] = {"product_id": product_id}
    for key, value in (
        ("price", price), ("is_available", is_available), ("name", name),
        ("category", category), ("is_bestseller", is_bestseller), ("is_veg", is_veg),
    ):
        if value is not None:
            payload[key] = value
    return _gated_menu_call("update_menu_item", payload, reason, _update_menu_item_impl)
