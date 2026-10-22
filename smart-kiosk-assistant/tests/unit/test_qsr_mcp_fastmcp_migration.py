"""Regression tests for the Smart Kiosk QSR MCP surface's migration off
``mcp_service_sdk`` and onto ``fastmcp`` directly.

Pins the exact tool/event contract the Central QSR Agent depends on (tool
names, descriptions, input schemas, and representative ``tools/call``
results), captured as a baseline against the pre-migration,
``mcp_service_sdk``-backed implementation before the rewrite (see
``docs/qsr-mcp.md``), plus event persistence/replay/rate-limit behaviour of
the in-tree durable log and policy gate that replaced the SDK's.

Runs the real ``fastmcp.FastMCP`` app in-process (``Client(mcp)``, no
network) against isolated temp files so it never touches a real
``qsr_mcp_events.db`` / ``qsr_board_state.json`` / ``kiosk.db``.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unittest.mock import MagicMock

sys.modules.setdefault("sounddevice", MagicMock())


@pytest.fixture
def qsr_env(monkeypatch, tmp_path):
    """Isolate the QSR MCP surface's on-disk state for one test."""
    monkeypatch.setenv("KIOSK_CORE_QSR_MCP_ENABLED", "true")
    monkeypatch.setenv(
        "KIOSK_CORE_QSR_MCP_LOG_PATH", str(tmp_path / "events.db")
    )
    monkeypatch.setenv(
        "KIOSK_CORE_QSR_MCP_BOARD_STATE_PATH", str(tmp_path / "board_state.json")
    )
    monkeypatch.setenv(
        "KIOSK_CORE_QSR_MENU_REQUESTS_PATH", str(tmp_path / "menu_requests.json")
    )
    monkeypatch.setenv("KIOSK_CORE_QUEUE_SERVICE_ENABLED", "false")
    monkeypatch.setenv("KIOSK_CORE_DB_PATH", str(tmp_path / "kiosk.db"))
    # Fresh state per test: mcp_service.py builds its log/policy/app at
    # import time from these env vars, so its module (and the singletons it
    # configures) must be *reloaded*, not just deleted-and-reimported.
    # ``from kiosk_core import config`` resolves via an attribute already
    # cached on the ``kiosk_core`` package object once anything has imported
    # it, so removing "kiosk_core.config" from sys.modules alone does not
    # force a reimport — ``importlib.reload`` re-executes each module in
    # place (updating that same object/attribute), which does.
    import kiosk_core.config as cfg_module
    importlib.reload(cfg_module)
    import kiosk_core.qsr_mcp.board_state as board_state_module
    importlib.reload(board_state_module)
    import kiosk_core.qsr_mcp.menu_requests as menu_requests_module
    importlib.reload(menu_requests_module)
    import kiosk_core.qsr_mcp.mcp_service as mcp_service
    importlib.reload(mcp_service)

    return mcp_service


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Baseline: tools/list — names, descriptions, input schemas must be
# byte-for-byte identical to the pre-migration mcp_service_sdk contract.
# ---------------------------------------------------------------------------

_EXPECTED_TOOLS = {
    "describe": {
        "description": "Describe this service's contract.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    },
    "subscribe": {
        "description": "Subscribe to an event type.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "event_type": {"type": "string"},
                "condition": {"type": "string"},
                "callback_url": {"type": "string"},
            },
            "required": ["event_type", "condition", "callback_url"],
        },
    },
    "get_queue_depth": {
        "description": (
            "Current kiosk queue depth (people waiting) and status "
            "(LOW/MEDIUM/HIGH)."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    },
    "get_queue_history": {
        "description": (
            "Queue-depth trend for a period (today|yesterday|all|YYYY-MM-DD): "
            "sample count, average, and max depth \u2014 answers 'is the queue "
            "always deep at this time?'."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"period": {"default": "today", "type": "string"}},
        },
    },
    "get_order_stats": {
        "description": (
            "Order activity for a period (today|yesterday|all|YYYY-MM-DD): "
            "confirmed order count, revenue, and top items."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "period": {"default": "today", "type": "string"},
                "top_n": {"default": 5, "type": "integer"},
            },
        },
    },
    "get_board_state": {
        "description": (
            "Current kiosk menu-board mode (full|simplified) and when/why "
            "it last changed."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    },
    "set_board_mode": {
        "description": (
            "[gate=automatic] Simplify or restore the kiosk's on-screen "
            "menu board (qsr-design action #6: deep queue -> simplify "
            "board)."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"type": "string"},
                "reason": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                },
            },
            "required": ["mode"],
        },
    },
}


# Tools added after the migration. Listed explicitly so the assertions below
# stay exact: a removal or an unintended addition still fails.
_POST_MIGRATION_TOOLS = {"add_menu_item", "update_menu_item", "get_menu_requests"}


def test_tools_list_matches_pre_migration_baseline(qsr_env):
    """`tools/list` still exposes the original 7 tools unchanged, with the
    same descriptions and input schemas as the mcp_service_sdk-backed
    server, plus the explicitly-listed menu tools added since."""
    from fastmcp import Client

    async def go():
        async with Client(qsr_env.mcp) as client:
            return await client.list_tools()

    tools = _run(go())
    actual = {t.name: {"description": t.description, "input_schema": t.input_schema} for t in tools}
    assert set(actual) == set(_EXPECTED_TOOLS) | _POST_MIGRATION_TOOLS
    for name, expected in _EXPECTED_TOOLS.items():
        assert actual[name] == expected, name


def test_describe_matches_pre_migration_baseline(qsr_env):
    """`describe()` keeps the exact event_types/read_tools/act_tools shape
    the SDK's ``ServiceServer.describe()`` produced, for every entry that
    existed then. Entries added since are asserted separately below so this
    test still fails if a pre-existing one changes or disappears."""
    from fastmcp import Client

    async def go():
        async with Client(qsr_env.mcp) as client:
            return await client.call_tool("describe", {})

    result = _run(go())
    baseline = {
        "service": "smart_kiosk",
        "store_id": "store-001",
        "event_types": {
            "queue_depth_sample": {
                "count": "int",
                "nearby": "int",
                "status": "str",
            },
            "board_mode_changed": {"mode": "str", "reason": "str|None"},
        },
        "read_tools": {
            "get_queue_depth": {
                "description": _EXPECTED_TOOLS["get_queue_depth"]["description"],
                "schema": {},
            },
            "get_queue_history": {
                "description": _EXPECTED_TOOLS["get_queue_history"]["description"],
                "schema": {"period": "str (default 'today')"},
            },
            "get_order_stats": {
                "description": _EXPECTED_TOOLS["get_order_stats"]["description"],
                "schema": {
                    "period": "str (default 'today')",
                    "top_n": "int (default 5)",
                },
            },
            "get_board_state": {
                "description": _EXPECTED_TOOLS["get_board_state"]["description"],
                "schema": {},
            },
        },
        "act_tools": {
            "set_board_mode": {
                "description": (
                    "Simplify or restore the kiosk's on-screen menu board "
                    "(qsr-design action #6: deep queue -> simplify board)."
                ),
                "schema": {
                    "mode": "str ('full'|'simplified')",
                    "reason": "str|None",
                },
                "gate": "automatic",
            }
        },
    }

    actual = result.data
    assert actual["service"] == baseline["service"]
    assert actual["store_id"] == baseline["store_id"]

    # Every pre-existing entry must be byte-identical...
    for section in ("event_types", "read_tools", "act_tools"):
        for key, expected in baseline[section].items():
            assert actual[section][key] == expected, f"{section}.{key}"

    # ...and the only additions are the documented menu ones.
    assert set(actual["read_tools"]) - set(baseline["read_tools"]) == {
        "get_menu_requests"
    }
    assert set(actual["act_tools"]) - set(baseline["act_tools"]) == {
        "add_menu_item",
        "update_menu_item",
    }
    assert set(actual["event_types"]) - set(baseline["event_types"]) == {
        "menu_item_added",
        "menu_item_updated",
        "menu_change_requested",
    }
    # add_menu_item must stay behind the approval gate by default.
    assert actual["act_tools"]["add_menu_item"]["gate"] == "needs_approval"
    assert actual["act_tools"]["update_menu_item"]["gate"] == "automatic"


def test_set_board_mode_wire_contract_and_board_state(qsr_env):
    """``set_board_mode`` keeps the SDK's `call_action` wire-level return
    shape (``executed``/``level``/``result``), and ``get_board_state``
    reflects the change."""
    from fastmcp import Client

    async def go():
        async with Client(qsr_env.mcp) as client:
            before = await client.call_tool("get_board_state", {})
            set_result = await client.call_tool(
                "set_board_mode", {"mode": "simplified", "reason": "queue deep"}
            )
            after = await client.call_tool("get_board_state", {})
            return before.data, set_result.data, after.data

    before, set_result, after = _run(go())
    assert before == {"mode": "full", "reason": None, "updated_at": None}
    assert set_result["executed"] is True
    assert set_result["level"] == "automatic"
    assert set_result["result"]["mode"] == "simplified"
    assert set_result["result"]["reason"] == "queue deep"
    assert after["mode"] == "simplified"
    assert after["reason"] == "queue deep"
    assert after["updated_at"] is not None


def test_set_board_mode_rate_limit_matches_baseline(qsr_env):
    """20 calls/60s succeed; the 21st is gated off with the SDK's exact
    rate-limit decision shape (``executed: false``, same ``level``)."""
    from fastmcp import Client

    async def go():
        async with Client(qsr_env.mcp) as client:
            results = []
            for i in range(21):
                r = await client.call_tool(
                    "set_board_mode", {"mode": "full", "reason": f"call-{i}"}
                )
                results.append(r.data)
            return results

    results = _run(go())
    assert [r["executed"] for r in results] == [True] * 20 + [False]
    assert results[-1] == {
        "executed": False,
        "level": "automatic",
        "reason": "rate limit exceeded",
    }


def test_subscribe_returns_ack(qsr_env):
    """`subscribe` keeps returning the SDK's `{"subscribed", "condition"}`
    acknowledgement (Kiosk, unlike Order Accuracy, does not strip it)."""
    from fastmcp import Client

    async def go():
        async with Client(qsr_env.mcp) as client:
            return await client.call_tool(
                "subscribe",
                {
                    "event_type": "board_mode_changed",
                    "condition": "*",
                    "callback_url": "http://example.invalid/cb",
                },
            )

    result = _run(go())
    assert result.data == {"subscribed": "board_mode_changed", "condition": "*"}


# ---------------------------------------------------------------------------
# Event persistence + replay — the durable log that replaced
# mcp_service_sdk's SQLiteLog/JSONLFileLog.
# ---------------------------------------------------------------------------


def test_sqlite_log_persistence_ordering_and_idempotent_replay(tmp_path):
    from kiosk_core.qsr_mcp.event_log import SQLiteLog, new_event

    db_path = str(tmp_path / "events.db")
    log = SQLiteLog(path=db_path, service="smart_kiosk")
    ev1 = new_event("queue_depth_sample", "smart_kiosk", "store-001", {"count": 1})
    ev2 = new_event("queue_depth_sample", "smart_kiosk", "store-001", {"count": 2})
    seq1 = log.append(ev1)
    seq2 = log.append(ev2)
    assert seq2 == seq1 + 1

    # Idempotent on ref_id: re-appending the same event is a no-op.
    assert log.append(ev1) == seq1

    all_events = log.read()
    assert [e.payload["count"] for e in all_events] == [1, 2]

    replayed = list(log.replay())
    assert [e.ref_id for e in replayed] == [ev1.ref_id, ev2.ref_id]

    # Restart-safety: a fresh SQLiteLog opened on the same path sees
    # everything already persisted (no populated DB is lost/recreated).
    reopened = SQLiteLog(path=db_path, service="smart_kiosk")
    assert len(reopened.read()) == 2


def test_jsonl_log_persistence_ordering_and_idempotent_replay(tmp_path):
    from kiosk_core.qsr_mcp.event_log import JSONLFileLog, new_event

    path = str(tmp_path / "events.jsonl")
    log = JSONLFileLog(path=path, service="smart_kiosk")
    ev1 = new_event("board_mode_changed", "smart_kiosk", "store-001", {"mode": "full"})
    ev2 = new_event(
        "board_mode_changed", "smart_kiosk", "store-001", {"mode": "simplified"}
    )
    seq1 = log.append(ev1)
    seq2 = log.append(ev2)
    assert seq2 == seq1 + 1
    assert log.append(ev1) == seq1  # idempotent

    # Restart-safety: a fresh JSONLFileLog rebuilds its seen/seq state from
    # the file on disk, so a populated log file is never silently dropped.
    reopened = JSONLFileLog(path=path, service="smart_kiosk")
    assert [e.ref_id for e in reopened.read()] == [ev1.ref_id, ev2.ref_id]
    assert list(reopened.replay()) == reopened.read()


def test_new_log_reads_pre_migration_sqlite_file_unmodified(tmp_path):
    """A durable log file produced by the pre-migration mcp_service_sdk
    SQLiteLog (same table/columns/envelope JSON shape) must still be
    readable byte-for-byte by the new in-tree SQLiteLog — no migration step
    is required, and the populated DB is never recreated."""
    import json
    import sqlite3

    from kiosk_core.qsr_mcp.event_log import SQLiteLog

    db_path = str(tmp_path / "legacy_events.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE events (
            seq          INTEGER PRIMARY KEY AUTOINCREMENT,
            ref_id       TEXT UNIQUE NOT NULL,
            event_type   TEXT NOT NULL,
            ts_ms        INTEGER NOT NULL,
            envelope     TEXT NOT NULL
        )
        """
    )
    envelope = {
        "event_type": "board_mode_changed",
        "service": "smart_kiosk",
        "store_id": "store-001",
        "payload": {"mode": "simplified", "reason": "legacy"},
        "ref_id": "legacy-ref-id",
        "ts_ms": 1700000000000,
        "schema_version": 1,
    }
    conn.execute(
        "INSERT INTO events (ref_id, event_type, ts_ms, envelope) VALUES (?, ?, ?, ?)",
        (
            envelope["ref_id"],
            envelope["event_type"],
            envelope["ts_ms"],
            json.dumps(envelope, separators=(",", ":"), sort_keys=True),
        ),
    )
    conn.commit()
    conn.close()

    log = SQLiteLog(path=db_path, service="smart_kiosk")
    rows = log.read()
    assert len(rows) == 1
    assert rows[0].ref_id == "legacy-ref-id"
    assert rows[0].payload == {"mode": "simplified", "reason": "legacy"}
    assert rows[0].ts_ms == 1700000000000


# ---------------------------------------------------------------------------
# Policy gate parity
# ---------------------------------------------------------------------------


def test_policy_gate_blocks_unregistered_action():
    from kiosk_core.qsr_mcp.policy import GateLevel, PolicyGate

    gate = PolicyGate()
    decision = gate.evaluate("not_a_real_action", {})
    assert decision.allowed is False
    assert decision.level is GateLevel.BLOCKED
    assert decision.reason == "not on allow-list"


def test_policy_gate_automatic_rate_limit():
    from kiosk_core.qsr_mcp.policy import GateLevel, PolicyGate

    gate = PolicyGate()
    gate.register("do_thing", GateLevel.AUTOMATIC, max_calls=2, per_seconds=60.0)
    d1 = gate.evaluate("do_thing", {})
    d2 = gate.evaluate("do_thing", {})
    d3 = gate.evaluate("do_thing", {})
    assert [d1.allowed, d2.allowed, d3.allowed] == [True, True, False]
    assert d3.reason == "rate limit exceeded"
