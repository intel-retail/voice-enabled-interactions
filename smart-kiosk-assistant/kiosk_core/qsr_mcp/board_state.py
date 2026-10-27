"""Kiosk on-screen menu-board mode — a small, durable actuator state.

The Central QSR Agent's act tool ``set_board_mode`` (see ``mcp_service.py``)
flips the kiosk's menu board between ``"full"`` and ``"simplified"`` (e.g. in
response to a deep queue — qsr-design §7 action #6: "Kiosk queue deep/short
-> Kiosk board simplify"). This module is the single place that owns that
state so both the MCP act tool and (future) kiosk-ui/menu endpoints agree on
the current mode.

Kept intentionally simple relative to the ordering DB: this is one row of
state, not a relational schema, so a small JSON file (guarded by a lock) is
used instead of pulling in aiosqlite for a single toggle. Durable so a
restart does not silently forget an agent-set mode mid-shift.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from kiosk_core.qsr_mcp.atomic_file import write_json_atomic

logger = logging.getLogger(__name__)

VALID_MODES = ("full", "simplified")

_lock = threading.Lock()
_state: dict[str, Any] = {"mode": "full", "reason": None, "updated_at": None}
_path: Path | None = None
_loaded = False


def configure(path: str) -> None:
    """Set the backing file path and load any persisted state.

    Safe to call more than once (e.g. in tests) — reloads from ``path``.
    """
    global _path, _state, _loaded  # noqa: PLW0603
    with _lock:
        _path = Path(path)
        _path.parent.mkdir(parents=True, exist_ok=True)
        if _path.exists():
            try:
                _state = json.loads(_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("[BoardState] Could not read %s, starting fresh: %s", _path, exc)
                _state = {"mode": "full", "reason": None, "updated_at": None}
        _loaded = True


def get() -> dict[str, Any]:
    """Return the current board state snapshot."""
    with _lock:
        return dict(_state)


def set_mode(mode: str, reason: str | None = None) -> dict[str, Any]:
    """Set the board mode. Raises ``ValueError`` for an unknown mode."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    with _lock:
        _state["mode"] = mode
        _state["reason"] = reason
        _state["updated_at"] = time.time()
        snapshot = dict(_state)
        if _path is not None:
            try:
                write_json_atomic(_path, snapshot)
            except OSError as exc:
                logger.warning("[BoardState] Could not persist state to %s: %s", _path, exc)
    return snapshot
