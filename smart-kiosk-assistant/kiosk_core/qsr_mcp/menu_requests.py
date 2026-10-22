"""Menu changes awaiting operator approval.

``policy.PolicyGate`` default-denies a ``NEEDS_APPROVAL`` action because no
human-approval callback is wired. For ``add_menu_item`` that alone would make
the tool a dead end: the agent's request would be refused and lost. This
module gives the refusal somewhere to go — the request is recorded durably,
surfaced to the operator (and to the agent via ``get_menu_requests``), and can
later be approved or rejected.

Durable for the same reason as ``board_state``: a request raised mid-shift
must survive a restart. Uses a lock-guarded JSON file rather than a table —
this is a short operator work queue, not relational data.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"

# Cap the stored history so a long-running kiosk cannot grow the file without
# bound; pending requests are never evicted, only resolved ones.
MAX_RESOLVED = 100

_lock = threading.RLock()
_requests: list[dict[str, Any]] = []
_path: Path | None = None


def configure(path: str) -> None:
    """Set the backing file path and load any persisted requests.

    Safe to call more than once (e.g. in tests) — reloads from ``path``.
    """
    global _path, _requests  # noqa: PLW0603
    with _lock:
        _path = Path(path)
        _requests = []
        if _path.exists():
            try:
                loaded = json.loads(_path.read_text())
                if isinstance(loaded, list):
                    _requests = loaded
            except (OSError, json.JSONDecodeError):
                logger.exception(
                    "[QSR-MENU] Could not read menu requests from %s; starting empty",
                    _path,
                )
        logger.info(
            "[QSR-MENU] Menu request store at %s (%d pending)",
            _path,
            sum(1 for r in _requests if r["status"] == STATUS_PENDING),
        )


def _persist() -> None:
    if _path is None:
        return
    try:
        _path.parent.mkdir(parents=True, exist_ok=True)
        _path.write_text(json.dumps(_requests, indent=2))
    except OSError:
        logger.exception("[QSR-MENU] Could not persist menu requests to %s", _path)


def create(action: str, payload: dict[str, Any], reason: str | None = None) -> dict[str, Any]:
    """Record a menu change awaiting approval and return it."""
    request = {
        "request_id": uuid.uuid4().hex[:12],
        "action": action,
        "payload": payload,
        "reason": reason,
        "status": STATUS_PENDING,
        "created_at": time.time(),
        "resolved_at": None,
        "result": None,
    }
    with _lock:
        _requests.append(request)
        _prune()
        _persist()
    logger.info(
        "[QSR-MENU] Recorded %s request %s awaiting approval", action, request["request_id"]
    )
    return dict(request)


def _prune() -> None:
    resolved = [r for r in _requests if r["status"] != STATUS_PENDING]
    if len(resolved) <= MAX_RESOLVED:
        return
    drop = {id(r) for r in resolved[: len(resolved) - MAX_RESOLVED]}
    _requests[:] = [r for r in _requests if id(r) not in drop]


def get(request_id: str) -> dict[str, Any] | None:
    with _lock:
        for request in _requests:
            if request["request_id"] == request_id:
                return dict(request)
    return None


def list_requests(status: str | None = STATUS_PENDING) -> list[dict[str, Any]]:
    """List requests, pending-only by default."""
    with _lock:
        return [
            dict(r) for r in _requests
            if status is None or r["status"] == status
        ]


def resolve(
    request_id: str, status: str, result: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Mark a pending request approved or rejected.

    Returns the updated request, or ``None`` if it does not exist or was
    already resolved — the caller must treat ``None`` as "do not execute",
    which is what makes approval single-use under concurrent operators.
    """
    with _lock:
        for request in _requests:
            if request["request_id"] != request_id:
                continue
            if request["status"] != STATUS_PENDING:
                return None
            request["status"] = status
            request["resolved_at"] = time.time()
            request["result"] = result
            _persist()
            return dict(request)
    return None
