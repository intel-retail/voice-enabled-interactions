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

from kiosk_core.qsr_mcp.atomic_file import write_json_atomic

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_EXECUTING = "executing"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"

# `approved` must mean "done", not "claimed". A single state for both is
# indistinguishable after a crash between the claim and the write: the
# request would sit at `approved` with `result: None` forever, unretryable
# and with nothing to tell an operator it never actually ran. `executing` is
# the claim; `approved` is only written once the result is recorded, so a
# stale `executing` row is durable evidence of an interrupted execution.

# Cap the stored history so a long-running kiosk cannot grow the file without
# bound; pending requests are never evicted, only resolved ones.
MAX_RESOLVED = 100

# A claim held longer than this is treated as abandoned even within the same
# process run. Executing is only ever held across a single menu write, which
# is milliseconds, so anything near this is a stuck or crashed execution.
STALE_CLAIM_SECONDS = 300.0

# Identifies this process run. A request claimed by an *earlier* run can only
# be the result of an interrupted execution, which is what distinguishes a
# genuine crash from an approval that is simply still in flight.
_RUN_ID = uuid.uuid4().hex[:12]

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
        write_json_atomic(_path, _requests)
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
        "claimed_at": None,
        "claimed_by_run": None,
        "resolved_at": None,
        "executed_at": None,
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
    # `executing` is explicitly *not* prunable. A stuck executing request is
    # the only evidence that a change may have been half-applied, so evicting
    # it once 100 later requests resolve would discard exactly the record an
    # operator needs. Only genuinely finished requests are capped.
    resolved = [
        r for r in _requests
        if r["status"] not in (STATUS_PENDING, STATUS_EXECUTING)
    ]
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


def claim(request_id: str) -> dict[str, Any] | None:
    """Claim a pending request for execution (pending -> executing).

    This is the single-use gate: two concurrent operators cannot both claim
    the same request, so the underlying write happens at most once. The claim
    is deliberately *not* ``approved`` — see the status constants above.

    Args:
        request_id: The request to claim.

    Returns:
        The claimed request, or ``None`` if it does not exist or is no longer
        pending. The caller must treat ``None`` as "do not execute".
    """
    with _lock:
        for request in _requests:
            if request["request_id"] != request_id:
                continue
            if request["status"] != STATUS_PENDING:
                return None
            request["status"] = STATUS_EXECUTING
            # Provenance of the claim, so an execution that is merely still
            # running can be told apart from one that was interrupted.
            request["claimed_at"] = time.time()
            request["claimed_by_run"] = _RUN_ID
            _persist()
            return dict(request)
    return None


def release(request_id: str) -> dict[str, Any] | None:
    """Return a claimed request to pending so it can be retried.

    A claim is single-use, which is the right guarantee once the write has
    succeeded, but it would strand a request that was claimed and then failed
    for a transient reason (a locked database, an I/O error). This hands the
    claim back so an operator can approve it again.

    Args:
        request_id: The request to hand back.

    Returns:
        The updated request, or ``None`` if it is unknown or not currently
        claimed — in which case there is nothing to release.
    """
    with _lock:
        for request in _requests:
            if request["request_id"] != request_id:
                continue
            if request["status"] != STATUS_EXECUTING:
                return None
            request["status"] = STATUS_PENDING
            # Set rather than popped: every other record carries these keys,
            # so dropping them would make this one a different shape to any
            # reader or REST response.
            request["claimed_at"] = None
            request["claimed_by_run"] = None
            request["resolved_at"] = None
            request["executed_at"] = None
            request["result"] = None
            _persist()
            return dict(request)
    return None


def fail(request_id: str, error: str) -> dict[str, Any] | None:
    """Mark a claimed request rejected after a permanent failure.

    Used when the underlying write is refused for a reason that retrying
    cannot fix (an invalid price, a duplicate product id). Unlike
    :func:`release`, this is final.

    Args:
        request_id: The request to reject.
        error: The refusal reason, recorded on the request.

    Returns:
        The updated request, or ``None`` if it is unknown or not currently
        claimed.
    """
    with _lock:
        for request in _requests:
            if request["request_id"] != request_id:
                continue
            if request["status"] != STATUS_EXECUTING:
                return None
            request["status"] = STATUS_REJECTED
            request["resolved_at"] = time.time()
            request["result"] = {"error": error}
            _persist()
            return dict(request)
    return None


def complete(request_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
    """Record the outcome of a successfully executed claimed request.

    This is the only transition into ``approved``: the status and the result
    are written together, so an ``approved`` request always carries the
    outcome of the write that actually happened.

    Args:
        request_id: The claimed request.
        result: The outcome of the underlying menu write.

    Returns:
        The updated request, or ``None`` if it is unknown or not currently
        claimed.
    """
    with _lock:
        for request in _requests:
            if request["request_id"] != request_id:
                continue
            if request["status"] != STATUS_EXECUTING:
                return None
            now = time.time()
            request["status"] = STATUS_APPROVED
            request["resolved_at"] = now
            request["executed_at"] = now
            request["result"] = result
            _persist()
            return dict(request)
    return None


def stale_executing(now: float | None = None) -> list[dict[str, Any]]:
    """Requests left mid-execution, i.e. claimed but never completed.

    A claim is only held for the duration of a single menu write, so an
    executing request is almost always one that is still in flight. Reporting
    every one of them would mean any listing taken during an approval
    wrongly accused that approval of being interrupted.

    A claim is therefore only treated as abandoned when it cannot plausibly
    still be running: it was made by an **earlier process run** (so the owner
    is definitively gone), or it has been held for longer than
    :data:`STALE_CLAIM_SECONDS` (covering a thread that hung within this run).

    Whether the underlying write landed is unknown in either case, so these
    are surfaced for an operator to check rather than retried automatically.
    Use :func:`resolve_incomplete` to close one out.

    Args:
        now: Override the current time, for tests.

    Returns:
        The abandoned requests, oldest claim first.
    """
    moment = time.time() if now is None else now
    with _lock:
        stale = []
        for request in _requests:
            if request["status"] != STATUS_EXECUTING:
                continue
            claimed_at = request.get("claimed_at")
            # A record written before claims carried provenance has no way to
            # prove it is live, and a missing timestamp cannot be aged out —
            # treat both as abandoned rather than hiding them forever.
            if request.get("claimed_by_run") != _RUN_ID or claimed_at is None:
                stale.append(dict(request))
            elif moment - claimed_at >= STALE_CLAIM_SECONDS:
                stale.append(dict(request))
        return sorted(stale, key=lambda r: r.get("claimed_at") or 0.0)


# Outcomes an operator can record for an abandoned execution. The operator
# has to check the catalogue first: only they can tell whether the half-done
# write actually landed, which is why this cannot be automatic.
RESOLVE_APPLIED = "applied"
RESOLVE_NOT_APPLIED = "not_applied"
RESOLVE_ABANDONED = "abandoned"
RESOLVE_OUTCOMES = (RESOLVE_APPLIED, RESOLVE_NOT_APPLIED, RESOLVE_ABANDONED)


def resolve_incomplete(
    request_id: str, outcome: str, note: str | None = None
) -> dict[str, Any] | None:
    """Close out a request stuck in ``executing`` after a human has checked.

    Without this an abandoned claim is a dead end: ``approve`` and ``reject``
    only accept pending requests, so a stuck request could be seen in
    ``incomplete`` but never cleared, and would sit there forever.

    Args:
        request_id: The stuck request.
        outcome: One of :data:`RESOLVE_OUTCOMES`. ``applied`` records that the
            change did land (-> approved); ``not_applied`` hands it back for
            another attempt (-> pending); ``abandoned`` gives up on it
            (-> rejected).
        note: Free-text operator note recorded on the request.

    Returns:
        The updated request, or ``None`` if it is unknown or not currently
        executing.

    Raises:
        ValueError: If ``outcome`` is not a recognised outcome.
    """
    if outcome not in RESOLVE_OUTCOMES:
        raise ValueError(f"outcome must be one of {RESOLVE_OUTCOMES}, got {outcome!r}")

    detail = {"resolved_by": "operator", "outcome": outcome, "note": note}
    if outcome == RESOLVE_APPLIED:
        return complete(request_id, detail)
    if outcome == RESOLVE_ABANDONED:
        return fail(request_id, f"abandoned by operator: {note or 'no note'}")
    return release(request_id)
