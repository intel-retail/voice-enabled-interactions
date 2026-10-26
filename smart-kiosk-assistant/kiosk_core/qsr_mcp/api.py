"""Operator REST endpoints for QSR menu approvals.

Router prefix: /api/v1/qsr
Tags: qsr-menu

``add_menu_item`` is gated ``needs_approval`` by default, so the agent's
request is parked in :mod:`kiosk_core.qsr_mcp.menu_requests` instead of being
executed. These endpoints are how a human actions that queue.

Deliberately **not** exposed as MCP tools: the agent must not be able to
approve its own gated request, which would make the gate meaningless.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from kiosk_core.qsr_mcp import mcp_service, menu, menu_requests

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/qsr", tags=["qsr-menu"])


@router.get("/menu/requests")
async def list_menu_requests(
    status: str = Query(
        "pending", pattern="^(pending|executing|approved|rejected|all)$"
    ),
) -> dict:
    """List menu changes requested by the agent.

    ``incomplete`` lists requests claimed for execution that never finished —
    an execution interrupted by a crash, or one hung past
    ``STALE_CLAIM_SECONDS``. An approval that is merely still running is not
    listed. Whether the change landed is unknown, so they need a human to
    check (then ``POST .../resolve-incomplete``) rather than an automatic
    retry.
    """
    requested = None if status == "all" else status
    response = {"status": status, "requests": menu_requests.list_requests(requested)}
    incomplete = menu_requests.stale_executing()
    if incomplete:
        response["incomplete"] = incomplete
    return response


@router.post("/menu/requests/{request_id}/approve")
async def approve_menu_request(request_id: str) -> dict:
    """Approve and execute a pending menu change."""
    try:
        return mcp_service.approve_menu_request(request_id)
    except menu.MenuError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/menu/requests/{request_id}/reject")
async def reject_menu_request(request_id: str, reason: str | None = None) -> dict:
    """Reject a pending menu change without executing it."""
    resolved = menu_requests.resolve(
        request_id, menu_requests.STATUS_REJECTED, {"reason": reason}
    )
    if resolved is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown or already-resolved request: {request_id}",
        )
    return {"request_id": request_id, "status": resolved["status"]}


@router.post("/menu/requests/{request_id}/resolve-incomplete")
async def resolve_incomplete_menu_request(
    request_id: str,
    outcome: str = Query(..., pattern="^(applied|not_applied|abandoned)$"),
    note: str | None = None,
    force: bool = False,
) -> dict:
    """Close out a request stuck in ``executing`` after a human has checked.

    ``approve`` and ``reject`` only accept *pending* requests, so without
    this an abandoned claim could be seen in ``incomplete`` but never
    cleared. Only an operator can say whether the interrupted write landed,
    which is why the outcome is an argument rather than a guess:

    - ``applied`` — the change is in the catalogue (-> approved)
    - ``not_applied`` — it never landed, hand it back to be retried
      (-> pending)
    - ``abandoned`` — give up on it (-> rejected)

    Only requests listed under ``incomplete`` are accepted. Resolving a claim
    that is still running would corrupt it — ``not_applied`` would return it
    to pending, the running write would then lose its result, and approving
    it again would apply the change twice. ``force=true`` overrides this and
    asserts the owning execution is genuinely gone.
    """
    resolved = mcp_service.resolve_incomplete_request(
        request_id, outcome, note, force=force
    )
    if resolved is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"request {request_id} is unknown, not executing, or still "
                "appears to be running (pass force=true to override)"
            ),
        )
    return {
        "request_id": request_id,
        "status": resolved["status"],
        "outcome": outcome,
        "request": resolved,
    }


@router.post("/menu/items/{product_id}/release")
async def release_menu_item(product_id: str) -> dict:
    """Hand an agent-owned product back to ``products.yaml`` control.

    A change the YAML seeder would otherwise re-apply marks the row
    ``source='agent'`` so a restart cannot silently revert it. Without a way
    back, that is permanent: edits to ``products.yaml`` would be ignored for
    that item forever. This is the release valve.

    Operator-only, like approve/reject — the agent releasing its own claim
    would let a restart undo a change it just made.
    """
    try:
        item = menu.release_item(product_id)
    except menu.MenuError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"product_id": product_id, "source": item["source"], "item": item}
