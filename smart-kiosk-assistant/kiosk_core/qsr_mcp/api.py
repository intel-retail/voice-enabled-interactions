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
    only possible if the process died mid-write. Whether the change landed is
    unknown, so they need a human to check rather than an automatic retry.
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
