"""Operator REST surface for QSR menu approvals and YAML release.

These endpoints are the only way a human actions the approval queue or hands
an agent-claimed product back to ``products.yaml``, so they are covered
through the router rather than by calling the functions directly.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiosk_core.ordering.models import Product


@pytest.fixture()
def api_env(monkeypatch, tmp_path):
    monkeypatch.setenv("KIOSK_CORE_QSR_MCP_ENABLED", "true")
    monkeypatch.setenv("KIOSK_CORE_QSR_MCP_LOG_BACKEND", "memory")
    monkeypatch.setenv("KIOSK_CORE_QUEUE_SERVICE_ENABLED", "false")
    monkeypatch.setenv(
        "KIOSK_CORE_QSR_MENU_REQUESTS_PATH", str(tmp_path / "requests.json")
    )
    monkeypatch.setenv(
        "KIOSK_CORE_QSR_MCP_BOARD_STATE_PATH", str(tmp_path / "board.json")
    )
    monkeypatch.setenv("KIOSK_CORE_DB_PATH", str(tmp_path / "kiosk.db"))

    import kiosk_core.config as cfg_module
    importlib.reload(cfg_module)
    import kiosk_core.ordering.db as db_module
    importlib.reload(db_module)
    import kiosk_core.qsr_mcp.menu as menu_module
    importlib.reload(menu_module)
    import kiosk_core.qsr_mcp.menu_requests as requests_module
    importlib.reload(requests_module)
    import kiosk_core.qsr_mcp.board_state as board_module
    importlib.reload(board_module)
    import kiosk_core.qsr_mcp.mcp_service as mcp_service
    importlib.reload(mcp_service)
    mcp_service.init(force=True)
    import kiosk_core.qsr_mcp.api as api_module
    importlib.reload(api_module)

    from kiosk_core.ordering.repository import SqliteProductRepository

    async def setup():
        await db_module.init_db()
        async with db_module.get_db() as db:
            await SqliteProductRepository(db).seed_upsert(
                Product(
                    product_id="P1",
                    name="Veg Burger",
                    category="burgers",
                    price=100.0,
                )
            )
            await db.commit()

    asyncio.run(setup())

    app = FastAPI()
    app.include_router(api_module.router)

    class Env:
        client = TestClient(app)
        service = mcp_service
        menu = menu_module
        requests = requests_module

    return Env()


# ---------------------------------------------------------------------------
# Releasing a claimed product back to YAML control
# ---------------------------------------------------------------------------

def test_release_endpoint_hands_the_row_back_to_yaml(api_env):
    """Without a reachable release path, one price change makes an item
    ignore products.yaml forever."""
    api_env.service.update_menu_item("P1", price=175.0)
    assert api_env.menu.get_item("P1")["source"] == "agent"

    response = api_env.client.post("/api/v1/qsr/menu/items/P1/release")

    assert response.status_code == 200, response.text
    assert response.json()["source"] == "yaml"
    assert api_env.menu.get_item("P1")["source"] == "yaml"


def test_releasing_an_unknown_product_is_404(api_env):
    response = api_env.client.post("/api/v1/qsr/menu/items/NOPE/release")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Approval queue
# ---------------------------------------------------------------------------

def test_interrupted_execution_is_surfaced_as_incomplete(api_env):
    """A request claimed but never completed is only possible if the process
    died mid-write. It must be visible, or an operator cannot tell that the
    change never ran."""
    request = api_env.requests.create(
        "update_menu_item", {"product_id": "P1", "price": 9.0}, None
    )
    api_env.requests.claim(request["request_id"])  # simulate a crash after the claim

    body = api_env.client.get("/api/v1/qsr/menu/requests").json()

    assert body["incomplete"][0]["request_id"] == request["request_id"]
    assert body["incomplete"][0]["status"] == "executing"


def test_completed_approval_is_not_reported_as_incomplete(api_env):
    request = api_env.requests.create(
        "update_menu_item", {"product_id": "P1", "price": 9.0}, None
    )
    response = api_env.client.post(
        f"/api/v1/qsr/menu/requests/{request['request_id']}/approve"
    )

    assert response.status_code == 200, response.text
    body = api_env.client.get("/api/v1/qsr/menu/requests?status=approved").json()
    assert "incomplete" not in body
    approved = body["requests"][0]
    assert approved["status"] == "approved"
    assert approved["result"] is not None, "approved must always carry its result"
    assert approved["executed_at"] is not None


def test_executing_is_an_accepted_status_filter(api_env):
    assert api_env.client.get("/api/v1/qsr/menu/requests?status=executing").status_code == 200


def test_approving_an_unknown_request_is_400(api_env):
    assert api_env.client.post("/api/v1/qsr/menu/requests/nope/approve").status_code == 400
