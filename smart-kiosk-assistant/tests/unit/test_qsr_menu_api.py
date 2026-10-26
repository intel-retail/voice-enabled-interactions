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

def _crash_after_claim(api_env, **payload):
    """Claim a request, then make the claim look like it came from a process
    run that has since died."""
    request = api_env.requests.create("update_menu_item", payload, None)
    api_env.requests.claim(request["request_id"])
    for record in api_env.requests._requests:
        if record["request_id"] == request["request_id"]:
            record["claimed_by_run"] = "a-previous-run"
    return request


def test_interrupted_execution_is_surfaced_as_incomplete(api_env):
    """A request claimed by a run that is now gone was interrupted mid-write.
    It must be visible, or an operator cannot tell the change never ran."""
    request = _crash_after_claim(api_env, product_id="P1", price=9.0)

    body = api_env.client.get("/api/v1/qsr/menu/requests").json()

    assert body["incomplete"][0]["request_id"] == request["request_id"]
    assert body["incomplete"][0]["status"] == "executing"


def test_an_approval_still_running_is_not_called_incomplete(api_env):
    """stale_executing() is consulted on every listing, so reporting all
    executing requests would accuse a live approval of having crashed."""
    request = api_env.requests.create(
        "update_menu_item", {"product_id": "P1", "price": 9.0}, None
    )
    api_env.requests.claim(request["request_id"])  # this run, just now

    body = api_env.client.get("/api/v1/qsr/menu/requests").json()

    assert "incomplete" not in body


def test_a_claim_held_too_long_is_incomplete_even_in_this_run(api_env):
    """A hung thread never ends its run, so the run id alone is not enough."""
    request = api_env.requests.create(
        "update_menu_item", {"product_id": "P1", "price": 9.0}, None
    )
    claimed = api_env.requests.claim(request["request_id"])

    stale = api_env.requests.stale_executing(
        now=claimed["claimed_at"] + api_env.requests.STALE_CLAIM_SECONDS + 1
    )

    assert [r["request_id"] for r in stale] == [request["request_id"]]


# ---------------------------------------------------------------------------
# Closing out an interrupted execution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "outcome,expected_status",
    [("applied", "approved"), ("not_applied", "pending"), ("abandoned", "rejected")],
)
def test_operator_can_close_out_an_incomplete_request(
    api_env, outcome, expected_status
):
    """approve/reject only accept pending requests, so without this route a
    stuck request could be seen forever but never cleared."""
    request = _crash_after_claim(api_env, product_id="P1", price=9.0)

    response = api_env.client.post(
        f"/api/v1/qsr/menu/requests/{request['request_id']}/resolve-incomplete"
        f"?outcome={outcome}&note=checked+the+catalogue"
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == expected_status
    assert api_env.client.get("/api/v1/qsr/menu/requests?status=all").json().get(
        "incomplete"
    ) is None


def test_not_applied_hands_the_request_back_for_another_attempt(api_env):
    request = _crash_after_claim(api_env, product_id="P1", price=9.0)
    api_env.client.post(
        f"/api/v1/qsr/menu/requests/{request['request_id']}"
        "/resolve-incomplete?outcome=not_applied"
    )

    approved = api_env.client.post(
        f"/api/v1/qsr/menu/requests/{request['request_id']}/approve"
    )

    assert approved.status_code == 200, "a released request must be approvable again"


def test_resolve_incomplete_rejects_an_unknown_outcome(api_env):
    request = _crash_after_claim(api_env, product_id="P1", price=9.0)
    response = api_env.client.post(
        f"/api/v1/qsr/menu/requests/{request['request_id']}"
        "/resolve-incomplete?outcome=probably"
    )
    assert response.status_code == 422


def test_resolve_incomplete_refuses_a_request_that_is_not_executing(api_env):
    request = api_env.requests.create("update_menu_item", {"product_id": "P1"}, None)
    response = api_env.client.post(
        f"/api/v1/qsr/menu/requests/{request['request_id']}"
        "/resolve-incomplete?outcome=applied"
    )
    assert response.status_code == 409


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


def test_a_live_claim_cannot_be_resolved(api_env):
    """Resolving a running claim corrupts it: not_applied returns it to
    pending, the running write then loses its result, and approving again
    applies the change twice."""
    request = api_env.requests.create(
        "update_menu_item", {"product_id": "P1", "price": 9.0}, None
    )
    api_env.requests.claim(request["request_id"])  # this run, still running

    response = api_env.client.post(
        f"/api/v1/qsr/menu/requests/{request['request_id']}"
        "/resolve-incomplete?outcome=not_applied"
    )

    assert response.status_code == 409
    assert api_env.requests.get(request["request_id"])["status"] == "executing"


def test_force_overrides_the_live_claim_check(api_env):
    request = api_env.requests.create(
        "update_menu_item", {"product_id": "P1", "price": 9.0}, None
    )
    api_env.requests.claim(request["request_id"])

    response = api_env.client.post(
        f"/api/v1/qsr/menu/requests/{request['request_id']}"
        "/resolve-incomplete?outcome=abandoned&force=true"
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "rejected"


def test_resolving_as_applied_is_recorded_in_the_audit_log(api_env):
    """The interrupted run never emitted its menu_item_* event, so without
    this the log and the agent never learn of a change that did happen."""
    request = _crash_after_claim(api_env, product_id="P1", price=9.0)

    api_env.client.post(
        f"/api/v1/qsr/menu/requests/{request['request_id']}"
        "/resolve-incomplete?outcome=applied&note=verified"
    )

    events = api_env.service._log.read(event_type="menu_change_resolved")
    assert len(events) == 1
    payload = events[0].payload
    assert payload["request_id"] == request["request_id"]
    assert payload["outcome"] == "applied"
    assert payload["status"] == "approved"
    assert payload["note"] == "verified"


def test_each_resolution_is_logged_separately(api_env):
    """A request sent back by not_applied can be resolved again; a reused
    ref_id would make the log drop the second one."""
    request = _crash_after_claim(api_env, product_id="P1", price=9.0)
    base = f"/api/v1/qsr/menu/requests/{request['request_id']}/resolve-incomplete"

    api_env.client.post(f"{base}?outcome=not_applied")
    api_env.requests.claim(request["request_id"])
    for record in api_env.requests._requests:
        if record["request_id"] == request["request_id"]:
            record["claimed_by_run"] = "a-previous-run"
    api_env.client.post(f"{base}?outcome=applied")

    events = api_env.service._log.read(event_type="menu_change_resolved")
    assert [e.payload["outcome"] for e in events] == ["not_applied", "applied"]


def test_approve_reports_a_conflict_when_its_result_cannot_be_recorded(api_env):
    """If an operator resolves the claim mid-write, the write still happened.
    Reporting a plain success would hide that the record disagrees."""
    request = api_env.requests.create(
        "update_menu_item", {"product_id": "P1", "price": 150.0}, None
    )
    real_update = api_env.service._update_menu_item_impl

    def steal_the_claim(payload, reason=None):
        result = real_update(payload, reason)
        # Simulate an operator resolving it while the write was in flight.
        api_env.requests.resolve_incomplete(
            request["request_id"], "abandoned", force=True
        )
        return result

    api_env.service._update_menu_item_impl = steal_the_claim
    try:
        response = api_env.service.approve_menu_request(request["request_id"])
    finally:
        api_env.service._update_menu_item_impl = real_update

    assert response["executed"] is True
    assert "conflict" in response, "a silent success would mislead the operator"
    assert api_env.requests.get(request["request_id"])["status"] == "rejected"
