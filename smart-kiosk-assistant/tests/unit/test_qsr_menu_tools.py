"""Tests for the QSR menu act tools — add_menu_item / update_menu_item.

Covers the three things that make this feature safe rather than merely
present:

1. An unavailable item is hidden from listings **and** refused at order time.
2. The YAML seeder does not revert an agent's change on the next restart.
3. ``add_menu_item`` is gated, and a gated request is parked for an operator
   rather than silently dropped.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from kiosk_core.ordering.models import CreateOrderRequest, OrderItemIn, Product


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def menu_env(monkeypatch, tmp_path):
    """Isolate the catalogue DB and the menu-request store for one test."""
    monkeypatch.setenv("KIOSK_CORE_QSR_MCP_ENABLED", "false")
    monkeypatch.setenv("KIOSK_CORE_QSR_MCP_LOG_BACKEND", "memory")
    monkeypatch.setenv("KIOSK_CORE_QUEUE_SERVICE_ENABLED", "false")
    monkeypatch.setenv("KIOSK_CORE_DB_PATH", str(tmp_path / "kiosk.db"))
    monkeypatch.setenv(
        "KIOSK_CORE_QSR_MENU_REQUESTS_PATH", str(tmp_path / "menu_requests.json")
    )
    monkeypatch.setenv(
        "KIOSK_CORE_QSR_MCP_BOARD_STATE_PATH", str(tmp_path / "board.json")
    )

    # These modules read config at import time, so reload in dependency order.
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

    from kiosk_core.ordering.repository import SqliteProductRepository

    async def setup():
        await db_module.init_db()
        async with db_module.get_db() as db:
            repo = SqliteProductRepository(db)
            await repo.seed_upsert(
                Product(product_id="P1", name="Veg Burger", category="burgers", price=100.0)
            )
            await repo.seed_upsert(
                Product(product_id="P2", name="Cola", category="beverages", price=50.0)
            )
            await db.commit()

    _run(setup())
    return mcp_service


@pytest.fixture()
def ordering(menu_env):
    from kiosk_core.ordering.service import OrderingService

    return OrderingService("configs/ordering/upsell_rules.yaml")


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def test_new_products_are_available_by_default(menu_env):
    from kiosk_core.qsr_mcp import menu

    assert menu.get_item("P1")["is_available"] is True


def test_unavailable_item_is_hidden_from_listings(menu_env, ordering):
    menu_env.update_menu_item("P1", is_available=False, reason="out of stock")

    listed = [p.product_id for p in _run(ordering.list_products())]
    assert "P1" not in listed
    assert "P2" in listed


def test_unavailable_item_cannot_be_ordered(menu_env, ordering):
    """The guard that actually protects revenue.

    Hiding an item from listings is not enough — a stale cart or a direct API
    call can still carry the id.
    """
    menu_env.update_menu_item("P1", is_available=False)

    with pytest.raises(ValueError, match="unavailable"):
        _run(
            ordering.place_order(
                CreateOrderRequest(
                    user_id="u1", items=[OrderItemIn(product_id="P1", quantity=1)]
                )
            )
        )


def test_unavailable_item_does_not_resolve(menu_env, ordering):
    """Resolution must fail so the agent offers an alternative instead of
    promising an item that place_order will then refuse."""
    menu_env.update_menu_item("P1", is_available=False)

    assert _run(ordering.resolve_product_detailed("P1")).status == "NOT_FOUND"
    assert _run(ordering.resolve_product_detailed("Veg Burger")).status == "NOT_FOUND"


def test_restoring_availability_makes_the_item_orderable_again(menu_env, ordering):
    menu_env.update_menu_item("P1", is_available=False)
    menu_env.update_menu_item("P1", is_available=True)

    order = _run(
        ordering.place_order(
            CreateOrderRequest(
                user_id="u1", items=[OrderItemIn(product_id="P1", quantity=1)]
            )
        )
    )
    assert order.total == 100.0


# ---------------------------------------------------------------------------
# Durability against the YAML seeder
# ---------------------------------------------------------------------------

def test_seeder_does_not_revert_agent_changes(menu_env):
    """A restart must not silently undo an agent's price change."""
    from kiosk_core.ordering.db import get_db
    from kiosk_core.ordering.repository import SqliteProductRepository
    from kiosk_core.qsr_mcp import menu

    menu_env.update_menu_item("P1", price=175.0, is_available=False)

    async def reseed():
        async with get_db() as db:
            repo = SqliteProductRepository(db)
            written = await repo.seed_upsert(
                Product(product_id="P1", name="Veg Burger", category="burgers", price=100.0)
            )
            await db.commit()
            return written

    assert _run(reseed()) is False
    item = menu.get_item("P1")
    assert item["price"] == 175.0
    assert item["is_available"] is False


def test_seeder_still_updates_yaml_owned_products(menu_env):
    """Editing products.yaml must remain the normal operator workflow for
    any item the agent has not taken over."""
    from kiosk_core.ordering.db import get_db
    from kiosk_core.ordering.repository import SqliteProductRepository
    from kiosk_core.qsr_mcp import menu

    async def reseed():
        async with get_db() as db:
            repo = SqliteProductRepository(db)
            written = await repo.seed_upsert(
                Product(product_id="P2", name="Cola", category="beverages", price=60.0)
            )
            await db.commit()
            return written

    assert _run(reseed()) is True
    assert menu.get_item("P2")["price"] == 60.0


def test_availability_only_change_does_not_claim_the_row(menu_env):
    """86'ing an item is usually temporary. The seeder never writes
    ``is_available``, so an availability-only flip must leave the product
    under YAML control rather than permanently detaching it."""
    from kiosk_core.qsr_mcp import menu

    assert menu.get_item("P1")["source"] == "yaml"
    menu_env.update_menu_item("P1", is_available=False)

    item = menu.get_item("P1")
    assert item["is_available"] is False
    assert item["source"] == "yaml", "a stock-out must not claim the row"


def test_availability_only_change_still_survives_a_reseed(menu_env):
    """Leaving the row YAML-owned is only safe because the seeder does not
    write is_available — pin that, since the ownership rule depends on it."""
    from kiosk_core.ordering.db import get_db
    from kiosk_core.ordering.repository import SqliteProductRepository
    from kiosk_core.qsr_mcp import menu

    menu_env.update_menu_item("P1", is_available=False)

    async def reseed():
        async with get_db() as db:
            repo = SqliteProductRepository(db)
            await repo.seed_upsert(
                Product(product_id="P1", name="Veg Burger", category="burgers", price=100.0)
            )
            await db.commit()

    _run(reseed())
    assert menu.get_item("P1")["is_available"] is False


def test_seeded_field_change_does_claim_the_row(menu_env):
    """A price change would be reverted by the seeder, so it must claim."""
    from kiosk_core.qsr_mcp import menu

    menu_env.update_menu_item("P1", price=175.0)
    assert menu.get_item("P1")["source"] == "agent"


def test_release_hands_a_claimed_row_back_to_yaml(menu_env):
    """Without a release path a product is agent-owned for good, and later
    YAML edits for it are ignored forever."""
    from kiosk_core.ordering.db import get_db
    from kiosk_core.ordering.repository import SqliteProductRepository
    from kiosk_core.qsr_mcp import menu

    menu_env.update_menu_item("P1", price=175.0)
    assert menu.get_item("P1")["source"] == "agent"

    released = menu.release_item("P1")
    assert released["source"] == "yaml"

    async def reseed():
        async with get_db() as db:
            repo = SqliteProductRepository(db)
            written = await repo.seed_upsert(
                Product(product_id="P1", name="Veg Burger", category="burgers", price=100.0)
            )
            await db.commit()
            return written

    assert _run(reseed()) is True
    assert menu.get_item("P1")["price"] == 100.0


def test_releasing_an_unknown_product_is_rejected(menu_env):
    from kiosk_core.qsr_mcp import menu

    with pytest.raises(menu.MenuError, match="not found"):
        menu.release_item("NOPE")


def test_failed_update_leaves_no_open_transaction(menu_env):
    """The rollback path must release BEGIN IMMEDIATE's write lock, or every
    later menu write would block until the connection was garbage-collected."""
    from kiosk_core.qsr_mcp import menu

    with pytest.raises(menu.MenuError):
        menu.update_item("NOPE", price=5.0)

    # Would raise "database is locked" if the failed call held the lock.
    menu.update_item("P1", price=123.0)
    assert menu.get_item("P1")["price"] == 123.0


# ---------------------------------------------------------------------------
# Policy gate + approval queue
# ---------------------------------------------------------------------------

def test_add_is_gated_and_parked_for_approval(menu_env):
    from kiosk_core.qsr_mcp import menu

    result = menu_env.add_menu_item("P9", "New Wrap", "wraps", 149.0, reason="promo")

    assert result["executed"] is False
    assert result["level"] == "needs_approval"
    assert result["status"] == "pending_approval"
    assert result["request_id"]
    # Nothing was written.
    assert menu.get_item("P9") is None


def test_pending_request_is_visible_to_the_operator(menu_env):
    result = menu_env.add_menu_item("P9", "New Wrap", "wraps", 149.0)

    pending = menu_env.get_menu_requests()["requests"]
    assert [r["request_id"] for r in pending] == [result["request_id"]]
    assert pending[0]["action"] == "add_menu_item"


def test_approval_executes_the_parked_request(menu_env, ordering):
    from kiosk_core.qsr_mcp import menu

    requested = menu_env.add_menu_item("P9", "New Wrap", "wraps", 149.0)
    outcome = menu_env.approve_menu_request(requested["request_id"])

    assert outcome["executed"] is True
    assert menu.get_item("P9")["is_available"] is True
    order = _run(
        ordering.place_order(
            CreateOrderRequest(
                user_id="u1", items=[OrderItemIn(product_id="P9", quantity=1)]
            )
        )
    )
    assert order.total == 149.0


def test_approval_is_single_use(menu_env):
    """Two operators hitting approve must not create the item twice."""
    from kiosk_core.qsr_mcp import menu

    requested = menu_env.add_menu_item("P9", "New Wrap", "wraps", 149.0)
    menu_env.approve_menu_request(requested["request_id"])

    with pytest.raises(menu.MenuError, match="already"):
        menu_env.approve_menu_request(requested["request_id"])


def test_rejected_request_is_never_executed(menu_env):
    from kiosk_core.qsr_mcp import menu, menu_requests

    requested = menu_env.add_menu_item("P9", "New Wrap", "wraps", 149.0)
    menu_requests.resolve(requested["request_id"], menu_requests.STATUS_REJECTED)

    assert menu.get_item("P9") is None
    with pytest.raises(menu.MenuError):
        menu_env.approve_menu_request(requested["request_id"])


def test_update_runs_automatically(menu_env):
    result = menu_env.update_menu_item("P1", price=120.0, reason="price rise")

    assert result["executed"] is True
    assert result["level"] == "automatic"
    assert result["result"]["changed"]["price"] == {"from": 100.0, "to": 120.0}


# ---------------------------------------------------------------------------
# Validation — rejections must be reportable tool outcomes, not crashes
# ---------------------------------------------------------------------------

def test_adding_an_existing_product_is_rejected(menu_env, monkeypatch):
    """Add must not silently overwrite an existing product's price."""
    from kiosk_core.qsr_mcp.policy import GateLevel

    menu_env._policy.register("add_menu_item", GateLevel.AUTOMATIC, max_calls=50, per_seconds=60.0)
    result = menu_env.add_menu_item("P1", "Clashing", "burgers", 10.0)

    assert result["executed"] is False
    assert "already exists" in result["error"]
    # The original row is untouched.
    from kiosk_core.qsr_mcp import menu
    assert menu.get_item("P1")["price"] == 100.0


@pytest.mark.parametrize("price", [0, -5.0, "free", True])
def test_invalid_price_is_rejected(menu_env, price):
    result = menu_env.update_menu_item("P1", price=price)

    assert result["executed"] is False
    assert "price" in result["error"]


def test_updating_an_unknown_product_is_rejected(menu_env):
    result = menu_env.update_menu_item("NOPE", price=10.0)

    assert result["executed"] is False
    assert "not found" in result["error"]


def test_update_with_no_fields_is_rejected(menu_env):
    result = menu_env.update_menu_item("P1")

    assert result["executed"] is False
    assert "no fields" in result["error"]


def test_product_id_cannot_be_changed(menu_env):
    """Renaming an id would orphan historical order_items rows."""
    from kiosk_core.qsr_mcp import menu

    assert "product_id" not in menu.UPDATABLE_FIELDS
    with pytest.raises(menu.MenuError, match="unknown field"):
        menu.update_item("P1", source="agent")
