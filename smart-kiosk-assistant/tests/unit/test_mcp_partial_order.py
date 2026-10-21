"""Regression tests for partial order fulfilment in the MCP ordering tools.

A batch add that contained one fabricated product id used to be rejected in
full, so the real items the customer asked for never reached the cart and the
agent then "confirmed" an empty order. These tests pin the corrected
behaviour: valid items are always added, only the fabricated reference is
refused, and an empty cart can never be confirmed.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kiosk_core.ordering import mcp_server  # noqa: E402
from kiosk_core.ordering.models import Order, OrderItem  # noqa: E402


class _FakeProduct:
    def __init__(self, product_id: str, name: str, price: float, category: str) -> None:
        self.product_id = product_id
        self.name = name
        self.price = price
        self.category = category


_CATALOGUE = {
    "chicken_burger": _FakeProduct("chicken_burger", "Chicken Burger", 120.0, "burgers"),
    "classic_french_fries": _FakeProduct("classic_french_fries", "Classic French Fries", 80.0, "sides"),
    "cold_coffee": _FakeProduct("cold_coffee", "Cold Coffee", 90.0, "beverages"),
}


class _FakeService:
    """Minimal OrderingService stand-in recording what actually got ordered."""

    def __init__(self) -> None:
        self.placed_items: list = []
        self.current_order: Order | None = None
        self.confirmed_ids: list[int] = []
        # dry_run value seen by each mutating call, in call order.
        self.place_order_dry_runs: list[bool] = []
        self.confirm_order_dry_runs: list[bool] = []

    async def resolve_product(self, ref: str):
        return _CATALOGUE.get(ref)

    async def suggest_products(self, ref: str, n: int = 5, dietary: str | None = None, min_score: float = 0.5):
        return list(_CATALOGUE.values())

    async def list_products(self, category: str | None = None):
        # Mirrors OrderingService.list_products — used by mcp_server's
        # _resolve_category to discover real menu categories, and by the
        # ambiguous-reference path to offer choices within one.
        if category is None:
            return list(_CATALOGUE.values())
        return [p for p in _CATALOGUE.values() if p.category == category]

    async def place_order(self, req, dry_run: bool = False):
        # Mirrors OrderingService.place_order(request, dry_run=False). The
        # speculative-draft path calls every mutating tool with dry_run=True
        # so an unconfirmed ASR preview can never persist a real order; the
        # flag is recorded here so tests can assert that contract.
        self.place_order_dry_runs.append(dry_run)
        self.placed_items = list(req.items)
        items = [
            OrderItem(
                id=n,
                order_id=1,
                product_id=i.product_id,
                product_name=_CATALOGUE[i.product_id].name,
                quantity=i.quantity,
                price=_CATALOGUE[i.product_id].price,
                subtotal=_CATALOGUE[i.product_id].price * i.quantity,
            )
            for n, i in enumerate(req.items, start=1)
        ]
        from datetime import datetime, timezone

        return Order(
            order_id=1,
            user_id=req.user_id,
            status="draft",
            total=sum(i.price * i.quantity for i in items),
            created_at=datetime.now(timezone.utc),
            items=items,
        )

    async def get_current_order(self, user_id: str):
        return self.current_order

    async def confirm_order(self, order_id: int, dry_run: bool = False):
        self.confirm_order_dry_runs.append(dry_run)
        self.confirmed_ids.append(order_id)
        assert self.current_order is not None
        return self.current_order.model_copy(update={"status": "confirmed"})

    async def get_upsell_suggestions(self, *a, **k):
        return []


@pytest.fixture
def svc(monkeypatch):
    fake = _FakeService()
    monkeypatch.setattr(mcp_server, "_svc", lambda: fake)
    return fake


def _call(tool, **kwargs):
    """Invoke a FastMCP-decorated tool's underlying async function."""
    import asyncio

    fn = getattr(tool, "fn", tool)
    return asyncio.run(fn(**kwargs))


# ---------------------------------------------------------------------------
# The actual regression
# ---------------------------------------------------------------------------


def test_valid_items_are_added_even_when_one_reference_is_fabricated(svc):
    """The exact production failure: 3 real lines + 1 hallucinated id."""
    result = _call(
        mcp_server.place_order,
        user_id="kiosk-user",
        items=[
            {"product_id": "chicken_burger", "quantity": 4},
            {"product_id": "classic_french_fries", "quantity": 3},
            {"product_id": "cold_coffee", "quantity": 1},
            {"product_id": "petty_fries", "quantity": 4},
        ],
    )

    # The three real items must have reached the cart.
    ordered = {i.product_id: i.quantity for i in svc.placed_items}
    assert ordered == {
        "chicken_burger": 4,
        "classic_french_fries": 3,
        "cold_coffee": 1,
    }
    # It is a success, not an error.
    assert "error" not in result
    assert result["items"]
    # ...but the fabricated one is reported so the agent can be honest.
    assert result["unavailable_items"] == ["petty_fries"]
    assert "petty_fries" in result["unavailable_message"]


def test_all_valid_items_still_succeed_without_a_rejection_notice(svc):
    result = _call(
        mcp_server.place_order,
        user_id="kiosk-user",
        items=[{"product_id": "chicken_burger", "quantity": 2}],
    )
    assert "error" not in result
    assert "unavailable_items" not in result


def test_wholly_fabricated_batch_is_still_a_hard_error(svc):
    result = _call(
        mcp_server.place_order,
        user_id="kiosk-user",
        items=[{"product_id": "petty_fries", "quantity": 1}],
    )
    assert "error" in result
    assert svc.placed_items == []
    assert result["unavailable_items"] == ["petty_fries"]


# ---------------------------------------------------------------------------
# Empty-cart confirmation
# ---------------------------------------------------------------------------


def test_empty_cart_cannot_be_confirmed(svc):
    from datetime import datetime, timezone

    svc.current_order = Order(
        order_id=82,
        user_id="kiosk-user",
        status="draft",
        total=0.0,
        created_at=datetime.now(timezone.utc),
        items=[],
    )
    result = _call(mcp_server.confirm_active_order, user_id="kiosk-user")
    assert "error" in result
    assert svc.confirmed_ids == [], "an empty order must never be confirmed"


def test_non_empty_cart_confirms_normally(svc):
    from datetime import datetime, timezone

    svc.current_order = Order(
        order_id=82,
        user_id="kiosk-user",
        status="draft",
        total=120.0,
        created_at=datetime.now(timezone.utc),
        items=[
            OrderItem(id=1, order_id=82, product_id="chicken_burger",
                      product_name="Chicken Burger", quantity=1, price=120.0, subtotal=120.0)
        ],
    )
    result = _call(mcp_server.confirm_active_order, user_id="kiosk-user")
    assert "error" not in result
    assert svc.confirmed_ids == [82]


# ---------------------------------------------------------------------------
# Partial implausible-quantity rejection (per item, not all-or-nothing)
# ---------------------------------------------------------------------------
#
# Live regression: "one mango lassi and 100 pepsi" — a two-item place_order
# call — used to abort the ENTIRE call the instant it saw the implausible
# 100, so even the perfectly clear "one mango lassi" was silently dropped.
# ``_split_implausible_quantities`` now filters per line, mirroring the
# existing off-menu per-reference handling above.


def test_valid_quantity_item_is_added_even_when_another_lines_quantity_is_implausible(svc):
    """The exact production failure: 1 clear line + 1 absurd quantity."""
    result = _call(
        mcp_server.place_order,
        user_id="kiosk-user",
        items=[
            {"product_id": "cold_coffee", "quantity": 1},
            {"product_id": "chicken_burger", "quantity": 2001},
        ],
    )

    # The clear line must have reached the cart.
    ordered = {i.product_id: i.quantity for i in svc.placed_items}
    assert ordered == {"cold_coffee": 1}
    # It is a success, not an error.
    assert "error" not in result
    assert result["items"]
    # ...but the implausible line is reported so the agent can be honest.
    assert result["quantity_rejected_items"] == ["chicken_burger"]
    assert "chicken_burger" in result["quantity_rejected_message"]
    assert result["max_quantity"] == mcp_server.MAX_ITEM_QUANTITY


def test_wholly_implausible_batch_is_still_a_hard_error(svc):
    result = _call(
        mcp_server.place_order,
        user_id="kiosk-user",
        items=[{"product_id": "chicken_burger", "quantity": 2001}],
    )
    assert "error" in result
    assert svc.placed_items == []
    assert result["quantity_rejected_items"] == ["chicken_burger"]


def test_plausible_quantity_at_the_limit_is_not_rejected(svc):
    """The boundary itself (MAX_ITEM_QUANTITY) must still be treated as
    trusted — only quantities strictly greater than it are implausible."""
    result = _call(
        mcp_server.place_order,
        user_id="kiosk-user",
        items=[{"product_id": "chicken_burger", "quantity": mcp_server.MAX_ITEM_QUANTITY}],
    )
    assert "error" not in result
    assert "quantity_rejected_items" not in result
    ordered = {i.product_id: i.quantity for i in svc.placed_items}
    assert ordered == {"chicken_burger": mcp_server.MAX_ITEM_QUANTITY}


def test_off_menu_and_implausible_quantity_can_both_be_reported_at_once(svc):
    """Both partial-rejection shapes can occur in the same multi-item call,
    and the result must carry both — neither must silently override the
    other, since the agent-side guard discloses each independently."""
    result = _call(
        mcp_server.place_order,
        user_id="kiosk-user",
        items=[
            {"product_id": "cold_coffee", "quantity": 1},
            {"product_id": "chicken_burger", "quantity": 2001},
            {"product_id": "petty_fries", "quantity": 1},
        ],
    )
    assert "error" not in result
    ordered = {i.product_id: i.quantity for i in svc.placed_items}
    assert ordered == {"cold_coffee": 1}
    assert result["quantity_rejected_items"] == ["chicken_burger"]
    assert result["unavailable_items"] == ["petty_fries"]

