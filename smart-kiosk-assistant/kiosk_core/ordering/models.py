"""Pydantic v2 DTOs for the ordering domain."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    is_bestseller: bool = False
    is_veg: bool = True
    # False removes the item from the menu and blocks it at order time.
    is_available: bool = True


class ProductResolution(BaseModel):
    """Explicit outcome of resolving a free-form product reference.

    Returned by ``OrderingService.resolve_product_detailed`` alongside the
    simpler ``resolve_product`` (which still returns ``Product | None`` for
    existing callers). Kept separate from ``Product`` because a caller that
    wants to react differently to "ambiguous" vs "not found" — e.g. offering
    the ambiguous candidates back to the customer instead of a generic
    refusal — needs that distinction; ``Product | None`` alone collapses both
    into the same "nothing" result.

    Attributes:
        status: ``"MATCH"``, ``"AMBIGUOUS"``, or ``"NOT_FOUND"``.
        product: The resolved product when ``status == "MATCH"``, else None.
        confidence: A rough 0-1 confidence for the match. Exact/normalised-
            equality matches are 1.0; substring/token-subset matches are 0.9;
            difflib fallback matches use the actual character-similarity
            ratio. None when there is no match.
        candidates: For ``"AMBIGUOUS"``, the tied products that could not be
            distinguished. Always empty for ``"MATCH"``/``"NOT_FOUND"``.
    """

    status: Literal["MATCH", "AMBIGUOUS", "NOT_FOUND"]
    product: Product | None = None
    confidence: float | None = None
    candidates: list[Product] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Order items
# ---------------------------------------------------------------------------


class OrderItemIn(BaseModel):
    product_id: str
    quantity: int = Field(default=1, ge=1)


class RemoveOrderItem(BaseModel):
    """A cart line the customer wants taken off their order.

    Distinct from :class:`OrderItemIn` because removal has an extra state that
    addition does not: "remove all of it". ``OrderItemIn.quantity`` is
    constrained to ``>= 1``, so it cannot express that.

    Attributes:
        product_id: Resolved catalogue product id to remove.
        quantity: Units to remove, or ``None`` to remove the entire line
            regardless of how many units it holds.
    """

    product_id: str
    quantity: int | None = Field(default=None, ge=1)


class OrderItem(BaseModel):
    id: int
    order_id: int
    product_id: str
    product_name: str
    quantity: int
    price: float
    subtotal: float


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


OrderStatus = Literal["draft", "confirmed"]


class Order(BaseModel):
    order_id: int
    user_id: str
    status: OrderStatus
    total: float
    created_at: datetime
    items: list[OrderItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request / Response schemas used by REST endpoints
# ---------------------------------------------------------------------------


class CreateOrderRequest(BaseModel):
    user_id: str = "anonymous"
    items: list[OrderItemIn]


class UpdateOrderItemsRequest(BaseModel):
    items: list[OrderItemIn]


class UpsellRequest(BaseModel):
    product_ids: list[str] = Field(description="Products currently in the cart")


# ---------------------------------------------------------------------------
# Upsell
# ---------------------------------------------------------------------------


class UpsellSuggestion(BaseModel):
    product: Product
    reason: str
