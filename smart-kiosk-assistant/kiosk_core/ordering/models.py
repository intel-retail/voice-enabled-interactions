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


# ---------------------------------------------------------------------------
# Order items
# ---------------------------------------------------------------------------


class OrderItemIn(BaseModel):
    product_id: str
    quantity: int = Field(default=1, ge=1)


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
