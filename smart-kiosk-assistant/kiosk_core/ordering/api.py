"""REST API endpoints for the ordering feature.

Router prefix: /api/v1
Tags: ordering
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from kiosk_core.ordering.models import (
    CreateOrderRequest,
    Order,
    Product,
    UpdateOrderItemsRequest,
    UpsellRequest,
    UpsellSuggestion,
)
from kiosk_core.ordering.service import OrderingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ordering"])

# Module-level singleton injected at startup via `init_ordering_service()`
_ordering_service: OrderingService | None = None


def init_ordering_service(service: OrderingService) -> None:
    """Called from main.py lifespan to inject the service singleton."""
    global _ordering_service
    _ordering_service = service
    logger.info("[ORDERING-API] OrderingService registered")


def get_ordering_service() -> OrderingService:
    if _ordering_service is None:
        raise RuntimeError("OrderingService not initialised. Call init_ordering_service() first.")
    return _ordering_service


ServiceDep = Annotated[OrderingService, Depends(get_ordering_service)]


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


@router.get("/products", response_model=list[Product], summary="List products")
async def list_products(
    service: ServiceDep,
    category: str | None = Query(default=None, description="Filter by category"),
) -> list[Product]:
    """Return the product catalogue, optionally filtered by category."""
    logger.info("[ORDERING-API] GET /products category=%s", category)
    return await service.list_products(category=category)


@router.get("/products/{product_id}", response_model=Product, summary="Get product")
async def get_product(product_id: str, service: ServiceDep) -> Product:
    """Return a single product by its ID."""
    product = await service.get_product(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=f"Product not found: {product_id}")
    return product


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@router.post("/orders", response_model=Order, status_code=201, summary="Place order")
async def place_order(request: CreateOrderRequest, service: ServiceDep) -> Order:
    """Create a new draft order with the given items.

    Returns the created order including calculated totals.
    """
    logger.info(
        "[ORDERING-API] POST /orders user=%s items=%d",
        request.user_id,
        len(request.items),
    )
    try:
        return await service.place_order(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/orders/{order_id}", response_model=Order, summary="Get order")
async def get_order(order_id: int, service: ServiceDep) -> Order:
    """Retrieve an order including all line items and totals."""
    order = await service.get_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order not found: {order_id}")
    return order


@router.get("/users/{user_id}/orders/current", response_model=Order, summary="Get current draft order")
async def get_current_order(user_id: str, service: ServiceDep) -> Order:
    """Retrieve the latest draft order for a user."""
    logger.info("[ORDERING-API] GET /users/%s/orders/current", user_id)
    order = await service.get_current_order(user_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"No active order for user: {user_id}")
    return order


@router.patch("/orders/{order_id}/items", response_model=Order, summary="Update order items")
async def update_order_items(
    order_id: int, request: UpdateOrderItemsRequest, service: ServiceDep
) -> Order:
    """Add or increment items in an existing draft order."""
    logger.info("[ORDERING-API] PATCH /orders/%d/items count=%d", order_id, len(request.items))
    try:
        return await service.update_order_items(order_id, request.items)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/orders/{order_id}/confirm", response_model=Order, summary="Confirm order")
async def confirm_order(order_id: int, service: ServiceDep) -> Order:
    """Confirm a draft order. Returns the finalised order with its Order ID."""
    logger.info("[ORDERING-API] POST /orders/%d/confirm", order_id)
    try:
        confirmed = await service.confirm_order(order_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info("[ORDERING-API] Order ORD-%05d confirmed ✓", confirmed.order_id)
    return confirmed


# ---------------------------------------------------------------------------
# Upsell
# ---------------------------------------------------------------------------


@router.post("/upsell", response_model=list[UpsellSuggestion], summary="Get upsell suggestions")
async def get_upsell_suggestions(
    request: UpsellRequest, service: ServiceDep
) -> list[UpsellSuggestion]:
    """Return rule-based upsell/cross-sell suggestions for the given cart."""
    logger.info("[ORDERING-API] POST /upsell cart=%s", request.product_ids)
    return await service.get_upsell_suggestions(request)
