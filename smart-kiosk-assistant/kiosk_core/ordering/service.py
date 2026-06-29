"""OrderingService — orchestrates the product catalogue, cart, and upsell.

The service is the single entry point for all ordering business logic.
Repositories are created per-call (each call opens/closes the DB connection).
"""

from __future__ import annotations

import asyncio
import logging

from kiosk_core.ordering.db import get_db
from kiosk_core.ordering.models import (
    CreateOrderRequest,
    Order,
    OrderItemIn,
    Product,
    UpsellRequest,
    UpsellSuggestion,
)
from kiosk_core.ordering.repository import (
    SqliteOrderRepository,
    SqliteProductRepository,
)
from kiosk_core.ordering.upsell import UpsellEngine

logger = logging.getLogger(__name__)


class OrderingService:
    """Business logic for product catalogue, cart management, and upsell.

    Args:
        upsell_rules_path: Path to ``upsell_rules.yaml``.
    """

    def __init__(self, upsell_rules_path: str):
        self._upsell_engine = UpsellEngine(upsell_rules_path)

    # ------------------------------------------------------------------
    # Products
    # ------------------------------------------------------------------

    async def list_products(self, category: str | None = None) -> list[Product]:
        async with get_db() as db:
            repo = SqliteProductRepository(db)
            products = await repo.list_all(category=category)
        logger.debug("[SERVICE] list_products category=%s → %d result(s)", category, len(products))
        return products

    async def get_product(self, product_id: str) -> Product | None:
        async with get_db() as db:
            repo = SqliteProductRepository(db)
            return await repo.get(product_id)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(self, request: CreateOrderRequest) -> Order:
        """Create a new draft order with the given items."""
        async with get_db() as db:
            prod_repo = SqliteProductRepository(db)
            order_repo = SqliteOrderRepository(db)

            order_id = await order_repo.create(request.user_id)

            for item_in in request.items:
                product = await prod_repo.get(item_in.product_id)
                if product is None:
                    raise ValueError(f"Product not found: {item_in.product_id}")
                await order_repo.add_item(order_id, item_in, product.price)

            total = await order_repo.update_total(order_id)
            await db.commit()

        logger.info("[SERVICE] Placed order_id=%d user=%s total=%.2f", order_id, request.user_id, total)
        order = await self.get_order(order_id)
        return order  # type: ignore[return-value]

    async def get_order(self, order_id: int) -> Order | None:
        async with get_db() as db:
            repo = SqliteOrderRepository(db)
            return await repo.get(order_id)

    async def get_current_order(self, user_id: str) -> Order | None:
        """Return the latest draft order for a user, if one exists."""
        async with get_db() as db:
            repo = SqliteOrderRepository(db)
            order = await repo.get_current_draft(user_id)
        logger.info("[SERVICE] Retrieved current draft order for user=%s", user_id)
        return order

    async def update_order_items(self, order_id: int, items: list[OrderItemIn]) -> Order:
        """Add or increment items on an existing draft order."""
        async with get_db() as db:
            prod_repo = SqliteProductRepository(db)
            order_repo = SqliteOrderRepository(db)

            # Verify order exists and is still a draft
            order = await order_repo.get(order_id)
            if order is None:
                raise ValueError(f"Order not found: {order_id}")
            if order.status != "draft":
                raise ValueError(f"Order {order_id} is already {order.status} and cannot be modified")

            for item_in in items:
                product = await prod_repo.get(item_in.product_id)
                if product is None:
                    raise ValueError(f"Product not found: {item_in.product_id}")
                await order_repo.add_item(order_id, item_in, product.price)

            total = await order_repo.update_total(order_id)
            await db.commit()

        logger.info("[SERVICE] Updated order_id=%d new_total=%.2f", order_id, total)
        updated = await self.get_order(order_id)
        return updated  # type: ignore[return-value]

    async def confirm_order(self, order_id: int) -> Order:
        """Confirm a draft order → status becomes 'confirmed'."""
        async with get_db() as db:
            repo = SqliteOrderRepository(db)
            order = await repo.get(order_id)
            if order is None:
                raise ValueError(f"Order not found: {order_id}")
            if order.status != "draft":
                raise ValueError(f"Order {order_id} is already {order.status}")
            await repo.confirm(order_id)
            await db.commit()

        logger.info("[SERVICE] Confirmed order_id=%d for user=%s", order_id, order.user_id)
        confirmed = await self.get_order(order_id)
        return confirmed  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Upsell
    # ------------------------------------------------------------------

    async def get_upsell_suggestions(self, request: UpsellRequest) -> list[UpsellSuggestion]:
        """Return rule-based upsell suggestions for the given cart."""
        async with get_db() as db:
            prod_repo = SqliteProductRepository(db)
            all_products_list = await prod_repo.list_all()
            all_products = {p.product_id: p for p in all_products_list}

        cart_products = [all_products[pid] for pid in request.product_ids if pid in all_products]
        return self._upsell_engine.get_suggestions(
            cart_product_ids=request.product_ids,
            cart_products=cart_products,
            all_products=all_products,
        )

    # ------------------------------------------------------------------
    # Sync bridge for startup (seed)
    # ------------------------------------------------------------------

    def run_seed(self, products_yaml_path: str) -> int:
        """Synchronous wrapper to seed products from YAML on startup."""
        from kiosk_core.ordering.seed import seed_products
        return asyncio.run(seed_products(products_yaml_path))
