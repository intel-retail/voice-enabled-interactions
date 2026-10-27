"""Repository layer — abstract interfaces + SQLite implementations.

Keeps all SQL in one place so the service layer stays SQL-free.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import aiosqlite

from kiosk_core.ordering.models import (
    Order,
    OrderItem,
    OrderItemIn,
    Product,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interfaces (dependency-inversion)
# ---------------------------------------------------------------------------


class AbstractProductRepository(ABC):
    @abstractmethod
    async def list_all(
        self,
        category: str | None = None,
        is_veg: bool | None = None,
        include_unavailable: bool = False,
    ) -> list[Product]:
        ...

    @abstractmethod
    async def get(self, product_id: str) -> Product | None:
        ...

    @abstractmethod
    async def upsert(self, product: Product) -> None:
        ...

    @abstractmethod
    async def seed_upsert(self, product: Product) -> bool:
        ...

    @abstractmethod
    async def list_bestsellers(
        self,
        category: str | None = None,
        is_veg: bool | None = None,
        include_unavailable: bool = False,
    ) -> list[Product]:
        ...


class AbstractOrderRepository(ABC):
    @abstractmethod
    async def create(self, user_id: str) -> int:
        ...

    @abstractmethod
    async def get(self, order_id: int) -> Order | None:
        ...

    @abstractmethod
    async def get_current_draft(self, user_id: str) -> Order | None:
        ...

    @abstractmethod
    async def add_item(self, order_id: int, item: OrderItemIn, price: float) -> None:
        ...

    @abstractmethod
    async def remove_item(self, order_id: int, product_id: str, quantity: int | None) -> int:
        ...

    @abstractmethod
    async def update_total(self, order_id: int) -> float:
        ...

    @abstractmethod
    async def confirm(self, order_id: int) -> None:
        ...

    @abstractmethod
    async def delete_draft_orders(self, user_id: str) -> int:
        ...

    @abstractmethod
    async def consolidate_drafts(self, user_id: str) -> int | None:
        ...


# ---------------------------------------------------------------------------
# SQLite implementations
# ---------------------------------------------------------------------------


class SqliteProductRepository(AbstractProductRepository):
    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    _COLUMNS = "product_id, name, category, price, is_bestseller, is_veg, is_available"

    @staticmethod
    def _row_to_product(r) -> Product:
        return Product(
            product_id=r[0], name=r[1], category=r[2], price=r[3],
            is_bestseller=bool(r[4]), is_veg=bool(r[5]), is_available=bool(r[6]),
        )

    async def list_all(
        self,
        category: str | None = None,
        is_veg: bool | None = None,
        include_unavailable: bool = False,
    ) -> list[Product]:
        """List catalogue products.

        Args:
            include_unavailable: Unavailable items are hidden by default so
                the agent never offers something that cannot be ordered.
                Only operator/menu-management callers should pass ``True``.
        """
        clauses: list[str] = []
        params: list[object] = []
        if not include_unavailable:
            clauses.append("is_available = 1")
        if category:
            clauses.append("category = ?")
            params.append(category)
        if is_veg is not None:
            clauses.append("is_veg = ?")
            params.append(int(is_veg))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await self._db.execute(
            f"SELECT {self._COLUMNS} FROM products{where} ORDER BY category, name",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_product(r) for r in rows]

    async def get(self, product_id: str) -> Product | None:
        cursor = await self._db.execute(
            f"SELECT {self._COLUMNS} FROM products WHERE product_id = ?",
            (product_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_product(row)

    async def upsert(self, product: Product) -> None:
        await self._db.execute(
            """
            INSERT INTO products (product_id, name, category, price, is_bestseller, is_veg)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                name          = excluded.name,
                category      = excluded.category,
                price         = excluded.price,
                is_bestseller = excluded.is_bestseller,
                is_veg        = excluded.is_veg
            """,
            (
                product.product_id, product.name, product.category, product.price,
                int(product.is_bestseller), int(product.is_veg),
            ),
        )

    async def seed_upsert(self, product: Product) -> bool:
        """Upsert a product from ``products.yaml`` without clobbering the agent.

        ``products.yaml`` is the *default* catalogue and is re-applied on every
        startup. A row changed through the QSR menu tools is marked
        ``source='agent'`` and is deliberately left alone here — otherwise an
        agent's price change or an 86'd item would silently revert on the next
        restart. Rows still owned by YAML keep tracking the file, so editing
        the YAML remains the normal operator workflow.

        Returns:
            ``True`` if the row was inserted or refreshed from YAML, ``False``
            if it was skipped because the agent now owns it.
        """
        cursor = await self._db.execute(
            """
            INSERT INTO products (product_id, name, category, price, is_bestseller, is_veg, source)
            VALUES (?, ?, ?, ?, ?, ?, 'yaml')
            ON CONFLICT(product_id) DO UPDATE SET
                name          = excluded.name,
                category      = excluded.category,
                price         = excluded.price,
                is_bestseller = excluded.is_bestseller,
                is_veg        = excluded.is_veg
            WHERE products.source = 'yaml'
            """,
            (
                product.product_id, product.name, product.category, product.price,
                int(product.is_bestseller), int(product.is_veg),
            ),
        )
        return cursor.rowcount > 0

    async def list_bestsellers(
        self,
        category: str | None = None,
        is_veg: bool | None = None,
        include_unavailable: bool = False,
    ) -> list[Product]:
        """Return only the products flagged ``is_bestseller`` in the catalogue.

        This is the authoritative, hallucination-free source for "what's
        popular / most ordered / your favourites" questions — the data comes
        from the same catalogue used for prices and availability, never from
        free-text knowledge-base retrieval, so the model cannot pad the
        result with dishes that do not exist (see agentic/ordering_agent.py
        Rule 7 for how this is wired into the agent).

        Args:
            is_veg: When the customer has stated a vegetarian/vegan
                preference, restricts the result to ``is_veg = true`` items
                only, so "show me your favourites" never surfaces a
                non-vegetarian dish to someone who said they're vegetarian.
        """
        clauses = ["is_bestseller = 1"]
        params: list[object] = []
        if not include_unavailable:
            clauses.append("is_available = 1")
        if category:
            clauses.append("category = ?")
            params.append(category)
        if is_veg is not None:
            clauses.append("is_veg = ?")
            params.append(int(is_veg))
        where = " AND ".join(clauses)
        cursor = await self._db.execute(
            f"SELECT {self._COLUMNS} FROM products WHERE {where} ORDER BY category, name",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_product(r) for r in rows]



class SqliteOrderRepository(AbstractOrderRepository):
    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def create(self, user_id: str) -> int:
        # Ensure user row exists (minimal upsert)
        await self._db.execute(
            "INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,)
        )
        cursor = await self._db.execute(
            "INSERT INTO orders (user_id, status, total) VALUES (?, 'draft', 0.0)",
            (user_id,),
        )
        order_id = cursor.lastrowid
        logger.info("[ORDER-REPO] Created draft order_id=%d for user=%s", order_id, user_id)
        return order_id

    async def get(self, order_id: int) -> Order | None:
        cursor = await self._db.execute(
            "SELECT order_id, user_id, status, total, created_at FROM orders WHERE order_id = ?",
            (order_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        items_cursor = await self._db.execute(
            """
            SELECT oi.id, oi.order_id, oi.product_id, p.name, oi.quantity, oi.price
            FROM order_items oi
            JOIN products p ON oi.product_id = p.product_id
            WHERE oi.order_id = ?
            ORDER BY oi.id
            """,
            (order_id,),
        )
        item_rows = await items_cursor.fetchall()
        items = [
            OrderItem(
                id=r[0],
                order_id=r[1],
                product_id=r[2],
                product_name=r[3],
                quantity=r[4],
                price=r[5],
                subtotal=round(r[4] * r[5], 2),
            )
            for r in item_rows
        ]
        return Order(
            order_id=row[0],
            user_id=row[1],
            status=row[2],
            total=row[3],
            created_at=row[4],
            items=items,
        )

    async def get_current_draft(self, user_id: str) -> Order | None:
        cursor = await self._db.execute(
            """
            SELECT order_id
            FROM orders
            WHERE user_id = ? AND status = 'draft'
            ORDER BY order_id DESC
            LIMIT 1
            """,
            (user_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return await self.get(row[0])

    async def add_item(self, order_id: int, item: OrderItemIn, price: float) -> None:
        # Upsert: if the product is already in the order, increment quantity
        cursor = await self._db.execute(
            "SELECT id, quantity FROM order_items WHERE order_id = ? AND product_id = ?",
            (order_id, item.product_id),
        )
        existing = await cursor.fetchone()
        if existing:
            await self._db.execute(
                "UPDATE order_items SET quantity = quantity + ? WHERE id = ?",
                (item.quantity, existing[0]),
            )
            logger.debug("[ORDER-REPO] Updated qty for product=%s in order=%d", item.product_id, order_id)
        else:
            await self._db.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, price) VALUES (?, ?, ?, ?)",
                (order_id, item.product_id, item.quantity, price),
            )
            logger.debug("[ORDER-REPO] Added product=%s to order=%d", item.product_id, order_id)

    async def remove_item(self, order_id: int, product_id: str, quantity: int | None) -> int:
        """Remove a product from an order, or decrement its quantity.

        Args:
            order_id: Target order.
            product_id: Catalogue product id to remove.
            quantity: Units to remove. ``None`` removes the whole line
                regardless of quantity ("take the burger off"); an explicit
                value decrements and only deletes the line when it reaches
                zero ("remove one of the two burgers").

        Returns:
            Units actually removed; ``0`` when the product was not in the order.
        """
        cursor = await self._db.execute(
            "SELECT id, quantity FROM order_items WHERE order_id = ? AND product_id = ?",
            (order_id, product_id),
        )
        existing = await cursor.fetchone()
        if existing is None:
            logger.debug(
                "[ORDER-REPO] product=%s not present in order=%d — nothing to remove",
                product_id, order_id,
            )
            return 0

        item_id, current_qty = existing[0], existing[1]
        if quantity is None or quantity >= current_qty:
            await self._db.execute("DELETE FROM order_items WHERE id = ?", (item_id,))
            logger.info(
                "[ORDER-REPO] Removed product=%s (x%d) from order=%d",
                product_id, current_qty, order_id,
            )
            return current_qty

        await self._db.execute(
            "UPDATE order_items SET quantity = quantity - ? WHERE id = ?",
            (quantity, item_id),
        )
        logger.info(
            "[ORDER-REPO] Decremented product=%s by %d in order=%d",
            product_id, quantity, order_id,
        )
        return quantity

    async def update_total(self, order_id: int) -> float:
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(quantity * price), 0.0) FROM order_items WHERE order_id = ?",
            (order_id,),
        )
        row = await cursor.fetchone()
        total = round(row[0], 2)
        await self._db.execute(
            "UPDATE orders SET total = ? WHERE order_id = ?", (total, order_id)
        )
        return total

    async def confirm(self, order_id: int) -> None:
        await self._db.execute(
            "UPDATE orders SET status = 'confirmed' WHERE order_id = ? AND status = 'draft'",
            (order_id,),
        )
        logger.info("[ORDER-REPO] Confirmed order_id=%d", order_id)

    async def delete_draft_orders(self, user_id: str) -> int:
        """Delete every still-open ('draft') order (and its items) for a user.

        Confirmed orders are left untouched — this only clears abandoned/stale
        carts, e.g. when a fresh conversation starts.
        """
        cursor = await self._db.execute(
            "SELECT order_id FROM orders WHERE user_id = ? AND status = 'draft'",
            (user_id,),
        )
        rows = await cursor.fetchall()
        order_ids = [r[0] for r in rows]
        if not order_ids:
            return 0

        placeholders = ",".join("?" * len(order_ids))
        await self._db.execute(
            f"DELETE FROM order_items WHERE order_id IN ({placeholders})", order_ids
        )
        await self._db.execute(
            f"DELETE FROM orders WHERE order_id IN ({placeholders})", order_ids
        )
        logger.info(
            "[ORDER-REPO] Cleared %d stale draft order(s) for user=%s: %s",
            len(order_ids), user_id, order_ids,
        )
        return len(order_ids)

    async def consolidate_drafts(self, user_id: str) -> int | None:
        """Collapse every open draft order for a user into a single cart.

        A customer must only ever have one live cart. If the agent creates a
        second draft (e.g. it calls ``place_order`` again instead of
        ``update_order`` after losing the ``order_id`` from its context), the
        earlier draft is silently shadowed by ``get_current_draft`` and its
        items would never be charged. This folds all extra drafts into the
        oldest one, merging duplicate line items by quantity.

        Args:
            user_id: Owner of the carts to consolidate.

        Returns:
            The surviving draft ``order_id``, or ``None`` if the user has no
            open draft. Does not commit — the caller owns the transaction.
        """
        cursor = await self._db.execute(
            "SELECT order_id FROM orders WHERE user_id = ? AND status = 'draft' ORDER BY order_id ASC",
            (user_id,),
        )
        order_ids = [r[0] for r in await cursor.fetchall()]
        if not order_ids:
            return None

        target_id, extra_ids = order_ids[0], order_ids[1:]
        if not extra_ids:
            return target_id

        placeholders = ",".join("?" * len(extra_ids))
        cursor = await self._db.execute(
            f"SELECT product_id, quantity, price FROM order_items WHERE order_id IN ({placeholders})",
            extra_ids,
        )
        for product_id, quantity, price in await cursor.fetchall():
            await self.add_item(
                target_id, OrderItemIn(product_id=product_id, quantity=quantity), price
            )

        await self._db.execute(
            f"DELETE FROM order_items WHERE order_id IN ({placeholders})", extra_ids
        )
        await self._db.execute(
            f"DELETE FROM orders WHERE order_id IN ({placeholders})", extra_ids
        )
        await self.update_total(target_id)
        logger.warning(
            "[ORDER-REPO] Consolidated split cart for user=%s: merged %s into order_id=%d",
            user_id, extra_ids, target_id,
        )
        return target_id
