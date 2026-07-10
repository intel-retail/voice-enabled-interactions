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
    async def list_all(self, category: str | None = None) -> list[Product]:
        ...

    @abstractmethod
    async def get(self, product_id: str) -> Product | None:
        ...

    @abstractmethod
    async def upsert(self, product: Product) -> None:
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
    async def update_total(self, order_id: int) -> float:
        ...

    @abstractmethod
    async def confirm(self, order_id: int) -> None:
        ...


# ---------------------------------------------------------------------------
# SQLite implementations
# ---------------------------------------------------------------------------


class SqliteProductRepository(AbstractProductRepository):
    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def list_all(self, category: str | None = None) -> list[Product]:
        if category:
            cursor = await self._db.execute(
                "SELECT product_id, name, category, price FROM products WHERE category = ? ORDER BY name",
                (category,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT product_id, name, category, price FROM products ORDER BY category, name"
            )
        rows = await cursor.fetchall()
        return [Product(product_id=r[0], name=r[1], category=r[2], price=r[3]) for r in rows]

    async def get(self, product_id: str) -> Product | None:
        cursor = await self._db.execute(
            "SELECT product_id, name, category, price FROM products WHERE product_id = ?",
            (product_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return Product(product_id=row[0], name=row[1], category=row[2], price=row[3])

    async def upsert(self, product: Product) -> None:
        await self._db.execute(
            """
            INSERT INTO products (product_id, name, category, price)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                name     = excluded.name,
                category = excluded.category,
                price    = excluded.price
            """,
            (product.product_id, product.name, product.category, product.price),
        )


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
