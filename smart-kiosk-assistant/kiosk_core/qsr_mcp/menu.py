"""Menu actuation for the QSR MCP surface — add and update catalogue items.

The Central QSR Agent's ``event-menu-decisions`` skill recommends bounded
menu changes in response to events (deep queue, weather, stock). This module
is the write path behind those decisions: it owns the SQL for creating a new
product and for updating an existing one's price/availability.

**Why raw sqlite3 and not the ordering repository.** The QSR MCP server runs
on its own thread and event loop (see ``mcp_server.py``), so it cannot await
the ordering module's aiosqlite repository from inside a sync tool call. The
same reasoning already applies to ``get_order_stats`` in ``mcp_service.py``.
WAL mode (set by ``kiosk_core.ordering.db.init_db``) makes these short-lived
writes safe alongside the async ordering writers.

**Ownership.** Every row written here is marked ``source='agent'`` so the
YAML seeder leaves it alone on the next restart — otherwise an agent's price
change or an 86'd item would silently revert (see
``SqliteProductRepository.seed_upsert``).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any, Optional

from kiosk_core import config as cfg

logger = logging.getLogger(__name__)

# Fields an update is allowed to touch. product_id is the identity and is
# deliberately immutable: order_items rows reference it, so renaming an id
# would orphan historical orders.
UPDATABLE_FIELDS = ("name", "category", "price", "is_available", "is_bestseller", "is_veg")


class MenuError(ValueError):
    """Raised when a menu write is rejected for a caller-fixable reason."""


def _connect() -> sqlite3.Connection:
    if not os.path.exists(cfg.KIOSK_DB_PATH):
        raise MenuError("catalogue database is not initialised yet")
    conn = sqlite3.connect(cfg.KIOSK_DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "product_id": row["product_id"],
        "name": row["name"],
        "category": row["category"],
        "price": row["price"],
        "is_bestseller": bool(row["is_bestseller"]),
        "is_veg": bool(row["is_veg"]),
        "is_available": bool(row["is_available"]),
        "source": row["source"],
    }


def get_item(product_id: str) -> dict[str, Any] | None:
    """Return one catalogue item, available or not, or ``None``."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT product_id, name, category, price, is_bestseller, is_veg, "
            "is_available, source FROM products WHERE product_id = ?",
            (product_id,),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_dict(row) if row else None


def _validate_price(price: Any) -> float:
    # bool is an int subclass; True would otherwise become a 1.0 price.
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        raise MenuError("price must be a number")
    value = float(price)
    if value <= 0:
        raise MenuError("price must be greater than 0")
    return value


def _validate_text(field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MenuError(f"{field} must be a non-empty string")
    return value.strip()


def add_item(
    product_id: str,
    name: str,
    category: str,
    price: float,
    is_veg: bool = True,
    is_bestseller: bool = False,
    is_available: bool = True,
) -> dict[str, Any]:
    """Create a new catalogue item.

    Raises:
        MenuError: If a field is invalid or ``product_id`` already exists.
            Adding is deliberately not an upsert — silently overwriting an
            existing product's price under an "add" call is how a catalogue
            gets corrupted. Use :func:`update_item` to change one.
    """
    product_id = _validate_text("product_id", product_id)
    name = _validate_text("name", name)
    category = _validate_text("category", category)
    price = _validate_price(price)

    conn = _connect()
    try:
        existing = conn.execute(
            "SELECT 1 FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
        if existing:
            raise MenuError(f"product already exists: {product_id}")
        conn.execute(
            "INSERT INTO products (product_id, name, category, price, "
            "is_bestseller, is_veg, is_available, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'agent')",
            (
                product_id, name, category, price,
                int(bool(is_bestseller)), int(bool(is_veg)), int(bool(is_available)),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "[QSR-MENU] Added product %s (%s, %s, %.2f)", product_id, name, category, price
    )
    return get_item(product_id)  # type: ignore[return-value]


def update_item(product_id: str, **fields: Any) -> dict[str, Any]:
    """Update an existing catalogue item in place.

    Only :data:`UPDATABLE_FIELDS` may be set; ``None`` values are treated as
    "leave unchanged" so a caller can pass a sparse update.

    Raises:
        MenuError: If the product does not exist, an unknown field is given,
            or no field was actually supplied.
    """
    product_id = _validate_text("product_id", product_id)

    unknown = set(fields) - set(UPDATABLE_FIELDS)
    if unknown:
        raise MenuError(
            f"unknown field(s): {', '.join(sorted(unknown))}; "
            f"updatable: {', '.join(UPDATABLE_FIELDS)}"
        )

    updates = {k: v for k, v in fields.items() if v is not None}
    if not updates:
        raise MenuError("no fields to update")

    if "price" in updates:
        updates["price"] = _validate_price(updates["price"])
    for text_field in ("name", "category"):
        if text_field in updates:
            updates[text_field] = _validate_text(text_field, updates[text_field])
    for flag in ("is_available", "is_bestseller", "is_veg"):
        if flag in updates:
            updates[flag] = int(bool(updates[flag]))

    before = get_item(product_id)
    if before is None:
        raise MenuError(f"product not found: {product_id}")

    assignments = ", ".join(f"{k} = ?" for k in updates)
    conn = _connect()
    try:
        # source='agent' marks the row as agent-owned so the YAML seeder
        # stops overwriting it on restart.
        conn.execute(
            f"UPDATE products SET {assignments}, source = 'agent' WHERE product_id = ?",
            (*updates.values(), product_id),
        )
        conn.commit()
    finally:
        conn.close()

    after = get_item(product_id)
    changed = {k: {"from": before.get(k), "to": after.get(k)} for k in updates
               if before.get(k) != after.get(k)}
    logger.info("[QSR-MENU] Updated product %s: %s", product_id, changed or "no-op")
    return {"product": after, "changed": changed}
