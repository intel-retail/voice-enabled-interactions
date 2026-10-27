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

**Ownership.** A row is marked ``source='agent'`` when the agent changes a
field that ``products.yaml`` would otherwise re-apply on restart (name,
category, price, is_bestseller, is_veg) — without it an agent's price change
would silently revert (see ``SqliteProductRepository.seed_upsert``). An
availability-only change does *not* claim the row, because the seeder never
writes ``is_available``; that keeps a temporary 86 from permanently detaching
the item from YAML. Use :func:`release_item` to hand a claimed row back.
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

# Fields that ``SqliteProductRepository.seed_upsert`` rewrites from
# products.yaml on every startup. Only a change to one of these needs to
# claim the row as agent-owned; note ``is_available`` is deliberately absent,
# because the seeder never writes it. That matters: 86'ing an item is usually
# temporary, and claiming the row for it would permanently detach the product
# from YAML control as a side effect of a one-shift stock-out.
_SEEDED_FIELDS = frozenset({"name", "category", "price", "is_bestseller", "is_veg"})

_COLUMNS = (
    "SELECT product_id, name, category, price, is_bestseller, is_veg, "
    "is_available, source FROM products WHERE product_id = ?"
)


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
        row = conn.execute(_COLUMNS, (product_id,)).fetchone()
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

    assignments = ", ".join(f"{k} = ?" for k in updates)
    # Only claim the row when the change is one the YAML seeder would
    # otherwise revert (see _SEEDED_FIELDS); an availability-only flip leaves
    # ownership where it is so the item can still track the file.
    claims_ownership = bool(set(updates) & _SEEDED_FIELDS)
    source_clause = ", source = 'agent'" if claims_ownership else ""

    conn = _connect()
    try:
        # One connection and one BEGIN IMMEDIATE for read-modify-read: with a
        # connection per statement, a concurrent writer could land between the
        # before-read and the after-read and be reported as this call's diff.
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(_COLUMNS, (product_id,)).fetchone()
            if row is None:
                raise MenuError(f"product not found: {product_id}")
            before = _row_to_dict(row)
            conn.execute(
                f"UPDATE products SET {assignments}{source_clause} "
                f"WHERE product_id = ?",
                (*updates.values(), product_id),
            )
            after = _row_to_dict(conn.execute(_COLUMNS, (product_id,)).fetchone())
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()

    changed = {k: {"from": before.get(k), "to": after.get(k)} for k in updates
               if before.get(k) != after.get(k)}
    logger.info("[QSR-MENU] Updated product %s: %s", product_id, changed or "no-op")
    return {"product": after, "changed": changed}


def release_item(product_id: str) -> dict[str, Any]:
    """Hand an agent-owned product back to ``products.yaml`` control.

    Without this, a row is agent-owned for good once any seeded field is
    changed, and later YAML edits for that item are ignored forever. Calling
    this restores ``source='yaml'`` so the next startup re-applies the file.

    Args:
        product_id: The product to release.

    Returns:
        The product as it stands after the release.

    Raises:
        MenuError: If the product does not exist.
    """
    product_id = _validate_text("product_id", product_id)
    conn = _connect()
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        try:
            if conn.execute(_COLUMNS, (product_id,)).fetchone() is None:
                raise MenuError(f"product not found: {product_id}")
            conn.execute(
                "UPDATE products SET source = 'yaml' WHERE product_id = ?",
                (product_id,),
            )
            after = _row_to_dict(conn.execute(_COLUMNS, (product_id,)).fetchone())
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()

    logger.info("[QSR-MENU] Released product %s back to YAML control", product_id)
    return after
