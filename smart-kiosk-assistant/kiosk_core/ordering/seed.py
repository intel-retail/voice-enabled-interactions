"""Seed the SQLite database from YAML configuration files.

Loads ``products.yaml`` and upsert-inserts every product so the catalogue is
up-to-date after a restart.  This is **idempotent** — safe to call on every
startup.

``products.yaml`` supplies the *defaults*. Products that have since been
changed through the QSR menu tools are marked ``source='agent'`` and are
skipped here, so an agent's price change or an 86'd item is not silently
reverted on the next restart.

products.yaml format::

    products:
      - product_id: BURGER-001
        name: "Crispy Veg Patty Burger"
        category: burgers
        price: 129.0
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from kiosk_core.ordering.db import get_db
from kiosk_core.ordering.models import Product
from kiosk_core.ordering.repository import SqliteProductRepository

logger = logging.getLogger(__name__)


async def seed_products(products_yaml_path: str) -> int:
    """Load products from YAML and upsert into the database.

    Returns the number of products processed.
    """
    path = Path(products_yaml_path)
    if not path.exists():
        logger.warning("[SEED] products.yaml not found at %s — skipping seed", path)
        return 0

    with path.open() as fh:
        data = yaml.safe_load(fh) or {}

    raw_products = data.get("products", [])
    if not raw_products:
        logger.warning("[SEED] No products found in %s", path)
        return 0

    count = 0
    skipped = 0
    async with get_db() as db:
        repo = SqliteProductRepository(db)
        for entry in raw_products:
            product = Product(
                product_id=entry["product_id"],
                name=entry["name"],
                category=entry["category"],
                price=float(entry["price"]),
                is_bestseller=bool(entry.get("is_bestseller", False)),
                is_veg=bool(entry.get("is_veg", True)),
            )
            if await repo.seed_upsert(product):
                count += 1
            else:
                skipped += 1
        await db.commit()

    logger.info("[SEED] Seeded %d product(s) from %s", count, path)
    if skipped:
        logger.info(
            "[SEED] Left %d agent-managed product(s) untouched (source='agent')",
            skipped,
        )
    return count
