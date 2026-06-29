"""SQLite database initialisation and connection management.

Uses aiosqlite for async access.  The database file path is configured via
``KIOSK_CORE_DB_PATH`` (default: ``./kiosk.db``).
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

logger = logging.getLogger(__name__)

_DB_PATH: str = os.getenv("KIOSK_CORE_DB_PATH", "./kiosk.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLES_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,
    price      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name    TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT    NOT NULL DEFAULT 'anonymous',
    status     TEXT    NOT NULL DEFAULT 'draft',
    total      REAL    NOT NULL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS order_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id   INTEGER NOT NULL,
    product_id TEXT    NOT NULL,
    quantity   INTEGER NOT NULL DEFAULT 1,
    price      REAL    NOT NULL,
    FOREIGN KEY (order_id)   REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""


async def init_db(db_path: str | None = None) -> None:
    """Create tables if they don't exist.

    Safe to call on every startup — all statements are idempotent.
    """
    path = db_path or _DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("[DB] Initialising database at %s", path)
    async with aiosqlite.connect(path) as db:
        await db.executescript(_CREATE_TABLES_SQL)
        await db.commit()
    logger.info("[DB] Schema bootstrap complete")


@asynccontextmanager
async def get_db(db_path: str | None = None) -> AsyncIterator[aiosqlite.Connection]:
    """Async context manager yielding a configured aiosqlite connection."""
    path = db_path or _DB_PATH
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        yield db


def run_init_db(db_path: str | None = None) -> None:
    """Synchronous wrapper for use in FastAPI startup events."""
    asyncio.run(init_db(db_path))
