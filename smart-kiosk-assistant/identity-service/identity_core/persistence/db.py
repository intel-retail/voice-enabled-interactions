"""SQLite persistence for loyalty profiles.

The identity-service shares the same ``kiosk.db`` file mounted by kiosk-core.
We only own the ``loyalty_profiles`` table here; kiosk-core owns its own tables
(products / orders / users).  All statements are idempotent so both services can
bootstrap the shared database independently.

Embeddings are **never** stored here — only the FAISS index offsets
(``face_faiss_id`` / ``voice_faiss_id``) that point into the FAISS indices.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLES_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS loyalty_profiles (
    user_id        TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    favorites      TEXT NOT NULL DEFAULT '[]',   -- JSON-encoded list[str]
    restrictions   TEXT NOT NULL DEFAULT '[]',   -- JSON-encoded list[str]
    face_faiss_id  INTEGER,
    voice_faiss_id INTEGER,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_loyalty_face_faiss
    ON loyalty_profiles(face_faiss_id);
CREATE INDEX IF NOT EXISTS idx_loyalty_voice_faiss
    ON loyalty_profiles(voice_faiss_id);
"""


async def init_db(db_path: str) -> None:
    """Create the ``loyalty_profiles`` table if it does not exist.

    Safe to call on every startup — all statements are idempotent and do not
    touch tables owned by other services sharing the same database file.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("[DB] Initialising loyalty_profiles in %s", db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_CREATE_TABLES_SQL)
        await db.commit()
    logger.info("[DB] loyalty_profiles schema ready")


@asynccontextmanager
async def get_db(db_path: str) -> AsyncIterator[aiosqlite.Connection]:
    """Async context manager yielding a configured aiosqlite connection."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        yield db
