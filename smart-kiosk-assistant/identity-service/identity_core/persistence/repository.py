"""Repository layer for loyalty profiles.

Keeps all SQL in one place behind an abstract interface so the service layer
stays SQL-free.  ``favorites`` and ``restrictions`` are persisted as JSON text
columns and (de)serialised here.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class ProfileRecord:
    """A persisted loyalty profile (metadata + FAISS offsets, no embeddings)."""

    user_id: str
    name: str
    favorites: list[str] = field(default_factory=list)
    restrictions: list[str] = field(default_factory=list)
    face_faiss_id: int | None = None
    voice_faiss_id: int | None = None


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class AbstractProfileRepository(ABC):
    @abstractmethod
    async def exists(self, user_id: str) -> bool:
        ...

    @abstractmethod
    async def get_by_user_id(self, user_id: str) -> ProfileRecord | None:
        ...

    @abstractmethod
    async def get_by_face_faiss_id(self, face_faiss_id: int) -> ProfileRecord | None:
        ...

    @abstractmethod
    async def get_by_voice_faiss_id(self, voice_faiss_id: int) -> ProfileRecord | None:
        ...

    @abstractmethod
    async def insert(self, record: ProfileRecord) -> None:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------

_COLUMNS = "user_id, name, favorites, restrictions, face_faiss_id, voice_faiss_id"


class SqliteProfileRepository(AbstractProfileRepository):
    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> ProfileRecord:
        return ProfileRecord(
            user_id=row["user_id"],
            name=row["name"],
            favorites=json.loads(row["favorites"] or "[]"),
            restrictions=json.loads(row["restrictions"] or "[]"),
            face_faiss_id=row["face_faiss_id"],
            voice_faiss_id=row["voice_faiss_id"],
        )

    async def exists(self, user_id: str) -> bool:
        cursor = await self._db.execute(
            "SELECT 1 FROM loyalty_profiles WHERE user_id = ?", (user_id,)
        )
        return await cursor.fetchone() is not None

    async def get_by_user_id(self, user_id: str) -> ProfileRecord | None:
        cursor = await self._db.execute(
            f"SELECT {_COLUMNS} FROM loyalty_profiles WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_record(row) if row else None

    async def get_by_face_faiss_id(self, face_faiss_id: int) -> ProfileRecord | None:
        cursor = await self._db.execute(
            f"SELECT {_COLUMNS} FROM loyalty_profiles WHERE face_faiss_id = ?",
            (face_faiss_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_record(row) if row else None

    async def get_by_voice_faiss_id(self, voice_faiss_id: int) -> ProfileRecord | None:
        cursor = await self._db.execute(
            f"SELECT {_COLUMNS} FROM loyalty_profiles WHERE voice_faiss_id = ?",
            (voice_faiss_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_record(row) if row else None

    async def insert(self, record: ProfileRecord) -> None:
        await self._db.execute(
            """
            INSERT INTO loyalty_profiles
                (user_id, name, favorites, restrictions, face_faiss_id, voice_faiss_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.user_id,
                record.name,
                json.dumps(record.favorites),
                json.dumps(record.restrictions),
                record.face_faiss_id,
                record.voice_faiss_id,
            ),
        )
        logger.info("[PROFILE-REPO] Inserted loyalty profile user_id=%s", record.user_id)

    async def count(self) -> int:
        cursor = await self._db.execute("SELECT COUNT(*) FROM loyalty_profiles")
        row = await cursor.fetchone()
        return int(row[0]) if row else 0
