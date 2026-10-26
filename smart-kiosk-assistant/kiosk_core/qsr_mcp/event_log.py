"""Durable event log for the Smart Kiosk QSR MCP surface.

This is a self-contained reimplementation of ``mcp_service_sdk``'s
``envelope``/``log`` modules (the only pieces of the SDK that
``mcp_service.py`` actually depended on for event persistence). The SQLite
table name/columns and JSON envelope shape are kept byte-for-byte identical
to the SDK's so an existing, populated ``qsr_mcp_events.db`` (or JSONL file)
keeps working unmodified after this migration — nothing here changes the
on-disk format, so no migration/backfill step is needed.

Each service owns its own log (no shared component): restart-safe and
ordered (monotonic ``seq``), idempotent on ``ref_id`` (safe replay),
queryable for comparative history (``read``), and replayable in original
order (``replay``).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Protocol


@dataclass(frozen=True)
class EventEnvelope:
    """A single durable, replayable event.

    ``ref_id`` makes replay idempotent: re-emitting the same ``ref_id`` is a
    no-op at the log layer.
    """

    event_type: str
    service: str
    store_id: str
    payload: dict[str, Any]
    ref_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Wall-clock time in epoch milliseconds; ordering is by log sequence, not this.
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    schema_version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "EventEnvelope":
        data = json.loads(raw)
        return cls(**data)


def new_event(
    event_type: str,
    service: str,
    store_id: str,
    payload: dict[str, Any],
    ref_id: str | None = None,
) -> EventEnvelope:
    """Convenience factory so callers never build the envelope by hand."""
    kwargs: dict[str, Any] = {
        "event_type": event_type,
        "service": service,
        "store_id": store_id,
        "payload": payload,
    }
    if ref_id is not None:
        kwargs["ref_id"] = ref_id
    return EventEnvelope(**kwargs)


class DurableLog(Protocol):
    """The storage contract callers depend on. Swappable per deployment."""

    def append(self, event: EventEnvelope) -> int:
        """Persist an event, return its sequence number. Idempotent on ref_id."""
        ...

    def read(
        self,
        event_type: str | None = None,
        since_seq: int = 0,
        limit: int | None = 1000,
        since_ts_ms: int | None = None,
        until_ts_ms: int | None = None,
    ) -> list[EventEnvelope]:
        """Return stored events, oldest first.

        Args:
            event_type: Only return events of this type.
            since_seq: Only return events with a sequence number above this.
            limit: Maximum number of events, or ``None`` for no cap. Use a cap
                unless the other filters already bound the result.
            since_ts_ms: Only return events at or after this epoch-ms.
            until_ts_ms: Only return events strictly before this epoch-ms.
        """
        ...

    def replay(self, from_seq: int = 0) -> Iterator[EventEnvelope]:
        """Re-emit events in original order for benchmarking / debugging."""
        ...


class SQLiteLog:
    """Single-box durable log. Thread-safe via a process-local lock."""

    def __init__(self, path: str = ":memory:", service: str = "unknown") -> None:
        self._service = service
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ref_id       TEXT UNIQUE NOT NULL,
                    event_type   TEXT NOT NULL,
                    ts_ms        INTEGER NOT NULL,
                    envelope     TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)"
            )

    def append(self, event: EventEnvelope) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT seq FROM events WHERE ref_id = ?", (event.ref_id,)
            )
            row = cur.fetchone()
            if row is not None:
                # Idempotent replay: same ref_id already stored.
                return int(row[0])
            cur = self._conn.execute(
                "INSERT INTO events (ref_id, event_type, ts_ms, envelope) "
                "VALUES (?, ?, ?, ?)",
                (event.ref_id, event.event_type, event.ts_ms, event.to_json()),
            )
            return int(cur.lastrowid)

    def read(
        self,
        event_type: str | None = None,
        since_seq: int = 0,
        limit: int | None = 1000,
        since_ts_ms: int | None = None,
        until_ts_ms: int | None = None,
    ) -> list[EventEnvelope]:
        query = "SELECT envelope FROM events WHERE seq > ?"
        params: list[object] = [since_seq]
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)
        # Filtering the time window here rather than in the caller keeps the
        # whole log off the heap: a period query reads only its own rows.
        if since_ts_ms is not None:
            query += " AND ts_ms >= ?"
            params.append(since_ts_ms)
        if until_ts_ms is not None:
            query += " AND ts_ms < ?"
            params.append(until_ts_ms)
        query += " ORDER BY seq ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [EventEnvelope.from_json(r[0]) for r in rows]

    def replay(self, from_seq: int = 0) -> Iterator[EventEnvelope]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT envelope FROM events WHERE seq > ? ORDER BY seq ASC",
                (from_seq,),
            ).fetchall()
        for r in rows:
            yield EventEnvelope.from_json(r[0])

    def close(self) -> None:
        self._conn.close()


class JSONLFileLog:
    """File-backed durable log (one JSON object per line).

    Same ``DurableLog`` contract as ``SQLiteLog``: ordered by ``seq``,
    idempotent on ``ref_id``, restart-safe (state is rebuilt from the file on
    open), replayable.
    """

    def __init__(self, path: str, service: str = "unknown") -> None:
        self._service = service
        self._path = path
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._seq = 0
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._rebuild_state()

    def _rebuild_state(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._seq = max(self._seq, int(rec["seq"]))
                self._seen.add(rec["event"]["ref_id"])

    def append(self, event: EventEnvelope) -> int:
        with self._lock:
            if event.ref_id in self._seen:
                return self._seq_of(event.ref_id)
            self._seq += 1
            rec = {"seq": self._seq, "event": json.loads(event.to_json())}
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._seen.add(event.ref_id)
            return self._seq

    def _seq_of(self, ref_id: str) -> int:
        for rec in self._iter_records():
            if rec["event"]["ref_id"] == ref_id:
                return int(rec["seq"])
        return 0

    def _iter_records(self) -> Iterator[dict]:
        if not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def read(
        self,
        event_type: str | None = None,
        since_seq: int = 0,
        limit: int | None = 1000,
        since_ts_ms: int | None = None,
        until_ts_ms: int | None = None,
    ) -> list[EventEnvelope]:
        out: list[EventEnvelope] = []
        with self._lock:
            for rec in self._iter_records():
                if int(rec["seq"]) <= since_seq:
                    continue
                ev = rec["event"]
                if event_type is not None and ev["event_type"] != event_type:
                    continue
                if since_ts_ms is not None and ev["ts_ms"] < since_ts_ms:
                    continue
                if until_ts_ms is not None and ev["ts_ms"] >= until_ts_ms:
                    continue
                out.append(EventEnvelope(**ev))
                if limit is not None and len(out) >= limit:
                    break
        return out

    def replay(self, from_seq: int = 0) -> Iterator[EventEnvelope]:
        with self._lock:
            records = [r for r in self._iter_records() if int(r["seq"]) > from_seq]
        for rec in records:
            yield EventEnvelope(**rec["event"])


def build_log(backend: str, path: str, service: str) -> DurableLog:
    """Build the configured log backend (mirrors the old ``ServiceConfig``
    ``log_backend`` selection: ``sqlite`` | ``jsonl`` | ``memory``)."""
    if backend in ("sqlite", "memory"):
        db_path = ":memory:" if backend == "memory" else path
        return SQLiteLog(path=db_path, service=service)
    if backend == "jsonl":
        return JSONLFileLog(path=path, service=service)
    raise ValueError(f"unknown log_backend: {backend}")
