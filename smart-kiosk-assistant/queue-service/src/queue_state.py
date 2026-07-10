"""Thread-safe singleton that stores the latest queue count and status.

Written by queue_counter.py (GLib/GStreamer thread) and read by api.py
(uvicorn/asyncio thread) without any shared coroutine state — a plain
threading.Lock is the correct primitive here.
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_count: int = 0
_status: str = "unknown"
_timestamp: float = 0.0


def set_count(count: int, medium_threshold: int = 3, high_threshold: int = 7) -> None:
    """Update the queue count (called from the GStreamer thread)."""
    global _count, _status, _timestamp  # noqa: PLW0603
    if count <= medium_threshold:
        status = "LOW"
    elif count <= high_threshold:
        status = "MEDIUM"
    else:
        status = "HIGH"
    with _lock:
        _count = count
        _status = status
        _timestamp = time.time()


def get() -> dict:
    """Return the latest snapshot (called from asyncio/uvicorn thread)."""
    with _lock:
        return {
            "count": _count,
            "status": _status,
            "timestamp": _timestamp,
        }
