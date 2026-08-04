"""Per-turn LLM timing accumulator.

Google ADK issues more than one LLM round-trip per tool-calling turn (one to
decide the tool call, one to compose the final reply).  The agent's overall
latency therefore mixes LLM time with MCP tool time and ADK overhead, which
made the UI's "LLM" figure meaningless.

This module accumulates genuine LLM time for the *current* turn.  A
``ContextVar`` is used rather than an instance attribute so that concurrent
requests handled by the same singleton agent never contaminate each other's
measurements.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Holds {"ms": float, "calls": int} for the in-flight turn.
_llm_stats: ContextVar[Dict[str, float] | None] = ContextVar("_llm_stats", default=None)

# Holds {"ms": float, "calls": int} of retrieval work done by tools this turn.
_retrieval_stats: ContextVar[Dict[str, float] | None] = ContextVar(
    "_retrieval_stats", default=None
)


def reset() -> None:
    """Begin a fresh measurement window for the current turn."""
    _llm_stats.set({"ms": 0.0, "calls": 0})
    _retrieval_stats.set({"ms": 0.0, "calls": 0})


def record(elapsed_ms: float) -> None:
    """Add one completed LLM round-trip to the current window."""
    stats = _llm_stats.get()
    if stats is None:
        return
    stats["ms"] += elapsed_ms
    stats["calls"] += 1


def record_retrieval(elapsed_ms: float) -> None:
    """Add one completed knowledge-base retrieval to the current window."""
    stats = _retrieval_stats.get()
    if stats is None:
        return
    stats["ms"] += elapsed_ms
    stats["calls"] += 1


def retrieval_snapshot() -> Dict[str, Any]:
    """Return accumulated retrieval timings for the current turn.

    Returns:
        dict with ``ms`` (float or None when nothing was retrieved) and
        ``calls`` (int). ``ms`` stays None when retrieval never ran so the UI
        can distinguish "not invoked" from "took 0 ms".
    """
    stats = _retrieval_stats.get()
    if stats is None or stats["calls"] == 0:
        return {"ms": None, "calls": 0}
    return {"ms": round(stats["ms"], 1), "calls": int(stats["calls"])}


def snapshot() -> Dict[str, Any]:
    """Return the accumulated timings for the current turn.

    Returns:
        dict with ``ms`` (float, cumulative LLM time) and ``calls`` (int).
        ``ms`` is None when no LLM call was recorded, so callers can
        distinguish "not measured" from "zero".
    """
    stats = _llm_stats.get()
    if stats is None or stats["calls"] == 0:
        return {"ms": None, "calls": 0}
    return {"ms": round(stats["ms"], 1), "calls": int(stats["calls"])}
