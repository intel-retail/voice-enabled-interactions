"""Guard against a spoken total that disagrees with the tool's own number.

Background
----------
``place_order``/``update_order``/``confirm_active_order``/``get_current_order``
results all carry an authoritative ``total`` computed by kiosk-core from the
real catalogue and cart contents. The common, successful path never risks a
wrong number: ``reply_templates.speak()`` builds the reply directly from that
same payload, so the spoken total and the billed total are always the same
value.

The risk is the *fallback* path — a turn where no template applies (an
unrecognised tool, or the model free-generating text about an order already
described earlier this conversation) lets the model restate a total from its
own context window rather than from this turn's tool result. Language models
transpose and round digits under exactly these conditions; a customer told a
different total than what is actually charged is a real billing complaint
waiting to happen, matching ``kiosk-voice-lab-main``'s ``total_guard()``
(``pipeline/cart.py``).

Design
------
Mirrors ``menu_guard``/``confirm_guard``/``removal_guard``: per-turn state in
a ``contextvars.ContextVar``, a ``record_tool_result()`` call at the same
call site as those guards, and a ``validate_reply()`` used by the whole-reply
post-hoc guard chain in ``ordering_agent.chat()``.

Unlike those guards, a wrong total is *corrected* rather than blocked or
replaced wholesale: the actual total is already known with certainty, so
there is no need to fall back to a generic refusal — swapping in the right
number is strictly better for the customer than a full reply substitution
that has to re-describe the whole order from scratch.
"""

from __future__ import annotations

import contextvars
import logging
import re
from dataclasses import dataclass
from typing import Any

from agentic import action_result

logger = logging.getLogger(__name__)

# Tools whose result carries an authoritative ``total`` for the customer's
# current order. Read-only lookups (``get_current_order``, ``get_order``)
# are included: a stale total restated from an earlier turn is exactly the
# failure mode this guard exists to catch. ``remove_from_order`` is included
# too — the MCP server returns an authoritative total for it just like the
# other mutation tools (kiosk_core/ordering/mcp_server.py). ``confirm_order``
# (order-id variant of ``confirm_active_order``) also returns the order's
# authoritative total and must be covered the same way.
_TOTAL_TOOLS = frozenset(
    {
        "place_order",
        "update_order",
        "confirm_active_order",
        "confirm_order",
        "get_current_order",
        "get_order",
        "remove_from_order",
    }
)

# Matches a sentence clause naming the *order total* (not a per-item price):
# "your total is 169", "total is now ₹169", "total comes to 169 rupees",
# "the total's 169.50". Deliberately anchored on the word "total" so a
# per-item price mention (already covered by reply_templates' own per-item
# formatting) is never misread as an order-total claim.
_TOTAL_MENTION_RE = re.compile(
    r"\btotal\b(?:'s|\s+\w+){0,4}?\s*(?:is|of|comes?\s+to|now)?\s*"
    r"₹?\s*(\d[\d,]*(?:\.\d{1,2})?)\s*(?:rupees?)?",
    re.IGNORECASE,
)


@dataclass
class _TurnState:
    """What this turn's own tool results say the order total actually is."""

    total: float | None = None


_turn_state: contextvars.ContextVar[_TurnState | None] = contextvars.ContextVar(
    "price_guard_turn_state", default=None
)


def begin_turn() -> _TurnState:
    """Reset per-turn total tracking.

    Must be called at the start of every ``OrderingAgent.chat()`` turn, same
    as ``menu_guard.begin_turn()`` — without a reset, a total from an earlier
    turn would validate (or fail to flag) this turn's reply.

    Returns:
        The fresh state object, mainly so tests can inspect it.
    """
    state = _TurnState()
    _turn_state.set(state)
    return state


def current_state() -> _TurnState:
    """Return the total-tracking state for the turn running in this context.

    See ``menu_guard.current_state()`` for why a fresh object — never the
    ContextVar's shared default — is returned when ``begin_turn()`` was never
    called in this context.
    """
    state = _turn_state.get()
    if state is None:
        state = _TurnState()
        _turn_state.set(state)
    return state


def record_tool_result(tool_name: str, raw: Any) -> None:
    """Capture the authoritative order total from one tool result, if present.

    Args:
        tool_name: Name of the MCP tool that was just invoked.
        raw: The raw value returned by ``mcp_client.call_tool``.
    """
    if tool_name not in _TOTAL_TOOLS:
        return
    result = action_result.classify(tool_name, raw)
    if not result.success:
        return
    total = result.data.get("total")
    if total is None:
        return
    try:
        current_state().total = float(total)
    except (TypeError, ValueError):
        logger.warning(
            "[PRICE-GUARD] Non-numeric total in %s result: %r", tool_name, total,
        )


def _format_total(total: float) -> str:
    """Render a whole-rupee total as an int, matching reply_templates' style."""
    return str(int(total)) if total.is_integer() else f"{total:.2f}"


def mismatched(sentence: str, state: "_TurnState | None" = None) -> bool:
    """Return True when ``sentence`` states a total that disagrees with the
    authoritative value already known for this turn.

    Used by ``_SentenceGate`` to withhold a sentence that ``validate_reply()``
    would otherwise have to correct after the fact — speech cannot be
    recalled, so a mismatched total must never be released early.
    """
    state = state or current_state()
    if state.total is None:
        return False
    match = _TOTAL_MENTION_RE.search(sentence)
    if not match:
        return False
    try:
        spoken = float(match.group(1).replace(",", ""))
    except ValueError:
        return False
    return abs(spoken - state.total) > 0.005


def validate_reply(reply: str, state: "_TurnState | None" = None) -> tuple[str, bool]:
    """Correct a hallucinated order total in ``reply``, if this turn knows one.

    Args:
        reply: The reply assembled so far (after every earlier guard).
        state: Explicit state, for tests. Defaults to the current turn's.

    Returns:
        ``(corrected_reply, corrected)``. ``corrected`` is True only when a
        total mention was found *and* disagreed with the authoritative value,
        so callers can log/telemetry exactly like every other guard here.
    """
    state = state or current_state()
    if not reply or state.total is None:
        return reply, False
    match = _TOTAL_MENTION_RE.search(reply)
    if not match:
        return reply, False
    try:
        spoken = float(match.group(1).replace(",", ""))
    except ValueError:
        return reply, False
    if abs(spoken - state.total) <= 0.005:
        return reply, False
    start, end = match.span(1)
    corrected = reply[:start] + _format_total(state.total) + reply[end:]
    logger.warning(
        "[PRICE-GUARD] Corrected hallucinated total %.2f → %.2f | reply=%r",
        spoken, state.total, reply[:160],
    )
    return corrected, True
