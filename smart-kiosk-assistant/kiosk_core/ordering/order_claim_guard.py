"""Guard against the agent claiming order actions it never performed.

Background
----------
On 2026-07-29 a full ordering conversation was observed in which the agent
told the customer "I've added the Paneer Tikka Burger (₹159) to your order"
and then "Your order is confirmed! Your Order ID is ORD-XXXXX." — while the
kiosk-core MCP server recorded only a single ``list_products`` call and the
database gained no row at all. ``ORD-XXXXX`` is a literal placeholder that
appeared in the agent's system prompt; real order ids are integers.

Prompt wording alone had already failed to prevent this: the instruction
'Never say "I've added…" unless the call succeeded this turn' was present and
ignored. Language models will reproduce a success template they have been
shown, because the template is the most probable continuation.

This module therefore enforces the invariant deterministically, outside the
model: **a reply may only claim an order action that the turn's tool calls
actually support.** It is model-independent, needs no extra LLM round-trip,
and cannot itself hallucinate.

The guard is intentionally *pure* — it takes the reply text and the list of
tools invoked during the turn, and returns corrected text. It performs no I/O,
so it can run inside the synchronous HTTP client boundary without dragging
database access into a layer that must not own it.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Tools whose invocation legitimises a "your order now contains X" claim.
_MUTATING_TOOLS = frozenset({"place_order", "update_order"})

# Tool whose invocation legitimises a "your order is confirmed" claim.
# `confirm_active_order` resolves the customer's open draft and confirms it in
# one step; rag-service also invokes it deterministically when the model fails
# to emit a confirm call. Omitting it here would rewrite a genuine, database-
# backed confirmation into a failure message — the inverse of this guard's
# purpose, and worse, because the order really is confirmed.
_CONFIRM_TOOLS = frozenset({"confirm_order", "confirm_active_order"})

# Claims that an item was put into the cart.
_ADDED_PATTERNS = (
    re.compile(r"\bI(?:'ve| have)\s+added\b", re.IGNORECASE),
    re.compile(r"\b(?:has|have)\s+been\s+added\s+to\s+your\s+order\b", re.IGNORECASE),
    re.compile(r"\badded\s+(?:the|a|an)\b.{0,60}?\bto\s+your\s+order\b", re.IGNORECASE),
)

# Claims that the order was placed/confirmed.
_CONFIRMED_PATTERNS = (
    re.compile(r"\byour\s+order\s+(?:is|has\s+been)\s+(?:confirmed|placed)\b", re.IGNORECASE),
    re.compile(r"\border\s+(?:is|has\s+been)\s+successfully\s+(?:placed|confirmed)\b", re.IGNORECASE),
    re.compile(r"\bOrder\s+ID\s+is\b", re.IGNORECASE),
)

# The literal placeholder from the system prompt. This string is never a valid
# order id under any circumstance, so it is stripped even when confirm_order
# genuinely ran — otherwise the customer is read a fake reference number.
_PLACEHOLDER_ORDER_ID = re.compile(r"\bORD-X+\b", re.IGNORECASE)

_ADDED_REPLACEMENT = (
    "Sorry — I wasn't able to add that to your order just now. "
    "Could you tell me the item name again?"
)

_CONFIRMED_REPLACEMENT = (
    "Sorry — I couldn't confirm your order just now. "
    "Please say 'confirm' once more and I'll place it."
)


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def validate_reply(reply: str, tool_calls: list[str] | None) -> tuple[str, bool]:
    """Reconcile an agent reply against the tools actually invoked this turn.

    Args:
        reply: The agent's composed reply text.
        tool_calls: Names of the tools invoked during this turn, as reported
            by rag-service. An empty or missing list means no tool ran.

    Returns:
        A ``(reply, corrected)`` pair. ``corrected`` is True when the reply
        made an unsupported claim and was rewritten, so callers can log or
        surface the event.
    """
    if not reply:
        return reply, False

    invoked = {name for name in (tool_calls or [])}
    corrected = False

    claims_confirmed = _matches_any(reply, _CONFIRMED_PATTERNS)
    claims_added = _matches_any(reply, _ADDED_PATTERNS)

    # A confirmation claim is the most damaging: the customer walks away
    # believing food is coming. Check it first and replace the whole reply,
    # because a confirmation sentence is rarely salvageable in part.
    if claims_confirmed and not (invoked & _CONFIRM_TOOLS):
        logger.error(
            "[ORDER-GUARD] Reply claimed order confirmation but confirm_order was "
            "not invoked this turn (tools=%s). Replacing reply: %r",
            sorted(invoked), reply[:160],
        )
        return _CONFIRMED_REPLACEMENT, True

    if claims_added and not (invoked & _MUTATING_TOOLS):
        logger.error(
            "[ORDER-GUARD] Reply claimed an item was added but no ordering tool was "
            "invoked this turn (tools=%s). Replacing reply: %r",
            sorted(invoked), reply[:160],
        )
        return _ADDED_REPLACEMENT, True

    # Strip the prompt's placeholder order id even on a legitimate confirmation.
    if _PLACEHOLDER_ORDER_ID.search(reply):
        logger.warning(
            "[ORDER-GUARD] Reply contained the placeholder order id from the prompt "
            "template; removing it (tools=%s).", sorted(invoked),
        )
        reply = _PLACEHOLDER_ORDER_ID.sub("", reply)
        # Tidy the punctuation left behind by the removal.
        reply = re.sub(r"\bYour\s+Order\s+ID\s+is\s*[.,]?\s*", "", reply, flags=re.IGNORECASE)
        reply = re.sub(r"\s{2,}", " ", reply).strip()
        corrected = True

    return reply, corrected
