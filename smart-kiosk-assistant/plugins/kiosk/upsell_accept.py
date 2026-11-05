"""Add the offered item when the customer accepts the kiosk's upsell offer.

After an item is added, the kiosk ends its reply with "Would you also like
<item> (<price>)?". That sentence is composed in code from the tool result
(``directive_mode._facts_tail`` and ``reply_templates.speak_order_mutation``),
so the item the customer is answering about is known exactly.

A short "yes" was still left to the model. Observed on the default menu:

    kiosk:    ... Would you also like Classic French Fries (Regular) (₹89)?
    customer: Yes, please go ahead and add that.

The model did not resolve "that" to the fries. It emitted a confirm, the
order closed without the fries, and the customer's next turn ("you didn't add
the french fries") started a new order. The same reply came back on 10 of 10
replays.

This module reads the offer back from the last assistant turn and recognises
a whole-utterance acceptance, so ``OrderingAgent.chat`` can add the item in
code with no LLM call. Anything that is not a plain acceptance (a different
item, a question, a decline) falls through to the normal path.
"""

from __future__ import annotations

import re

# The offer sentence at the end of an assistant turn. The display string is
# "<name> (<currency><price>)" (see kiosk_core/ordering/mcp_server.py), and a
# name can carry its own parentheses ("Classic French Fries (Regular)"), so
# only the LAST parenthesised group, the one holding a number, is the price.
_OFFER_RE = re.compile(
    r"Would you also like (?P<name>.+?)\s*\([^()\d]{0,4}\d[\d.,]*\)\?\s*$",
    re.IGNORECASE | re.DOTALL,
)

_YES = (
    r"(?:yes|yeah|yep|yup|ya|sure|ok(?:ay)?|alright|all right|please|"
    r"absolutely|definitely|of course|sounds good|why not|go ahead)"
)
_ADD_IT = (
    r"(?:(?:go ahead\s+(?:and\s+)?)?"
    r"(?:add|include|get me|give me|i'?ll (?:take|have)|i'?d like)\s+"
    r"(?:that|it|those|them|one|that one)"
    r"(?:\s+(?:too|as well|also|to (?:it|that|my order|the order)))?)"
)
_POLITE = r"(?:please|thanks?|thank you)"

# The WHOLE utterance must be an acceptance: "Yes", "Sure, add that",
# "Yes, please go ahead and add that.", "I'll take it". Anchored at both ends
# so "yes, and a coke" or "yes but large" never match.
_ACCEPT_RE = re.compile(
    rf"^\s*(?:{_YES}[\s,.!]*)*(?:{_ADD_IT}[\s,.!]*)?(?:{_POLITE}[\s,.!]*)*$",
    re.IGNORECASE,
)
# Every part of _ACCEPT_RE is optional, so a turn must also hold a yes-word or
# an add phrase. "Thanks" or "Thank you" alone is left to the decline path.
_HAS_YES_RE = re.compile(rf"\b{_YES}\b|{_ADD_IT}", re.IGNORECASE)


def offered_item(history: list[dict[str, str]] | None) -> str | None:
    """Return the item named in the last assistant turn's upsell offer.

    Args:
        history: Prior turns as ``{"role", "content"}`` dicts, oldest first.

    Returns:
        The offered item's menu name, or ``None`` when the last assistant
        turn did not end with an offer.
    """
    for turn in reversed(history or []):
        if turn.get("role") != "assistant":
            continue
        m = _OFFER_RE.search(turn.get("content") or "")
        return m.group("name").strip() if m else None
    return None


def is_acceptance(message: str) -> bool:
    """Return True when the whole utterance accepts an offer and says nothing else."""
    return bool(message and _ACCEPT_RE.match(message) and _HAS_YES_RE.search(message))
