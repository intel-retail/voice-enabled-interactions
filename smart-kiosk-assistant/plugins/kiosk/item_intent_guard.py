"""Guard against adding a stale/pending item instead of the one just named.

Background
----------
Observed live: the customer had an item pending confirmation ("Would you like
to proceed with ordering French fries?"), then said "Go ahead and add a
pizza." The model's ``update_order``/``place_order`` tool call still carried
``"french fries"`` as the item reference — it resolved successfully (French
fries genuinely are on the menu), so ``menu_guard`` sees a clean success and
has nothing to correct. The customer is told a pizza was added while French
fries land in the cart, and the wrong item is written to the database.

This is not an off-menu problem (``menu_guard``'s territory) or a false
confirm-without-a-tool-call problem (``order_claim_guard``/
``_ORDER_CLAIM_FALLBACK``). It is the tool call itself carrying a stale
argument the model failed to update against what the customer just said.

Design
------
Rather than detect the bad write after it lands and speak a correction (which
still leaves a wrong row in the cart until something rolls it back), this
guard corrects the **tool-call argument before the call is made** — the same
philosophy as the dietary-preference injection in ``ordering_agent.py``:
deterministic ground truth from the current turn's raw utterance overrides
whatever the model decided to pass.

Scope is deliberately narrow to avoid false corrections:

* Only single-item ``place_order``/``update_order`` calls are considered.
  A multi-item call ("fries and a pizza") is left untouched — safely
  disambiguating which of several references is stale is out of scope, and
  leaving the call alone preserves today's (correct, most of the time)
  behaviour rather than risking making it worse.
* The utterance must contain an explicit, extractable "add/order X" phrase.
  A bare confirmation ("yes", "go ahead", "sounds good") has no such phrase,
  so the guard does nothing and the pending item is correctly used as-is.
* The correction only fires when the named phrase shares **no** meaningful
  token overlap with the tool call's reference — a paraphrase of the same
  item ("the classic burger" vs. "classic chicken burger") still overlaps and
  is left alone; only a clearly *different* item swaps the reference.

The module performs no I/O and makes no LLM round-trip, so it can be unit
tested as pure functions over text.
* The named phrase must not be *anaphoric*. This is the constraint whose
  absence made the first version of this guard actively harmful. "Yes, one of
  those" and "I'd like to try all of them" both match the add/order pattern,
  but they point at products established earlier in the conversation, which
  the model had already resolved correctly. Substituting the literal words
  produced replies like "Sorry, we don't have of those on the menu" and threw
  away a valid four-item order. When the customer is vague, the model's
  context-resolved reference wins.
"""

from __future__ import annotations

import logging
import re

from agentic import action_result

logger = logging.getLogger(__name__)

# Verbs that introduce a fresh item reference. Deliberately anchored to the
# start of an imperative/desire clause — "go ahead and add a pizza",
# "I'd like a pizza", "order me a pizza", "get me a pizza" — rather than any
# mention of the verb anywhere in the sentence, to avoid capturing unrelated
# clauses.
_NAMED_ITEM_RE = re.compile(
    r"""
    \b(?:
        add(?:ing)?|order|get\s+me|bring\s+me|
        i(?:'d|'ll|\s+would|\s+will)?\s+(?:like|want|take|have)|
        i\s+want|can\s+i\s+(?:get|have|order)
    )\b
    \s+(?:to\s+)?(?:order\s+)?
    (?:me\s+)?
    (?:a|an|one|\d+|the|some)?\s*
    (?P<item>[a-z][a-z\s'-]{2,40}?)
    (?:\s*(?:please|now|instead|as\s+well|too|for\s+me))?\s*[.!?]*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bare agreement/confirmation phrases carry no new item and must never be
# treated as naming one, even though some contain a word like "order".
_CONFIRMATION_ONLY_RE = re.compile(
    r"^\s*(?:"
    r"yes|yeah|yep|sure|ok(?:ay)?|go\s+ahead|sounds?\s+good|"
    r"please\s+(?:proceed|confirm|do)|proceed|confirm(?:\s+it)?|"
    r"that('?s| is)\s+(?:right|correct|it)|correct"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)

# Tentative/exploratory phrases — the customer is browsing, not placing an
# order. Matched against the whole utterance; when detected, extract_named_item
# returns None so the guard never treats the embedded item as a direct order
# reference. Examples:
#   "I was thinking of ordering a burger"
#   "Maybe a pizza"
#   "I might want fries"
#   "I'm considering a coffee"
_TENTATIVE_RE = re.compile(
    r"""^\s*(?:
        i\s+(?:was|am|'m)\s+(?:thinking|considering|looking|going)\s+(?:of|about|at|to)
        |i\s+might(?:\s+want|\s+like|\s+have|\s+get)?
        |maybe\s+a?(?:\s+i\s+(?:could|can|would))?
        |possibly\s+a?
        |what\s+about\s+a?
        |how\s+about\s+a?
        |i\s+(?:could|can|would)\s+(?:try|have|get)\s+a?
        |(?:just\s+)?(?:browsing|looking)
        |i\s+was\s+(?:going\s+to|planning\s+to)\s+(?:get|order|have)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Common stopwords stripped before token-overlap comparison so they never
# count as "overlap" between two different items.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "one", "some", "please", "order", "add", "get",
        "me", "to", "want", "like", "would", "will", "now", "instead",
        "well", "as", "too", "for", "have", "take", "can", "i",
    }
)


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens with stopwords removed, for overlap comparison."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


# Anaphora and collective quantifiers. These point at something established
# earlier in the conversation rather than naming a product, so the model's own
# reference — which was resolved from that context — must win. Matched against
# the whole extracted phrase, so "one of those" is rejected while a real
# product containing a listed word (e.g. "Cold Coffee") is unaffected.
_ANAPHORIC_RE = re.compile(
    r"""^\s*(?:
        (?:try\s+|have\s+|take\s+)?
        (?:all|both|each|any|either|one|some|two|three|\d+)?\s*
        (?:of\s+)?
        (?:them|those|these|it|that|this|the\s+(?:same|other|first|second|last|one))
        (?:\s+one)?
        |everything|anything|something|the\s+usual|same\s+again|same|usual
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

def _is_concrete_item(phrase: str) -> bool:
    """Return True when ``phrase`` could plausibly identify a real product.

    Anaphora ("one of those", "all of them") points at something established
    earlier in the conversation, which the model already resolved into a
    concrete product. Substituting the literal words destroys that resolution
    and guarantees a spurious off-menu refusal, so those are rejected.

    A bare *category* ("pizza") is deliberately **not** rejected: it is exactly
    the original motivating bug ("go ahead and add a pizza" while fries are
    pending), where letting the stale reference through would silently add the
    wrong item. The category is allowed to flow into the tool call, where
    ``mcp_server._resolve_items`` turns it into a "which pizza?" disambiguation
    rather than an off-menu refusal.

    Args:
        phrase: The item phrase extracted from the customer's utterance.

    Returns:
        True when the phrase names something specific enough to act on,
        False for anaphora and collective quantifiers.
    """
    if _ANAPHORIC_RE.match(phrase):
        return False
    return bool(_tokens(phrase))


def extract_named_item(utterance: str) -> str | None:
    """Return the item the customer explicitly named this turn, if any.

    Args:
        utterance: The raw customer message for the current turn (not the
            ``[dietary=...]``/``[customer_name=...]`` tag-prefixed version).

    Returns:
        The captured item phrase (untrimmed of internal words), or ``None``
        when the utterance is a bare confirmation, does not match an explicit
        "add/order X" pattern, or names something too vague to identify a
        product (anaphora or a bare category).
    """
    if not utterance or _CONFIRMATION_ONLY_RE.match(utterance):
        return None
    # Exploratory/tentative utterances ("I was thinking of ordering a burger",
    # "maybe a pizza") are NOT order commands — the model should browse and
    # clarify. Returning None here prevents the stale-reference guard from
    # treating the embedded item as a confirmed item reference, which would
    # make the guard appear to confirm a tentative statement as an order intent.
    if _TENTATIVE_RE.match(utterance):
        return None
    match = _NAMED_ITEM_RE.search(utterance)
    if not match:
        return None
    item = match.group("item").strip()
    if not item or not _tokens(item):
        return None
    if not _is_concrete_item(item):
        # Vague reference: the model's context-resolved product wins.
        return None
    return item


# A reference made only of digits. When the model does not recognise the
# spoken name, it fills product_id with a made-up number instead. Observed:
# "One Perry chicken wrap." -> place_order(product_id="12345"), then
# "and a fresh lime soda, please." -> place_order(product_id="67890"), and
# the customer heard "Sorry, we don't have 12345 on the menu" twice.
_PLACEHOLDER_REF_RE = re.compile(r"^\s*\d+\s*$")

# Words before the item in a request with no ordering verb: "and a ...",
# "also one ...", "uh, two ...".
_BARE_ITEM_RE = re.compile(
    r"""^\s*
    (?:(?:and|also|plus|then|um+|uh+|er+|oh)[\s,]+)*
    (?:a|an|one|two|three|four|five|\d+|the|some)?\s*
    (?P<item>[a-z][a-z\s'-]{2,40}?)
    (?:[\s,]*(?:please|too|as\s+well))?\s*[.!?]*\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_placeholder_reference(reference: object, utterance: str) -> bool:
    """Return True when ``reference`` is a number the customer never said.

    Args:
        reference: The product reference in the tool call.
        utterance: The raw customer message for the current turn.

    Returns:
        True for an all-digit reference that does not appear in the
        utterance.
    """
    text = str(reference)
    return bool(_PLACEHOLDER_REF_RE.match(text)) and text.strip() not in (utterance or "")


def extract_bare_item(utterance: str) -> str | None:
    """Return the item in a short request with no ordering verb.

    Used only to replace a placeholder reference (see
    :func:`is_placeholder_reference`). "One Perry chicken wrap." gives
    "Perry chicken wrap"; kiosk-core's resolver matches the spoken name.

    Args:
        utterance: The raw customer message for the current turn.

    Returns:
        The item phrase, or ``None`` when the utterance is not a bare item
        or names something too vague to identify a product.
    """
    if not utterance or _CONFIRMATION_ONLY_RE.match(utterance):
        return None
    # "maybe a pizza" is browsing, not an order; see extract_named_item.
    if _TENTATIVE_RE.match(utterance):
        return None
    match = _BARE_ITEM_RE.match(utterance)
    if not match:
        return None
    item = match.group("item").strip()
    # "the second one" loses its article above, so check both forms.
    if not _is_concrete_item(item) or not _is_concrete_item(f"the {item}"):
        return None
    return item


def mismatches(reference: str, named_item: str) -> bool:
    """Return True when ``named_item`` shares no meaningful token with ``reference``.

    Args:
        reference: The product reference currently in the tool-call args
            (what the model is about to send to kiosk-core).
        named_item: The item phrase just extracted from the customer's
            current utterance via :func:`extract_named_item`.

    Returns:
        True when the two token sets are disjoint, meaning the tool call is
        very likely still carrying a stale/pending reference rather than the
        item the customer just named.
    """
    ref_tokens = _tokens(reference)
    named_tokens = _tokens(named_item)
    if not ref_tokens or not named_tokens:
        return False
    return ref_tokens.isdisjoint(named_tokens)


def corrected_reference(
    tool_name: str, items: object, utterance: str
) -> str | None:
    """Compute a corrected single-item reference, if a correction is warranted.

    Args:
        tool_name: The MCP tool about to be called (only ``place_order`` and
            ``update_order`` carry an ``items`` argument worth checking).
        items: The tool call's ``items`` argument as supplied by the model.
        utterance: The raw customer message for the current turn.

    Returns:
        The customer's just-named item phrase when it should replace the
        existing single-item reference, or ``None`` when nothing should
        change (multi-item calls, bare confirmations, or no mismatch).
    """
    if tool_name not in action_result.CLAIM_TOOLS[action_result.ITEM_ADDED]:
        return None
    if not isinstance(items, list) or len(items) != 1:
        # Multi-item calls are left untouched — see module docstring.
        return None
    item = items[0]
    if not isinstance(item, dict):
        return None
    reference = item.get("product_id") or item.get("name") or item.get("product") or ""
    if not reference:
        return None

    if is_placeholder_reference(reference, utterance):
        spoken_item = extract_named_item(utterance) or extract_bare_item(utterance)
        if spoken_item:
            logger.warning(
                "[ITEM-INTENT-GUARD] tool=%s placeholder reference=%r — using the "
                "customer's words %r instead",
                tool_name, reference, spoken_item,
            )
        return spoken_item

    named_item = extract_named_item(utterance)
    if not named_item:
        return None

    if not mismatches(reference, named_item):
        return None

    logger.warning(
        "[ITEM-INTENT-GUARD] tool=%s stale reference=%r did not match customer's "
        "just-named item=%r — correcting to the named item",
        tool_name, reference, named_item,
    )
    return named_item
