"""Customer-facing sentences for deterministic order-mutation tool outcomes.

Background
----------
Every mutating tool call today costs **two** LLM round-trips: one to decide
the tool and its arguments, and a second, purely to narrate a result the
server already knows in full — order_id, item names, total, upsell display
strings. On Panther Lake's single iGPU that second call is ~half the LLM time
in a typical turn (see tests/benchmarks/results/replay-qwen3-4b-postfix.json).

Google ADK supports skipping that second call natively: a tool sets
``tool_context.actions.skip_summarization = True`` and returns a plain string,
and ``Event.is_final_response()`` treats that as the end of the turn — no
second model call happens. See ``agentic/ordering_agent.py::_make_mcp_callable``
for where this module is used.

**Why these templates are written fresh here rather than reusing
``kiosk_core.ordering.mcp_server``'s ``unavailable_message`` /
``choice_message`` strings**: those strings are authored *for the model*, not
the customer. They read like "Do not invent them and do not ask the customer
to try again. Tell them those are unavailable and offer these real
alternatives instead: ..." — instructive prose meant to steer the LLM's next
completion. Speaking that verbatim over TTS would be nonsense. Every sentence
below is built only from *structured* fields (names, prices, totals, ids) that
the server already validated, never from those instruction strings.

Design
------
Every function here returns ``None`` when the tool outcome does not cleanly
match one of the templates below (e.g. a partial success mixed with a
rejection). Returning ``None`` means "fall through to the normal LLM
narration call" — the safe default. A function is never called on a result it
cannot describe with total confidence; when in doubt, the LLM still speaks.

Pure functions, no I/O, no LLM round-trip — unit-testable exactly like the
guards in this package.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agentic import domain_config
from agentic.action_result import unwrap, unwrap_any

# How many upsell/alternative suggestions to mention in one spoken sentence.
# Matches the ceiling already used by menu_guard and mcp_server for the same
# reason: a voice customer stops retaining a list beyond this.
_MAX_SPOKEN_ALTERNATIVES = 3


def _money(value: Any) -> str:
    """Render a price/total as ``"79"`` or ``"78.50"`` — never ``"79.00"``.

    TTS pronounces ``79.00`` as "seventy-nine point zero zero"; whole-rupee
    values are the overwhelming majority in this catalogue.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f.is_integer() else f"{f:.2f}"


def _join_names(names: list[str]) -> str:
    """Join a list of item names into natural spoken English."""
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def speak_catalogue(payload: Any) -> str | None:
    """Template a reply for a ``list_products`` / ``list_categories`` result.

    Catalogue results are the single largest deterministic win available in
    this agent. ``_AGENT_INSTRUCTION`` Rule 4 already dictates the exact
    sentence the model must produce from this data — "We have <Name> (<price>),
    ... Which one would you like to try?" — so the second LLM call is spending
    ~3 s of Qwen inference reproducing a string that is fully determined by
    the tool result. Every name and price here is copied verbatim from
    kiosk-core's response, which makes this strictly *more* faithful than the
    model's paraphrase, never less: it cannot drop, rename, or re-price a row.

    Three payload shapes are produced by ``mcp_server.list_products``:

    * ``[{product_id, name, category, price}, ...]`` — a browsed category.
    * ``[{category, item_count}, ...]`` — the category summary, also what
      ``list_categories`` returns.
    * ``{category_not_found, requested, categories, message}`` — the customer
      asked for something the kiosk does not carry at all ("dosa", "sushi").

    Args:
        payload: The tool's own JSON result, already unwrapped from the MCP
            transport envelope.

    Returns:
        A spoken sentence, or ``None`` to defer to normal LLM narration when
        the shape is not one of the three above.
    """
    # "We don't carry that" — built from `requested` + `categories` only. The
    # payload's own `message` field is deliberately NOT spoken: it is authored
    # for the model ("Do not invent a product...") and is nonsense over TTS.
    # Templating this case also removes the specific failure it was written to
    # catch, where the model cherry-picked unrelated items from the catalogue
    # and presented them as an answer.
    if isinstance(payload, dict):
        if not payload.get("category_not_found"):
            return None
        categories = [str(c) for c in (payload.get("categories") or []) if c]
        requested = str(payload.get("requested") or "").strip()
        if not categories or not requested:
            return None
        tpl = domain_config.get_reply_template("not_found")
        if tpl:
            return tpl.format(requested=requested, categories=_join_names(categories))
        return (
            f"Sorry, we don't have {requested} on the menu. "
            f"We do serve {_join_names(categories)}. "
            f"Which would you like to see?"
        )

    if not isinstance(payload, list) or not payload:
        return None

    # Category summary — entries carry a category and a count and, crucially,
    # NO product ``name``. Testing only for "has category, lacks price" would
    # swallow a real product row whose price failed to serialise, reporting
    # the customer's category back at them as if it were the whole answer.
    if all(
        isinstance(e, dict) and e.get("category") and not e.get("name")
        for e in payload
    ):
        parts = [
            f"{e['category']} ({e.get('item_count', e.get('count'))} items)"
            if e.get("item_count", e.get("count")) is not None
            else str(e["category"])
            for e in payload
        ]
        tpl = domain_config.get_reply_template("category_list")
        if tpl:
            return tpl.format(categories=_join_names(parts))
        return (
            f"We have {_join_names(parts)}. "
            f"Which category would you like to explore?"
        )

    # A browsed category — every entry must carry a name and a price, or we
    # cannot describe the row faithfully and the model should speak instead.
    if all(
        isinstance(e, dict) and e.get("name") and e.get("price") is not None
        for e in payload
    ):
        currency = domain_config.get_currency_symbol()
        parts = [f"{e['name']} ({currency}{_money(e['price'])})" for e in payload]
        tpl = domain_config.get_reply_template("product_list")
        if tpl:
            return tpl.format(products=_join_names(parts))
        return f"We have {_join_names(parts)}. Which one would you like to try?"

    return None


def speak_order_mutation(payload: dict[str, Any]) -> str | None:
    """Template a reply for a clean ``place_order``/``update_order`` outcome.

    Args:
        payload: The tool's own JSON result (already unwrapped from the MCP
            transport envelope), as produced by ``mcp_server.place_order`` /
            ``update_order``.

    Returns:
        A spoken sentence for an unambiguous success, or ``None`` when the
        result is a rejection, a category-disambiguation, or a mix of
        success and failure — those are left to the LLM, which can weave a
        compound reply together better than a fixed template can.
    """
    if not isinstance(payload, dict):
        return None
    # Any of these mean the outcome is not a clean, fully-resolved success —
    # fall through and let the model narrate the nuance.
    if "error" in payload or "needs_choice" in payload or "unavailable_message" in payload:
        return None
    just_added = payload.get("just_added")
    total = payload.get("total")
    if not just_added or total is None:
        return None

    names = _join_names([it.get("name", "") for it in just_added])
    if not names:
        return None

    currency = domain_config.get_currency_symbol()
    tpl = domain_config.get_reply_template("item_added")
    if tpl:
        sentence = tpl.format(items=names, total=_money(total), currency=currency)
    else:
        sentence = f"I've added {names} to your order. Your total is now {currency}{_money(total)}."

    upsell = payload.get("upsell_suggestions") or []
    if upsell and isinstance(upsell[0], dict) and upsell[0].get("display"):
        sentence += f" Would you also like {upsell[0]['display']}?"
    else:
        sentence += " Would you like anything else?"
    return sentence


def speak_confirm(payload: dict[str, Any]) -> str | None:
    """Template a reply for a clean order confirmation.

    Args:
        payload: The tool's own JSON result from ``confirm_order`` /
            ``confirm_active_order``.

    Returns:
        A spoken confirmation sentence, or ``None`` for any error — the
        error paths here (empty cart, order not found, already confirmed)
        are worded for the model, not the customer, so they always fall
        through to normal narration.
    """
    if not isinstance(payload, dict) or "error" in payload:
        return None
    order_id = payload.get("order_id")
    total = payload.get("total")
    if order_id is None or total is None or payload.get("status") != "confirmed":
        return None
    currency = domain_config.get_currency_symbol()
    tpl = domain_config.get_reply_template("order_confirmed")
    if tpl:
        return tpl.format(order_id=order_id, total=_money(total), currency=currency)
    return (
        f"Your order is confirmed! Order number {order_id}, total {currency}{_money(total)}. "
        f"Thank you!"
    )


def speak_removal(payload: dict[str, Any]) -> str | None:
    """Template a reply for a clean ``remove_from_order`` outcome.

    Args:
        payload: The tool's own JSON result from ``remove_from_order``.

    Returns:
        A spoken sentence for a fully-resolved removal (every requested item
        was actually in the cart), or ``None`` when nothing was removed or
        the removal was only partial — those read better as an LLM-narrated
        explanation of what was and wasn't found.
    """
    if not isinstance(payload, dict) or "error" in payload:
        return None
    removed = payload.get("removed")
    not_in_cart = payload.get("not_in_cart")
    total = payload.get("total")
    if not removed or not_in_cart or total is None:
        return None

    names = _join_names(removed)
    currency = domain_config.get_currency_symbol()
    if payload.get("cart_empty"):
        return f"I've removed {names}. Your cart is now empty."
    tpl = domain_config.get_reply_template("item_removed")
    if tpl:
        return tpl.format(items=names, total=_money(total), currency=currency)
    return f"I've removed {names}. Your new total is {currency}{_money(total)}."


def speak_current_order(payload: dict[str, Any]) -> str | None:
    """Template a reply for a ``get_current_order`` cart read.

    "What's in my cart" / "what's my total" is the most common read in the
    ordering flow, and every field needed to answer it is already present in
    the tool result. Narrating it with a second LLM call costs a full
    ~1.2 s round-trip to restate data the template can copy verbatim — which
    is also strictly more faithful, since a template cannot drop a line,
    rename an item, or mis-add a total.

    Args:
        payload: The tool's own JSON result (already unwrapped from the MCP
            transport envelope) — an ``Order`` dump with ``items``/``total``,
            or ``None`` when the customer has no open order.

    Returns:
        A spoken sentence, or ``None`` to defer to normal LLM narration when
        the payload is not a recognisable order shape.
    """
    currency = domain_config.get_currency_symbol()

    # A transport-level failure is surfaced by ``unwrap`` as ``{"error": ...}``.
    # It must NOT fall through to the empty-cart reply below: "your cart is
    # empty" would be a false claim about a cart we simply failed to read,
    # and the customer's real items would silently vanish from the
    # conversation. Defer to the model, which will report the failure.
    if isinstance(payload, dict) and "error" in payload:
        return None

    # No open draft at all. ``get_current_order`` returns null here, which
    # ``unwrap`` surfaces as None, so this is reached via the explicit
    # empty-cart call in ``speak()`` rather than through this branch alone.
    if payload is None or (isinstance(payload, dict) and not payload.get("items")):
        return domain_config.get_reply_template("cart_empty") or (
            "Your cart is empty at the moment. What would you like to order?"
        )
    if not isinstance(payload, dict):
        return None

    total = payload.get("total")
    if total is None:
        return None

    # Speak quantities only when there is more than one, so a single item
    # reads "a Classic Chicken Burger" rather than the robotic "1 Classic
    # Chicken Burger".
    parts: list[str] = []
    for item in payload["items"]:
        if not isinstance(item, dict):
            return None
        name = str(item.get("product_name") or "").strip()
        if not name:
            return None
        quantity = item.get("quantity") or 1
        parts.append(f"{quantity} {name}" if quantity > 1 else name)

    names = _join_names(parts)
    if not names:
        return None

    tpl = domain_config.get_reply_template("cart_summary")
    if tpl:
        return tpl.format(items=names, total=_money(total), currency=currency)
    return f"Sure. You have {names}. Your total is {currency}{_money(total)}."


# Dispatch table used by ordering_agent.py — keeps the "which tools are
# speakable" decision in one place, next to the templates themselves.
_TEMPLATES = {
    "place_order": speak_order_mutation,
    "update_order": speak_order_mutation,
    "confirm_order": speak_confirm,
    "confirm_active_order": speak_confirm,
    "remove_from_order": speak_removal,
    "list_products": speak_catalogue,
    "list_categories": speak_catalogue,
    # Returns the same [{product_id, name, category, price}, ...] shape as
    # list_products, so it reuses the catalogue templater verbatim.
    "get_popular_products": speak_catalogue,
    "get_current_order": speak_current_order,
}

# Public: which tools this module can ever speak for. Checked by
# ordering_agent.py before even attempting ``speak()``, so a tool this module
# doesn't know about is never a candidate for skipping narration.
SPEAKABLE_TOOLS = frozenset(_TEMPLATES)

# Read-only tools are speakable only when the customer was actually browsing.
#
# Templating sets ``skip_summarization``, which ADK treats as the END of the
# turn. For a mutating tool that is safe — the mutation already happened and
# the template describes it. For a *read* like ``list_products`` or
# ``get_current_order`` it is only safe if the read was the customer's goal.
# Were the model to use one as an intermediate lookup before ``place_order``,
# ending the turn at the lookup would drop the order.
#
# Measured across the 234-turn replay corpus that has never happened (zero
# multi-tool turns; 56 single catalogue turns), and the failure mode is a
# stale menu recital rather than a false claim, so no truthfulness invariant
# rests on this. It is still gated, because the gate is free.
_READ_TOOLS = frozenset({
    "list_products",
    "list_categories",
    "get_popular_products",
    "get_current_order",
})

# Tools whose result is a top-level JSON *array*. The mutation tools and
# ``get_current_order`` all return an object, so they use the dict-narrowing
# ``unwrap`` that the guards expect; these need the list-preserving decoder.
_LIST_RESULT_TOOLS = frozenset({
    "list_products",
    "list_categories",
    "get_popular_products",
})

# Explicit cart-mutation intent. Deliberately narrow: a false NEGATIVE here
# just costs one LLM call (today's behaviour), while a false POSITIVE could
# swallow an order. Bare "order" is not enough — "what's in my order" is a
# read; only "order a/one/the <thing>" is an instruction to buy.
_MUTATION_INTENT_RE = re.compile(
    r"\b(?:add|adding|buy|purchase|remove|removing|delete|drop|cancel|"
    r"confirm|checkout|check\s+out|place\s+(?:the|my)\s+order|"
    r"order\s+(?:me\s+)?(?:a|an|one|two|three|four|five|\d+|the)\b)",
    re.IGNORECASE,
)


def is_browse_intent(utterance: str) -> bool:
    """Return True when a catalogue read may be spoken as the whole turn.

    Args:
        utterance: The customer's raw message for this turn.

    Returns:
        ``False`` when the customer asked to change the cart, in which case a
        catalogue tool can only have been an intermediate step and the turn
        must be allowed to continue to the mutating call.
    """
    return not _MUTATION_INTENT_RE.search(utterance or "")


# Single-item price/cost questions only — deliberately narrow. General
# catalogue browsing ("what desserts do you have") still needs the
# authoritative list_products tool-calling path (see _CATALOGUE_QUERY_RE in
# ordering_agent.py); only a "how much does X cost" style lookup is safe to
# answer from directive mode's cached catalogue, because that lookup is a
# single exact-name match against the same static menu already trusted for
# <act> item names (see directive_mode._execute_price).
_PRICE_QUERY_RE = re.compile(
    r"\bhow\s+much\s+(?:does|is|for|would|are)\b|"
    r"\bwhat(?:'s|s| is| does)\b.{0,40}\b(?:cost|costs|price)\b|"
    r"\b(?:price|cost)\s+of\b",
    re.IGNORECASE,
)


def is_price_query_intent(utterance: str) -> bool:
    """Return True when the utterance is a single-item price/cost question.

    Args:
        utterance: The customer's raw message for this turn.

    Returns:
        ``True`` for "how much does the brownie cost?" style questions, which
        directive mode may answer via a fast cached-catalogue lookup instead
        of falling through to the slower tool-calling path.
    """
    return bool(_PRICE_QUERY_RE.search(utterance or ""))


def speak(tool_name: str, raw_result: Any, utterance: str = "") -> str | None:
    """Return a spoken reply for ``tool_name``'s result, or None to defer to the LLM.

    Args:
        tool_name: The MCP tool that was just called.
        raw_result: The raw value returned by ``mcp_client.call_tool`` (the
            MCP transport envelope, not yet decoded).
        utterance: The customer's raw message this turn. Only consulted for
            catalogue reads — see ``_READ_TOOLS``. Defaults to empty,
            which is treated as browse intent, so existing callers and the
            mutation templates are unaffected.

    Returns:
        A ready-to-speak sentence, or ``None`` when this tool/outcome
        combination is not one of the templates above.
    """
    template = _TEMPLATES.get(tool_name)
    if template is None:
        return None
    if tool_name in _READ_TOOLS and not is_browse_intent(utterance):
        return None
    decode = unwrap_any if tool_name in _LIST_RESULT_TOOLS else unwrap
    payload = decode(raw_result)
    if payload is None:
        # For every other tool a null result means "no usable data, let the
        # model narrate". For get_current_order it is the answer itself:
        # the tool returns null precisely when the customer has no open
        # order, so this is an empty cart, not a decode failure.
        #
        # But unwrap() collapses FOUR cases to None: a genuine JSON null, a
        # non-dict envelope, an empty/missing result, and a JSONDecodeError.
        # Speaking "your cart is empty" for a decode failure would be a false
        # statement about order state — the exact failure this deterministic
        # path exists to prevent — and skip_summarization means the model never
        # gets to correct it. So require positive evidence of a successful call
        # that really returned null before speaking the empty-cart line.
        if tool_name == "get_current_order" and _is_successful_null(raw_result):
            return speak_current_order(None)
        return None
    return template(payload)


def _is_successful_null(raw_result: Any) -> bool:
    """True when the envelope reports success and its payload is literally null.

    Args:
        raw_result: The raw ``call_tool`` envelope, before unwrapping.

    Returns:
        True only for a well-formed success envelope whose ``result`` decodes
        to JSON ``null`` — never for a malformed, errored, or undecodable one.
    """
    if not isinstance(raw_result, dict) or "error" in raw_result:
        return False
    if raw_result.get("status") not in (None, "success"):
        return False
    result = raw_result.get("result")
    if result is None:
        return True
    if not isinstance(result, str):
        return False
    try:
        return json.loads(result) is None
    except (json.JSONDecodeError, TypeError):
        return False


# ── Structured root-fact answers ─────────────────────────────────────────
#
# Restaurant name and operating hours are answered by having the LLM
# paraphrase the pinned knowledge-base root section (see
# agentic/tools/knowledge_lookup_tool.py). Measured live, that paraphrase is
# not reliable even at temperature=0.0: two identical requests to the same
# quantised model produced "8 AM to 11 PM Monday through Saturday" and later
# "8 AM to 12 AM on weekdays", both merging the source's genuinely distinct
# Mon-Thu/Fri-Sat/Sun ranges into one, and the second additionally invented
# an unsupported "closed on public holidays" claim. For a restaurant's own
# hours, a wrong answer is a worse defect than an unstyled one, so these two
# facts are spoken directly from the parsed structured value — no LLM call,
# no paraphrase, no possibility of merging or inventing.
#
# Deliberately narrow: only the two facts empirically shown to be unreliable.
# Every other root-level fact (parking, wifi, delivery, ...) keeps using the
# existing pre-grounded LLM narration path, since simple string facts like
# "shared parking lot" carry little risk of being merged or hallucinated.
_ROOT_FACT_NAME_RE = re.compile(
    r"\b(?:restaurant'?s? name|name of (?:the|this) (?:restaurant|outlet|place)|"
    r"what(?:'s| is) (?:the|this) (?:restaurant|outlet|place)(?:'s)? (?:called|name)|"
    r"what(?:'s| is) the name of)\b",
    re.IGNORECASE,
)
_ROOT_FACT_BREAKFAST_RE = re.compile(r"\bbreakfast (?:hours?|timings?)\b", re.IGNORECASE)
_ROOT_FACT_HOURS_RE = re.compile(
    r"\b(?:opening hours?|closing hours?|operating hours?|business hours?|"
    r"open(?:ing)?\s+and\s+clos(?:e|ing)(?:\s+(?:hours?|time))?|"
    r"open\s+and\s+close|opens?\s+and\s+closes?|"
    r"what time (?:do|does) (?:you|it|the restaurant|the outlet).{0,10}(?:open|close)|"
    r"(?:are you|is it) open|\btimings?\b)\b",
    re.IGNORECASE,
)
# General "about the restaurant" overview questions — handled via a curated
# structured reply (no LLM) so the model cannot echo raw KB markdown verbatim.
# Every branch requires the object noun to actually be the restaurant/place/
# outlet/kiosk itself. Earlier versions of the first two branches stopped at
# "your" without requiring that noun, so "tell me about your PARKING" (or
# seating, wifi, anything else) matched the generic overview path and got the
# canned name/hours/delivery intro instead of an answer to what was actually
# asked. Requiring the noun everywhere means this template only ever claims
# genuine "about the restaurant itself" questions; every other topic — in any
# phrasing, present or future — falls through to the grounded knowledge_lookup
# path below, which is what actually has the relevant fact.
_ROOT_FACT_OVERVIEW_RE = re.compile(
    r"\b(?:"
    r"tell (?:me|us) (?:something |a bit |more )?about (?:the|this|your) "
    r"(?:restaurant|place|outlet|kiosk)|"
    r"(?:something|anything) about (?:the|this|your) (?:restaurant|place|outlet|kiosk)|"
    r"about (?:the|this|your) (?:restaurant|place|outlet|kiosk)|"
    r"what (?:are|is) (?:this|the) (?:restaurant|place|outlet)"
    r"(?!'?s?\s+(?:name|called))|"
    r"what (?:kind of|type of) (?:restaurant|place)|"
    r"introduce (?:yourself|the restaurant)|"
    r"(?:give|can you give) (?:me |us )?(?:an? )?(?:overview|introduction|summary)|"
    r"describe (?:this|the) (?:restaurant|place|outlet)"
    r")\b",
    re.IGNORECASE,
)
# Hours/timing-ish words that don't match the strict phrases above. If one of
# these appears, the utterance IS asking about hours but in a paraphrase this
# classifier does not recognise confidently. Observed live: "what are the
# opening and closing of the restaurant? what is the restaurant name" matched
# only _ROOT_FACT_NAME_RE, so the fast path answered the name and silently
# dropped the hours half of the question — an incomplete answer is worse than
# a slow one. When this hint fires without a confident phrase match, abstain
# from the ENTIRE fast path (not just the unmatched fact) so the full
# pre-grounded LLM path sees and answers the whole question.
_HOURS_HINT_RE = re.compile(r"\b(?:hours?|timing\w*|open(?:ing)?|clos(?:e|ing|ed))\b", re.IGNORECASE)
# "Address" is not a structured root fact this module formats — it lives only
# in the free-form KB text, answered via knowledge_lookup. But
# _ROOT_FACT_OVERVIEW_RE's "what is the restaurant" branch only excludes a
# trailing "'s name/called" in its lookahead, so "what is the restaurant
# address?" still matched it and got the canned name/hours/delivery intro
# with the address silently omitted. Same failure mode as the hours-hint
# case above for compound questions like "the restaurant address and its
# operating hours" — matched only "hours" and dropped address. Abstaining
# the whole fast path whenever "address"/"located"/"location" is mentioned
# routes to the full pre-grounded LLM path, which now answers correctly
# since the KB has the Address field.
_ADDRESS_HINT_RE = re.compile(r"\baddress\b|\blocated?\b|\blocation\b", re.IGNORECASE)


def classify_root_facts(utterance: str) -> list[str]:
    """Return which structured root facts ``utterance`` is asking for.

    Args:
        utterance: The customer's raw message.

    Returns:
        A list drawn from ``{"name", "hours", "breakfast_hours", "overview"}``
        in the order a spoken reply should cover them. Empty when the utterance
        does not ask for any of these specific facts, OR when it hints at
        hours/timing without matching a known phrase confidently — in both
        cases the caller must fall back to the normal (pre-grounded LLM) path
        rather than risk answering only part of a compound question.
    """
    if not utterance:
        return []
    # Address questions are never a structured root fact — abstain the whole
    # fast path so the grounded knowledge_lookup path answers them (and any
    # fact asked alongside them) instead of the overview template silently
    # omitting the address or a compound question losing half its answer.
    if _ADDRESS_HINT_RE.search(utterance):
        return []
    # General overview questions take priority — they absorb the whole reply
    # so there is no need to check individual fact patterns.
    if _ROOT_FACT_OVERVIEW_RE.search(utterance):
        return ["overview"]
    facts: list[str] = []
    if _ROOT_FACT_NAME_RE.search(utterance):
        facts.append("name")
    if _ROOT_FACT_BREAKFAST_RE.search(utterance):
        facts.append("breakfast_hours")
    elif _ROOT_FACT_HOURS_RE.search(utterance):
        facts.append("hours")
    elif _HOURS_HINT_RE.search(utterance):
        return []
    return facts


def speak_root_fact(facts_wanted: list[str], root_facts: dict[str, str]) -> str | None:
    """Deterministically format a reply for the facts in ``facts_wanted``.

    Args:
        facts_wanted: Output of ``classify_root_facts`` — non-empty.
        root_facts: Output of ``knowledge_lookup_tool.root_facts()``.

    Returns:
        A ready-to-speak sentence, or ``None`` when a requested fact is not
        present in ``root_facts`` (e.g. the knowledge base lacks that field,
        or the pin failed) — the caller must fall back to the LLM path rather
        than speak an incomplete answer.
    """
    parts: list[str] = []
    for fact in facts_wanted:
        if fact == "overview":
            name = root_facts.get("brand name", "").strip().rstrip(".")
            cuisine = root_facts.get("cuisine", "").strip().rstrip(".")
            hours = root_facts.get("hours", "").strip()
            delivery = root_facts.get("delivery", "").strip()
            if not name:
                return None
            intro = f"Welcome to {name}"
            if cuisine:
                intro += f", a {cuisine} restaurant"
            intro += "."
            sentences = [intro]
            if hours:
                spoken = hours.replace(chr(0xB7), ",").replace(" ,", ",")
                sentences.append(f"We're open {spoken}.")
            if delivery:
                sentences.append(
                    "We offer dine-in, takeaway, and delivery via Swiggy, Zomato, and our app."
                )
            parts.append(" ".join(sentences))
        elif fact == "name":
            name = root_facts.get("brand name")
            if not name:
                return None
            parts.append(f"We're called {name}.")
        elif fact == "hours":
            hours = root_facts.get("hours")
            if not hours:
                return None
            spoken_hours = hours.replace(chr(0xB7), ",").replace(" ,", ",")
            parts.append(f"Our hours are {spoken_hours}.")
        elif fact == "breakfast_hours":
            hours = root_facts.get("breakfast hours")
            if not hours:
                return None
            parts.append(f"Breakfast is served {hours}.")
    if not parts:
        return None
    return " ".join(parts) + " Would you like to know anything else?"
