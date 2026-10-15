"""OrderingAgent — Google ADK LlmAgent for the kiosk ordering flow.

The agent:
  1. Answers menu/FAQ questions via the ``knowledge_lookup`` tool (RAG pipeline).
  2. Places, updates, gets, and confirms orders via MCP tools on kiosk-core.
  3. Proactively surfaces upsell suggestions after items are added.
  4. Confirms orders with a friendly message and the Order ID.

Usage::

    agent = OrderingAgent()
    await agent.bootstrap()
    reply = await agent.chat(
        session_id="user-session-1",
        user_id="user123",
        message="I'd like a Paneer Tikka Burger please",
    )
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import logging
import re
import time
from typing import Any

from agentic import action_result
from plugins.kiosk import cart_state_guard
from agentic import config as agent_cfg
from plugins.kiosk import confirm_guard
from agentic import domain_config
from plugins.kiosk import directive_mode
from plugins.kiosk import item_intent_guard
from agentic import llm_metrics
from plugins.kiosk import menu_guard
from plugins.kiosk import price_guard
from plugins.kiosk import removal_guard
from plugins.kiosk import reply_templates
from agentic.adk_runtime import create_adk_model, create_runner, create_session_service
from agentic.mcp_client import MCPTool, bootstrap_mcp_tools, call_tool, get_all_tools
from agentic.tools import knowledge_lookup_tool as knowledge_tool
from agentic.tools.knowledge_lookup_tool import knowledge_lookup

logger = logging.getLogger(__name__)

# Current turn's dietary preference (or None), read by ``_mcp_fn`` so
# place_order/update_order calls get it without the LLM having to supply it
# as a tool argument. A ContextVar (not an instance attribute) because
# ``_mcp_fn`` is a bare async function shared by every concurrent session —
# same rationale as menu_guard's ``_turn_state``, scoped per asyncio task.
_dietary_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "dietary_ctx", default=None
)

# The current turn's customer user_id — injected into MCP tool calls that
# accept it so the model never needs to generate "user_id":"anonymous" in
# every tool-call JSON.  Removing user_id from the tool schema saves ~8
# tokens per call (~444ms at 18 tps) and up to 19 tokens for confirm/cancel
# (~1,050ms) where user_id was the *only* argument.
_user_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "user_id_ctx", default="anonymous"
)

# Tools that accept user_id and have it injected server-side.  user_id is
# stripped from their ADK schema so the model never generates it.
# Defaults match Smart Kiosk tool names; updated in-place after MCP bootstrap
# via classify_discovered_tools() using role patterns from agent_profile.yaml.
_USER_ID_INJECTED_TOOLS: action_result._MutableToolSet = action_result._MutableToolSet(
    frozenset({"place_order", "remove_from_order", "cancel_order",
               "confirm_active_order", "get_current_order"})
)

# Tools where dietary is always injected server-side from _dietary_ctx and
# stripped from the model-visible schema.  Without stripping, the model fills
# the optional dietary field with hallucinated values (e.g. "vegetarian" for a
# chicken burger), wasting 5–7 decode tokens per call.  Dietary preferences are
# expressed by the user in natural language, stored in _dietary_prefs by the
# extraction layer, and injected here — the model never needs to generate them.
# Also covers list_products/get_popular_products so a customer who has stated
# a vegetarian/vegan preference gets a veg-only catalogue/bestseller list for
# open-ended "suggest me some dishes"/"what are your favourites" questions,
# without the model needing to ask again or filter free-hand.
# Defaults match Smart Kiosk; updated in-place after MCP bootstrap.
_DIETARY_INJECTED_TOOLS: action_result._MutableToolSet = action_result._MutableToolSet(
    frozenset({"place_order", "update_order", "list_products", "get_popular_products"})
)

# The current turn's raw customer utterance (untouched by the
# ``[customer_name=...]``/``[dietary=...]`` tag prefixes), read by ``_mcp_fn``
# so it can catch a stale item reference in a single-item place_order/
# update_order call — see ``agentic/item_intent_guard.py``. Same ContextVar
# rationale as ``_dietary_ctx`` above.
_utterance_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "utterance_ctx", default=""
)


# Tools that write to the orders/order_items tables. ``dry_run`` is forced
# onto every call to one of these while a speculative draft is running (see
# ``_speculative_ctx`` below), regardless of what the LLM generates — dry_run
# is stripped from the model-visible schema entirely (like user_id/dietary),
# so the model can never see or set it itself. This is a static, mechanical
# list (unlike the role-classified sets above) because "never persist during
# speculation" must hold for every DB-writing tool unconditionally, not just
# ones matching a configurable guard role pattern.
_MUTATING_TOOLS: frozenset[str] = frozenset({
    "place_order", "update_order", "confirm_active_order",
    "remove_from_order", "cancel_order", "confirm_order",
})


def _injected_param_names(tool_name: str) -> set[str]:
    """Return the parameters ``_mcp_fn`` supplies itself for ``tool_name``.

    These are exactly the parameters hidden from the model-visible schema in
    :meth:`OrderingAgent._make_mcp_callable`. Kept as one function so the
    schema and the description can never disagree about what is hidden.

    Args:
        tool_name: MCP tool being wrapped.

    Returns:
        Parameter names the runtime injects for this tool.
    """
    hidden: set[str] = set()
    if tool_name in _USER_ID_INJECTED_TOOLS:
        hidden.add("user_id")
    if tool_name in _DIETARY_INJECTED_TOOLS:
        hidden.add("dietary")
    if tool_name in _MUTATING_TOOLS:
        hidden.add("dry_run")
    return hidden


# Matches a Google-style ``Args:`` entry, capturing its indent and name.
_ARGS_ENTRY_RE = re.compile(r"^(\s*)(\w+)\s*(?:\([^)]*\))?:\s")


def _scrub_injected_args(description: str, hidden: set[str]) -> str:
    """Remove the ``Args:`` entries for runtime-injected parameters.

    Hiding a parameter from the JSON schema is not enough on its own: the tool
    description still documents it, and the model follows the prose. Measured
    against OVMS, ``place_order`` kept emitting ``"user_id": "kiosk-user"``
    purely because its docstring described the argument — 27 emitted tokens
    instead of 17, i.e. ~360ms of avoidable decode on the voice-to-voice
    critical path (1,438ms → 1,079ms).

    Only the entries for ``hidden`` parameters are removed; every other line,
    including the behavioural guidance the ordering guards depend on, is left
    untouched. Blanket-truncating descriptions was measured to be faster still
    but made the model emit malformed calls (``{"items": {"classic chicken":
    "1"}}``), so it is deliberately not done here.

    Args:
        description: The MCP tool's ``prompt_description``.
        hidden:      Parameter names the runtime injects.

    Returns:
        The description with those parameters' ``Args:`` entries removed.
    """
    if not hidden or not description:
        return description

    lines = description.splitlines()
    kept: list[str] = []
    dropping_indent: int | None = None
    for line in lines:
        match = _ARGS_ENTRY_RE.match(line)
        if match is not None:
            indent = len(match.group(1))
            # A new entry at or above the dropped entry's level ends the drop.
            if dropping_indent is not None and indent <= dropping_indent:
                dropping_indent = None
            if match.group(2) in hidden and dropping_indent is None:
                dropping_indent = indent
                continue
        elif dropping_indent is not None:
            # Continuation lines are indented past the entry they belong to;
            # a blank line or an outdent ends the entry.
            if line.strip() and len(line) - len(line.lstrip()) > dropping_indent:
                continue
            dropping_indent = None
        if dropping_indent is None:
            kept.append(line)
    return "\n".join(kept)

# Set for the duration of a speculative (cache-warming / draft) agent call —
# see ``OrderingAgent.chat(..., speculative=True)``. When true, ``_mcp_fn``
# forces ``dry_run=True`` onto every tool in ``_MUTATING_TOOLS`` so a
# speculative run — triggered from a still-changing, unconfirmed ASR preview
# — can never write a real row to the orders database. Defaults to False so
# a real (non-speculative) turn always persists normally, fail-safe.
_speculative_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "speculative_ctx", default=False
)

# The tool_name+kwargs actually dispatched this turn, captured by _mcp_fn so
# a speculative draft's tool call(s) can be replayed for real (dry_run=False)
# at commit time without asking the LLM to decide again — see
# ``OrderingAgent.chat()``'s ``tool_call_detail`` return field and the
# replay path in ``kiosk_core/audio_session.py``. Reset per-turn in chat().
_tool_call_log_ctx: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("tool_call_log_ctx", default=None)
)


class _CartState:
    """Per-session cart/upsell memory, read and written by ``_mcp_fn``.

    Unlike ``_dietary_ctx``/``_utterance_ctx`` (reset every turn), this must
    survive *across* turns for the same conversation — see
    ``agentic/cart_state_guard.py`` for why: catching a place_order/
    update_order call that silently restates an already-cart-resident item
    or auto-accepts an unconfirmed upsell suggestion requires knowing what
    was in the cart, and what was last offered, *before* this turn's call.

    One instance is kept per ``session_id`` in ``OrderingAgent._cart_states``
    and handed to ``_mcp_fn`` via ``_cart_state_ctx`` (a ContextVar bound to
    the same mutable object each turn, not a fresh one) — the same
    shared-mutable-object technique ``_ToolCallCounter`` uses to cross the
    ADK sibling-task boundary, applied here to cross the turn boundary
    instead.
    """

    __slots__ = ("known_items", "pending_upsell")

    def __init__(self) -> None:
        self.known_items: list[dict[str, Any]] = []
        self.pending_upsell: dict[str, Any] | None = None


# The current session's cart/upsell memory (see _CartState above), read and
# updated by ``_mcp_fn`` around each place_order/update_order/remove_from_order
# call. Bound once per turn in ``chat()`` to the same per-session ``_CartState``
# object every time, so mutations persist to the next turn.
_cart_state_ctx: contextvars.ContextVar[_CartState | None] = contextvars.ContextVar(
    "cart_state_ctx", default=None
)


class _ToolCallCounter:
    """Mutable per-turn counter, shared across a turn's parallel tool calls.

    ADK executes multiple tool calls from a single model response as sibling
    ``asyncio.create_task`` coroutines (see
    ``google.adk.flows.llm_flows.functions.handle_function_call_list_async``),
    each of which snapshots the current ``contextvars`` bindings *at task
    creation*. A plain ``int`` ContextVar's ``.set()`` from inside one such
    task would therefore be invisible to its siblings — each would see
    "calls_this_turn == 1" independently. Using a shared mutable object
    (bound once by ``begin_turn`` before any tasks exist) sidesteps this: all
    sibling tasks hold a reference to the *same* counter instance, exactly
    the pattern ``menu_guard._TurnState`` already relies on.
    """

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def increment(self) -> int:
        self.count += 1
        return self.count


# How many MCP tools have been invoked so far this turn, read by ``_mcp_fn`` to
# decide whether it's safe to skip the narration LLM call. Only the turn's
# *first* tool call may be templated: if the model calls a second tool, the
# customer's utterance needed more than one action, and a fixed template for
# the first call alone would silently drop whatever the second call answers.
# Reset at the top of every ``chat()`` turn, same rationale as ``_dietary_ctx``.
_tool_call_count_ctx: contextvars.ContextVar[_ToolCallCounter] = contextvars.ContextVar(
    "tool_call_count_ctx", default=None
)

# Qwen3 hybrid-reasoning models wrap chain-of-thought in explicit <think> tags.
# Match them precisely rather than guessing from paragraph structure.
#
# An earlier heuristic split the reply on blank lines and kept only the last
# paragraph. That silently destroyed every legitimate multi-paragraph answer:
# a 786-char menu listing was reduced to its trailing 53-char follow-up
# question ("Would you like to explore any of these items further?"), so the
# customer heard a question about a list they were never told. The service runs
# with enable_thinking=False by default, meaning no <think> block is emitted at
# all and that heuristic could only ever cause harm.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"^\s*<think>.*?(?:\n\n|\Z)", re.DOTALL | re.IGNORECASE)

# A turn that announces a lookup but calls no tool is a dead end — the customer
# is told "let me check that" and never gets the answer.
_PROMISE_RE = re.compile(
    r"\b(?:let me (?:look|check|find|see|get|pull)"
    r"|i(?:'ll| will) (?:look|check|find|see|get|pull)"
    r"|i can (?:look|check|find|tell you)"
    r"|checking (?:that|on that)"
    r"|one moment|hold on|give me a moment)\b",
    re.IGNORECASE,
)

_RETRY_NUDGE = (
    "You replied without calling a tool, so the customer received no answer. "
    "Call the appropriate tool now and answer using only its result: "
    "use list_products for menu items, prices, or availability; "
    "use knowledge_lookup for hours, ingredients, allergens, or policies."
)

# ---------------------------------------------------------------------------
# Single-add utterance detection — used to decide whether place_order /
# update_order can safely skip the 2nd LLM narration call.
#
# Safety contract (conservative by design):
#   False-positive (compound treated as simple) → wrong: customer says
#     "add X and remove Y" but only X is added; Y is never removed.
#   False-negative (simple treated as compound) → wasteful: a 2nd LLM call
#     runs unnecessarily, costing ~2.7s.
#
# We only allow the shortcut when BOTH removal-intent and multi-item ordering
# keywords are absent from the utterance.  Any doubt → fall through to the
# normal 2nd-LLM path.
# ---------------------------------------------------------------------------

# Keywords that signal a REMOVE, CANCEL, or SWAP action is also requested.
_REMOVAL_IN_UTTERANCE_RE = re.compile(
    r"\b(?:remove|take\s+off|drop\s+the|i\s+don'?t\s+want|cancel\s+the|"
    r"swap|instead\s+of|in\s+place\s+of|replace|no\s+more)\b",
    re.IGNORECASE,
)

# "and a/an/one/the <word>" or "and fries" → a second item in the same turn.
_MULTI_ADD_IN_UTTERANCE_RE = re.compile(
    r"\b(?:and|also|plus)\s+(?:(?:a|an|one|the|\d+)\s+)?\w{3,}",
    re.IGNORECASE,
)


def _is_single_add_utterance(utterance: str) -> bool:
    """Return True when the utterance almost certainly expresses ONE add only.

    This is used to gate the deterministic-narration shortcut for
    ``place_order``/``update_order``.  The check is intentionally strict:
    any sign of a compound action (remove + add, multiple adds) causes it to
    return False so the normal 2nd-LLM path handles the ambiguity.

    Args:
        utterance: The raw customer utterance for this turn.

    Returns:
        True when neither removal-intent nor multi-add patterns are found,
        meaning it is safe to template the reply without a 2nd LLM call.
    """
    if not utterance:
        return False
    if _REMOVAL_IN_UTTERANCE_RE.search(utterance):
        return False
    if _MULTI_ADD_IN_UTTERANCE_RE.search(utterance):
        return False
    return True

# A "we don't have that information" reply is only ever legitimate when it is
# grounded in a lookup that actually ran this turn. Emitted with no tool call it
# means the model replayed an earlier refusal from conversation history, which
# is exactly how a single mis-transcribed question ("hours" heard as "arts")
# used to poison every later attempt in the same session.
_REFUSAL_RE = re.compile(
    r"(?:do(?:n't| not) have|no information|not sure|unable to find|"
    r"couldn(?:'t|t) find|could not find|isn(?:'t| not) available|"
    r"can(?:'t|not| not) (?:provide|tell|give|share|answer)|"
    r"no (?:tool|data) available)",
    re.IGNORECASE,
)

# Questions the knowledge base answers. When one of these is asked and the
# model produces no tool call, the answer is invented — observed: confidently
# stating "11:00 AM to 10:00 PM, seven days a week" for a restaurant whose
# real hours differ every day. Unlike a refusal this reads as authoritative,
# so it cannot be detected from the reply text alone; it is detected from the
# question instead.
_KNOWLEDGE_QUERY_RE = domain_config.build_knowledge_regex() or re.compile(
    r"\b(?:open(?:ing)?|clos(?:e|ing)|hours?|timing|breakfast|"
    r"restaurant name|name of (?:the|your) restaurant|address|located?|location|"
    r"parking|deliver(?:y|ies)?|takeaway|dine[- ]?in|wifi|contact|phone|"
    r"halal|vegetarian|vegan|allergen|gluten|ingredient|spicy|"
    r"payment|upi|card|cash|policy|refund|"
    r"restrooms?|bathrooms?|washrooms?|toilets?|facilit(?:y|ies)|"
    r"wheelchair|accessib\w*|seating|"
    r"(?:tell|know|hear|learn) (?:me )?(?:something |anything )?about "
    r"(?:the|your|this) (?:restaurant|outlet|place|kiosk)|"
    r"about (?:the|your|this) (?:restaurant|outlet|place))\b",
    re.IGNORECASE,
)

_KNOWLEDGE_NUDGE = (
    "You answered a question about the outlet from memory. You have no such "
    "memory — anything you stated is invented and may be wrong. Call "
    "knowledge_lookup now with the customer's question, then answer in one or "
    "two spoken sentences using only what it returns."
)

# Catalogue questions must always be answered from list_products. The model is
# otherwise happy to invent a confident, well-formatted menu — one turn asking
# "can you tell me the desserts?" produced eight fictional items at a uniform
# price when the catalogue holds two. That fabrication then propagates: the
# customer tries to order an item that does not exist and every subsequent
# place_order fails. Neither _PROMISE_RE nor _REFUSAL_RE catches this, because a
# hallucinated menu neither promises a lookup nor refuses one.
_CATALOGUE_QUERY_RE = domain_config.build_catalogue_regex() or re.compile(
    r"\b(?:menu|item|items|dish|dishes|serve|serves|offer|offers|available|"
    r"option|options|price|prices|cost|costs|how much|rate|rates|"
    r"burger|burgers|pizza|pizzas|wrap|wraps|side|sides|dessert|desserts|"
    r"desert|deserts|sweet|sweets|beverage|beverages|drink|drinks|combo|combos|"
    r"(?:do|does) you (?:have|serve|sell|offer|carry|make|got)|"
    r"have you got|any (?:sandwiches?|food|snacks?))\b",
    re.IGNORECASE,
)

_CATALOGUE_NUDGE = (
    "You answered a menu, item, or price question without calling a catalogue "
    "tool, so your answer is not grounded in the real catalogue and may contain "
    "items that do not exist. If the customer named a category or product, call "
    "list_products now. If they asked about the menu in general without naming a "
    "category, call list_categories instead and offer them the categories. Reply "
    "using ONLY what the tool returns — do not add, rename, or re-price anything."
)

# Claiming an order was confirmed without calling confirm_order is the most
# damaging failure the kiosk can produce: the customer walks away believing the
# order is placed, the kitchen never receives it, and the recited total can
# include upsell items that were only ever suggestions. Prompt wording alone did
# not stop it, so the claim is detected and corrected deterministically.
_ORDER_ACTION_RE = re.compile(
    r"\b(?:confirm|confirmed|place (?:the |my )?order|checkout|check out|"
    r"finali[sz]e|proceed|that(?:'s| is) all|that(?:'s| is) it|"
    r"i(?:'m| am) done|cancel|remove|pay)\b",
    re.IGNORECASE,
)

# A read-only order-status query ("what is the status of my order?", "track my
# order", "did my order go through?"). Distinct from _ORDER_ACTION_RE, which
# only recognises mutation/confirmation verbs — none of which appear here, so
# this phrasing previously matched no intent regex at all. Observed live: with
# no tool call and no pre-grounding, the model either produced an ungrounded
# reply that fell through to the generic out-of-scope refusal, or on a later
# turn picked the wrong tool entirely (remove_from_order) for the identical
# question. Rule 5 in the system prompt names get_current_order for "show my
# order"/"what's my total" but never for "status" phrasing — this regex closes
# that gap by forcing the same _ORDER_NUDGE retry (which already names
# get_current_order) whenever the model answers one of these without a tool.
_ORDER_STATUS_RE = re.compile(
    r"status\s+of\s+(?:my|the|this)\s+order|track\s+(?:my|the)\s+order|"
    r"how(?:'s| is)\s+(?:my|the)\s+order|did\s+(?:my|the)\s+order\s+go\s+through|"
    r"what(?:'s| is)\s+happening\s+with\s+(?:my|the)\s+order",
    re.IGNORECASE,
)

# A reset intent ("start a new order", "start over", "let's start fresh",
# "begin a new order") — the customer wants the current draft cleared, not a
# complaint-style cancellation. Rule 6b in the system prompt already names
# "start over" as a cancel_order trigger, but only as prompt guidance with no
# deterministic backstop, the same gap _ORDER_STATUS_RE closed for status
# phrasing. Distinct from _ORDER_ACTION_RE's bare "cancel" — none of these
# phrases contain that word, so this regex cannot double-fire against it; it
# routes to the same cancel_order tool via _ORDER_NUDGE.
_START_OVER_RE = re.compile(
    r"start\s+(?:over|fresh|a\s+new\s+order|again)|"
    r"begin\s+(?:a\s+)?new\s+order|"
    r"new\s+order\s+please",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Scope guard
# ---------------------------------------------------------------------------
# Every recovery guard above is *intent-keyed*: the missing-tool retry,
# _force_knowledge and _force_catalogue only fire when the message matches one
# of the intent regexes. A question matching none of them therefore reaches the
# customer with no grounding check whatsoever, straight from the model's
# parametric memory. Observed live on a kiosk turn with tool_calls=[] and no
# pre-grounding: "who is the president of India?" was answered in full
# ("Droupadi Murmu... took office on 17 July 2022"), as was "what is the size
# of a football player?". Both are confident and entirely unverifiable. A kiosk
# that will state arbitrary world facts cannot be trusted on the facts that
# matter — prices, allergens, what is in the cart.
#
# The test is deliberately *positive*: a turn stays on the existing path when
# the customer's words touch the kiosk's world at all. Only a question with no
# such connection is routed through a real knowledge lookup and refused when
# the knowledge base cannot support it. Parametric memory is never spoken.
_QUESTION_RE = re.compile(
    r"\?|\b(?:who|what|when|where|why|which|whose|whom)\b|"
    r"^\s*(?:can|could|do|does|did|is|are|was|were|will|would|should|tell)\b",
    re.IGNORECASE,
)

# Vocabulary marking a turn as belonging to the kiosk's world. Kept broad on
# purpose: classifying a turn in-domain merely leaves it on today's path, while
# classifying it out-of-domain refuses it. The asymmetry is resolved in the
# customer's favour, so this list errs towards inclusion.
_IN_DOMAIN_RE = domain_config.build_domain_regex() or re.compile(
    r"\b(?:order|orders|cart|bill|total|checkout|pay|payment|price|cost|"
    r"menu|item|items|food|eat|drink|drinks|meal|combo|snack|"
    r"restaurant|outlet|kiosk|store|shop|kitchen|staff|table|"
    r"seat|seats|seating|sit|sitting|"
    r"add|added|remove|cancel|confirm|serve|serves|recommend|suggest|"
    r"veg|vegan|vegetarian|halal|allergen|gluten|spicy|calorie|calories|"
    r"burger|burgers|pizza|pizzas|wrap|wraps|fries|dessert|desserts|"
    r"beverage|beverages|coffee|lassi|soda|pepsi|roll|rolls|"
    r"open|opening|close|closing|hours?|timing|address|location|parking|"
    r"wifi|delivery|takeaway|offer|offers|discount|"
    r"washroom|restroom|toilet|water|napkin|straw|takeout|parcel)\b",
    re.IGNORECASE,
)

# A contextual follow-up ("what else?", "which one?") carries no domain
# vocabulary of its own — it inherits it from the previous turn. Refusing those
# would break ordinary conversation, and they are far too short to be the
# world-knowledge questions this guard exists to catch, which always name their
# subject ("who is the president of India"). Four words is comfortably below
# the shortest such question and comfortably above the longest follow-up.
_MIN_OUT_OF_SCOPE_WORDS = 4

# Short social turns are not questions to be grounded — refusing "how are you?"
# with a scope message would be brusque and pointless. The bounded tail keeps
# this to genuinely short utterances so it cannot swallow a real question that
# merely opens with "ok" or "sorry".
_SMALLTALK_RE = re.compile(
    r"^\s*(?:hi|hey|hello|good (?:morning|afternoon|evening)|"
    r"how (?:are|r) (?:you|u)|how(?:'s| is) it going|what(?:'s| is) up|"
    r"thanks|thank you|thankyou|cheers|bye|goodbye|see you|"
    r"ok|okay|yes|no|yeah|yep|nope|sure|please|sorry)\b.{0,24}$",
    re.IGNORECASE | re.DOTALL,
)

_OUT_OF_SCOPE_FALLBACK = domain_config.get_out_of_scope_fallback()



def _is_out_of_scope(message: str) -> bool:
    """Report whether a customer turn falls outside the kiosk's remit.

    A turn is out of scope when it *asks* something but contains no vocabulary
    connecting it to the restaurant, its menu, or an order. Statements are
    never out of scope: "my name is Ravi" wants a conversational reply, not a
    refusal.

    Args:
        message: The customer's utterance.

    Returns:
        ``True`` when the turn must be grounded or refused rather than answered
        from the model's own memory.
    """
    if not message or _SMALLTALK_RE.match(message):
        return False
    if len(re.findall(r"[A-Za-z']+", message)) < _MIN_OUT_OF_SCOPE_WORDS:
        return False
    if (
        _IN_DOMAIN_RE.search(message)
        or _KNOWLEDGE_QUERY_RE.search(message)
        or _CATALOGUE_QUERY_RE.search(message)
        or _ORDER_ACTION_RE.search(message)
    ):
        return False
    return bool(_QUESTION_RE.search(message))


_ORDER_CLAIM_RE = re.compile(
    r"(?:order (?:is |has been )?(?:confirmed|placed)|"
    r"confirmed your order|order id|order number|order #)",
    re.IGNORECASE,
)

# A question about a PAST order action ("what did you remove earlier?", "in
# this conversation what did you remove?") — a WH-question with "you"/"I" as
# subject and an order verb as a past participle. This is a read-only recall
# from the model's own conversation history, not a fresh action request, even
# though it contains an _ORDER_ACTION_RE keyword like "remove".
#
# Observed live: "What did you remove from my cart earlier?" matched
# _ORDER_ACTION_RE (bare "remove") in _needs_tool_retry, forcing a retry and
# then _ORDER_CLAIM_FALLBACK, which replaced an already-correct, truthful
# answer ("I removed Aloo Tikki Burger from your cart.") with "Sorry, I could
# not complete that just now" — a wrong answer to a question the model had
# already answered correctly, because no tool exists (or is needed) to answer
# "what did you do a moment ago"; that's just the turn's own memory.
_HISTORY_QUERY_RE = re.compile(
    r"\bwhat\b[^.?!]{0,30}\b(?:did|have|has)\b[^.?!]{0,15}\b(?:you|i)\b[^.?!]{0,20}"
    r"\b(?:remove|removed|add|added|order|ordered|place|placed|confirm|confirmed|"
    r"cancel|cancelled)\b",
    re.IGNORECASE,
)

# Tools that actually finalise an order. Only these make a "your order is
# confirmed" sentence true. Sourced from action_result.CLAIM_TOOLS — the
# single registry also consumed by confirm_guard.py and menu/removal guards,
# so a new confirm-type tool only needs to be added in one place.
_CONFIRM_TOOLS = action_result.CLAIM_TOOLS[action_result.ORDER_CONFIRMED]


# A claim that the order is *finalised*, as opposed to merely added to. This is
# narrower than _ORDER_CLAIM_RE: adding an item legitimately mentions the order,
# but must never say it is confirmed.
#
# The gap between "order" and the confirmation verb is deliberately bounded but
# non-trivial: a real reply says "Your order for 1 Classic Chicken Burger is
# confirmed", not "Your order is confirmed" — the item description sits between
# the noun and the verb. An earlier version of this regex required them to be
# adjacent and consequently never matched real replies, silently letting every
# "place_order ran, order is still draft" false-confirmation through. Caught by
# replaying a real conversation and diffing the spoken reply against the actual
# tool result (status stayed "draft" both times "confirmed" was said aloud).
_CONFIRM_CLAIM_RE = re.compile(
    r"order\b[^.!?]{0,80}?\b(?:is|has been|was)\b[^.!?]{0,15}?\b(?:now\s+)?"
    r"(?:confirmed|placed|finalis(?:ed)?|finaliz(?:ed)?|complete)\b"
    r"|confirmed your order"
    r"|order (?:id|number|#)",
    re.IGNORECASE,
)

# Substituted for a stripped confirmation claim on a turn that only added items:
# the cart is real, it simply has not been confirmed yet.
_UNCONFIRMED_TAIL = "Would you like to confirm your order?"


def _strip_false_confirmation(reply: str, tool_calls: list[str]) -> tuple[str, bool]:
    """Remove "your order is confirmed" claims that no confirm tool *actually
    completed*.

    ``place_order`` and ``update_order`` leave the order in ``draft``. A reply
    that nonetheless says the order is confirmed sends the customer away
    believing the kitchen has their food, which is the single worst failure this
    kiosk can produce.

    Checking tool *names* alone is not enough: the model can call
    ``confirm_order`` with a hallucinated ``order_id`` (observed live: the
    literal guess ``12345``), which fails with an "Order not found" error —
    nothing is confirmed, yet ``confirm_order`` is still in ``tool_calls``.
    This checks ``confirm_guard``'s recorded *result* instead of the bare tool
    name, so a failed confirm attempt is treated the same as no attempt at all.

    Only the offending sentences are dropped, not the whole reply: the item
    names, prices, and upsell lines around them came from a real tool result and
    are worth keeping.

    Args:
        reply: The assistant's drafted reply.
        tool_calls: Tools invoked this turn, in call order. Unused for the
            confirm decision itself (kept for the caller's logging) — the
            source of truth is ``confirm_guard.current_state()``.

    Returns:
        ``(reply, changed)`` — the cleaned reply and whether anything was cut.
    """
    if not reply or confirm_guard.current_state().succeeded:
        return reply, False
    if not _CONFIRM_CLAIM_RE.search(reply):
        return reply, False

    # A confirm was attempted this turn and failed (e.g. a hallucinated
    # order_id) rather than never being attempted at all — say so honestly
    # instead of the generic "would you like to confirm?" invitation, which
    # reads as if the customer hasn't asked yet when they just did.
    fallback = (
        confirm_guard.build_refusal()
        if confirm_guard.current_state().attempted
        else _UNCONFIRMED_TAIL
    )

    sentences = [s.strip() for s in _SENTENCE_END_RE.split(reply) if s.strip()]
    kept = [s for s in sentences if not _CONFIRM_CLAIM_RE.search(s)]
    if not kept:
        # The entire reply was the false claim — there is nothing truthful left
        # to keep, so ask for confirmation instead of inventing content.
        return fallback, True

    cleaned = " ".join(kept)
    if "confirm" not in cleaned.lower():
        cleaned = f"{cleaned} {fallback}"
    return cleaned, True


_ORDER_NUDGE = (
    "You described a change to the order without calling a tool, so nothing was "
    "actually recorded and any order id or total you stated is invented. Call the "
    "correct tool now — confirm_active_order to confirm (or confirm_order if you have the id), update_order to change items, "
    "cancel_order to clear the whole cart (including a request to start over/start a new order), "
    "get_current_order to read the current order (never guess an order_id for this) "
    "— and reply using ONLY its result. "
    "Never list an upsell suggestion as if the customer had ordered it."
)


_NUDGE_NAMES = {
    id(_CATALOGUE_NUDGE): "catalogue",
    id(_ORDER_NUDGE): "order",
    id(_RETRY_NUDGE): "generic",
}

# Sent when the model wrote a tool call as prose instead of invoking it. The
# int4 model does this intermittently, most often on compound questions
# ("What is the restaurant name and your opening hours?"), where it emits
# `knowledge_lookup {...}` as text and makes no function call at all.
_TOOL_SYNTAX_NUDGE = (
    "You wrote the name of a tool and its arguments as plain text instead of "
    "calling it. That is not a reply and the customer cannot hear it. Invoke "
    "the tool properly now using the function-calling interface, then answer "
    "the customer in one or two spoken sentences using only the tool result. "
    "If the question has two parts, call the tool once and answer both parts "
    "from that single result."
)
_NUDGE_NAMES[id(_TOOL_SYNTAX_NUDGE)] = "tool-syntax"
_NUDGE_NAMES[id(_KNOWLEDGE_NUDGE)] = "knowledge"

# Narrower than _ORDER_ACTION_RE: only utterances that unambiguously mean
# "finalise my order". Deliberately excludes cancel/remove/pay, which must
# never be resolved by auto-confirming.
# Singular forms are enough: kiosk-core's list_products normalises singulars
# and synonyms ("drink" -> beverages) before querying.
_CATEGORY_KEYWORDS = domain_config.get_category_names() or (
    "burger", "pizza", "wrap", "side", "beverage", "drink", "dessert", "fries",
)

# Deliberately excludes ambiguous phrases like "that's it" / "that's all" /
# "I'm done" — those are commonly used to decline an upsell, not to confirm
# an order.  Only phrases that unambiguously express a desire to place/confirm
# the order are kept here, to prevent _force_confirm() from firing when the
# customer is merely wrapping up a browse turn.
_CONFIRM_INTENT_RE = re.compile(
    r"\b(?:confirm|confirmed|checkout|check out|finali[sz]e|place (?:the |my )?order)\b",
    re.IGNORECASE,
)

# Tools that actually mutate or read an order. A reply claiming an order was
# placed or confirmed is only trustworthy if one of these ran this turn.
# Sourced from action_result.ORDER_TOOLS (== every claim-bearing tool, plus
# the read-only get_order) — the single registry also consumed by every
# guard, so a new order-related tool only needs to be added in one place.
_ORDER_TOOLS = action_result.ORDER_TOOLS

_ORDER_CLAIM_FALLBACK = domain_config.get_order_claim_fallback()



def _needs_tool_retry(reply: str, message: str) -> tuple[bool, str]:
    """Decide whether a tool-less turn should be retried once.

    Args:
        reply: The concatenated reply text produced without any tool call.
        message: The customer utterance that triggered the turn.

    Returns:
        A ``(should_retry, nudge)`` pair. ``nudge`` is the correction to send
        back to the model and is empty when no retry is needed.
    """
    # Most specific signal first: the model named a tool but never called it.
    if _TOOL_MENTION_RE.search(reply):
        return True, _TOOL_SYNTAX_NUDGE
    if (
        _ORDER_ACTION_RE.search(message) and not _HISTORY_QUERY_RE.search(message)
    ) or _ORDER_STATUS_RE.search(message) or _START_OVER_RE.search(message) or _ORDER_CLAIM_RE.search(reply):
        return True, _ORDER_NUDGE
    if _CATALOGUE_QUERY_RE.search(message):
        return True, _CATALOGUE_NUDGE
    if _KNOWLEDGE_QUERY_RE.search(message) and not _is_dietary_statement_only(message):
        return True, _KNOWLEDGE_NUDGE
    if _PROMISE_RE.search(reply) or _REFUSAL_RE.search(reply):
        return True, _RETRY_NUDGE
    return False, ""


def _strip_thinking(reply: str) -> str:
    """Remove Qwen3 ``<think>`` reasoning blocks from a reply.

    Args:
        reply: Raw concatenated model output.

    Returns:
        The reply with any thinking block removed. Text outside the tags is
        preserved verbatim — multi-paragraph answers stay intact.
    """
    if not reply:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", reply)
    # An unterminated <think> (truncated generation) would otherwise be spoken.
    if "<think>" in cleaned.lower():
        cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    cleaned = cleaned.strip()
    if cleaned != reply.strip():
        logger.debug("[AGENT] Stripped thinking block (%d → %d chars)", len(reply), len(cleaned))
    return cleaned


# Tools the model may name. Defaults to Smart Kiosk tool names; rebuilt after
# MCP bootstrap by _apply_bootstrap_tool_classification() so a domain that
# adds a new MCP tool automatically gets it guarded against prose leakage.
# knowledge_lookup is always included (it is never in a cart category).
_TOOL_NAMES: tuple[str, ...] = (
    "knowledge_lookup",
    "place_order", "update_order", "remove_from_order", "cancel_order",
    "confirm_order", "confirm_active_order",
    "get_order", "get_current_order",
    "list_categories", "list_products", "get_popular_products", "get_product_details",
    "get_upsell_suggestions",
)

# A snake_case tool name has no place in anything spoken to a customer, so its
# presence anywhere in a reply is a leak regardless of surrounding syntax.
# Observed prose form (JSON regex below does not match it):
#     "I will call the list_categories tool to show the available categories."
_TOOL_MENTION_RE = re.compile("|".join(_TOOL_NAMES))

# Observed on out-of-domain questions ("What is the capital of France?"): the
# model answers in ~0.9 s with a single LLM call and no tool execution, and the
# reply text is literally
#     knowledge_lookup
#     {"question": "What is the capital of France?"}
# On a voice kiosk that JSON is spoken aloud to the customer. The turn is
# already lost at that point, so the only question is what the customer hears.
_TOOL_SYNTAX_RE = re.compile(
    r"(?:^|\n)\s*(?:```[a-zA-Z]*\s*)?(?:functions[.:]\s*)?"
    r"(?:" + "|".join(_TOOL_NAMES) + r")"
    r"\s*(?:\(\s*)?\{.*?\}\s*\)?\s*(?:```)?",
    re.DOTALL,
)

# The model sometimes emits a tool error payload verbatim as its reply, e.g.
# `<error> {"error": "No relevant tool found", "available_products": []} </error>`.
# Spoken aloud this is gibberish and tells the customer nothing.
_ERROR_PAYLOAD_RE = re.compile(
    r"<\s*error\s*>|\{\s*\"error\"\s*:", re.IGNORECASE
)

_TOOL_SYNTAX_FALLBACK = (
    "I can help with our menu, opening hours, and taking your order. "
    "What would you like?"
)


def _apply_bootstrap_tool_classification(discovered_tool_names: list[str]) -> None:
    """Classify MCP-discovered tools by role patterns and update all module globals.

    Called once after MCP bootstrap with the full list of tool names returned
    by the MCP server(s).  Uses ``domain_config.classify_discovered_tools()``
    to match names against the role patterns in ``agent_profile.yaml``'s
    ``tool_roles`` section.  All mutable holders (_USER_ID_INJECTED_TOOLS,
    _DIETARY_INJECTED_TOOLS, action_result.CLAIM_TOOLS) are updated in-place so
    guards that captured a reference at import time see the new contents
    automatically.  Regex globals (_TOOL_NAMES, _TOOL_MENTION_RE,
    _TOOL_SYNTAX_RE) are rebound since re.Pattern is immutable.

    This eliminates the need to list MCP tool names in agent_profile.yaml —
    the only per-domain knowledge required is the role patterns (regex), not
    the exact tool names that the MCP server exposes.
    """
    global _TOOL_NAMES, _TOOL_MENTION_RE, _TOOL_SYNTAX_RE  # noqa: PLW0603

    classified = domain_config.classify_discovered_tools(discovered_tool_names)

    # 1. Update action_result guards (CLAIM_TOOLS, MUTATING_TOOLS, ORDER_TOOLS)
    action_result.update_claim_tools_from_classified(classified)

    # 2. Update injection sets in-place (mutable holders — guards hold a ref)
    if classified.get("user_id_injected"):
        _USER_ID_INJECTED_TOOLS.update(classified["user_id_injected"])
    if classified.get("dietary_injected"):
        _DIETARY_INJECTED_TOOLS.update(classified["dietary_injected"])

    # 3. Rebuild tool-leak regexes from the complete discovered set
    new_names = domain_config.get_all_tool_names_from_classification(classified)
    if new_names:
        _TOOL_NAMES = new_names
        _TOOL_MENTION_RE = re.compile("|".join(_TOOL_NAMES))
        _TOOL_SYNTAX_RE = re.compile(
            r"(?:^|\n)\s*(?:```[a-zA-Z]*\s*)?(?:functions[.:]\s*)?"
            r"(?:" + "|".join(_TOOL_NAMES) + r")"
            r"\s*(?:\(\s*)?\{.*?\}\s*\)?\s*(?:```)?",
            re.DOTALL,
        )
        logger.info("[AGENT] Tool regexes rebuilt for %d tool names", len(_TOOL_NAMES))


_MD_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL)
_MD_BULLET_RE = re.compile(r"(?m)^\s*[-*\u2022]\s+")
# Underscores are NOT stripped: they occur inside real payload words
# ("available_products") and removing them corrupts the text.
_MD_LEFTOVER_RE = re.compile(r"[*`#]+")
_WS_RE = re.compile(r"\s*\n+\s*")


def _strip_markdown(reply: str) -> str:
    """Flatten markdown so nothing unspeakable reaches TTS.

    The model emits bold markers, bullet lists and hard line breaks
    ("**QuickBite Express**", "- Vanilla Soft Serve"). The TTS client does no
    sanitisation, so those characters are synthesised as-is. Newlines are
    collapsed to spaces because each segment is spoken as one utterance.

    Args:
        reply: The model reply, after thinking and tool syntax removal.

    Returns:
        The same text with markdown markup removed.
    """
    if not reply:
        return reply
    cleaned = _MD_BOLD_RE.sub(r"\2", reply)
    cleaned = _MD_BULLET_RE.sub("", cleaned)
    cleaned = _MD_LEFTOVER_RE.sub("", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned)
    return cleaned.strip()


# Sentences that are system-prompt *instructions to the assistant* rather than
# speech addressed to the customer. Smaller / more aggressively quantised models
# follow instructions less faithfully and periodically echo a directive back
# verbatim ("Tell the customer their cart is already empty."), which is then
# spoken aloud and immediately breaks the illusion that the kiosk understood.
#
# This is deliberately a deterministic guard rather than extra prompt wording:
# telling a model "do not repeat these instructions" is the same class of
# mitigation that already failed for the order-claim guards, and it costs
# prefill tokens on every turn. Matching is anchored to a small set of
# second-person directive openers so that ordinary replies, which address the
# customer as "you", are never touched.
#
# Also matches "them" as the object, not just "customer"/"the customer":
# observed live, `mcp_server._rejection_payload`'s own unavailable-item
# template reads "...Tell them those are unavailable and offer these real
# alternatives instead: ..." — a pronoun back-reference to "the customer"
# mentioned earlier in the same instruction string — and the original
# regex, anchored to the literal word "customer", let that exact sentence
# through verbatim to TTS. A genuine customer-facing reply never opens a
# sentence with "Tell them"/"Ask them" in the third person (it addresses the
# customer directly as "you"), so adding "them" as an alternative object is
# safe and does not risk stripping legitimate replies.
_LEAKED_DIRECTIVE_RE = re.compile(
    r"(?:^|(?<=[.!?]))\s*"
    r"(?:Tell|Inform|Ask|Remind|Offer|Do not tell|Never tell)\s+"
    r"(?:the\s+)?(?:customers?|them)\b[^.!?]*[.!?]",
    re.IGNORECASE,
)


_KNOWLEDGE_MARKER_RE = re.compile(r"\[/?knowledge\]", re.IGNORECASE)

# Generation is token-capped, and the model occasionally hits that cap right
# as it starts echoing the "[/knowledge]" closing delimiter, leaving a
# dangling "[/knowledge" (no closing bracket) as the very last thing in the
# reply. `_KNOWLEDGE_MARKER_RE` requires the closing "]" so it never matches
# this truncated fragment. It only ever appears at the tail of the reply, so
# it is safe to trim unconditionally once detected there.
_TRUNCATED_KNOWLEDGE_TAIL_RE = re.compile(r"\s*\[/?knowledge\b[^\]]*$", re.IGNORECASE)

# knowledge_lookup's docstring (and the system prompt) tell the model the
# retrieved excerpts are numbered "[1]", "[2]" and warn it never to speak that
# numbering aloud — but a smaller quantised model imitates the citation
# *style* even on turns where the pinned block (which carries no numbering)
# is the only source, hallucinating a leading "[1]" it never actually saw.
_CITATION_MARKER_RE = re.compile(r"\[\d{1,2}\]\s*")


def _strip_citation_markers(reply: str) -> str:
    """Remove hallucinated ``[1]``/``[2]``-style citation markers.

    Args:
        reply: Model output.

    Returns:
        The reply with any leading numeric citation markers removed. Falls
        back to the original text if stripping would leave nothing speakable.
    """
    if not reply or not _CITATION_MARKER_RE.search(reply):
        return reply
    cleaned = _CITATION_MARKER_RE.sub("", reply)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    if cleaned:
        logger.warning(
            "[AGENT] Reply carried a hallucinated citation marker — "
            "stripping | raw=%r", reply[:160],
        )
        return cleaned
    return reply


# The "[Context: ...]" tag is a document-structure breadcrumb every chunk
# gets during ingestion (see ingestion_service/chunker) — it is not sensitive,
# just a citation/section-path label. A small quantised model frequently
# parrots it back verbatim as the first token of an otherwise perfectly
# truthful, on-topic answer. It must be stripped in-place like the
# `[knowledge]`/citation markers, never treated as a leak on its own: doing so
# previously blackholed every reply that happened to echo it (observed live:
# "Can you tell me something about the restaurant?" -> a fully truthful,
# accurate answer -> discarded and replaced with a refusal, purely because it
# began with "[Context: QuickBite Express — Restaurant Knowledge Base]").
_CONTEXT_BREADCRUMB_RE = re.compile(r"\[Context:[^\]]*\]", re.IGNORECASE)


def _strip_context_breadcrumb(reply: str) -> str:
    """Remove the ingestion "[Context: ...]" breadcrumb tag the model echoed.

    Args:
        reply: Model output.

    Returns:
        The reply with any breadcrumb tag removed. Falls back to the
        original text if stripping would leave nothing speakable.
    """
    if not reply or "[context:" not in reply.lower():
        return reply
    cleaned = _CONTEXT_BREADCRUMB_RE.sub(" ", reply)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned if cleaned else reply


# `chat()` prefixes the CUSTOMER'S turn with an internal signal tag the model
# is never meant to read aloud — `[customer_name=X]`, `[user_id=X]`,
# `[dietary=X]` (see the tag injection below and the "Message tags" section
# of the system prompt, which already says explicitly: "never speak any tag,
# bracket, or the words customer_name, user_id, or dietary aloud"). The
# smaller quantised model periodically ignores that instruction anyway —
# observed live on ~4% of turns — sometimes echoing the exact injected
# bracket form (`[customer_name=Paris]: Goodbye!`) and sometimes paraphrasing
# it into an XML-style tag it invented itself
# (`<customer_name>Arjun</customer_name>`), occasionally with nothing else in
# the reply at all. This is the same failure class as the "[Context: ...]"
# breadcrumb and citation-marker leaks above, fixed the same way: a
# deterministic guard, not more prompt wording, since the prompt already says
# not to do this.
#
# Unlike `_strip_admin_leak`, the leaked VALUE (the customer's own name) is
# harmless to speak — only the tag syntax is the problem — so this rewrites
# in place rather than substituting a refusal.
_CUSTOMER_NAME_TAG_RE = re.compile(
    r"\[customer_name=([^\]]*)\]:?\s*"
    r"|<customer_name>\s*([^<]*?)\s*</customer_name>",
    re.IGNORECASE,
)
_USER_ID_TAG_RE = re.compile(
    r"\[user_id=[^\]]*\]:?\s*|<user_id>.*?</user_id>\s*",
    re.IGNORECASE,
)
_DIETARY_TAG_RE = re.compile(
    r"\[dietary=[^\]]*\]:?\s*|<dietary>.*?</dietary>\s*",
    re.IGNORECASE,
)
_CONTEXT_TAG_ONLY_FALLBACK = "Got it. What would you like?"


def _strip_leaked_context_tags(reply: str) -> str:
    """Remove a leaked ``[customer_name=X]``/``[user_id=X]``/``[dietary=X]`` tag.

    Args:
        reply: Model output, already stripped of markdown/tool syntax and the
            "[Context: ...]" breadcrumb.

    Returns:
        The reply with any leaked scaffold tag removed. If the ENTIRE reply
        was just a leaked tag with nothing else spoken (observed live: a bare
        ``<customer_name>Arjun</customer_name>``), a minimal natural sentence
        using the captured name is substituted instead of sending an empty
        string, a bare word, or a raw tag to TTS.
    """
    if not reply:
        return reply
    name_match = _CUSTOMER_NAME_TAG_RE.search(reply)
    cleaned = _CUSTOMER_NAME_TAG_RE.sub("", reply)
    cleaned = _USER_ID_TAG_RE.sub("", cleaned)
    cleaned = _DIETARY_TAG_RE.sub("", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    if cleaned:
        if cleaned != reply.strip():
            logger.warning(
                "[AGENT] Reply leaked a customer_name/user_id/dietary scaffold "
                "tag — stripping in place | raw=%r", reply[:160],
            )
        return cleaned
    name = (name_match.group(1) or name_match.group(2)) if name_match else None
    logger.warning(
        "[AGENT] Reply was ENTIRELY a leaked scaffold tag — substituting a "
        "minimal reply | raw=%r", reply[:160],
    )
    if name and name.strip():
        return f"Got it, {name.strip()}."
    return _CONTEXT_TAG_ONLY_FALLBACK



# (FSSAI food-safety license, GST tax registration, internal outlet code,
# parent-company legal name) that must NEVER be read to a customer regardless
# of what question triggered them. Observed live: a "where can I find
# parking" question pre-grounded with the pinned root section, and the model
# echoed almost the entire raw block back verbatim — license numbers, tax
# registration, and all — instead of answering only about parking. A partial
# paraphrase failure is recoverable by rephrasing; reading out a food-safety
# license or tax ID to a customer is not, so this is a hard substitution, not
# a strip-in-place. (The "[Context: ...]" breadcrumb tag is handled
# separately by `_strip_context_breadcrumb` — it is cosmetic, not sensitive.)
_ADMIN_LEAK_RE = re.compile(
    r"FSSAI License|GST Registration|Outlet Code|Parent Company",
    re.IGNORECASE,
)
_ADMIN_LEAK_FALLBACK = (
    "I don't have that detail phrased simply right now — could you ask again?"
)


def _strip_admin_leak(reply: str) -> str:
    """Replace a reply that leaked raw internal/administrative KB fields.

    Args:
        reply: Model output, already stripped of markdown/tool syntax and the
            "[Context: ...]" breadcrumb (see `_strip_context_breadcrumb`).

    Returns:
        ``reply`` unchanged, or ``_ADMIN_LEAK_FALLBACK`` when it contains a
        license number, tax registration, or internal outlet code — these
        must never reach a customer, and removing only the offending
        fragment would still leave the rest of an un-paraphrased document
        dump.
    """
    if not reply or not _ADMIN_LEAK_RE.search(reply):
        return reply
    logger.error(
        "[AGENT] Reply leaked raw internal/administrative KB fields — "
        "substituting fallback | raw=%r", reply[:200],
    )
    return _ADMIN_LEAK_FALLBACK


def _strip_leaked_directives(reply: str) -> str:
    """Remove system-prompt directives the model echoed into its reply.
    Args:
        reply: Model output, already stripped of markdown and tool syntax.

    Returns:
        The reply without leaked instruction sentences. If stripping would
        leave nothing speakable, the original text is returned unchanged so
        that a customer never receives silence.
    """
    if not reply:
        return reply
    cleaned = _LEAKED_DIRECTIVE_RE.sub(" ", reply)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned if cleaned else reply


def _strip_knowledge_markers(reply: str, *, pregrounded: bool) -> tuple[str, bool]:
    """Unwrap a `[knowledge]` block the model echoed instead of answering from.

    Pre-grounding prefixes the user turn with a ``[knowledge] ... [/knowledge]``
    block. The model is instructed to answer *from* that block, but a smaller
    quantised model sometimes parrots it back verbatim, sending the literal
    delimiters to TTS.

    When the turn really was pre-grounded the markers are removed and the inner
    text kept: that text came from the authoritative knowledge base, so speaking
    it is truthful and strictly better than substituting a fallback.

    **That premise only holds when a block was actually injected.** This
    function previously unwrapped unconditionally, which let the model forge its
    own grounding certificate: on a turn with no retrieval at all it emitted
    ``[knowledge] The President of India is Droupadi Murmu...`` and the unwrap
    laundered pure parametric memory into a clean, confident, grounded-looking
    answer. This is the same failure class the menu and confirm guards exist to
    prevent — there, *invocation is not success*; here, **a marker is not
    grounding**. So the caller is told, and replaces the reply.

    Args:
        reply: Model output.
        pregrounded: Whether a real ``[knowledge]`` block was injected into this
            turn's prompt.

    Returns:
        ``(cleaned_reply, forged)``. ``forged`` is ``True`` when markers were
        present on a turn that was never pre-grounded, meaning the cleaned text
        is ungrounded and must not be spoken as-is.
    """
    if not reply:
        return reply, False
    lowered = reply.lower()
    has_full_marker = "[knowledge]" in lowered or "[/knowledge]" in lowered
    has_truncated_tail = bool(_TRUNCATED_KNOWLEDGE_TAIL_RE.search(reply))
    if not has_full_marker and not has_truncated_tail:
        return reply, False
    cleaned = _KNOWLEDGE_MARKER_RE.sub(" ", reply)
    cleaned = _TRUNCATED_KNOWLEDGE_TAIL_RE.sub("", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    if not pregrounded:
        logger.error(
            "[AGENT] Model forged a [knowledge] block on a turn with no "
            "pre-grounding — content is ungrounded | raw=%r", reply[:160],
        )
        return (cleaned or reply), True
    if cleaned:
        logger.warning(
            "[AGENT] Model echoed the pre-grounded knowledge block — "
            "unwrapping markers | raw=%r", reply[:160],
        )
        return cleaned, False
    return reply, False


def _dedupe_repeated_sentences(reply: str) -> str:
    """Collapse a fact the model stated twice in the same reply.

    Observed live on knowledge/pre-grounding turns: the model both echoes the
    ``[knowledge]...[/knowledge]`` block verbatim (unwrapped in-place by
    ``_strip_knowledge_markers``/``_strip_citation_markers``, which keep the
    inner text) *and* separately paraphrases the same fact in its own words,
    e.g. "The restaurant has a seating capacity of 50 people. The restaurant
    has a seating capacity of 50 people." Both halves are truthful — they came
    from the same grounded source — so the fix is to keep exactly one
    occurrence, never fall back to silence or a generic refusal.

    This only needs to run on the buffered whole-reply path, not mirrored into
    ``_SentenceGate._is_safe()``: a duplicate requires at least two sentences
    to compare, and every turn that can produce this pattern (a knowledge
    answer that never calls a tool) is unconditionally withheld from early
    release by gate condition (a) — ``_is_safe`` returns False until a tool has
    run, so these turns always reach the buffered path where this runs.

    Args:
        reply: Model output, already stripped of thinking/markdown/knowledge/
            citation markers.

    Returns:
        ``reply`` with exact-duplicate sentences (compared case/whitespace
        insensitively) collapsed to their first occurrence. Returns ``reply``
        unchanged when nothing was duplicated.
    """
    if not reply:
        return reply
    sentences = [s.strip() for s in _SENTENCE_END_RE.split(reply) if s.strip()]
    if len(sentences) < 2:
        return reply
    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        norm = re.sub(r"\s+", " ", sentence).strip().lower()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(sentence)
    if len(deduped) == len(sentences):
        return reply
    logger.warning(
        "[AGENT] Reply repeated the same sentence — collapsing duplicates | "
        "raw=%r", reply[:200],
    )
    return " ".join(deduped)


def _strip_tool_syntax(reply: str) -> str:
    """Remove un-executed tool calls that the model emitted as plain text.

    Args:
        reply: Model output, already stripped of thinking blocks.

    Returns:
        The reply with any literal tool-call syntax removed, or a safe
        fallback when nothing speakable remains.
    """
    if not reply:
        return reply
    cleaned = _TOOL_SYNTAX_RE.sub(" ", reply).strip()
    if cleaned == reply.strip() and not _TOOL_MENTION_RE.search(cleaned):
        return reply
    # A prose mention ("I will call the list_categories tool...") survives the
    # JSON strip above but is still unspeakable. Usually this means the turn
    # produced no tool result to answer from, so wholesale substitution is
    # correct. But observed live: a real tool DID run and the model prefixed
    # a genuine, grounded answer with throwaway narration, e.g.
    #     "Call `get_current_order` to check the status of the order.
    #      Your cart is empty. Would you like to start a new order?"
    # Wholesale substitution there discards a truthful answer. Drop only the
    # sentence(s) naming a tool and keep the rest if anything speakable
    # remains; fall back wholesale only when nothing does.
    if _TOOL_MENTION_RE.search(cleaned):
        sentences = [s.strip() for s in _SENTENCE_END_RE.split(cleaned) if s.strip()]
        kept = [s for s in sentences if not _TOOL_MENTION_RE.search(s)]
        remainder = " ".join(kept).strip()
        if remainder and re.search(r"[A-Za-z]{3,}", remainder):
            logger.warning(
                "[AGENT] Reply narrated a tool name alongside a real answer — "
                "dropping the narration sentence | raw=%r kept=%r",
                reply[:160], remainder[:160],
            )
            return remainder
        logger.warning(
            "[AGENT] Reply narrated a tool name instead of calling it — "
            "substituting fallback | raw=%r", reply[:160],
        )
        return _TOOL_SYNTAX_FALLBACK
    # Leftover fragments are usually punctuation or a stray brace.
    if len(cleaned) < 15 or not re.search(r"[A-Za-z]{3,}", cleaned):
        logger.warning(
            "[AGENT] Reply was raw tool syntax with no speakable text — "
            "substituting fallback | raw=%r", reply[:160],
        )
        return _TOOL_SYNTAX_FALLBACK
    logger.warning(
        "[AGENT] Stripped literal tool syntax from reply (%d → %d chars)",
        len(reply), len(cleaned),
    )
    return cleaned


# Sentence terminators used to decide when a streamed fragment is speakable.
# A sentence is only released once its terminator has arrived, so TTS never
# receives a half-formed clause.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Phrase (clause) boundaries: sentence terminators, plus a comma that is not
# part of digit-grouping (e.g. the "," in "1,234" must never split). Releasing
# at comma boundaries — not just full sentences — lets TTS start on "Sure,"
# or "Your total is now 169 rupees," well before the model finishes the
# sentence, shrinking time-to-first-informative-audio the same way
# kiosk-voice-lab-main's ``speak_streaming()`` phrase splitter does. Every
# safety condition in ``_SentenceGate._is_safe()`` still applies per phrase —
# they are local (word-boundary) regex checks, so a claim never straddles a
# phrase boundary undetected.
_PHRASE_BREAK_RE = re.compile(r"(?<=[.!?])\s+|(?<!\d)(?<=,)\s+")


def _normalise_for_compare(text: str) -> str:
    """Collapse whitespace so streamed and final text can be prefix-compared.

    The streamed sentences are re-joined with single spaces, whereas the final
    reply keeps the model's original spacing, so a raw prefix test would report
    a spurious mismatch.
    """
    return _WS_RE.sub(" ", text).strip()


class _SentenceGate:
    """Releases complete sentences early, but only when they are provably safe.

    ``chat()`` applies several whole-reply guards *after* generation finishes.
    Some of them replace the reply outright (an unbacked order claim, leaked
    tool syntax) or regenerate the turn (the missing-tool-call retry). Speech
    cannot be recalled, so a sentence may only be released when none of those
    paths can still fire for this turn.

    The conditions below mirror the guards in ``chat()`` one-for-one. They are
    evaluated per sentence against the tool calls observed *so far*, which is
    sound because ADK emits every ``function_call`` part before the final text
    parts: by the time text streams, the turn's tool set is already known.

    The gate is one-way. Once a sentence fails a check the gate closes for the
    remainder of the turn and the caller falls back to the buffered reply,
    because a later guard may rewrite text we would otherwise have spoken.
    """

    def __init__(self, message: str, emit, *, pregrounded: bool = False) -> None:
        """Initialise the gate.

        Args:
            message:     The customer's utterance, needed to detect confirm intent.
            emit:        Callback invoked with each sentence cleared for speech.
            pregrounded: Whether an authoritative ``[knowledge]`` block was
                injected into this turn's prompt. Such a turn answers from that
                block and is *expected* to call no tool, which relaxes
                condition (a) below.
        """
        self._message = message
        self._emit = emit
        self._pregrounded = pregrounded
        self._buffer = ""
        self._released: list[str] = []
        self._open = True
        # A confirm intent can trigger _force_confirm(), which replaces the
        # whole reply. Nothing in such a turn may be spoken early.
        self._confirm_intent = bool(_CONFIRM_INTENT_RE.search(message))

    @property
    def released_text(self) -> str:
        """Concatenation of everything already handed to the caller."""
        return " ".join(self._released)

    def close(self) -> None:
        """Stop releasing sentences for the rest of this turn."""
        self._open = False

    def _is_safe(self, sentence: str, tool_calls: list[str]) -> bool:
        """Return True when no post-hoc guard in chat() can rewrite ``sentence``."""
        # (a) Every recovery path in chat() is gated on `not tool_calls`.
        #     Until a tool has run, any of them may still replace the reply.
        #     This is also the mirror for the grounding guard (forged
        #     [knowledge] markers / out-of-scope questions): it likewise only
        #     fires when `not tool_calls`, so a sentence it would replace can
        #     never have been released early.
        #
        #     A *pre-grounded* turn is the one exception, and it is exempt for
        #     the same reason chat() exempts it: the authoritative knowledge was
        #     already placed in the prompt, so "no tool call" is the intended
        #     outcome rather than a grounding failure. Every guard this
        #     condition mirrors is itself disabled on such a turn — the
        #     missing-tool-call retry, _force_knowledge(), and the ungrounded
        #     reply guard are all additionally gated on `not pregrounded`.
        #
        #     The one whole-reply guard that is NOT pregrounded-exempt is the
        #     catalogue-promise recovery, and it cannot collide here:
        #     pre-grounding is only applied to a knowledge question that does
        #     *not* match the catalogue regex, so the two are mutually
        #     exclusive by construction.
        #
        #     That leaves _strip_knowledge_markers(), which does still rewrite a
        #     pre-grounded reply when the model parrots the injected block back.
        #     Condition (d2) below withholds exactly those sentences, so it is
        #     load-bearing now rather than merely defensive.
        if not tool_calls and not self._pregrounded:
            return False
        # (b) _force_confirm() replaces the reply on confirm-intent turns.
        if self._confirm_intent:
            return False
        # (b2) chat() strips leaked system-prompt directives from the whole
        #      reply. Speech cannot be recalled, so a sentence that the strip
        #      would delete must never be released early.
        if _strip_leaked_directives(sentence) != sentence.strip():
            return False
        # (c) An order claim is only trustworthy if an order tool actually ran;
        #     otherwise chat() substitutes _ORDER_CLAIM_FALLBACK.
        if _ORDER_CLAIM_RE.search(sentence) and not any(
            tool in _ORDER_TOOLS for tool in tool_calls
        ):
            return False
        # (c2) A *confirmation* claim additionally needs a confirm tool that
        #      actually succeeded — not merely one that was invoked. A
        #      hallucinated order_id (e.g. confirm_order(12345)) still shows
        #      up in tool_calls but fails server-side; checking invocation
        #      alone reintroduces the exact bug _strip_false_confirmation now
        #      guards against post-hoc. See agentic/confirm_guard.py.
        if _CONFIRM_CLAIM_RE.search(sentence) and not confirm_guard.current_state().succeeded:
            return False
        # (c3) menu_guard.validate_reply() can still rewrite the reply whenever
        #      this turn had an off-menu/ambiguous rejection — either because
        #      a sentence falsely claims an addition, because the reply never
        #      surfaces the real alternatives the tool provided (see
        #      _mentions_alternative in menu_guard.py), or because a
        #      *partial* success (some items added, one refused) never
        #      discloses the refusal (see _reconcile_partial_success). All
        #      three can only be known once the full reply is assembled, so
        #      no sentence from a turn with ANY rejection — full or partial —
        #      may be released early: a partial success is corrected by
        #      appending a disclosure the customer must still hear before any
        #      of the turn's sentences are considered final.
        #      Tool results are already recorded by this point: ADK emits every
        #      function_call part, and _mcp_fn awaits the call, before text streams.
        _menu_state = menu_guard.current_state()
        if _menu_state.has_rejection or _menu_state.has_partial_rejection:
            return False
        # (c4) A removal claim is only true if remove_from_order actually took
        #      something off the cart. chat() replaces the whole reply
        #      otherwise (agentic/removal_guard.py), so it must never reach
        #      TTS first.
        if removal_guard.claims_removal(sentence) and not removal_guard.current_state().succeeded:
            return False
        # (c5) A restated total is only trustworthy when it matches this
        #      turn's own tool result; chat() corrects a mismatch post-hoc via
        #      price_guard.validate_reply(). See agentic/price_guard.py.
        if price_guard.mismatched(sentence):
            return False
        # (d) Anything that would be stripped or substituted wholesale.
        if _ERROR_PAYLOAD_RE.search(sentence) or _TOOL_SYNTAX_RE.search(sentence):
            return False
        # (d2) A parroted pre-grounding delimiter is rewritten by
        #      _strip_knowledge_markers(). On a pre-grounded turn condition (a)
        #      no longer withholds the reply, so this is the sole check
        #      standing between a parroted "[knowledge]" and TTS.
        if _KNOWLEDGE_MARKER_RE.search(sentence):
            return False
        # (d3) A hallucinated citation marker is rewritten by
        #      _strip_citation_markers(); same reasoning as (d2).
        if _CITATION_MARKER_RE.search(sentence):
            return False
        # (d4) A leaked license/tax/internal field is substituted wholesale
        #      by _strip_admin_leak(); a partial sentence containing it must
        #      never be released early.
        if _ADMIN_LEAK_RE.search(sentence):
            return False
        if _TOOL_MENTION_RE.search(sentence) or "<think" in sentence.lower():
            return False
        return True

    def feed(self, delta: str, tool_calls: list[str]) -> None:
        """Accumulate a streamed fragment and release any complete safe phrases.

        Releases at both sentence and comma boundaries (see
        ``_PHRASE_BREAK_RE``) so TTS can start on an early clause instead of
        waiting for the whole sentence.

        Args:
            delta:      Newly generated text.
            tool_calls: Tools invoked so far this turn, in call order.
        """
        if not self._open or not delta:
            return
        self._buffer += delta
        parts = _PHRASE_BREAK_RE.split(self._buffer)
        # The trailing element has no terminator yet, so it stays buffered.
        self._buffer = parts.pop() if parts else ""
        for sentence in parts:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not self._is_safe(sentence, tool_calls):
                logger.info(
                    "[AGENT][STREAM] Gate closed — sentence withheld for "
                    "buffered validation | tools=%s sentence=%r",
                    tool_calls, sentence[:120],
                )
                self.close()
                return
            spoken = _strip_markdown(sentence)
            if not spoken:
                continue
            self._released.append(spoken)
            self._emit(spoken)


# Maps JSON-schema primitive types to Python types so ADK can build an
# accurate function-call declaration (parameter names + types) for each MCP
# tool. Without this, a ``**kwargs`` wrapper advertises zero parameters and the
# LLM invokes every tool with empty arguments.
_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _infer_json_type(pspec: dict[str, Any]) -> type:
    """Resolve a JSON-schema property spec to a Python type for ADK.

    Most MCP tool parameters have a direct ``"type"`` key. An **optional**
    parameter (e.g. ``items: list[dict] | None = None``) is instead emitted by
    FastMCP as ``{"anyOf": [{"type": "array", ...}, {"type": "null"}],
    "default": None}`` — no top-level ``"type"`` at all. Falling through to the
    ``str`` default there was the actual cause of a reported "cannot remove
    items" bug: ``remove_from_order``'s ``items`` parameter is optional, so ADK
    was told it takes a *string*, the model dutifully passed a JSON-encoded or
    comma-joined string instead of a real list, and kiosk-core's Pydantic
    validation rejected it outright — while ``place_order``/``update_order``
    (whose ``items`` is required, so no ``anyOf``) worked fine.

    Args:
        pspec: One property's JSON-schema spec from the tool's input schema.

    Returns:
        The best-matching Python type; the first non-null branch of an
        ``anyOf`` wins, falling back to ``str`` only when nothing usable is
        found.
    """
    direct_type = pspec.get("type")
    if direct_type:
        return _JSON_TO_PY.get(direct_type, str)

    for branch in pspec.get("anyOf", None) or ():
        branch_type = (branch or {}).get("type")
        if branch_type and branch_type != "null":
            return _JSON_TO_PY.get(branch_type, str)

    return str

# ---------------------------------------------------------------------------
# Tool result compression
# ---------------------------------------------------------------------------


def _compress_tool_result(tool_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Compress a tool result before it is stored in ADK session history.

    The full JSON returned by kiosk-core is large.  Only a fraction of it
    is needed in subsequent turns (the LLM needs order_id, item names/prices,
    and upsell display strings — never product_id, category, user_id, reason,
    or the full nested product object inside upsell_suggestions).

    Slimming the stored result cuts the per-turn input token count by ~70 %
    for order tools and ~80 % for list_products, which directly reduces OVMS
    prefill latency on every subsequent turn.

    Rules:
    - Error responses are NEVER compressed — return unchanged.
    - If JSON parsing or compression fails for any reason, return the original
      unchanged (safe fallback — never break the ordering flow).
    - ``knowledge_lookup`` is excluded: it returns plain text, already minimal.
    """
    if "error" in raw:
        return raw

    result_text = raw.get("result", "")
    if not result_text:
        return raw

    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return raw  # not valid JSON — return as-is

    try:
        compressed: Any = None

        def _price(value: Any) -> Any:
            """Render whole-rupee prices as ints.

            ``169.0`` serialises as ``169.00`` in the tool result, which the LLM
            echoes verbatim.  That wastes output tokens (~2 per price) and makes
            TTS pronounce "one hundred sixty-nine point zero zero".
            """
            try:
                if value is not None and float(value).is_integer():
                    return int(value)
            except (TypeError, ValueError):
                pass
            return value

        if tool_name == "list_products":
            # Keep only name + price — product_id and category never appear in replies.
            # An unfiltered call returns {category, item_count} entries instead of
            # products; compressing those to {name, price} would null out every
            # field and the agent would report an empty menu.
            if isinstance(data, list):
                if all(isinstance(p, dict) and "name" in p for p in data):
                    compressed = [
                        {"name": p.get("name"), "price": _price(p.get("price"))}
                        for p in data
                    ]

        elif tool_name == "get_popular_products":
            # Same shape as list_products, minus the {category, item_count}
            # summary branch — this tool never returns that shape.
            if isinstance(data, list):
                compressed = [
                    {"name": p.get("name"), "price": _price(p.get("price"))}
                    for p in data
                    if isinstance(p, dict) and "name" in p
                ]

        elif tool_name in (action_result.CLAIM_TOOLS[action_result.ITEM_ADDED] | action_result.ORDER_READ_TOOLS):
            if isinstance(data, dict):
                items = [
                    {
                        "name": it.get("product_name") or it.get("name"),
                        "qty": it.get("quantity", 1),
                        "price": _price(it.get("price")),
                    }
                    for it in data.get("items", [])
                ]
                compressed = {
                    "order_id": data.get("order_id"),
                    "total": _price(data.get("total")),
                    "items": items,
                }
                # Upsell display strings are copied verbatim into the reply template —
                # keep them; drop the full product object and reason.
                upsell_displays = [
                    s.get("display", "")
                    for s in data.get("upsell_suggestions", [])
                    if s.get("display")
                ]
                if upsell_displays:
                    compressed["upsell"] = upsell_displays
                # Guard inputs must survive compression: dropping these makes
                # the model report an empty menu or claim an item was added
                # when the call actually needs the customer to choose.
                for key in (
                    "needs_choice", "choice_message", "category",
                    "available_products", "unavailable_message",
                ):
                    if key in data:
                        compressed[key] = data[key]

        elif tool_name in action_result.CLAIM_TOOLS[action_result.ITEM_REMOVED]:
            if isinstance(data, dict) and "error" not in data:
                compressed = {
                    "removed": data.get("removed", []),
                    "not_in_cart": data.get("not_in_cart", []),
                    "cart_empty": data.get("cart_empty", False),
                    "total": _price(data.get("total")),
                    "items": [
                        {
                            "name": it.get("product_name") or it.get("name"),
                            "qty": it.get("quantity", 1),
                            "price": _price(it.get("price")),
                        }
                        for it in data.get("items", [])
                    ],
                }

        elif tool_name in action_result.CLAIM_TOOLS[action_result.ORDER_CONFIRMED]:
            if isinstance(data, dict):
                compressed = {
                    "order_id": data.get("order_id"),
                    "status": data.get("status"),
                    "total": _price(data.get("total")),
                }

        elif tool_name == "get_upsell_suggestions":
            if isinstance(data, list):
                compressed = [
                    s.get("display", "")
                    for s in data
                    if s.get("display")
                ]

        if compressed is not None:
            return {"status": "success", "result": json.dumps(compressed)}

    except Exception:
        pass  # any compression failure → safe fallback

    return raw


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Customer name extraction
# ---------------------------------------------------------------------------

# Words that follow "I'm"/"I am" far more often than a name does. Without this
# guard "I'm hungry" would enrol the customer as "Hungry" and the kiosk would
# greet them by it for the rest of the conversation.
_NAME_STOPWORDS = frozenset({
    "a", "an", "the", "not", "no", "yes", "ok", "okay", "fine", "good", "great",
    "hungry", "thirsty", "sorry", "sure", "done", "ready", "here", "back",
    "looking", "trying", "thinking", "wondering", "going", "getting", "having",
    "just", "still", "already", "very", "so", "really", "quite", "too",
    "vegetarian", "vegan", "allergic", "lactose", "gluten", "diabetic",
    "waiting", "confused", "curious", "interested", "afraid", "glad", "happy",
    "new", "old", "sick", "tired", "full", "good", "bad", "sure", "certain",
    "from", "with", "for", "about", "in", "on", "at", "to", "of",
    "ordering", "asking", "calling", "speaking", "buying", "paying",
    "it", "that", "this", "there", "all", "done", "set", "next", "first",
})

_NAME_PATTERNS = (
    re.compile(r"\bmy name(?:'s| is| would be)\s+([a-z][a-z'\-]{1,20})", re.I),
    re.compile(r"\b(?:you can |please |just )?call me\s+([a-z][a-z'\-]{1,20})", re.I),
    re.compile(r"\bi am\s+([a-z][a-z'\-]{1,20})\b", re.I),
    re.compile(r"\bi'm\s+([a-z][a-z'\-]{1,20})\b", re.I),
    re.compile(r"\bthis is\s+([a-z][a-z'\-]{1,20})\s+(?:here|speaking)\b", re.I),
    re.compile(r"\bit'?s\s+([a-z][a-z'\-]{1,20})\s+(?:here|speaking)\b", re.I),
)

# "my name is …" and "call me …" are explicit enough to trust on their own;
# the "I'm …" forms are ambiguous and need the stopword screen.
_EXPLICIT_NAME_PATTERN_COUNT = 2


def _extract_customer_name(message: str) -> str | None:
    """Pull the customer's given name out of an utterance, if they stated one.

    Deterministic extraction is used rather than asking the LLM to remember,
    because a 4B model loses names across turns and cannot be relied on to
    call a "remember this" tool at the right moment.

    Args:
        message: The customer's transcribed utterance.

    Returns:
        The name in title case, or None if the utterance does not state one.
    """
    if not message:
        return None
    for index, pattern in enumerate(_NAME_PATTERNS):
        match = pattern.search(message)
        if not match:
            continue
        candidate = match.group(1).strip(" '-")
        if not candidate or not candidate.replace("'", "").replace("-", "").isalpha():
            continue
        if index >= _EXPLICIT_NAME_PATTERN_COUNT and candidate.lower() in _NAME_STOPWORDS:
            continue
        if len(candidate) < 2:
            continue
        return candidate[:1].upper() + candidate[1:].lower()
    return None


# ---------------------------------------------------------------------------
# Dietary preference extraction
# ---------------------------------------------------------------------------

# Checked before the positive patterns: "I'm not vegetarian" / "I eat
# non-veg" must clear a preference (capability statement — "I can eat
# anything"), not set "vegetarian" from the substring match that would
# otherwise fire.
_NON_VEG_PATTERN = re.compile(
    r"\b(?:i(?:'m| am)\s+not\s+vegetarian|i(?:'m| am)\s+non[- ]?veg(?:etarian)?|"
    r"i\s+eat\s+(?:meat|non[- ]?veg(?:etarian)?|chicken|fish))\b",
    re.IGNORECASE,
)

_VEGAN_PATTERN = re.compile(r"\bi(?:'m| am)\s+(?:a\s+)?vegan\b", re.IGNORECASE)
_VEGETARIAN_PATTERN = re.compile(
    r"\bi(?:'m| am)\s+(?:a\s+)?vegetarian\b|\bno meat\s+(?:for me|please)?\b",
    re.IGNORECASE,
)


def _is_dietary_statement_only(message: str) -> bool:
    """Return True if ``message`` is a bare dietary capability statement.

    "I'm vegetarian" / "I'm vegan" / "I eat meat" / "I'm not vegetarian" match
    ``_KNOWLEDGE_QUERY_RE`` on the word "vegetarian"/"vegan" alone, even though
    they are not a question about the outlet's menu — they are Rule 0.6
    (acknowledge, no tool call). Both the pre-grounding step and
    ``_needs_tool_retry`` must treat them as exempt from the knowledge path,
    or the model gets irrelevant [knowledge] context (e.g. opening hours)
    injected ahead of, or forced into, a plain preference statement.
    """
    return bool(
        _NON_VEG_PATTERN.search(message)
        or _VEGAN_PATTERN.search(message)
        or _VEGETARIAN_PATTERN.search(message)
    )


# Ad-hoc "suggest me veg dishes" / "show me non-veg options" — the customer
# names veg/non-veg as a filter on THIS request rather than stating an
# ongoing capability ("I'm vegetarian"). Distinct from the patterns above:
# "non veg dishes" is a request to see ONLY non-vegetarian items (a positive
# filter), not "I'm not vegetarian" (which means no restriction at all — see
# _NON_VEG_PATTERN). Scoped to fire only alongside a food-request context
# word so a plain order like "I'll have the veg burger" (Rule 1 DIRECT ORDER,
# resolved by name/id, never reaches list_products/get_popular_products)
# does not get misread as a lasting dietary statement.
_DIET_ADHOC_CONTEXT_RE = re.compile(
    r"\b(?:dish|dishes|item|items|option|options|food|menu|thing|things|stuff|"
    r"suggest|suggestion|suggestions|recommend|recommendation|recommendations|"
    r"favou?rite|favou?rites|bestseller|bestsellers)\b",
    re.IGNORECASE,
)
_NON_VEG_ADHOC_PATTERN = re.compile(r"\bnon[- ]?veg(?:etarian)?\b", re.IGNORECASE)
_VEG_ADHOC_PATTERN = re.compile(r"\b(?:veg|vegetarian|vegan)\b", re.IGNORECASE)


def _extract_dietary_pref(message: str) -> str | None:
    """Pull a stated dietary preference out of an utterance, if present.

    Deterministic extraction (same rationale as :func:`_extract_customer_name`):
    a small model cannot be trusted to recall "I am vegetarian" several turns
    later, so the preference is captured once here and re-injected on every
    later turn instead of relying on conversation memory.

    Returns:
        ``"vegetarian"``, ``"vegan"``, ``"non_vegetarian"`` (positive filter —
        show only non-veg items, e.g. "suggest me non veg dishes"),
        ``"none"`` (explicit non-veg capability statement, e.g. "I'm not
        vegetarian" — clears any earlier preference, no restriction; also
        returned when BOTH veg and non-veg are named in the same turn, e.g.
        "veg and non-veg options" — a single positive filter would silently
        drop half of what was asked for), or ``None`` if nothing was stated.
    """
    if not message:
        return None
    if _NON_VEG_PATTERN.search(message):
        return "none"
    if _VEGAN_PATTERN.search(message):
        return "vegan"
    if _VEGETARIAN_PATTERN.search(message):
        return "vegetarian"
    # Ad-hoc "veg dishes"/"non veg options" phrasing — only trusted alongside
    # a food-request context word (see _DIET_ADHOC_CONTEXT_RE docstring).
    if _DIET_ADHOC_CONTEXT_RE.search(message):
        has_non_veg = bool(_NON_VEG_ADHOC_PATTERN.search(message))
        # _VEG_ADHOC_PATTERN also matches the "veg" inside "non-veg", so a
        # standalone veg mention must be tested against the message with any
        # non-veg mention stripped out first — otherwise "non veg dishes"
        # alone looks like both were requested.
        veg_only_text = _NON_VEG_ADHOC_PATTERN.sub("", message)
        has_veg = bool(_VEG_ADHOC_PATTERN.search(veg_only_text))
        if has_non_veg and has_veg:
            # Customer named BOTH filters in the same turn, e.g. "veg option
            # and non-veg options" — observed live: this collapsed to
            # non_vegetarian only, silently hiding every veg item from a
            # customer who explicitly asked to see both.
            return "none"
        if has_non_veg:
            return "non_vegetarian"
        if has_veg:
            return "vegetarian"
    return None


# ---------------------------------------------------------------------------
# Agent instruction prompt
# ---------------------------------------------------------------------------
# The instruction is assembled from three sources:
#   1. Domain sections (YAML)  : persona, tool_rules, domain_tags
#      Owned by the consuming application — swap these to port to a new domain
#      (healthcare, education, retail) without touching any Python file.
#   2. Core framework (code)   : multi-action turns, knowledge block,
#      tool error handling, output format constraints.
#      Domain-agnostic and always present; rag-service owns and maintains it.
#
# See configs/rag-service/agent_profile.yaml → agent_instruction section.
# See agentic/domain_config.py → get_agent_instruction() for assembly logic.

_CORE_FRAMEWORK = """
## Multi-action turns
After each tool result, check whether another requested action remains. Call the next tool. Speak only after ALL actions are complete.

## [knowledge] block
When [knowledge]…[/knowledge] is present, answer from it directly — do not call knowledge_lookup. Never read the tags or "[1]" markers aloud. Summarise in 1–2 sentences.

## Tool errors
{"error": …, "available_products": […]} means the requested item is not available. Never say "try again". Offer the available_products alternatives by name and price.
knowledge_lookup returns excerpts, not a finished answer — write the reply yourself in 1–2 sentences from those excerpts only.

## Output
Responses are spoken aloud. Keep replies concise — under 200 characters for routine turns, longer only when listing catalogue items or confirming a completed transaction.
Never open with "Sure!", "Of course!", or restate the question. Start with the answer.
Only use names, prices, and transaction IDs that appeared in a tool result this turn.
""".strip()

_AGENT_INSTRUCTION: str = domain_config.get_agent_instruction(_CORE_FRAMEWORK)


# ---------------------------------------------------------------------------
# OrderingAgent
# ---------------------------------------------------------------------------


class OrderingAgent:
    """Wraps the ADK LlmAgent for the ordering flow.

    Call ``await bootstrap()`` once before using ``chat()``.
    """

    def __init__(self) -> None:
        self._agent = None
        self._runner = None
        self._session_service = None
        self._bootstrapped = False
        # Customer given name per conversation. Keyed on session_id, which the
        # kiosk UI regenerates for every new conversation, so a name never
        # survives into the next customer's session.
        self._customer_names: dict[str, str] = {}
        # Stated dietary preference ("vegetarian"/"vegan") per conversation,
        # same lifecycle/rationale as _customer_names above.
        self._dietary_prefs: dict[str, str] = {}
        # Cart/upsell memory per conversation, same lifecycle/rationale as
        # _customer_names above — see cart_state_guard.py and _CartState.
        self._cart_states: dict[str, _CartState] = {}

    async def bootstrap(self) -> None:
        """Initialise the ADK model, MCP tools, and runner.

        Safe to call multiple times — subsequent calls are no-ops.
        MCP tool discovery may silently return 0 tools if kiosk-core is not
        yet ready; ``chat()`` will trigger re-discovery automatically.
        """
        if self._bootstrapped:
            return

        logger.info("[AGENT] Bootstrapping OrderingAgent …")

        # 1. Discover MCP tools from kiosk-core (best-effort at startup;
        #    will be retried on first chat() call if kiosk-core isn't ready yet)
        mcp_tools = await bootstrap_mcp_tools(agent_cfg.MCP_CONFIG_PATH)
        logger.info("[AGENT] MCP tools: %s", list(mcp_tools))

        # Classify discovered tools into semantic roles and update all guards.
        # Tool names come exclusively from the MCP server — agent_profile.yaml
        # only provides the role patterns (regex) for classification.
        _apply_bootstrap_tool_classification(list(mcp_tools.keys()))

        # 2. Build ADK FunctionTools
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool

        # knowledge_lookup is a native Python async function
        adk_tools = [FunctionTool(knowledge_lookup)]

        # Wrap each MCP tool as an async Python callable → FunctionTool
        for tool_name, mcp_tool in mcp_tools.items():
            adk_tools.append(FunctionTool(self._make_mcp_callable(tool_name, mcp_tool)))

        # 3. Create ADK agent
        model = create_adk_model()
        self._agent = LlmAgent(
            name=domain_config.get_agent_name(),
            model=model,
            description=domain_config.get_agent_description(),
            instruction=_AGENT_INSTRUCTION,
            tools=adk_tools,
        )

        # 4. Runner + session service
        self._session_service = create_session_service()
        self._runner = create_runner(self._agent, self._session_service)

        self._bootstrapped = True
        logger.info("[AGENT] OrderingAgent ready ✓")

    async def _refresh_mcp_tools(self) -> None:
        """Re-discover MCP tools and rebuild the agent if tools are missing.

        Called automatically from ``chat()`` when no MCP tools are registered —
        this recovers from the startup race where rag-service starts before
        kiosk-core, as well as from kiosk-core restarts mid-session.
        """
        if get_all_tools():
            return  # already have tools, nothing to do

        logger.info("[AGENT] No MCP tools registered — retrying discovery from kiosk-core …")
        mcp_tools = await bootstrap_mcp_tools(agent_cfg.MCP_CONFIG_PATH)
        if not mcp_tools:
            logger.warning("[AGENT] MCP re-discovery returned 0 tools — kiosk-core may still be starting")
            return

        logger.info("[AGENT] MCP re-discovery succeeded: %s", list(mcp_tools))

        # Re-classify newly discovered tools and update guards + injection sets.
        _apply_bootstrap_tool_classification(list(mcp_tools.keys()))

        # Rebuild the agent with the newly discovered tools
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool

        adk_tools = [FunctionTool(knowledge_lookup)]
        for tool_name, mcp_tool in mcp_tools.items():
            adk_tools.append(FunctionTool(self._make_mcp_callable(tool_name, mcp_tool)))

        model = create_adk_model()
        self._agent = LlmAgent(
            name=domain_config.get_agent_name(),
            model=model,
            description=domain_config.get_agent_description(),
            instruction=_AGENT_INSTRUCTION,
            tools=adk_tools,
        )
        self._session_service = create_session_service()
        self._runner = create_runner(self._agent, self._session_service)
        logger.info("[AGENT] Agent rebuilt with %d MCP tool(s) ✓", len(mcp_tools))

    async def warmup(self, max_attempts: int = 30, delay_seconds: float = 2.0) -> bool:
        """Poll kiosk-core until MCP tools are discovered.

        rag-service and kiosk-core start concurrently, so the startup
        ``bootstrap()`` frequently finds zero MCP tools.  Without this warmup
        the *first customer turn* pays for tool discovery plus a full agent
        rebuild (~1.3 s measured), which is the worst possible moment.

        Intended to run as a background task so it never delays startup.

        Args:
            max_attempts:   How many discovery attempts before giving up.
            delay_seconds:  Delay between attempts.

        Returns:
            True once MCP tools are registered, False if all attempts failed.
        """
        if not self._bootstrapped:
            await self.bootstrap()

        for attempt in range(1, max_attempts + 1):
            if get_all_tools():
                logger.info("[AGENT] Warmup complete — MCP tools ready (attempt %d) ✓", attempt)
                return True
            try:
                await self._refresh_mcp_tools()
            except Exception as exc:
                logger.debug("[AGENT] Warmup attempt %d failed: %s", attempt, exc)
            if get_all_tools():
                logger.info("[AGENT] Warmup complete — MCP tools ready (attempt %d) ✓", attempt)
                return True
            await asyncio.sleep(delay_seconds)

        logger.warning(
            "[AGENT] Warmup gave up after %d attempts — first turn will retry discovery",
            max_attempts,
        )
        return False

    @staticmethod
    def _make_mcp_callable(tool_name: str, mcp_tool: MCPTool):
        """Dynamically create an async function wrapping an MCP tool call.

        The wrapper is given an explicit ``__signature__`` and
        ``__annotations__`` derived from the MCP tool's JSON input schema so
        that Google ADK advertises the real parameter names and types to the
        LLM. A bare ``**kwargs`` signature would otherwise be introspected as
        a zero-parameter tool, causing the model to call every tool with empty
        arguments (e.g. ``list_products()`` instead of
        ``list_products(category="burgers")``).

        A ``tool_context: ToolContext`` parameter is also added to the
        signature. ADK detects it by its type annotation (not by name) and
        injects the running turn's ``ToolContext`` without exposing it to the
        LLM as a callable argument — used below to skip the 2nd LLM call for
        deterministic outcomes. See ``agentic/reply_templates.py``.
        """
        from google.adk.tools import ToolContext

        async def _mcp_fn(tool_context: ToolContext | None = None, **kwargs: Any) -> Any:
            cart_state = _cart_state_ctx.get()
            # Inject user_id from turn context — it is stripped from the
            # model-visible schema so the model never generates it, saving
            # 8–19 tokens per call at 18 tps (~444–1,050ms).
            if tool_name in _USER_ID_INJECTED_TOOLS:
                kwargs["user_id"] = _user_id_ctx.get() or "anonymous"
            if tool_name in _DIETARY_INJECTED_TOOLS:
                # Overwrite rather than merge: this is the deterministically
                # captured preference, always more trustworthy than anything
                # the model might guess and pass itself. Applies to
                # place_order/update_order (cart writes) as well as
                # list_products/get_popular_products (catalogue reads), so a
                # customer who has stated a veg/vegan preference gets a
                # veg-only result for "suggest me some dishes" without the
                # model needing to ask again or filter free-hand.
                diet = _dietary_ctx.get()
                if diet:
                    kwargs["dietary"] = diet
            if tool_name in _MUTATING_TOOLS:
                # Force dry_run onto every DB-writing tool while a speculative
                # draft is running — never trust/expose this to the LLM (it
                # is stripped from the schema below). Explicitly forced False
                # otherwise, so a real turn always persists normally even if
                # some stale value leaked in from elsewhere.
                kwargs["dry_run"] = _speculative_ctx.get()
            if tool_name in action_result.CLAIM_TOOLS[action_result.ITEM_ADDED]:
                # Catch a stale/pending item reference the model failed to
                # update against what the customer just said this turn (e.g.
                # a pending "French fries" confirmation still in the call
                # after the customer said "add a pizza"). See
                # agentic/item_intent_guard.py for the full rationale.
                corrected = item_intent_guard.corrected_reference(
                    tool_name, kwargs.get("items"), _utterance_ctx.get()
                )
                if corrected:
                    kwargs["items"][0]["product_id"] = corrected
                    kwargs["items"][0].pop("name", None)
                    kwargs["items"][0].pop("product", None)
                # Drop any item in a *multi*-item call that silently restates
                # something already in the cart (e.g. re-adding the burger
                # the customer already has) or auto-accepts an upsell
                # suggestion the customer never confirmed — both are real,
                # additive DB writes the customer never asked for this turn.
                # See agentic/cart_state_guard.py for the full rationale and
                # why this is deliberately narrower than item_intent_guard.
                items_arg = kwargs.get("items")
                if cart_state is not None and isinstance(items_arg, list):
                    filtered = cart_state_guard.filter_stale_and_unconfirmed_items(
                        items_arg,
                        _utterance_ctx.get(),
                        cart_state.known_items,
                        cart_state.pending_upsell,
                    )
                    if len(filtered) != len(items_arg):
                        kwargs["items"] = filtered
            logger.info("[AGENT→MCP] tool=%s args=%s", tool_name, kwargs)
            result = await call_tool(tool_name, kwargs)
            # Capture the exact tool_name+kwargs+result dispatched this turn
            # so a speculative draft can be replayed for real (dry_run=False,
            # same args) at commit time instead of re-running the LLM — see
            # ``chat()``'s ``tool_call_detail`` return and the replay path in
            # ``kiosk_core/audio_session.py``. Only meaningful for mutating
            # tools; harmless to record reads too (never replayed).
            _log = _tool_call_log_ctx.get()
            if _log is not None:
                _log.append({"tool_name": tool_name, "kwargs": dict(kwargs), "result": result})
            # Record the outcome *before* compression: the menu guard needs the
            # tool's own error payload, and compression is free to reshape a
            # successful result.
            _guard_start = time.monotonic()
            menu_guard.record_tool_result(tool_name, result)
            removal_guard.record_tool_result(tool_name, result)
            confirm_guard.record_tool_result(tool_name, result)
            price_guard.record_tool_result(tool_name, result)
            llm_metrics.record_guard((time.monotonic() - _guard_start) * 1000)

            # Refresh this session's cart/upsell memory from whatever the
            # tool actually returned — the ground truth for the *next* turn's
            # cart_state_guard check above. Read after menu_guard/etc. so a
            # failed/off-menu call (no ``items``) simply leaves prior state
            # untouched rather than wiping it. ``remove_from_order`` updates
            # the known cart but leaves ``pending_upsell`` alone — it carries
            # no upsell suggestions of its own, and a removal doesn't resolve
            # (accept or decline) whatever was previously offered.
            if cart_state is not None and tool_name in (
                action_result.CLAIM_TOOLS[action_result.ITEM_ADDED] | action_result.CLAIM_TOOLS[action_result.ITEM_REMOVED]
            ):
                payload = action_result.unwrap(result)
                if isinstance(payload, dict) and isinstance(payload.get("items"), list):
                    cart_state.known_items = payload["items"]
                    if tool_name not in action_result.CLAIM_TOOLS[action_result.ITEM_REMOVED]:
                        upsell = payload.get("upsell_suggestions") or []
                        cart_state.pending_upsell = (
                            upsell[0].get("product")
                            if upsell and isinstance(upsell[0], dict)
                            else None
                        )
            elif cart_state is not None and tool_name in action_result.CLAIM_TOOLS[action_result.ORDER_CONFIRMED]:
                # A confirmed order is closed; place_order's own docstring
                # says a *new* draft opens for whatever comes next, so this
                # session's known cart must not keep matching against a
                # finished order.
                payload = action_result.unwrap(result)
                if isinstance(payload, dict) and payload.get("status") == "confirmed":
                    cart_state.known_items = []
                    cart_state.pending_upsell = None


            # Skip the 2nd LLM call (pure narration) when this is the turn's
            # only tool call and the outcome cleanly matches an authored
            # template — see agentic/reply_templates.py for why templating
            # is at least as accurate as the model's own paraphrase, never
            # less. Any turn that calls a 2nd tool, or whose outcome doesn't
            # cleanly match (partial success, unexpected shape), falls
            # through unchanged to today's LLM-narrated path.
            #
            # Cart-mutating tools (``action_result.MUTATING_TOOLS`` —
            # place_order, update_order, remove_from_order, cancel_order,
            # confirm_order, confirm_active_order) are NEVER eligible for this
            # shortcut, full stop, regardless of what the customer said.
            # Templating here sets `skip_summarization = True`, which ends the
            # ADK turn immediately instead of feeding the tool result back to
            # the model — and that feedback step is the model's ONLY chance to
            # notice the turn isn't finished and call a second tool.
            #
            # An earlier version of this gate tried to detect a "compound"
            # utterance ("remove X and add Y") with a regex before deciding
            # whether to skip — that is exactly the wrong layer to make this
            # decision at: it can only recognise phrasings someone already
            # thought to write down, and a customer's real words are
            # infinitely more varied than any fixed pattern. The single
            # observed bug (`remove classic chicken burger and add one
            # margherita pizza instead` silently dropping the pizza) is one of
            # unbounded many possible phrasings for the same intent. Removing
            # the shortcut unconditionally for every mutating tool means the
            # model — which has the full conversation and every tool result —
            # always gets to decide whether another action is still owed,
            # using the request and the tool results, never surface wording.
            # Read-only catalogue tools (list_products/list_categories) keep
            # the shortcut: a pure browse has no cart side effect to lose if
            # the turn ends right after it.
            #
            # ``tool_context`` is only absent when this function is invoked
            # directly (e.g. unit tests bypassing ADK) — there's no
            # ``actions.skip_summarization`` to set in that case, so the
            # templating path is skipped entirely rather than guessed at.
            counter = _tool_call_count_ctx.get()
            calls_this_turn = counter.increment() if counter is not None else None

            # Mutating tools that are safe to template without a 2nd LLM call
            # when the utterance is provably a single-add.  Removal / compound
            # actions fall through to the 2nd-LLM path as before.
            _SINGLE_ORDER_TEMPLATE_TOOLS = frozenset({"place_order", "update_order"})

            utterance_now = _utterance_ctx.get()
            single_add_eligible = (
                tool_name in _SINGLE_ORDER_TEMPLATE_TOOLS
                and _is_single_add_utterance(utterance_now)
            )

            if (
                tool_context is not None
                and calls_this_turn == 1
                and tool_name in reply_templates.SPEAKABLE_TOOLS
                and (
                    tool_name not in action_result.MUTATING_TOOLS
                    or single_add_eligible
                )
            ):
                _tpl_start = time.monotonic()
                spoken = reply_templates.speak(tool_name, result, _utterance_ctx.get())
                llm_metrics.record_template((time.monotonic() - _tpl_start) * 1000)
                if spoken is not None:
                    logger.info(
                        "[AGENT→MCP] tool=%s templated reply (%s), skipping narration call: %r",
                        tool_name,
                        "single_add" if single_add_eligible else "read_only",
                        spoken[:160],
                    )
                    tool_context.actions.skip_summarization = True
                    llm_metrics.set_template_reply(spoken)
                    return spoken

            result = _compress_tool_result(tool_name, result)
            logger.debug("[AGENT→MCP] tool=%s compressed_result=%s", tool_name, str(result)[:200])
            return result

        _mcp_fn.__name__ = tool_name
        _mcp_fn.__doc__ = _scrub_injected_args(
            mcp_tool.prompt_description or tool_name,
            _injected_param_names(tool_name),
        )

        # Build an explicit signature from the MCP JSON input schema so ADK
        # introspection produces a correct function-call declaration.
        input_schema = mcp_tool.input_schema or {}
        properties: dict[str, Any] = input_schema.get("properties", {}) or {}
        required = set(input_schema.get("required", []) or [])

        # tool_context is detected by its ToolContext type annotation (not by
        # name) and excluded from what ADK shows the LLM as a callable
        # argument — see the class docstring above.
        params: list[inspect.Parameter] = [
            inspect.Parameter(
                "tool_context",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=ToolContext,
                default=None,
            )
        ]
        annotations: dict[str, Any] = {"tool_context": ToolContext}
        for pname, pspec in properties.items():
            # user_id is injected server-side from _user_id_ctx for the tools
            # in _USER_ID_INJECTED_TOOLS — strip it from the model-visible
            # schema so the model never generates it, saving 8–19 decode
            # tokens per call (444–1,050ms at 18 tps on iGPU).
            if pname == "user_id" and tool_name in _USER_ID_INJECTED_TOOLS:
                continue
            # dietary is injected server-side from _dietary_ctx for
            # place_order/update_order — strip it to prevent the model from
            # hallucinating dietary values (e.g. "vegetarian" for chicken).
            if pname == "dietary" and tool_name in _DIETARY_INJECTED_TOOLS:
                continue
            # dry_run is a server/agent-internal flag (see _MUTATING_TOOLS
            # above) — never expose it to the LLM; it is force-set by
            # _mcp_fn based on _speculative_ctx, not by anything the model
            # generates.
            if pname == "dry_run" and tool_name in _MUTATING_TOOLS:
                continue
            pytype = _infer_json_type(pspec or {})
            annotations[pname] = pytype
            if pname in required:
                params.append(
                    inspect.Parameter(
                        pname, inspect.Parameter.KEYWORD_ONLY, annotation=pytype
                    )
                )
            else:
                params.append(
                    inspect.Parameter(
                        pname,
                        inspect.Parameter.KEYWORD_ONLY,
                        annotation=pytype,
                        default=None,
                    )
                )
        annotations["return"] = Any

        _mcp_fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            params, return_annotation=Any
        )
        _mcp_fn.__annotations__ = annotations
        return _mcp_fn

    async def chat(
        self,
        message: str,
        session_id: str,
        user_id: str = "anonymous",
        history: list[dict[str, str]] | None = None,
        on_safe_sentence=None,
        speculative: bool = False,
    ) -> dict[str, Any]:
        """Run one conversational turn and return the agent's response.

        Args:
            message:    The user's transcribed utterance.
            session_id: Opaque session identifier (maps to ADK session).
            user_id:    The customer's user identifier.
            history:    Previous turns [{role, content}, …] — used to seed
                        the ADK session when it does not yet exist (e.g.
                        after a rag-service restart).
            speculative: If True, this is a speculative draft run — every
                        mutating MCP tool this turn is forced to ``dry_run``
                        (see ``_MUTATING_TOOLS``/``_speculative_ctx``), so it
                        can never write a real order row. Also captures the
                        exact tool_name+kwargs+result dispatched, returned as
                        ``tool_call_detail``, so a caller can replay the same
                        call for real later without re-running the LLM.

        Returns:
            dict with keys:
              - ``reply``:     str — the agent's text response.
              - ``tool_calls``: list[str] — tools invoked this turn.
              - ``tool_call_detail``: list[dict] — {tool_name, kwargs, result}
                for every tool dispatched this turn, in call order. Always
                populated (not just for speculative turns) so a real turn's
                calls can also be inspected if ever needed.
              - ``llm_ms``:      float | None — cumulative genuine LLM time
                (prefill + decode, i.e. the full round-trip).
              - ``llm_ttft_ms``: float | None — cumulative prefill time only.
              - ``llm_calls``: int — number of LLM round-trips this turn.
              - ``retrieval_ms``: float | None — cumulative knowledge-base
                retrieval time (None when no retrieval ran this turn).
              - ``mcp_ms``: float | None — cumulative MCP tool round-trip
                time (network + kiosk-core request handling; None when no
                tool was called). This is the only vantage point rag-service
                has on kiosk-core/SQLite time, since it never queries SQLite
                directly.
              - ``mcp_calls``: int — number of MCP tool round-trips this turn.
              - ``guard_ms``: float | None — cumulative time spent in the
                truthfulness guards (menu_guard/removal_guard/confirm_guard
                result recording + whole-reply validation). Tracked to rule
                guard overhead in or out as a latency contributor, since
                these are pure in-process functions and should stay small.
              - ``template_ms``: float | None — time spent rendering a
                deterministic reply template (None when none was attempted).
              - ``templated``: bool — True when a template was rendered, i.e.
                the second (narration) LLM call was skipped for this turn.
        """
        if not self._bootstrapped:
            await self.bootstrap()

        # If MCP tools weren't available at startup (race with kiosk-core),
        # attempt re-discovery before this turn so ordering tools work.
        await self._refresh_mcp_tools()

        logger.info(
            "[AGENT] chat session=%s user=%s speculative=%s message=%r",
            session_id, user_id, speculative, message[:120],
        )

        # Scoped to this call's asyncio task (each chat() invocation is its
        # own top-level task, so this never leaks into a concurrent/real
        # turn) — forces dry_run onto every mutating tool call below and
        # starts a fresh capture list for tool_call_detail.
        speculative_token = _speculative_ctx.set(speculative)
        tool_call_log_token = _tool_call_log_ctx.set([])

        # Must happen BEFORE the directive-mode branch below, not after it
        # (the original position, further down this function). Directive
        # mode returns early on a hit, and every accumulator in llm_metrics
        # (_mcp_stats, _llm_stats, etc.) starts as None until reset() runs —
        # record_mcp()/record() silently no-op on a None accumulator. With
        # reset() only called after the directive-mode return, every
        # directive-mode turn's real call_tool() timing (confirmed via logs
        # to actually execute — e.g. place_order) was captured into a dead
        # accumulator and discarded, so mcp_ms/mcp_calls always came back
        # None/0 even though the MCP round-trip genuinely happened.
        llm_metrics.reset()

        # Directive mode: one tool-free completion that carries both the cart
        # mutation and the speech, so TTS can start while the model is still
        # generating. Restricted to turns with explicit mutation intent, plus
        # single-item price questions (<check_price> — answered from the same
        # cached catalogue already trusted for <act> item names, so no LLM
        # tool-calling round-trip is needed) — a fallback costs a wasted
        # generation, and other questions/browse turns have nothing to gain
        # here. Never used for speculative drafts, which rely on dry_run
        # plumbing this path does not implement.
        if (
            directive_mode.DIRECTIVE_MODE
            and not speculative
            and (
                not reply_templates.is_browse_intent(message)
                or reply_templates.is_price_query_intent(message)
            )
        ):
            try:
                menu_block = await directive_mode.get_menu_block()
                if menu_block:
                    result = await directive_mode.run_turn(
                        message=message,
                        user_id=user_id,
                        history=history,
                        menu_block=menu_block,
                        base_instruction=_AGENT_INSTRUCTION,
                        on_safe_sentence=on_safe_sentence,
                    )
                    if result is not None:
                        _speculative_ctx.reset(speculative_token)
                        _tool_call_log_ctx.reset(tool_call_log_token)
                        return result
            except Exception:
                # Never let this optimisation break a turn — the
                # tool-calling path below is the authoritative one.
                logger.warning(
                    "[DIRECTIVE] path failed, falling back to tools", exc_info=True
                )

        from google.genai import types as genai_types

        # Seed the ADK session with prior history if the session does not
        # yet exist (rag-service restart scenario).
        #
        # A speculative draft runs against a DERIVED, throwaway session id.
        # ``run_async`` permanently appends its message, function calls and
        # reply to whatever session it is given, and a draft's tool results are
        # fabricated (dry_run=True) from a partial, still-changing preview
        # transcript. Writing those into the real conversation would leave the
        # following genuine turn looking at an exchange that already "added"
        # the items — and, as reset_sessions' own docstring warns, the model
        # then replays that conclusion instead of calling the tool again, so
        # the customer is told the order was placed while the DB has nothing.
        # Prefix-cache warming is unaffected: the cache is keyed on the token
        # prefix, which _ensure_session reproduces by seeding the same history.
        adk_session_id = f"{session_id}::spec" if speculative else session_id
        await self._ensure_session(user_id, adk_session_id, history)

        # Prefix the user_id into the first turn so the LLM (and ordering
        # tools) know which customer is speaking without needing a dedicated
        # user-lookup tool.
        full_message = message
        if user_id != "anonymous":
            full_message = f"[user_id={user_id}] {message}"

        # Remember a name the customer states, and re-inject it on every later
        # turn: the model cannot be trusted to carry it through a long history,
        # but it will happily use a name placed in the current turn.
        spoken_name = _extract_customer_name(message)
        if spoken_name:
            if self._customer_names.get(session_id) != spoken_name:
                logger.info(
                    "[AGENT] Customer name captured session=%s name=%s", session_id, spoken_name
                )
            self._customer_names[session_id] = spoken_name
        known_name = self._customer_names.get(session_id)
        if known_name:
            full_message = f"[customer_name={known_name}] {full_message}"

        # Same rationale as the name above: remember a stated dietary
        # preference deterministically and re-inject it every turn, rather
        # than trusting the model to recall it from conversation history.
        spoken_diet = _extract_dietary_pref(message)
        if spoken_diet:
            if spoken_diet == "none":
                if self._dietary_prefs.pop(session_id, None) is not None:
                    logger.info("[AGENT] Dietary preference cleared session=%s", session_id)
            elif self._dietary_prefs.get(session_id) != spoken_diet:
                logger.info(
                    "[AGENT] Dietary preference captured session=%s diet=%s", session_id, spoken_diet
                )
                self._dietary_prefs[session_id] = spoken_diet
        known_diet = self._dietary_prefs.get(session_id)
        if known_diet:
            full_message = f"[dietary={known_diet}] {full_message}"
        # Read by _mcp_fn this turn so place_order/update_order get the
        # preference without the LLM having to supply it as a tool argument.
        dietary_token = _dietary_ctx.set(known_diet)
        # Read by _mcp_fn to inject user_id into every tool that accepts it,
        # removing it from the model-visible schema so the model never wastes
        # tokens generating "user_id":"anonymous" (saves 8–19 tokens / call).
        user_id_token = _user_id_ctx.set(user_id or "anonymous")
        # Read by _mcp_fn this turn to catch a stale item reference against
        # what the customer actually just said. The raw ``message`` (not the
        # tag-prefixed ``full_message``) is used deliberately — the tags are
        # not something the customer said and would only add noise here.
        utterance_token = _utterance_ctx.set(message)
        # Bound to the same per-session _CartState object every turn (not a
        # fresh one) so place_order/update_order results recorded by _mcp_fn
        # persist into the next turn — see _CartState docstring.
        #
        # A speculative draft gets a THROWAWAY _CartState instead. Its tool
        # calls run with dry_run=True, so the payloads it sees describe an
        # uncommitted cart built from a partial, still-changing preview
        # transcript. Writing those into the shared object would make
        # cart_state_guard.filter_stale_and_unconfirmed_items treat the
        # customer's genuine add as "already in cart" on the real turn that
        # follows, and silently drop it.
        cart_state_token = _cart_state_ctx.set(
            _CartState() if speculative
            else self._cart_states.setdefault(session_id, _CartState())
        )

        # ── Structured root-fact fast path ──────────────────────────────
        # Restaurant name and hours are answered directly from the parsed
        # knowledge-base root section — zero LLM calls — rather than via
        # pre-grounded narration below. See reply_templates.speak_root_fact
        # for why: narration measurably merges distinct hour ranges and
        # invents unsupported details even at temperature=0.0, which is
        # unacceptable for a restaurant's own operating hours.
        facts_wanted = (
            reply_templates.classify_root_facts(message)
            if not _CATALOGUE_QUERY_RE.search(message)
            and not _ORDER_ACTION_RE.search(message)
            else []
        )
        if facts_wanted:
            _tpl_start = time.monotonic()
            try:
                facts = knowledge_tool.root_facts()
            except Exception:  # noqa: BLE001 — fast path is best-effort
                logger.exception(
                    "[AGENT] Structured root-fact lookup failed | session=%s", session_id
                )
                facts = {}
            spoken = reply_templates.speak_root_fact(facts_wanted, facts)
            if spoken:
                llm_metrics.reset()
                llm_metrics.record_template((time.monotonic() - _tpl_start) * 1000)
                template = llm_metrics.template_snapshot()
                logger.info(
                    "[AGENT] Structured root-fact reply | session=%s facts=%s",
                    session_id, facts_wanted,
                )
                _dietary_ctx.reset(dietary_token)
                _user_id_ctx.reset(user_id_token)
                _utterance_ctx.reset(utterance_token)
                _cart_state_ctx.reset(cart_state_token)
                _speculative_ctx.reset(speculative_token)
                _tool_call_log_ctx.reset(tool_call_log_token)
                return {
                    "reply": spoken,
                    "tool_calls": [],
                    "tool_call_detail": [],
                    "llm_ms": None,
                    "llm_ttft_ms": None,
                    "llm_calls": 0,
                    "retrieval_ms": None,
                    "mcp_ms": None,
                    "mcp_calls": 0,
                    "guard_ms": None,
                    "template_ms": template["ms"],
                    "templated": True,
                    "streamed": "",
                }
            # A requested fact was missing from the parsed data (e.g. pin
            # unavailable) — fall through to the normal pre-grounded path
            # below rather than speak an incomplete answer.

        # ── Pre-grounding ────────────────────────────────────────────────
        # For an unambiguous outlet question, retrieve the facts BEFORE the
        # model speaks rather than hoping it chooses knowledge_lookup and
        # then paying for a retry when it doesn't. Retrieval is ~110 ms; the
        # inference it removes is ~3 s. See agentic/config.py.
        #
        # Catalogue and order intents are excluded: those are answered by
        # list_products/place_order, and prose context would only invite the
        # model to invent products from marketing copy.
        #
        # A dietary preference STATEMENT ("I'm vegetarian", "I'm not
        # vegetarian") is also excluded: it matches _KNOWLEDGE_QUERY_RE on
        # the word "vegetarian"/"vegan" but is not a question about the
        # outlet's offerings — it is Rule 0.6 (acknowledge, no tool call).
        # Pre-grounding it injected irrelevant [knowledge] prose ahead of a
        # plain statement and produced garbled replies (observed: the model
        # echoing a bare "[dietary=...]" tag instead of a real sentence).
        pregrounded = False
        if (
            agent_cfg.PREGROUND_KNOWLEDGE
            and _KNOWLEDGE_QUERY_RE.search(message)
            and not _CATALOGUE_QUERY_RE.search(message)
            and not _ORDER_ACTION_RE.search(message)
            and not _is_dietary_statement_only(message)
        ):
            try:
                context = await knowledge_lookup(message)
            except Exception:  # noqa: BLE001 — pre-grounding is best-effort
                logger.exception("[AGENT] Pre-grounding failed | session=%s", session_id)
                context = ""
            if context and context not in knowledge_tool.NON_CONTEXT_RESULTS:
                full_message = (
                    f"[knowledge]\n{context}\n[/knowledge]\n\n{full_message}"
                )
                pregrounded = True
                logger.info(
                    "[AGENT] Pre-grounded outlet question | session=%s context_chars=%d",
                    session_id, len(context),
                )

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=full_message)],
        )

        logger.info(
            "[AGENT→OVMS] Sending request | session=%s user=%s message_chars=%d",
            session_id, user_id, len(full_message),
        )
        t_start = time.perf_counter()
        llm_metrics.reset()
        # Tool *results* for this turn are tracked separately from tool names:
        # an off-menu item makes place_order run and fail, which every
        # name-based guard reads as success. See agentic/menu_guard.py.
        menu_guard.begin_turn()
        removal_guard.begin_turn()
        confirm_guard.begin_turn()
        price_guard.begin_turn()
        _tool_call_count_ctx.set(_ToolCallCounter())

        try:
            gate = (
                _SentenceGate(message, on_safe_sentence, pregrounded=pregrounded)
                if (on_safe_sentence is not None and agent_cfg.STREAM_SENTENCES)
                else None
            )
            reply_parts, tool_calls = await self._run_turn(
                user_id, adk_session_id, content, gate=gate
            )

            # A turn that answers a catalogue question, promises a lookup, or
            # refuses without calling any tool is ungrounded. Re-run once with
            # an explicit correction rather than speaking the bad reply.
            #
            # A pre-grounded turn is exempt: the authoritative knowledge was
            # already put in front of the model, so "no tool call" is the
            # intended outcome here, not a grounding failure. Retrying would
            # spend two more inferences reaching the same answer.
            #
            # A status query is retried even when a tool WAS called, as long
            # as it wasn't get_current_order. Observed live: right after a
            # cancel, "Can you tell me the status of my order?" made the model
            # call remove_from_order instead (there was nothing to remove, so
            # removal_guard's stock failure text — "wasn't able to remove
            # that" — was spoken as the "status" answer). tool_calls is
            # non-empty there, so the no-tool-call branch above never fires;
            # the wrong-tool case needs its own check. A "start over"/"start a
            # new order" request gets the identical treatment against
            # cancel_order for the same reason.
            should_retry, nudge_text = (
                _needs_tool_retry("".join(reply_parts), message)
                if not pregrounded and (
                    not tool_calls
                    or (
                        _ORDER_STATUS_RE.search(message)
                        and "get_current_order" not in tool_calls
                    )
                    or (
                        _START_OVER_RE.search(message)
                        and "cancel_order" not in tool_calls
                    )
                )
                else (False, "")
            )
            if agent_cfg.RETRY_ON_MISSING_TOOL_CALL and should_retry:
                logger.warning(
                    "[AGENT] Ungrounded reply with no tool call — retrying once | "
                    "session=%s nudge=%s",
                    session_id,
                    _NUDGE_NAMES.get(id(nudge_text), "generic"),
                )
                nudge = genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=nudge_text)],
                )
                retry_parts, retry_tools = await self._run_turn(user_id, adk_session_id, nudge)
                if retry_tools or "".join(retry_parts).strip():
                    reply_parts, tool_calls = retry_parts, retry_tools
                    logger.info("[AGENT] Retry produced tool_calls=%s", retry_tools)

            # Last line of defence. If the model still claims an order was
            # placed or confirmed while no order tool ran, that claim is false:
            # observed with "Yes confirm my order" -> "Your order ... is
            # confirmed" and tool_calls=[], leaving the order in `draft`. The
            # customer walks away believing the kitchen has their order.
            # Speaking an honest failure is strictly better than a false
            # confirmation, so the claim is never allowed through unbacked.
            if not any(tool in _ORDER_TOOLS for tool in tool_calls):
                claimed = "".join(reply_parts)
                # The customer's intent is unambiguous and the action is
                # deterministic, so recover it in code rather than making them
                # repeat themselves: resolve their open draft and confirm it.
                if _CONFIRM_INTENT_RE.search(message) and not _speculative_ctx.get():
                    forced = await self._force_confirm(user_id, session_id)
                    if forced:
                        reply_parts, tool_calls = [forced], ["confirm_active_order"]
                    elif _ORDER_CLAIM_RE.search(claimed):
                        # Nothing was confirmed, so any claim is still false.
                        logger.error(
                            "[AGENT] Confirm recovery failed and reply claims success — "
                            "replacing reply | session=%s claim=%r",
                            session_id, claimed[:160],
                        )
                        reply_parts = [_ORDER_CLAIM_FALLBACK]
                elif _ORDER_CLAIM_RE.search(claimed):
                    logger.error(
                        "[AGENT] Unbacked order claim with no order tool call — "
                        "replacing reply | session=%s tool_calls=%s claim=%r",
                        session_id, tool_calls, claimed[:160],
                    )
                    reply_parts = [_ORDER_CLAIM_FALLBACK]

            # A catalogue question that ended in a promise ("let me look that
            # up") with no tool call is a dead end for a voice customer: there
            # is no second speaker turn coming. Fetch the data and answer.
            # Same reasoning as the catalogue path: an outlet question with no
            # tool call is either invented or an unnecessary refusal, and the
            # customer has no way to ask a follow-up mid-turn.
            if not tool_calls and not pregrounded and _KNOWLEDGE_QUERY_RE.search(message):
                grounded = await self._force_knowledge(message, session_id)
                if grounded:
                    reply_parts, tool_calls = [grounded], ["knowledge_lookup"]

            if not tool_calls and _CATALOGUE_QUERY_RE.search(message):
                spoken = "".join(reply_parts)
                if _PROMISE_RE.search(spoken) or not spoken.strip():
                    recovered, used_tool = await self._force_catalogue(message, session_id)
                    if recovered:
                        reply_parts, tool_calls = [recovered], [used_tool]
        except Exception as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.error(
                "[AGENT←OVMS] Request failed | session=%s latency_ms=%.0f error=%s",
                session_id, latency_ms, exc, exc_info=True,
            )
            llm = llm_metrics.snapshot()
            return {
                "reply": "Sorry, I encountered an error. Please try again.",
                "tool_calls": [],
                "tool_call_detail": _tool_call_log_ctx.get() or [],
                "llm_ms": llm["ms"],
                "llm_ttft_ms": llm["ttft_ms"],
                "llm_calls": llm["calls"],
                "retrieval_ms": llm_metrics.retrieval_snapshot()["ms"],
            }

        latency_ms = (time.perf_counter() - t_start) * 1000
        llm = llm_metrics.snapshot()
        logger.info(
            "[AGENT←OVMS] Response received | session=%s latency_ms=%.0f llm_ms=%s "
            "llm_ttft_ms=%s llm_calls=%d tool_calls=%s reply_chars=%d",
            session_id, latency_ms, llm["ms"], llm["ttft_ms"], llm["calls"],
            tool_calls, len("".join(reply_parts)),
        )

        reply = "".join(reply_parts).strip()
        reply = _strip_thinking(reply)
        reply = _strip_tool_syntax(reply)
        reply = _strip_markdown(reply)
        reply, forged_knowledge = _strip_knowledge_markers(reply, pregrounded=pregrounded)
        reply = _strip_citation_markers(reply)
        reply = _strip_context_breadcrumb(reply)
        reply = _strip_leaked_context_tags(reply)
        reply = _dedupe_repeated_sentences(reply)
        reply = _strip_admin_leak(reply)

        # When skip_summarization=True, ADK emits no model text event after the
        # tool call, so reply_parts is empty (or all content was stripped).
        # Recover the deterministic template text stored by _mcp_fn.
        if not reply.strip():
            tpl_reply = llm_metrics.get_template_reply()
            if tpl_reply:
                logger.info(
                    "[AGENT] reply was empty after stripping — using template reply | "
                    "session=%s template_reply=%r",
                    session_id, tpl_reply[:160],
                )
                reply = tpl_reply
        reply = _strip_leaked_directives(reply)
        if _ERROR_PAYLOAD_RE.search(reply):
            logger.error(
                "[AGENT] Raw tool error payload leaked into reply — substituting "
                "fallback | session=%s raw=%r", session_id, reply[:160],
            )
            reply = _TOOL_SYNTAX_FALLBACK

        # Grounding guard: nothing may be spoken from the model's own memory.
        # A reply arrives here ungrounded in one of two ways:
        #   * the model wrapped its answer in [knowledge] markers on a turn
        #     where no block was ever injected — a forged grounding
        #     certificate (see _strip_knowledge_markers), or
        #   * the customer asked something outside the kiosk's remit, which
        #     matches none of the intent regexes and therefore triggers none of
        #     the intent-keyed recovery guards above (see _is_out_of_scope).
        # Both are resolved identically: attempt a real knowledge lookup, and
        # refuse honestly when the knowledge base cannot support an answer.
        # Requiring `not tool_calls` keeps this off every grounded turn, and is
        # also what makes _SentenceGate condition (a) a complete mirror of this
        # guard — a tool-less turn releases no sentence early.
        if not tool_calls and not pregrounded and (
            forged_knowledge or _is_out_of_scope(message)
        ):
            logger.error(
                "[AGENT] Ungrounded reply — no tool call and no pre-grounding | "
                "session=%s forged_knowledge=%s message=%r reply=%r",
                session_id, forged_knowledge, message[:80], reply[:160],
            )
            # An out-of-scope question is refused outright rather than sent to
            # _force_knowledge. That path delegates to the RAG pipeline, which
            # answers with its own LLM and is equally free to speak from
            # parametric memory: asked "who is the president of India?" it
            # replied "Droupadi Murmu, took office 7 July 2022" with an empty
            # knowledge base behind it. Routing an out-of-domain question there
            # relocates the hallucination instead of removing it.
            #
            # A forged [knowledge] block is different: the turn is usually
            # in-domain and the knowledge base plausibly covers it, so a real
            # lookup is worth attempting before refusing.
            grounded = ""
            if forged_knowledge and not _is_out_of_scope(message):
                grounded = await self._force_knowledge(message, session_id)
            if grounded:
                reply, tool_calls = grounded, ["knowledge_lookup"]
            else:
                logger.error(
                    "[AGENT] Nothing grounds this turn — refusing out of scope "
                    "| session=%s", session_id,
                )
                reply = _OUT_OF_SCOPE_FALLBACK

        # A draft cart is not a confirmed order. Strip any claim to the contrary
        # that no confirm tool backs — see _strip_false_confirmation.
        _guard_start = time.monotonic()
        reply, stripped = _strip_false_confirmation(reply, tool_calls)
        if stripped:
            logger.error(
                "[AGENT] Unbacked confirmation claim stripped — no confirm tool ran "
                "| session=%s tool_calls=%s cleaned=%r",
                session_id, tool_calls, reply[:160],
            )

        # An item the catalogue refused was never added, however confidently the
        # model narrates otherwise. This is the only guard that reads tool
        # results rather than tool names, so it is the only one that can catch it.
        reply, refused = menu_guard.validate_reply(reply)
        if refused:
            logger.error(
                "[AGENT] Off-menu addition claim replaced with grounded refusal "
                "| session=%s tool_calls=%s reply=%r",
                session_id, tool_calls, reply[:160],
            )

        # A "removed from your order" claim is only true if remove_from_order
        # actually took something off the cart, not merely that it was called.
        # A reference that never matched a cart line (or was never invoked at
        # all) means the item is still there — see agentic/removal_guard.py.
        reply, removal_refused = removal_guard.validate_reply(reply)
        if removal_refused:
            logger.error(
                "[AGENT] Unbacked removal claim replaced with grounded refusal "
                "| session=%s tool_calls=%s reply=%r",
                session_id, tool_calls, reply[:160],
            )

        # A restated total is only trustworthy when it matches this turn's own
        # tool result. Corrected in place (never blocked) — the right number
        # is already known with certainty, so there is nothing to fall back
        # to. See agentic/price_guard.py.
        reply, price_corrected = price_guard.validate_reply(reply)
        if price_corrected:
            logger.error(
                "[AGENT] Hallucinated total corrected against tool result "
                "| session=%s tool_calls=%s reply=%r",
                session_id, tool_calls, reply[:160],
            )
        llm_metrics.record_guard((time.monotonic() - _guard_start) * 1000)

        logger.info("[AGENT] Reply length=%d tool_calls=%s latency_ms=%.0f", len(reply), tool_calls, latency_ms)
        retrieval = llm_metrics.retrieval_snapshot()
        mcp = llm_metrics.mcp_snapshot()
        guard = llm_metrics.guard_snapshot()
        template = llm_metrics.template_snapshot()

        # Reconcile what was already spoken against the authoritative reply.
        # The gate is designed so that no post-hoc guard can rewrite a released
        # sentence; this check proves that held rather than assuming it. On a
        # mismatch the caller is told nothing was streamed, so it speaks the
        # authoritative reply in full — a repeated clause is recoverable, a
        # silently wrong one is not.
        streamed = gate.released_text if gate is not None else ""
        if streamed and not _normalise_for_compare(reply).startswith(
            _normalise_for_compare(streamed)
        ):
            logger.error(
                "[AGENT][STREAM] Released text is not a prefix of the final reply — "
                "discarding stream and replaying in full | session=%s "
                "streamed=%r final=%r",
                session_id, streamed[:160], reply[:160],
            )
            streamed = ""

        return {
            "reply": reply,
            "tool_calls": tool_calls,
            "tool_call_detail": _tool_call_log_ctx.get() or [],
            "llm_ms": llm["ms"],
            "llm_ttft_ms": llm["ttft_ms"],
            "llm_calls": llm["calls"],
            "retrieval_ms": retrieval["ms"],
            "mcp_ms": mcp["ms"],
            "mcp_calls": mcp["calls"],
            "guard_ms": guard["ms"],
            "template_ms": template["ms"],
            "templated": template["calls"] > 0,
            "streamed": streamed,
        }

    async def _force_confirm(self, user_id: str, session_id: str) -> str:
        """Confirm the customer's open draft order deterministically.

        Called only when the customer clearly asked to confirm and the model
        failed to emit a tool call on both the first attempt and the retry.
        The reply is built from the tool result, so what the customer hears
        always matches what is stored.

        Args:
            user_id: Owner of the draft order.
            session_id: Conversation id, for logging only.

        Returns:
            A spoken confirmation built from the tool result, or an empty
            string if there was nothing to confirm.
        """
        try:
            # dry_run must be forced here exactly as _mcp_fn does it. This is a
            # DIRECT call_tool, so it bypasses the _mcp_fn wrapper that normally
            # injects the flag — without this, a speculative draft running on a
            # partial preview transcript ("...and that's my order") would really
            # flip the customer's draft to confirmed, mid-utterance, before they
            # finished speaking.
            envelope = await call_tool(
                "confirm_active_order",
                {"user_id": user_id, "dry_run": _speculative_ctx.get()},
            )
        except Exception:
            logger.exception("[AGENT] Deterministic confirm failed | session=%s", session_id)
            return ""
        # call_tool returns {"status": ..., "result": "<json string>"}; the
        # order (or an {"error": ...}) lives inside that nested string. Reading
        # the envelope directly would treat a failure as a success and speak a
        # confirmation for an order that was never confirmed.
        result: Any = None
        if isinstance(envelope, dict):
            try:
                result = json.loads(envelope.get("result", "") or "null")
            except (json.JSONDecodeError, TypeError):
                result = None
        if not isinstance(result, dict) or result.get("error") or not result.get("order_id"):
            logger.warning(
                "[AGENT] Deterministic confirm found nothing to confirm | session=%s result=%s",
                session_id, str(envelope)[:200],
            )
            return ""
        if result.get("status") != "confirmed":
            logger.error(
                "[AGENT] Deterministic confirm returned status=%s — not claiming success | session=%s",
                result.get("status"), session_id,
            )
            return ""
        order_id = result.get("order_id")
        total = result.get("total")
        logger.info(
            "[AGENT] Model skipped confirm_order — confirmed deterministically | "
            "session=%s order_id=%s", session_id, order_id,
        )
        total_txt = f" The total is {int(total) if float(total).is_integer() else total} rupees." if total is not None else ""
        return f"Your order is confirmed. Your order number is {order_id}.{total_txt} Enjoy your meal!"

    async def _force_knowledge(self, message: str, session_id: str) -> str:
        """Answer an outlet question from the knowledge base, deterministically.

        Used when the customer asked something the knowledge base covers and
        the model produced no tool call on either attempt, leaving it free to
        invent hours or deny knowing the restaurant's own name. Delegates to
        the RAG pipeline's own answer path — the same grounded route the
        non-agent ``/api/v1/query`` endpoint uses.

        Args:
            message: The customer's question.
            session_id: Conversation id, for logging only.

        Returns:
            A grounded answer, or an empty string when the knowledge base has
            nothing relevant (in which case an honest refusal is correct).
        """
        try:
            from pipeline import get_shared_pipeline  # rag-service module

            pipeline = get_shared_pipeline()
            result = await asyncio.to_thread(pipeline.answer_question, message)
        except Exception:
            logger.exception("[AGENT] Deterministic knowledge lookup failed | session=%s", session_id)
            return ""
        answer = (result or {}).get("answer", "").strip()
        if not answer:
            return ""
        answer = _strip_thinking(answer)
        answer = _strip_markdown(answer)
        if _REFUSAL_RE.search(answer):
            # The knowledge base genuinely lacks this. Let the model's own
            # refusal stand rather than swapping in a second one.
            logger.info(
                "[AGENT] Knowledge base has no answer for %r | session=%s", message[:60], session_id,
            )
            return ""
        logger.info(
            "[AGENT] Model answered an outlet question without a tool — replaced "
            "with grounded RAG answer | session=%s chars=%d",
            session_id, len(answer),
        )
        return answer

    async def _force_catalogue(self, message: str, session_id: str) -> tuple[str, str]:
        """Answer a catalogue question deterministically from tool data.

        Used when the customer asked about items or prices and the model
        produced no tool call on either attempt. Rather than speaking a
        dead-end promise ("let me look that up"), fetch the data and answer.

        Args:
            message: The customer utterance, used to pick a category.
            session_id: Conversation id, for logging only.

        Returns:
            A ``(reply, tool_name)`` pair. ``reply`` is empty when the lookup
            produced nothing usable. ``tool_name`` is the tool actually called,
            so telemetry and the downstream order-claim guard both see the
            truth rather than an assumed name.
        """
        category = ""
        lowered = message.lower()
        for keyword in _CATEGORY_KEYWORDS:
            if keyword in lowered:
                category = keyword
                break
        tool = "list_products" if category else "list_categories"
        args = {"category": category} if category else {}
        try:
            envelope = await call_tool(tool, args)
        except Exception:
            logger.exception("[AGENT] Deterministic catalogue lookup failed | session=%s", session_id)
            return "", tool
        # Same formatter the skip-summarization path uses, so a catalogue
        # answer reads identically whether the model called the tool itself or
        # this recovery path had to. ``utterance`` is forced to browse intent:
        # reaching here already means the model produced no tool call at all,
        # so there is no in-flight mutation for a template to cut short.
        spoken = reply_templates.speak(tool, envelope, utterance="")
        if not spoken:
            return "", tool
        logger.info(
            "[AGENT] Model promised a lookup without calling a tool — answered "
            "deterministically | session=%s tool=%s category=%r",
            session_id, tool, category,
        )
        return spoken, tool

    async def _run_turn(
        self,
        user_id: str,
        session_id: str,
        content,
        gate: "_SentenceGate | None" = None,
    ) -> tuple[list[str], list[str]]:
        """Drive one ADK run and collect its text parts and tool invocations.

        Args:
            user_id:    Customer identifier.
            session_id: ADK session identifier.
            content:    ``genai_types.Content`` to send as the new message.
            gate:       Optional sentence gate. When supplied the run is made in
                        SSE streaming mode and completed sentences are released
                        early via the gate; the returned parts are unchanged.

        Returns:
            ``(reply_parts, tool_calls)`` — text fragments emitted by the model
            and the names of the tools it invoked, in call order.
        """
        reply_parts: list[str] = []
        tool_calls: list[str] = []

        run_kwargs = {}
        if gate is not None:
            from google.adk.agents.run_config import RunConfig, StreamingMode

            run_kwargs["run_config"] = RunConfig(streaming_mode=StreamingMode.SSE)

        # Use run_async — run() is documented as "local testing only"
        # and blocks the event loop thread via queue.Queue().get().
        async for event in self._runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
            **run_kwargs,
        ):
            # In SSE mode ADK emits incremental `partial` events followed by a
            # final aggregated event. Only the aggregated text is accumulated,
            # so reply_parts stays byte-identical to the non-streaming path and
            # every downstream guard sees exactly what it saw before.
            is_partial = bool(getattr(event, "partial", False))
            if hasattr(event, "content") and event.content:
                for part in getattr(event.content, "parts", []):
                    # ADK surfaces tool invocations as function_call parts on
                    # the event content — it never populates ``event.tool_call``.
                    fn_call = getattr(part, "function_call", None)
                    if fn_call is not None and getattr(fn_call, "name", None):
                        tool_calls.append(fn_call.name)
                        logger.info("[AGENT] Tool invoked: %s", fn_call.name)
                        # Any text accumulated so far in this turn was
                        # necessarily generated BEFORE this tool call — the
                        # model "thinking out loud" or asking a clarifying
                        # question it is about to answer itself once the tool
                        # result comes back (observed on compound/two-item
                        # requests, e.g. "Would you like one Peri Peri Fries
                        # and one Onion Rings?" immediately followed by "Got
                        # it. ... total is now ₹357."). Speaking both glued
                        # together with no separator produces a
                        # self-contradicting reply. Only text generated after
                        # the LAST tool call in the turn is the grounded final
                        # answer, so discard everything collected up to now.
                        reply_parts.clear()
                    if hasattr(part, "text") and part.text:
                        if is_partial:
                            if gate is not None:
                                gate.feed(part.text, tool_calls)
                        else:
                            reply_parts.append(part.text)

        return reply_parts, tool_calls

    async def reset_sessions(self) -> None:
        """Drop all in-memory conversation sessions.

        Called when the knowledge base changes. ADK's ``InMemorySessionService``
        retains every prior turn, and the model will happily replay a previous
        conclusion instead of calling a tool again. So once a lookup failed —
        e.g. because ASR mis-heard "hours" as "arts" and retrieval correctly
        returned nothing — the answer "we don't have information about the
        restaurant's opening times" stayed in history, and every later attempt
        returned that same sentence with ``tool_calls=[]``, even after the
        document was re-ingested. Clearing sessions forces the next turn to
        query the newly ingested content.
        """
        if not self._bootstrapped:
            return
        self._session_service = create_session_service()
        self._runner = create_runner(self._agent, self._session_service)
        self._customer_names.clear()
        self._dietary_prefs.clear()
        self._cart_states.clear()
        logger.info("[AGENT] Conversation sessions reset — knowledge base changed")

    def forget_conversation(self, session_id: str) -> None:
        """Drop any per-conversation memory held for ``session_id``.

        Called when a conversation ends or the kiosk screen is reset, so the
        next customer never inherits the previous customer's name.
        """
        if self._customer_names.pop(session_id, None) is not None:
            logger.info("[AGENT] Cleared customer name for session=%s", session_id)
        if self._dietary_prefs.pop(session_id, None) is not None:
            logger.info("[AGENT] Cleared dietary preference for session=%s", session_id)
        self._cart_states.pop(session_id, None)

    async def _ensure_session(
        self,
        user_id: str,
        session_id: str,
        history: list[dict[str, str]] | None,
    ) -> None:
        """Create the ADK session and optionally seed it with prior history.

        If the session already exists (normal multi-turn case) this is a
        no-op.  If it does not exist (first turn, or after a rag-service
        restart) we create it and replay any history provided by the caller
        so the agent retains conversation context.
        """
        from google.genai import types as genai_types

        try:
            existing = await self._session_service.get_session(
                app_name=self._agent.name,
                user_id=user_id,
                session_id=session_id,
            )
            if existing is not None:
                return
        except Exception:
            pass  # session service may raise if session not found

        # Build initial events from history so the agent has context
        initial_events: list[genai_types.Content] = []
        for turn in (history or []):
            role = turn.get("role", "")
            text = turn.get("content", "")
            if role in ("user", "assistant") and text:
                adk_role = "model" if role == "assistant" else "user"
                initial_events.append(
                    genai_types.Content(
                        role=adk_role,
                        parts=[genai_types.Part(text=text)],
                    )
                )

        await self._session_service.create_session(
            app_name=self._agent.name,
            user_id=user_id,
            session_id=session_id,
            state={"history_seeded": bool(initial_events)},
        )
        logger.debug(
            "[AGENT] Created session user=%s session=%s history_turns=%d",
            user_id, session_id, len(initial_events),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_agent_instance: OrderingAgent | None = None


def get_ordering_agent() -> OrderingAgent:
    """Return the module-level OrderingAgent singleton (created lazily)."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = OrderingAgent()
    return _agent_instance
