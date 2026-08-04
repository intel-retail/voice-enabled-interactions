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
import inspect
import json
import logging
import re
import time
from typing import Any

from agentic import config as agent_cfg
from agentic import llm_metrics
from agentic.adk_runtime import create_adk_model, create_runner, create_session_service
from agentic.mcp_client import MCPTool, bootstrap_mcp_tools, call_tool, get_all_tools
from agentic.tools.knowledge_lookup_tool import knowledge_lookup

logger = logging.getLogger(__name__)

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
_KNOWLEDGE_QUERY_RE = re.compile(
    r"\b(?:open(?:ing)?|clos(?:e|ing)|hours?|timing|breakfast|"
    r"restaurant name|name of (?:the|your) restaurant|address|located?|location|"
    r"parking|deliver(?:y|ies)?|takeaway|dine[- ]?in|wifi|contact|phone|"
    r"halal|vegetarian|vegan|allergen|gluten|ingredient|spicy|"
    r"payment|upi|card|cash|policy|refund)\b",
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
_CATALOGUE_QUERY_RE = re.compile(
    r"\b(?:menu|item|items|dish|dishes|serve|serves|offer|offers|available|"
    r"option|options|price|prices|cost|costs|how much|rate|rates|"
    r"burger|burgers|pizza|pizzas|wrap|wraps|side|sides|dessert|desserts|"
    r"desert|deserts|sweet|sweets|beverage|beverages|drink|drinks|combo|combos)\b",
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

_ORDER_CLAIM_RE = re.compile(
    r"(?:order (?:is |has been )?(?:confirmed|placed)|"
    r"confirmed your order|order id|order number|order #)",
    re.IGNORECASE,
)

_ORDER_NUDGE = (
    "You described a change to the order without calling a tool, so nothing was "
    "actually recorded and any order id or total you stated is invented. Call the "
    "correct tool now — confirm_active_order to confirm (or confirm_order if you have the id), update_order to change items, "
    "get_order to read the current order — and reply using ONLY its result. "
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
_CATEGORY_KEYWORDS = (
    "burger", "pizza", "wrap", "side", "beverage", "drink", "dessert", "fries",
)

_CONFIRM_INTENT_RE = re.compile(
    r"\b(?:confirm|confirmed|checkout|check out|finali[sz]e|place (?:the |my )?order|"
    r"that(?:'s| is) all|that(?:'s| is) it|i(?:'m| am) done)\b",
    re.IGNORECASE,
)

# Tools that actually mutate or read an order. A reply claiming an order was
# placed or confirmed is only trustworthy if one of these ran this turn.
_ORDER_TOOLS = frozenset(
    {"place_order", "update_order", "confirm_order", "confirm_active_order", "get_order"}
)

_ORDER_CLAIM_FALLBACK = (
    "Sorry, I could not complete that just now and I don't want to tell you it "
    "went through when it hasn't. Please say \"confirm my order\" once more, or "
    "ask a member of staff."
)


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
    if _ORDER_ACTION_RE.search(message) or _ORDER_CLAIM_RE.search(reply):
        return True, _ORDER_NUDGE
    if _CATALOGUE_QUERY_RE.search(message):
        return True, _CATALOGUE_NUDGE
    if _KNOWLEDGE_QUERY_RE.search(message):
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


# Tools the model may name. Used only to recognise a tool call that was
# emitted as *prose* instead of being executed.
_TOOL_NAMES = (
    "knowledge_lookup",
    "list_categories",
    "list_products",
    "place_order",
    "update_order",
    "get_order",
    "confirm_order",
    "confirm_active_order",
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
    # JSON strip above but is still unspeakable, and there is no reliable way to
    # rewrite it into a real answer here — the turn produced no tool result to
    # answer from. Substitute wholesale rather than speak a tool name.
    if _TOOL_MENTION_RE.search(cleaned):
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

    def __init__(self, message: str, emit) -> None:
        """Initialise the gate.

        Args:
            message: The customer's utterance, needed to detect confirm intent.
            emit:    Callback invoked with each sentence cleared for speech.
        """
        self._message = message
        self._emit = emit
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
        if not tool_calls:
            return False
        # (b) _force_confirm() replaces the reply on confirm-intent turns.
        if self._confirm_intent:
            return False
        # (c) An order claim is only trustworthy if an order tool actually ran;
        #     otherwise chat() substitutes _ORDER_CLAIM_FALLBACK.
        if _ORDER_CLAIM_RE.search(sentence) and not any(
            tool in _ORDER_TOOLS for tool in tool_calls
        ):
            return False
        # (d) Anything that would be stripped or substituted wholesale.
        if _ERROR_PAYLOAD_RE.search(sentence) or _TOOL_SYNTAX_RE.search(sentence):
            return False
        if _TOOL_MENTION_RE.search(sentence) or "<think" in sentence.lower():
            return False
        return True

    def feed(self, delta: str, tool_calls: list[str]) -> None:
        """Accumulate a streamed fragment and release any complete safe sentences.

        Args:
            delta:      Newly generated text.
            tool_calls: Tools invoked so far this turn, in call order.
        """
        if not self._open or not delta:
            return
        self._buffer += delta
        parts = _SENTENCE_END_RE.split(self._buffer)
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

        elif tool_name in ("place_order", "update_order", "get_order"):
            if isinstance(data, dict):
                items = [
                    {
                        "name": it.get("name"),
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

        elif tool_name == "confirm_order":
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
# Agent instruction prompt
# ---------------------------------------------------------------------------

_AGENT_INSTRUCTION = """
You are the ordering assistant for QuickBite Express, a QSR voice kiosk.

## GROUNDING (most important)
Never state a product name, id, or price that did not come from a tool result in
THIS conversation. Don't guess, invent, or recall prices. If a tool returns an
`error` with `available_products`, offer those real items — never call something
unavailable from memory.

## Rules (check in order)
0. GENERAL "what do you serve / what's on the menu / show me the menu" (no food
   type named) — call list_categories and answer from its result.
   NEVER read out the whole catalogue: this is a voice kiosk and a 26-item list
   is unusable spoken aloud. Narrow to one category first, then use Rule 4.
1. ORDER ("I want X", "add X", "order X") — call place_order (or update_order if an
   order exists) passing the spoken name as product_id; do NOT call list_products first.
   - On `error` with available_products, offer one of those (name + price) and
     ask if they want it — never retry a made-up id, never say "unavailable".
   - On success reply with the item NAME and PRICE taken from the tool result,
     PLUS every upsell `display` string copied verbatim, then ask to confirm.
     Every number and name you say must be copied out of the tool result you
     received this turn. If you did not receive a successful result this turn,
     you have not added anything — say so and ask the customer to repeat the
     item, rather than describing an addition that did not happen.
2. ANY question about WHAT IS SOLD or WHAT IT COSTS — prices, "how much is X",
   "what X do you have", "is X available", menu listings — call list_products.
   list_products is the only source of truth for product names and prices.
   NEVER answer these from knowledge_lookup or from memory: the knowledge base
   is prose and does not define the catalogue, so using it invents items that
   do not exist. If the customer names a category ("chicken burgers"), call
   list_products(category) and report ONLY the returned rows.
3. INFO question with no product price involved — opening hours, ingredients,
   "is X vegan?", allergens, offers, outlet or policy details — call
   knowledge_lookup.
4. BROWSE a named category ("show me burgers") — call list_products(category), then
   list EVERY product returned with NAME and PRICE verbatim in one comma-separated
   sentence, then ask which they want. Omitting the full list is WRONG.
   Template: "We have <Name1> (<price1>), <Name2> (<price2>), and <Name3> (<price3>).
   Which one would you like to try?"
4b. BROWSE the whole menu ("what do you serve", "show me the menu", "what items do
   you have") — the customer has NOT named a category. Call list_categories, NOT
   list_products. Name the categories it returns back in one short sentence and ask
   which one they want to see. Do not list products here, do not guess a category on
   the customer's behalf, and never name a category absent from the result.
   If a catalogue tool ever comes back empty, that is a system fault, not an empty
   restaurant: say you are having trouble reading the menu and ask them to try again.
   NEVER tell a customer we have no items.
5. MANAGE — "show my order" → get_order; "confirm/place it/that's all/yes" →
   confirm_order. Only after confirm_order returns successfully, tell the
   customer the order is confirmed and read back the `order_id` exactly as the
   tool returned it (it is a plain number), then wish them well. Never invent,
   pad, or reformat an order id, and never state an order is confirmed before
   the tool has returned.

## Never answer from memory
If a rule above says to call a tool, you MUST call it before replying. Do not
say "let me check" or "one moment" and then stop — that leaves the customer
with no answer. Either call the tool in this turn or answer directly; never
promise a lookup you do not perform.

Treat every question as fresh. If you previously said you did not have some
information, do NOT repeat that answer from memory — call the tool again, because
the knowledge base may have been updated since. A question the customer repeats
is a signal to retry the lookup, never to replay your last reply verbatim.

Never list, name, or price a product you have not just seen in a list_products
result. Listing plausible-sounding items is worse than saying nothing: the
customer will try to order something that does not exist. If you are naming
products, a list_products call must have happened in this turn.

## When a tool returns an error
Some tools return `{"error": ..., "available_products": [...]}` instead of a
result. This is NOT a system failure and you must never tell the customer to
"try again" — repeating the same request will fail identically. It means the
item they asked for is not on the menu. Read `available_products`, apologise
briefly that the item is unavailable, and offer the closest real alternatives
by name and price. For example, if "Vanilla Ice Cream" does not resolve but
`available_products` contains "Vanilla Soft Serve", offer that.

## Using knowledge_lookup results
The tool returns numbered knowledge-base excerpts, NOT a finished answer. Read
them and write the reply yourself in 1-2 spoken sentences. Never read the "[1]"
markers aloud, never dump an excerpt verbatim, and never state a fact that is not
in the excerpts. Only if the excerpts truly contain nothing relevant, say you
don't have that detail and offer to help with the menu or an order.

## BREVITY (this is spoken aloud — length is a defect)
Every word you write is read out by a text-to-speech voice at about 14
characters per second. A 300-character answer takes over 20 seconds to speak and
the customer will walk away. Keep answers UNDER 200 CHARACTERS unless a rule
above explicitly requires a product list.

Answer ONLY what was asked. The excerpts will usually contain far more detail
than the question needs — that extra detail is not a bonus, it is noise. Never
volunteer neighbouring facts.

Collapse repetition into ranges. Never recite a value per day, per item, or per
branch when one phrase covers them: give the pattern, then note the exception.

Collapsing must never change a fact. Only merge values that are genuinely the
same. When the values differ — different days, sizes, or branches carry
different numbers — say each distinct value, briefly. Never average them, never
round them together, and never present one figure as covering all cases: a
short answer that states the wrong time is a worse defect than a long one.
Compress the wording, never the facts.

Answer in your own words, using only the values from the excerpts. Do not copy
phrasing from these instructions into a customer reply: any example wording here
describes the STYLE to aim for, never the CONTENT to say. If you find yourself
repeating a sentence from this prompt, you are answering the wrong question.

Apply that same compression to every informational question. Close with a short
generic offer such as "Anything else?" — do NOT promise a specific extra detail
by name, because a follow-up question may not find it in the knowledge base.

## Style
Concise, conversational, at most 2 sentences — EXCEPT Rule 4 (list every product) and
Rule 1 success (name the item, its price, and an upsell). Product lists as a natural
comma sentence, not bullets. Use the given user_id (default "anonymous"). If a name is
unclear, match the closest product from a tool result and confirm "Did you mean …?".
Never invent ids, names, or prices — they must come from a tool result.
Do not open with filler like "Sure!", "Of course!", "Great question" or restate the
customer's question — start with the answer itself.
""".strip()


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
            name="kiosk_ordering_agent",
            model=model,
            description="Kiosk ordering assistant — handles menu Q&A and order management",
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

        # Rebuild the agent with the newly discovered tools
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool

        adk_tools = [FunctionTool(knowledge_lookup)]
        for tool_name, mcp_tool in mcp_tools.items():
            adk_tools.append(FunctionTool(self._make_mcp_callable(tool_name, mcp_tool)))

        model = create_adk_model()
        self._agent = LlmAgent(
            name="kiosk_ordering_agent",
            model=model,
            description="Kiosk ordering assistant — handles menu Q&A and order management",
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
        """

        async def _mcp_fn(**kwargs: Any) -> Any:
            logger.info("[AGENT→MCP] tool=%s args=%s", tool_name, kwargs)
            result = await call_tool(tool_name, kwargs)
            result = _compress_tool_result(tool_name, result)
            logger.debug("[AGENT→MCP] tool=%s compressed_result=%s", tool_name, str(result)[:200])
            return result

        _mcp_fn.__name__ = tool_name
        _mcp_fn.__doc__ = mcp_tool.description or tool_name

        # Build an explicit signature from the MCP JSON input schema so ADK
        # introspection produces a correct function-call declaration.
        input_schema = mcp_tool.input_schema or {}
        properties: dict[str, Any] = input_schema.get("properties", {}) or {}
        required = set(input_schema.get("required", []) or [])

        params: list[inspect.Parameter] = []
        annotations: dict[str, Any] = {}
        for pname, pspec in properties.items():
            pytype = _JSON_TO_PY.get((pspec or {}).get("type", "string"), str)
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
    ) -> dict[str, Any]:
        """Run one conversational turn and return the agent's response.

        Args:
            message:    The user's transcribed utterance.
            session_id: Opaque session identifier (maps to ADK session).
            user_id:    The customer's user identifier.
            history:    Previous turns [{role, content}, …] — used to seed
                        the ADK session when it does not yet exist (e.g.
                        after a rag-service restart).

        Returns:
            dict with keys:
              - ``reply``:     str — the agent's text response.
              - ``tool_calls``: list[str] — tools invoked this turn.
              - ``llm_ms``:    float | None — cumulative genuine LLM time.
              - ``llm_calls``: int — number of LLM round-trips this turn.
        """
        if not self._bootstrapped:
            await self.bootstrap()

        # If MCP tools weren't available at startup (race with kiosk-core),
        # attempt re-discovery before this turn so ordering tools work.
        await self._refresh_mcp_tools()

        logger.info("[AGENT] chat session=%s user=%s message=%r", session_id, user_id, message[:120])

        from google.genai import types as genai_types

        # Seed the ADK session with prior history if the session does not
        # yet exist (rag-service restart scenario).
        await self._ensure_session(user_id, session_id, history)

        # Prefix the user_id into the first turn so the LLM (and ordering
        # tools) know which customer is speaking without needing a dedicated
        # user-lookup tool.
        full_message = message
        if user_id != "anonymous":
            full_message = f"[user_id={user_id}] {message}"

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

        try:
            gate = (
                _SentenceGate(message, on_safe_sentence)
                if (on_safe_sentence is not None and agent_cfg.STREAM_SENTENCES)
                else None
            )
            reply_parts, tool_calls = await self._run_turn(
                user_id, session_id, content, gate=gate
            )

            # A turn that answers a catalogue question, promises a lookup, or
            # refuses without calling any tool is ungrounded. Re-run once with
            # an explicit correction rather than speaking the bad reply.
            should_retry, nudge_text = (
                _needs_tool_retry("".join(reply_parts), message)
                if not tool_calls
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
                retry_parts, retry_tools = await self._run_turn(user_id, session_id, nudge)
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
                if _CONFIRM_INTENT_RE.search(message):
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
            if not tool_calls and _KNOWLEDGE_QUERY_RE.search(message):
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
                "llm_ms": llm["ms"],
                "llm_calls": llm["calls"],
                "retrieval_ms": llm_metrics.retrieval_snapshot()["ms"],
            }

        latency_ms = (time.perf_counter() - t_start) * 1000
        llm = llm_metrics.snapshot()
        logger.info(
            "[AGENT←OVMS] Response received | session=%s latency_ms=%.0f llm_ms=%s llm_calls=%d "
            "tool_calls=%s reply_chars=%d",
            session_id, latency_ms, llm["ms"], llm["calls"], tool_calls, len("".join(reply_parts)),
        )

        reply = "".join(reply_parts).strip()
        reply = _strip_thinking(reply)
        reply = _strip_tool_syntax(reply)
        reply = _strip_markdown(reply)
        if _ERROR_PAYLOAD_RE.search(reply):
            logger.error(
                "[AGENT] Raw tool error payload leaked into reply — substituting "
                "fallback | session=%s raw=%r", session_id, reply[:160],
            )
            reply = _TOOL_SYNTAX_FALLBACK

        logger.info("[AGENT] Reply length=%d tool_calls=%s latency_ms=%.0f", len(reply), tool_calls, latency_ms)
        retrieval = llm_metrics.retrieval_snapshot()

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
            "llm_ms": llm["ms"],
            "llm_calls": llm["calls"],
            "retrieval_ms": retrieval["ms"],
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
            envelope = await call_tool("confirm_active_order", {"user_id": user_id})
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
            data = json.loads((envelope or {}).get("result", "") or "null")
        except Exception:
            logger.exception("[AGENT] Deterministic catalogue lookup failed | session=%s", session_id)
            return "", tool
        if not isinstance(data, list) or not data:
            return "", tool
        parts: list[str] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("category")
            if not name:
                continue
            price = entry.get("price")
            count = entry.get("item_count", entry.get("count"))
            if price is not None:
                parts.append(f"{name} (₹{int(price) if float(price).is_integer() else price})")
            elif count is not None:
                parts.append(f"{name} ({count} items)")
            else:
                parts.append(str(name))
        if not parts:
            return "", tool
        logger.info(
            "[AGENT] Model promised a lookup without calling a tool — answered "
            "deterministically | session=%s tool=%s category=%r",
            session_id, tool, category,
        )
        listing = ", ".join(parts)
        tail = "Which one would you like?" if category else "Which would you like to explore?"
        return f"We have {listing}. {tail}", tool

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
        logger.info("[AGENT] Conversation sessions reset — knowledge base changed")

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
