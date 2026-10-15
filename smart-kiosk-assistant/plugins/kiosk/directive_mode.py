"""Directive-mode ordering turn — one LLM call that yields action *and* speech.

Why this exists
---------------
The tool-calling path has a hard latency floor that no amount of tuning
removes. Measured against this exact OVMS/Qwen3-4B-int4 deployment:

* Passing ``tools`` costs a flat ~350 ms regardless of tool count (1 tool and
  12 tools both cost ~460 ms to first content; 0 tools costs ~106 ms). It is
  not guided generation — that is already disabled. Timing the raw SSE stream
  shows the model starts decoding at ~66 ms, and the ``hermes3`` parser simply
  cannot emit a structured delta until it has buffered the
  ``<tool_call>{"name": ..., "arguments":`` preamble.
* Worse, a tool call produces **no speech at all**. The arguments must arrive
  complete before a reply template can turn them into words, so TTS cannot
  overlap generation. First audio therefore cannot start before ~1,150 ms.

This module removes both costs by having the model emit a compact directive
inline, immediately followed by the prose it should say, in a single
generation with no ``tools`` parameter:

    <act>add|Classic Chicken Burger|1</act>One classic chicken burger, got it.

Measured on the same deployment: first content at 137 ms, three speakable
prose words by ~551 ms, whole generation 670 ms — against 1,154-1,574 ms
before *any* speech existed. The approach is taken from the reference
prototype in ``kiosk-voice-lab-main`` (``pipeline/cart.py``).

The safety trade
----------------
Free-form text is weaker than a JSON schema, so this module deliberately
narrows what the model is trusted with: **it never speaks a price or a
total.** The model produces only a short, price-free confirmation. Every
number the customer hears is appended afterwards from the tool's own result
by :mod:`reply_templates`. A directive that cannot be parsed, or that names an
item the catalogue cannot resolve, returns ``None`` so the caller falls back
to the authoritative tool-calling path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Callable

import httpx

from agentic import config as agent_cfg
from agentic import llm_metrics
from agentic.action_result import unwrap
from agentic.mcp_client import call_tool

logger = logging.getLogger(__name__)

# Express cart mutations as an inline <act> directive instead of a structured
# tool call. Lives here rather than in agentic/config.py because this is a
# kiosk-plugin feature, and because plugins/ is bind-mounted — toggling it
# needs only a restart, not an image rebuild.
#
# OFF by default. Free-form text is a weaker contract than a JSON schema, so
# this path never lets the model speak a price, and hands the turn back to the
# tool-calling path whenever the directive is missing, unparseable, or names
# an item the catalogue cannot resolve.
DIRECTIVE_MODE: bool = os.getenv(
    "AGENT_DIRECTIVE_MODE", "false"
).lower() in ("true", "1", "yes")

# Read timeout for the tool-free directive completion.
LLM_TIMEOUT: float = float(os.getenv("AGENT_LLM_TIMEOUT", "120"))

# Process-wide pooled client to OVMS. stream_completion() used to open a new
# httpx.AsyncClient (and therefore a new TCP connection) on every single turn
# — measured live, the directive that this module's docstring claims appears
# at ~137ms was instead landing at ~525ms, and connection setup on every call
# is exactly the kind of fixed per-call cost that would explain it. One
# pooled client, created lazily and reused for the life of the process, lets
# httpx keep the OVMS connection alive between turns instead of paying a
# fresh TCP (and any local proxy/DNS resolution) handshake each time.
_HTTP_CLIENT: httpx.AsyncClient | None = None
_HTTP_CLIENT_LOCK = asyncio.Lock()


async def _get_http_client() -> httpx.AsyncClient:
    """Return the process-wide pooled OVMS client, creating it on first use."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        async with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None:  # re-check inside the lock
                _HTTP_CLIENT = httpx.AsyncClient(
                    timeout=LLM_TIMEOUT,
                    limits=httpx.Limits(
                        max_keepalive_connections=4,
                        max_connections=8,
                        keepalive_expiry=60.0,
                    ),
                )
    return _HTTP_CLIENT


async def close_http_client() -> None:
    """Close the pooled OVMS client. Call once, at process shutdown."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        await _HTTP_CLIENT.aclose()
        _HTTP_CLIENT = None

# Sentence boundary for releasing speech to TTS early. Deliberately requires
# whitespace-or-end after the terminator so "₹169." or "No. 2" do not split.
_SENTENCE_RE = re.compile(r"(?<=[.!?])(?:\s+|$)")

# The two terminal directives a directive-mode generation may end with: a cart
# mutation (<act>) or a price lookup (<check_price>). Both are parsed the same
# way structurally (freeze prose the instant the tag opens, dispatch on close)
# even though what runs afterward differs — see run_turn's on_delta.
_ACT_TAG = "act"
_CHECK_PRICE_TAG = "check_price"
_DIRECTIVE_TAGS = (_ACT_TAG, _CHECK_PRICE_TAG)


def _partial_tag_pattern(tag: str) -> str:
    """Regex matching any partial prefix of ``<tag>`` anchored at end of string.

    Used to strip a directive tag that is still mid-stream (e.g. "<", "<ac",
    "<check_pr") from text about to be spoken — see _STRAY_DIRECTIVE_RE.

    Args:
        tag: The bare tag name, e.g. "act" or "check_price".

    Returns:
        A regex alternation of every non-empty prefix of ``<tag>``, longest
        first, anchored at end of string.
    """
    full = f"<{tag}>"
    return "(?:" + "|".join(re.escape(full[:i]) for i in range(len(full), 0, -1)) + ")$"


# Any further directive block after the first one, plus a trailing partial tag
# still being streamed ("<", "<act", "<act>add|Fries", "<check_pr"). Directive
# markup is an internal wire format — it must never reach TTS or the screen.
_STRAY_DIRECTIVE_RE = re.compile(
    "|".join(
        [rf"<(?:{'|'.join(_DIRECTIVE_TAGS)})>.*?</(?:{'|'.join(_DIRECTIVE_TAGS)})>",
         rf"<(?:{'|'.join(_DIRECTIVE_TAGS)})>.*$"]
        + [_partial_tag_pattern(t) for t in _DIRECTIVE_TAGS]
    ),
    re.DOTALL,
)


class _StopGeneration(Exception):
    """Raised from the delta callback to abort a stream that has gone wrong.

    Once the model starts a second directive there is nothing useful left in the
    turn — every further token is wasted latency — so the HTTP stream is closed
    rather than read to completion.
    """


def scrub_directives(text: str) -> str:
    """Strip directive markup from text destined for the customer.

    ``run_turn`` only consumes the *first* ``<act>…</act>``; everything after it
    is treated as speech. A model that emits a second directive would therefore
    have raw markup spoken aloud and printed on screen. Observed live:

        'Got it. One Classic French Fries. <act>add|Classic Chicken Burger|1</act>Got it. …'

    Args:
        text: Candidate prose.

    Returns:
        ``text`` with directive blocks and partial tags removed.
    """
    return _STRAY_DIRECTIVE_RE.sub("", text).strip()


# Verbs the parser accepts. Anything else is treated as unparseable and falls
# back, rather than being guessed at.
_ADD, _REMOVE, _CONFIRM = "add", "remove", "confirm"


def _split_sentences(text: str) -> tuple[list[str], str]:
    """Split ``text`` into complete sentences plus an unterminated remainder.

    Args:
        text: Prose accumulated so far this turn.

    Returns:
        ``(complete, remainder)`` where ``complete`` holds sentences that are
        safe to synthesise now and ``remainder`` is the partial tail still
        being generated.
    """
    parts = _SENTENCE_RE.split(text)
    if not parts:
        return [], text
    remainder = parts[-1]
    complete = [p.strip() for p in parts[:-1] if p.strip()]
    return complete, remainder


def parse_directive(raw: str) -> list[dict[str, Any]] | None:
    """Parse the ``<act>…</act>`` block into structured actions.

    Args:
        raw: The directive body, e.g. ``add|Classic Chicken Burger|1;confirm``.

    Returns:
        A list of action dicts, or ``None`` when any clause is malformed —
        a partially-understood directive is never applied, because dropping
        half of "remove the burger; add the wrap" would corrupt the order.
    """
    actions: list[dict[str, Any]] = []
    for clause in (raw or "").split(";"):
        clause = clause.strip()
        if not clause:
            continue
        fields = [f.strip() for f in clause.split("|")]
        verb = fields[0].lower()
        if verb == _CONFIRM:
            actions.append({"verb": _CONFIRM})
            continue
        if verb not in (_ADD, _REMOVE) or len(fields) < 2 or not fields[1]:
            return None
        quantity = 1
        if len(fields) >= 3 and fields[2]:
            if not fields[2].isdigit():
                return None
            quantity = int(fields[2])
            if quantity < 1:
                return None
        actions.append({"verb": verb, "name": fields[1], "quantity": quantity})
    return actions or None


def parse_price_directive(raw: str) -> str | None:
    """Parse the ``<check_price>…</check_price>`` block into an item name.

    Args:
        raw: The directive body, e.g. ``Chocolate Brownie``.

    Returns:
        The trimmed item name, or ``None`` when empty.
    """
    name = (raw or "").strip()
    return name or None


_MENU_PRICE_INDEX: dict[str, tuple[str, float]] | None = None
_MENU_LINE_RE = re.compile(r"^(.*)\s\([^\d]*([\d.]+)\)$")


def _menu_price_index() -> dict[str, tuple[str, float]]:
    """Parse the cached menu block (see get_menu_block) into a price lookup.

    Built once per process, lazily, from the same ``_MENU_CACHE`` already
    trusted to teach the model exact item names for ``<act>`` — reusing it
    here means a price lookup costs zero extra MCP round-trips.

    Returns:
        ``{name.lower(): (canonical_name, price)}``, or ``{}`` if the menu
        cache has not been populated yet (caller then falls back).
    """
    global _MENU_PRICE_INDEX
    if _MENU_PRICE_INDEX is not None:
        return _MENU_PRICE_INDEX
    index: dict[str, tuple[str, float]] = {}
    for line in (_MENU_CACHE or "").splitlines():
        match = _MENU_LINE_RE.match(line.strip())
        if not match:
            continue
        name = match.group(1).strip()
        try:
            price = float(match.group(2))
        except ValueError:
            continue
        index[name.lower()] = (name, price)
    if index:
        _MENU_PRICE_INDEX = index
    return index


async def _execute_price(item_name: str) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Resolve a customer's price question against the cached catalogue.

    Args:
        item_name: The item name the model echoed back, expected to be the
            exact catalogue name per the prompt's "Use item names EXACTLY as
            they appear in the menu" instruction (see build_directive_spec).

    Returns:
        ``(payload, tools_called, committed)``, matching ``_execute()``'s
        contract so ``run_turn`` can treat both directives identically.
        ``payload`` is ``{"name", "price"}`` on a resolved match, or ``None``
        when the name cannot be matched exactly — deliberately no
        fuzzy/partial matching, since a wrong guess here would speak an
        incorrect price with total confidence; the caller then falls back to
        the authoritative list_products tool-calling path. ``committed`` is
        always empty: a price lookup never mutates the cart, so unlike a
        failed ``<act>`` there is no double-apply hazard blocking that
        fallback.
    """
    match = _menu_price_index().get((item_name or "").strip().lower())
    if match is None:
        return None, [], []
    name, price = match
    return {"name": name, "price": price}, [], []


_MENU_CACHE: str | None = None


async def get_menu_block() -> str:
    """Return the catalogue rendered for the prompt, fetched once per process.

    The catalogue is static configuration seeded at startup, so re-fetching it
    every turn would add an MCP round-trip (~70 ms) to the critical path for
    data that never changes.

    Returns:
        One ``Name (currency price)`` entry per line, or ``""`` when the
        catalogue could not be read (the caller then falls back).
    """
    global _MENU_CACHE
    if _MENU_CACHE is not None:
        return _MENU_CACHE
    try:
        from agentic.action_result import unwrap_any
        from agentic import domain_config

        # list_products with no category returns a category *summary*
        # ([{category, item_count}, ...]), not products — so the catalogue has
        # to be walked one category at a time. Done once per process.
        summary = unwrap_any(await call_tool("list_products", {}))
        if not isinstance(summary, list) or not summary:
            logger.warning("[DIRECTIVE] catalogue summary empty — directive mode inactive")
            return ""
        categories = [
            c["category"] for c in summary
            if isinstance(c, dict) and c.get("category")
        ]
        currency = domain_config.get_currency_symbol()
        lines: list[str] = []
        for category in categories:
            products = unwrap_any(await call_tool("list_products", {"category": category}))
            if not isinstance(products, list):
                continue
            lines.extend(
                f"{p['name']} ({currency}{p['price']:g})"
                for p in products
                if isinstance(p, dict) and p.get("name") and p.get("price") is not None
            )
        if not lines:
            logger.warning("[DIRECTIVE] no products resolved — directive mode inactive")
            return ""
        _MENU_CACHE = "\n".join(lines)
        logger.info(
            "[DIRECTIVE] menu cached (%d items across %d categories)",
            len(lines), len(categories),
        )
        return _MENU_CACHE
    except Exception:
        logger.warning("[DIRECTIVE] menu fetch failed", exc_info=True)
        return ""


def build_directive_spec(menu_block: str) -> str:
    """Build the directive instructions appended to the system prompt.

    Args:
        menu_block: A one-line-per-item catalogue rendering, used so the model
            emits names the ordering tools can actually resolve.

    Returns:
        The prompt fragment describing the directive contract.
    """
    return f"""
ORDERING FORMAT — follow this exactly.
When the customer orders, changes, or cancels something, speak a SHORT confirmation of that
change FIRST, THEN end your reply with ONE directive describing ONLY what this sentence changed:
<act>add|EXACT ITEM NAME|QUANTITY</act>
<act>remove|EXACT ITEM NAME|QUANTITY</act>
<act>confirm</act>
The directive is the LAST thing in your reply, after all spoken words — never before them, never
in the middle, and never on its own with no spoken words. Several changes in one sentence go in
one directive separated by semicolons. Use item names EXACTLY as they appear in the menu below.
Never speak a directive out loud. Questions get no directive.

CRITICAL: never say any price, total, or amount of money. Do not offer or suggest extra items.
Those are added for you automatically.
Begin with a TWO-WORD acknowledgement sentence ending in a full stop
("Got it." / "Sure thing." / "All set."), then ONE short confirmation sentence naming the item,
THEN the directive.

Example:
Customer: two classic chicken burgers
You: Got it. Two chicken burgers.<act>add|Classic Chicken Burger|2</act>

PRICE QUESTIONS — follow this exactly.
When the customer asks how much a single menu item costs, begin with a SHORT generic
acknowledgement of TWO OR THREE WORDS ("Let me check." / "One moment." / "Sure, checking."),
THEN end your reply with the directive. Never state or hint at the price, total, or any amount of
money in that acknowledgement — it is spoken for you afterward, from the catalogue, not from you:
<check_price>EXACT ITEM NAME</check_price>
Use the item name EXACTLY as it appears in the menu below. If the question names more than one
item, or is not about a single item's price, do not use this directive.

Example:
Customer: how much does the chocolate brownie cost?
You: Let me check.<check_price>Chocolate Brownie</check_price>

MENU:
{menu_block}
"""


async def _execute(
    actions: list[dict[str, Any]], user_id: str
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Apply parsed actions through the existing MCP ordering tools.

    Args:
        actions: Output of :func:`parse_directive`.
        user_id: Customer identity injected into every ordering call.

    Returns:
        ``(payload, tools_called, committed)``. ``payload`` is the final tool
        result, or ``None`` when a call failed or was rejected. ``committed``
        lists only the tools that actually changed cart state.

        The distinction matters: a ``None`` payload makes the caller fall back
        to the authoritative tool-calling path, which re-interprets the SAME
        utterance and would re-apply anything already written. "add a burger,
        remove the fries" where the add succeeds and the remove fails would
        otherwise add a second burger. When ``committed`` is non-empty the
        caller must not fall back.
    """
    payload: dict[str, Any] | None = None
    called: list[str] = []
    committed: list[str] = []
    adds = [a for a in actions if a["verb"] == _ADD]
    if adds:
        payload = unwrap(await call_tool("place_order", {
            "user_id": user_id,
            "items": [{"product_id": a["name"], "quantity": a["quantity"]} for a in adds],
        }))
        called.append("place_order")
        if not isinstance(payload, dict) or "error" in payload:
            return None, called, committed
        # An add the catalogue could not resolve, or one that needs the
        # customer to choose between variants, is exactly the nuance the
        # template cannot narrate. Hand the turn back.
        if payload.get("needs_choice") or payload.get("unavailable_message"):
            return None, called, committed
        if not payload.get("just_added"):
            return None, called, committed
        committed.append("place_order")

    for action in (a for a in actions if a["verb"] == _REMOVE):
        payload = unwrap(await call_tool("remove_from_order", {
            "user_id": user_id,
            "items": [{"product_id": action["name"], "quantity": action["quantity"]}],
        }))
        called.append("remove_from_order")
        if not isinstance(payload, dict) or "error" in payload:
            return None, called, committed
        committed.append("remove_from_order")

    if any(a["verb"] == _CONFIRM for a in actions):
        payload = unwrap(await call_tool("confirm_active_order", {"user_id": user_id}))
        called.append("confirm_active_order")
        if not isinstance(payload, dict) or "error" in payload:
            return None, called, committed
        committed.append("confirm_active_order")

    if payload is None:
        return None, called, committed
    return payload, called, committed


_FACTS_TAIL_RE = re.compile(
    r"\s*(?:Your (?:new )?total is(?: now)?\b|Would you also like\b|Would you like anything else\b).*",
    re.IGNORECASE | re.DOTALL,
)

# Spoken when some cart writes committed but a later one failed. The turn
# cannot fall back (that would re-apply the committed writes) and cannot claim
# success either, so it asks the customer to re-state rather than guessing.
_PARTIAL_FAILURE_REPLY = (
    "Sorry, I only managed part of that. Could you tell me again what you'd "
    "like, and I'll check your order?"
)


def strip_facts_tail(content: str) -> str:
    """Remove the composed, price-bearing tail from a prior assistant reply.

    A directive-mode reply is ``model prose + _facts_tail(payload)``. The client
    displays that whole string and then feeds it back as one assistant history
    turn, so from the model's point of view it looks like *it* produced the
    prices and the upsell question. On the next turn it dutifully imitates the
    pattern — emitting a total and a closing question itself — which then gets
    the real tail appended on top.

    Observed live: turn 1 generated 2 sentences in 851 ms; turn 2 generated 4 in
    1473 ms and spoke "Your total is now Rs.258" twice. The loop compounds, so
    every turn is slower and more duplicated than the last.

    Stripping the tail out of history keeps the model's in-context examples
    matching the behaviour we actually want: short prose, no money.

    Args:
        content: A prior assistant turn as displayed to the customer.

    Returns:
        Just the prose portion, or the original string when no tail is found.
    """
    stripped = _FACTS_TAIL_RE.sub("", content).strip()
    return stripped or content


async def stream_completion(
    system_prompt: str,
    message: str,
    history: list[dict[str, str]] | None,
    on_delta: Callable[[str], None],
) -> str:
    """Stream one tool-free completion from the LLM, invoking ``on_delta``.

    Args:
        system_prompt: Full system instruction including the directive spec.
        message: The customer's utterance for this turn.
        history: Prior turns as ``{"role", "content"}`` dicts.
        on_delta: Called with each raw text delta as it arrives.

    Returns:
        The complete generated text.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in history or []:
        role, content = turn.get("role"), turn.get("content")
        if role in ("user", "assistant") and content:
            if role == "assistant":
                content = strip_facts_tail(content)
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    body = {
        "model": agent_cfg.LLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": agent_cfg.MAX_TOKENS,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": agent_cfg.ENABLE_THINKING},
    }
    url = agent_cfg.LLM_URL.rstrip("/") + "/chat/completions"

    out: list[str] = []
    client = await _get_http_client()
    async with client.stream("POST", url, json=body) as response:
        response.raise_for_status()
        try:
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                text = delta.get("content") or ""
                if text:
                    out.append(text)
                    on_delta(text)
        except _StopGeneration:
            # The callback decided the rest of the generation is unusable.
            # Returning here closes the stream and skips the wasted tokens.
            pass
    return "".join(out)


def _facts_tail(payload: dict[str, Any]) -> str:
    """Build the price-bearing part of the reply from the tool result.

    The model is forbidden from speaking money, so the total and the upsell
    offer are composed here, straight from the payload. This is why directive
    mode cannot misquote a price: the model never sees one to repeat.

    Args:
        payload: The ordering tool's result.

    Returns:
        A sentence or two of factual tail, or ``""`` when the payload carries
        no total (e.g. a removal that emptied the cart).
    """
    from agentic import domain_config
    from plugins.kiosk.reply_templates import _money

    total = payload.get("total")
    if total is None:
        return ""
    currency = domain_config.get_currency_symbol()
    out = f"Your total is now {currency}{_money(total)}."
    upsell = payload.get("upsell_suggestions") or []
    if upsell and isinstance(upsell[0], dict) and upsell[0].get("display"):
        out += f" Would you also like {upsell[0]['display']}?"
    else:
        out += " Would you like anything else?"
    return out


def _price_tail(payload: dict[str, Any]) -> str:
    """Build the spoken reply for a resolved ``<check_price>`` directive.

    Mirrors ``_facts_tail``: the model is told never to speak a price, so this
    is the only place a price answer is composed — straight from the cached
    catalogue lookup (see ``_execute_price``), never from the model's own text.

    Args:
        payload: ``{"name", "price"}`` from ``_execute_price``.

    Returns:
        A one-sentence spoken price answer.
    """
    from agentic import domain_config
    from plugins.kiosk.reply_templates import _money

    currency = domain_config.get_currency_symbol()
    return f"The {payload['name']} costs {currency}{_money(payload['price'])}."


async def run_turn(
    message: str,
    user_id: str,
    history: list[dict[str, str]] | None,
    menu_block: str,
    base_instruction: str,
    on_safe_sentence: Callable[[str], None] | None,
) -> dict[str, Any] | None:
    """Run one ordering turn in directive mode.

    Speech is released to TTS as prose sentences complete, and the directive
    is now the LAST thing the model emits (see ``build_directive_spec``) —
    reversed from the original "directive, then prose" layout. Measured live
    against this exact deployment: the model decodes at a very consistent
    ~28 ms/token, and the old layout forced "Got it." to wait behind the
    ~12-token ``<act>add|Classic Chicken Burger|1</act>`` directive before it
    could even start (~307ms of pure decode paid before the first speakable
    word), pushing first-audio out to ~600ms even though the raw first token
    landed at ~135-210ms. Putting prose first means the first sentence is
    gated only by however many tokens THAT sentence needs (~3-5 for "Got
    it."), cutting first-audio roughly in half. The cart mutation dispatch
    correspondingly moves later (after the confirmation prose instead of
    before it) — harmless, since it was always fire-and-forget and does not
    gate audio in either layout, only how soon the (price-bearing) tail is
    ready, which happens well after first-audio regardless.

    Args:
        message: The customer's utterance.
        user_id: Customer identity for the ordering tools.
        history: Prior conversation turns.
        menu_block: Catalogue rendering for the prompt.
        base_instruction: The agent's existing system instruction.
        on_safe_sentence: Callback invoked with each sentence cleared for TTS.

    Returns:
        A turn result dict compatible with the tool-calling path, or ``None``
        to signal the caller must fall back to that path.
    """


    t0 = time.monotonic()
    system_prompt = base_instruction + build_directive_spec(menu_block)

    state: dict[str, Any] = {
        "buf": "", "task": None, "spoken": [], "rest": "",
        # Which directive tag (see _DIRECTIVE_TAGS) first opened in the
        # stream, or None until one does. From that point on, prose is FROZEN
        # at buf.split(f"<{tag}>", 1)[0] — nothing after the tag is prose, so
        # this can only be set once.
        "tag": None,
        # True once the open tag's matching close tag has been seen and
        # dispatch attempted. Distinct from "tag" being set so a directive
        # left unclosed by a truncated/aborted generation is still correctly
        # treated as "no usable directive" by the fallback check below.
        "tag_dispatched": False,
        # Sentences pulled off the stream so far (>= len(spoken), because
        # facts-bearing sentences are consumed but never spoken).
        "consumed": 0,
        # Set once the model starts narrating prices; all further prose is
        # dropped in favour of the payload-derived tail.
        "prose_closed": False,
        # First-delta wall-clock, for llm_ttft_ms below — directive mode is a
        # single raw completion (no ADK LiteLlm wrapper), so nothing else
        # times its prefill separately.
        "first_delta_ms": None,
    }

    def _emit_prose(prose: str) -> None:
        """Pull any newly-completed sentences out of ``prose`` and speak them."""
        if state["prose_closed"]:
            return
        complete, remainder = _split_sentences(prose)
        state["rest"] = remainder
        new = complete[state["consumed"]:]
        state["consumed"] = len(complete)
        for sentence in new:
            # The model is told never to speak money, but in-context drift can
            # make it start doing so anyway. Anything from the first such
            # sentence onward duplicates the authoritative tail composed from
            # the tool payload, so refuse to speak it: stop accepting prose and
            # let _facts_tail() supply the numbers.
            if _FACTS_TAIL_RE.match(sentence):
                logger.info(
                    "[DIRECTIVE] model began speaking facts — suppressing prose from %r",
                    sentence[:60],
                )
                state["prose_closed"] = True
                return
            sentence = scrub_directives(sentence)
            if not sentence:
                continue
            state["spoken"].append(sentence)
            if on_safe_sentence is not None:
                on_safe_sentence(sentence)

    def on_delta(_text: str) -> None:
        if state["first_delta_ms"] is None:
            state["first_delta_ms"] = (time.monotonic() - t0) * 1000.0
        state["buf"] += _text
        buf: str = state["buf"]

        if state["tag"] is None:
            # Whichever directive tag (see _DIRECTIVE_TAGS) opens first in the
            # stream — at most one can ever apply per turn, per the prompt.
            opened = min(
                (idx, tag) for tag in _DIRECTIVE_TAGS
                if (idx := buf.find(f"<{tag}>")) != -1
            ) if any(f"<{tag}>" in buf for tag in _DIRECTIVE_TAGS) else None
            if opened:
                state["tag"] = opened[1]

        # Prose is everything before the first open tag — frozen the instant
        # that tag appears, since the directive is now expected to be the
        # last thing in the reply. Everything from here on speaks only what
        # was already buffered before the tag showed up.
        prose = buf.split(f"<{state['tag']}>", 1)[0] if state["tag"] else buf
        _emit_prose(prose)

        if state["prose_closed"]:
            raise _StopGeneration

        close_tag = f"</{state['tag']}>" if state["tag"] else None
        if state["tag"] and not state["tag_dispatched"] and close_tag in buf:
            state["tag_dispatched"] = True
            # Prose is frozen now (no further growth possible), so flush
            # whatever incomplete sentence is still sitting in the remainder
            # rather than silently dropping it — the model may not always
            # close the final sentence with punctuation before starting the
            # directive tag.
            leftover = (state["rest"] or "").strip()
            if leftover and not state["prose_closed"] and not _FACTS_TAIL_RE.match(leftover):
                leftover = scrub_directives(leftover)
                if leftover:
                    state["spoken"].append(leftover)
                    if on_safe_sentence is not None:
                        on_safe_sentence(leftover)
            state["rest"] = ""

            after_open = buf.split(f"<{state['tag']}>", 1)[1]
            directive_body, closed, _ = after_open.partition(close_tag)
            if closed:
                if state["tag"] == _ACT_TAG:
                    actions = parse_directive(directive_body)
                    if actions:
                        state["task"] = asyncio.create_task(_execute(actions, user_id))
                    else:
                        state["task"] = "unparseable"
                else:  # _CHECK_PRICE_TAG
                    item_name = parse_price_directive(directive_body)
                    if item_name:
                        state["task"] = asyncio.create_task(_execute_price(item_name))
                    else:
                        state["task"] = "unparseable"
            # Nothing usable follows the directive in this layout — it is
            # the last thing in the reply — so stop reading the stream
            # instead of paying for tail tokens that will only be discarded.
            raise _StopGeneration

    try:
        raw = await stream_completion(system_prompt, message, history, on_delta)
    except Exception:
        # The dispatch task is fire-and-forget, so an exception here would
        # otherwise leave it running (and possibly mutating the cart) while
        # chat() swallows the error and falls back — the same double-apply
        # hazard as the committed-then-failed path below. Settle it first.
        task = state.get("task")
        if isinstance(task, asyncio.Task):
            try:
                _, _, committed = await task
                if committed:
                    logger.error(
                        "[DIRECTIVE] generation failed after committing %s — "
                        "cart already changed; fallback will double-apply",
                        committed,
                    )
            except Exception:
                logger.exception("[DIRECTIVE] dispatch task failed during generation error")
        raise
    gen_ms = (time.monotonic() - t0) * 1000.0
    # Recorded here (not by a wrapper like ADK's _TimedLiteLlm) because
    # directive mode calls the raw completion API directly — nothing else
    # times this call. Without this, llm_metrics' accumulators stay at their
    # reset() defaults for every directive-mode turn, and mcp_ms/mcp_calls
    # below would report the current call_tool() round-trip as if it never
    # happened once read back via mcp_snapshot() (its own accumulator is
    # independent and unaffected, but keeping llm/mcp recorded the same way
    # for every code path is what makes the two comparable turn-to-turn).
    llm_metrics.record(gen_ms, ttft_ms=state["first_delta_ms"])

    if not state["tag_dispatched"] or state["task"] == "unparseable" or state["task"] is None:
        logger.info(
            "[DIRECTIVE] no usable directive (tag_dispatched=%s) — falling back | gen=%.0fms raw=%r",
            state["tag_dispatched"], gen_ms, raw[:120],
        )
        return None

    executed, tools_called, committed = await state["task"]
    if executed is None:
        # Falling back re-runs the SAME utterance through the tool-calling
        # path. That is only safe while nothing has been written yet — once a
        # cart mutation is committed, a fallback would apply it a second time
        # (the "add succeeded, remove failed" case). Speak an honest partial
        # failure instead and end the turn here.
        if committed:
            logger.error(
                "[DIRECTIVE] tools committed %s then a later call failed — "
                "cannot fall back without double-applying | gen=%.0fms",
                committed, gen_ms,
            )
            _mcp = llm_metrics.mcp_snapshot()
            return {
                "reply": _PARTIAL_FAILURE_REPLY,
                "streamed": " ".join(state["spoken"]).strip(),
                "tool_calls": tools_called,
                "llm_calls": 1,
                "llm_ms": gen_ms,
                "llm_ttft_ms": state["first_delta_ms"],
                "retrieval_ms": None,
                "mcp_ms": _mcp["ms"],
                "mcp_calls": _mcp["calls"],
                "guard_ms": None,
                "template_ms": None,
                "templated": True,
                "directive": True,
            }
        logger.info("[DIRECTIVE] tool execution rejected the action — falling back")
        return None
    payload = executed

    # The model spoke a price-free confirmation (a cart-change summary for
    # <act>, a short generic acknowledgement like "Let me check." for
    # <check_price>); every number the customer hears is appended here,
    # verbatim from the tool result / cached catalogue lookup. (state["rest"]
    # is always empty here — any trailing incomplete sentence was already
    # flushed and spoken the moment the directive tag appeared, since prose
    # is frozen at that point and cannot grow further.)
    spoken = " ".join(state["spoken"]).strip()
    tail = _price_tail(payload) if state["tag"] == _CHECK_PRICE_TAG else _facts_tail(payload)
    # Last line of defence: whatever happens above, directive markup never
    # reaches TTS or the screen.
    reply = scrub_directives(f"{spoken} {tail}".strip())

    logger.info(
        "[DIRECTIVE] turn complete | gen=%.0fms sentences_streamed=%d reply=%r",
        gen_ms, len(state["spoken"]), reply[:120],
    )
    _mcp = llm_metrics.mcp_snapshot()
    return {
        "reply": reply,
        "streamed": " ".join(state["spoken"]).strip(),
        "tool_calls": tools_called,
        "llm_calls": 1,
        "llm_ms": gen_ms,
        "llm_ttft_ms": state["first_delta_ms"],
        "retrieval_ms": None,
        "mcp_ms": _mcp["ms"],
        "mcp_calls": _mcp["calls"],
        "guard_ms": None,
        "template_ms": None,
        "templated": True,
        "directive": True,
    }
