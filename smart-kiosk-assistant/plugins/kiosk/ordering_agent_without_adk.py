"""Non-ADK ordering agent — one LLM call per turn, no ``tools=``, no framework.

Why this file exists
---------------------
This is the "lab-style" counterpart to :mod:`plugins.kiosk.ordering_agent`,
built for a controlled A/B: does removing Google ADK / LiteLLM / native
OpenAI ``tools=`` actually explain the ~600-740 ms LLM TTFT the production
agent reports, or does the floor sit somewhere else?

Phase 0 measurement on this exact deployment (OVMS, Qwen3-4B-int4-ov, this
host's Panther Lake NPU/iGPU) answered that before this file was written:

    arm                                          TTFT (first byte, median)
    no system prompt, no tools                        33 ms
    real agent instruction + 12 native tools=[]        44 ms
    real agent instruction, lab-style <act> directive  41 ms

Raw first-token latency is indistinguishable across all three. It is *not*
where the 600-740 ms production number comes from. What actually costs time
is (a) the production agent's **two** LLM round-trips per turn (tool-select
call, then a narration call) with each call's own decode cost, (b) a fresh
MCP streamable-HTTP session opened per tool call, and (c) ``llm_metrics``
summing TTFT *across* both calls into one reported number. This module tests
the fix implied by that finding: collapse every turn to **one** LLM call,
remove ``tools=`` entirely, and let the model emit its action as inline text
— exactly ``kiosk-voice-lab-main``'s ``<act>…</act>`` convention — instead of
a JSON tool-call delta.

What this reuses, unchanged, from the production plugin
---------------------------------------------------------
* :mod:`plugins.kiosk.directive_mode` — the single-call, tool-free ordering
  path (``<act>add|Item|qty;confirm</act>`` + prose). Already implements
  exactly the mechanism above for cart mutations; this file's job is to
  extend it to the turn types it does not cover (browse, FAQ, order-state
  reads) rather than reinvent it.
* :mod:`plugins.kiosk.reply_templates` — deterministic, LLM-free replies for
  root facts (hours/location) and catalogue reads.
* :mod:`agentic.mcp_client`, :mod:`agentic.action_result` — MCP transport and
  result envelope handling.
* :mod:`agentic.llm_metrics` — the same span accumulators the ADK path
  writes, so both agents are comparable turn-for-turn in
  ``kiosk_core.pipeline_latency``.
* :mod:`agentic.tools.knowledge_lookup_tool` — the existing RAG retrieval
  tool, called directly as a Python function (pre-grounding) rather than
  exposed to the model as a callable tool.

What this deliberately does NOT reproduce
-------------------------------------------
This is a benchmarking vehicle, not a production replacement. Relative to
``plugins.kiosk.ordering_agent.OrderingAgent`` it does not implement:
customer-name memory, dietary-preference memory, cart-state
staleness/confirmation guarding, speculative (dry-run) drafting, or the full
menu/removal/confirm truthfulness-guard chain. It reuses the guard chain only
where doing so is cheap and safe (see ``_apply_guards`` below); anywhere it
does not, this is called out in a comment.

Turn routing (all branches are at most one LLM call; several are zero)
------------------------------------------------------------------------
1. Root-fact questions ("what are your hours") — answered from the parsed
   knowledge-base root section. Zero LLM calls.
2. Order-mutation intent (add/remove/confirm/cancel) — dispatched to
   :func:`plugins.kiosk.directive_mode.run_turn` unchanged. One LLM call.
3. Everything else (browse/catalogue, FAQ, order-state reads) — one
   streamed completion with no ``tools=`` field, whose system prompt already
   contains the full menu (mirrors the lab's ``build_prompt()``: stable
   rules first, then context, then the question) plus, when the turn looks
   like a knowledge question, pre-fetched RAG context from
   ``knowledge_lookup()`` — mirroring the lab's retrieval-before-generation
   design instead of exposing retrieval as a callable tool.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from agentic import config as agent_cfg
from agentic import domain_config
from agentic import llm_metrics
from agentic.action_result import unwrap
from agentic.mcp_client import bootstrap_mcp_tools, call_tool, get_all_tools
from agentic.tools import knowledge_lookup_tool as knowledge_tool
from agentic.tools.knowledge_lookup_tool import knowledge_lookup

from plugins.kiosk import directive_mode
from plugins.kiosk import reply_templates

logger = logging.getLogger(__name__)


# Heuristic only — same intent surface directive_mode already uses to decide
# whether a turn is a mutation. Kept local (not imported) so this file has no
# hidden coupling to reply_templates' regex beyond the public is_browse_intent
# call below, which is the one guarantee we rely on.
_KNOWLEDGE_HINT_RE = re.compile(
    r"\b(hour|open|close|closing|opening|location|address|parking|wifi|wi-fi|"
    r"allerg|vegan|vegetarian|gluten|halal|kosher|ingredient|policy|refund|"
    r"return|delivery|reservation|book a table|contact|phone|email)\b",
    re.IGNORECASE,
)


class OrderingAgentWithoutADK:
    """Lab-style ordering agent: one LLM call per turn, no ADK, no ``tools=``.

    Public surface matches :class:`plugins.kiosk.ordering_agent.OrderingAgent`
    (``bootstrap``/``warmup``/``chat``/``reset_sessions``) so it is a drop-in
    swap behind the same plugin-loader seam, and so the existing benchmark
    harness's request/response shape needs no changes to point at it.
    """

    def __init__(self) -> None:
        self._bootstrapped = False
        # No ADK session service exists here, so conversation history is
        # whatever the caller passes in `history` each turn — this class
        # keeps no cross-turn state of its own beyond the MCP tool registry.

    async def bootstrap(self) -> None:
        """Discover MCP tools. No model/runner/session construction — there
        is no ADK agent to build; the LLM is called directly per turn."""
        if self._bootstrapped:
            return
        logger.info("[NOADK] Bootstrapping OrderingAgentWithoutADK …")
        mcp_tools = await bootstrap_mcp_tools(agent_cfg.MCP_CONFIG_PATH)
        logger.info("[NOADK] MCP tools: %s", list(mcp_tools))
        self._bootstrapped = True

    async def _refresh_mcp_tools(self) -> None:
        if get_all_tools():
            return
        logger.info("[NOADK] No MCP tools registered — retrying discovery …")
        mcp_tools = await bootstrap_mcp_tools(agent_cfg.MCP_CONFIG_PATH)
        if mcp_tools:
            logger.info("[NOADK] MCP re-discovery succeeded: %s", list(mcp_tools))

    async def warmup(self, max_attempts: int = 30, delay_seconds: float = 2.0) -> bool:
        """Poll kiosk-core until MCP tools and the menu cache are ready."""
        import asyncio

        if not self._bootstrapped:
            await self.bootstrap()
        for attempt in range(max_attempts):
            if get_all_tools():
                menu = await directive_mode.get_menu_block()
                if menu:
                    logger.info("[NOADK] warmup ready after %d attempt(s)", attempt + 1)
                    return True
            await asyncio.sleep(delay_seconds)
            await self._refresh_mcp_tools()
        logger.warning("[NOADK] warmup exhausted %d attempts", max_attempts)
        return False

    async def reset_sessions(self) -> None:
        """No ADK session store to clear; the menu cache is the only
        process-lifetime state and it only changes when the catalogue does."""
        directive_mode._MENU_CACHE = None  # noqa: SLF001 — same process, same intent as ADK's reset

    # -- turn execution ----------------------------------------------------

    async def chat(
        self,
        message: str,
        session_id: str,
        user_id: str = "anonymous",
        history: list[dict[str, str]] | None = None,
        on_safe_sentence=None,
        speculative: bool = False,
    ) -> dict[str, Any]:
        """Run one turn. See module docstring for the routing this follows.

        Contract matches ``OrderingAgent.chat()`` (see that docstring for the
        full field list); ``speculative`` is accepted for API compatibility
        but not implemented — speculative dry-run turns are always routed to
        the real production agent, never this one, until dry-run plumbing is
        added here.
        """
        if not self._bootstrapped:
            await self.bootstrap()
        await self._refresh_mcp_tools()

        logger.info(
            "[NOADK] chat session=%s user=%s message=%r", session_id, user_id, message[:120]
        )
        llm_metrics.reset()

        # ── 1. Root-fact fast path — zero LLM calls ─────────────────────
        facts_wanted = reply_templates.classify_root_facts(message)
        if facts_wanted:
            t0 = time.monotonic()
            try:
                facts = knowledge_tool.root_facts()
            except Exception:
                logger.exception("[NOADK] root_facts() failed")
                facts = {}
            spoken = reply_templates.speak_root_fact(facts_wanted, facts)
            if spoken:
                llm_metrics.record_template((time.monotonic() - t0) * 1000)
                template = llm_metrics.template_snapshot()
                return self._result(reply=spoken, llm_calls=0, template_ms=template["ms"], templated=True)

        # ── 2. Order-mutation intent — one LLM call, directive_mode as-is ──
        if not reply_templates.is_browse_intent(message):
            menu_block = await directive_mode.get_menu_block()
            if menu_block:
                try:
                    result = await directive_mode.run_turn(
                        message=message,
                        user_id=user_id,
                        history=history,
                        menu_block=menu_block,
                        base_instruction=_BASE_INSTRUCTION,
                        on_safe_sentence=on_safe_sentence,
                    )
                    if result is not None:
                        return result
                except Exception:
                    logger.warning("[NOADK] directive_mode.run_turn failed, falling back", exc_info=True)
            # Falls through to the generic single-call path below on a miss
            # (unparseable directive, empty menu, or an exception) — that
            # path still answers with no `tools=`, so the turn stays inside
            # the one-LLM-call design even on this fallback.

        # ── 3. Generic single-call path: browse / FAQ / order-state reads ──
        return await self._run_generic_turn(message, user_id, history, on_safe_sentence)

    async def _run_generic_turn(
        self,
        message: str,
        user_id: str,
        history: list[dict[str, str]] | None,
        on_safe_sentence,
    ) -> dict[str, Any]:
        """One streamed completion, no ``tools=``, context pre-fetched.

        Mirrors ``kiosk-voice-lab-main``'s ``build_prompt()``: stable system
        rules first, then retrieved context, then the customer's question —
        chosen deliberately so the stable prefix stays byte-identical turn to
        turn for OVMS's prefix cache, exactly as that prototype's own
        comments warn (``AGENTS.md`` — "don't move per-turn text earlier").
        """
        t0 = time.monotonic()
        menu_block = await directive_mode.get_menu_block()

        retrieval_ms: float | None = None
        knowledge_block = ""
        if _KNOWLEDGE_HINT_RE.search(message):
            r0 = time.monotonic()
            try:
                excerpt = await knowledge_lookup(message)
                if excerpt and not excerpt.lower().startswith("no relevant"):
                    knowledge_block = f"\n\n[knowledge]\n{excerpt}\n[/knowledge]"
            except Exception:
                logger.warning("[NOADK] knowledge_lookup pre-fetch failed", exc_info=True)
            retrieval_ms = (time.monotonic() - r0) * 1000.0

        order_block = ""
        mcp_ms = None
        mcp_calls = 0
        if _ORDER_STATE_RE.search(message):
            m0 = time.monotonic()
            try:
                current = unwrap(await call_tool("get_current_order", {"user_id": user_id}))
                mcp_calls = 1
                if isinstance(current, dict):
                    order_block = f"\n\n[current_order]\n{current}\n[/current_order]"
            except Exception:
                logger.warning("[NOADK] get_current_order pre-fetch failed", exc_info=True)
            mcp_ms = (time.monotonic() - m0) * 1000.0

        system_prompt = (
            _NO_TOOLS_QA_INSTRUCTION
            + (f"\n\nMENU:\n{menu_block}" if menu_block else "")
            + knowledge_block
            + order_block
        )

        spoken_sentences: list[str] = []
        first_delta_ms: list[float | None] = [None]

        def on_delta(text: str) -> None:
            if first_delta_ms[0] is None:
                first_delta_ms[0] = (time.monotonic() - t0) * 1000.0

        raw = await directive_mode.stream_completion(system_prompt, message, history, on_delta)
        gen_ms = (time.monotonic() - t0) * 1000.0
        llm_metrics.record(gen_ms, ttft_ms=first_delta_ms[0])

        reply = directive_mode.scrub_directives(raw).strip()
        if on_safe_sentence is not None and reply:
            on_safe_sentence(reply)
            spoken_sentences.append(reply)

        return self._result(
            reply=reply or "Sorry, could you say that again?",
            llm_calls=1,
            llm_ms=gen_ms,
            llm_ttft_ms=first_delta_ms[0],
            retrieval_ms=retrieval_ms,
            mcp_ms=mcp_ms,
            mcp_calls=mcp_calls,
            streamed=" ".join(spoken_sentences),
        )

    @staticmethod
    def _result(
        reply: str,
        llm_calls: int,
        llm_ms: float | None = None,
        llm_ttft_ms: float | None = None,
        retrieval_ms: float | None = None,
        mcp_ms: float | None = None,
        mcp_calls: int = 0,
        template_ms: float | None = None,
        templated: bool = False,
        streamed: str = "",
    ) -> dict[str, Any]:
        return {
            "reply": reply,
            "tool_calls": [],
            "tool_call_detail": [],
            "llm_ms": llm_ms,
            "llm_ttft_ms": llm_ttft_ms,
            "llm_calls": llm_calls,
            "retrieval_ms": retrieval_ms,
            "mcp_ms": mcp_ms,
            "mcp_calls": mcp_calls,
            "guard_ms": None,
            "template_ms": template_ms,
            "templated": templated,
            "streamed": streamed,
            "directive": llm_calls <= 1,
        }


_ORDER_STATE_RE = re.compile(
    r"\b(what(?:'s| is) in my (?:cart|order)|my (?:current )?order|"
    r"what did i order|order status|my cart)\b",
    re.IGNORECASE,
)

# Same core-framework text as plugins/kiosk/ordering_agent.py's
# _AGENT_INSTRUCTION, assembled through the same domain_config seam so both
# agents start from an identical prompt and the A/B isolates the runtime, not
# the prompt content.
_CORE_FRAMEWORK = """
## Multi-action turns
After each tool result, check whether another requested action remains. Call the next tool. Speak only after ALL actions are complete.

## [knowledge] block
When [knowledge]…[/knowledge] is present, answer from it directly — do not call knowledge_lookup. Never read the tags or "[1]" markers aloud. Summarise in 1–2 sentences.

## [current_order] block
When [current_order]…[/current_order] is present, answer the customer's question about their order from it directly. Never invent items or totals not in that block.

## Output
Responses are spoken aloud. Keep replies concise — under 200 characters for routine turns, longer only when listing catalogue items or confirming a completed transaction.
Never open with "Sure!", "Of course!", or restate the question. Start with the answer.
Only use names, prices, and transaction IDs that appeared in a tool result or context block this turn.
""".strip()

_BASE_INSTRUCTION: str = domain_config.get_agent_instruction(_CORE_FRAMEWORK)

# Dedicated no-tools instruction for the generic (browse/FAQ/order-state-read)
# turn — see kiosk-voice-lab-main's QSR_QA_SYSTEM_PROMPT, which this mirrors.
# _BASE_INSTRUCTION above pulls in domain_config's "tool_rules" section
# (list_products / place_order / etc.), which is correct for the ADK agent
# but actively wrong here: with no `tools=` field on this call the model has
# nothing to call, so instructed to "call list_products" it just writes
# "list_products (category=burgers)" as its spoken reply instead of an
# answer. This prompt only reuses the persona and tells the model to answer
# directly from the MENU/[knowledge]/[current_order] blocks already placed in
# the prompt, and never to write a function/tool name.
_NO_TOOLS_QA_INSTRUCTION: str = (
    domain_config.get_agent_instruction_section("persona")
    + "\n\n"
    + """
Answer using ONLY the MENU, [knowledge], and [current_order] blocks below — you have no tools or functions to call this turn, so never write a function/tool name (e.g. "list_products", "get_current_order") as part of your reply.
If asked about a category or the whole menu, name up to four items with prices in one short sentence, never as a list.
If asked about your current order, answer from [current_order] only; never invent items or totals not in that block.
If the blocks do not contain the answer, say you are not sure and offer to get a team member.
Keep replies to one or two short spoken sentences. Never restate the question, never open with "Sure!"/"Of course!", start with the answer.
""".strip()
).strip()


_agent_singleton: OrderingAgentWithoutADK | None = None


def get_ordering_agent_without_adk() -> OrderingAgentWithoutADK:
    """Factory matching the ``agent_module``/``agent_factory`` plugin-loader
    contract in ``configs/rag-service/agent_profile.yaml``. Returns a
    process-wide singleton, same lifecycle as ``get_ordering_agent()``."""
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = OrderingAgentWithoutADK()
    return _agent_singleton
