"""knowledge_lookup_tool — retrieves kiosk knowledge-base context for the agent.

The tool deliberately returns *retrieved context*, not a generated answer.

Earlier this wrapped the full RAG pipeline (``pipeline.stream_answer``), which
ran a complete LLM generation inside the tool. Because Google ADK already makes
two LLM round-trips per tool-calling turn (one to pick the tool, one to compose
the reply), that meant three sequential generations per question — the answer
was written by the RAG pipeline and then rewritten by the agent. Measured on a
"what are your opening hours?" turn, the tool alone cost 8.76 s of an 18.3 s
turn while retrieval itself was only 91 ms.

Returning context lets the agent compose the reply once, from the same
knowledge, inside its existing second call.
"""

from __future__ import annotations

import logging
import os
import re
import time

from agentic import llm_metrics
from utils.latency_store import retrieval_latency

logger = logging.getLogger(__name__)

# Char budget for the context handed back to the agent. The agent's reply call
# must prefill whatever we return, so this trades answer coverage against
# time-to-first-token.
#
# It is also the only reliable brake on reply length. The 4B int4 model
# paraphrases roughly everything it is handed, so context size sets answer size:
# measured on "opening and closing hours", 2001 chars of context produced a
# 277-char reply — 20 seconds of speech on a voice kiosk. Prompt instructions to
# Lower means shorter spoken answers and less prefill; too low starts
# dropping facts the customer actually asked for. Measured on that same
# question: 2000 chars -> 315-char reply / 6835 ms LLM; 900 -> 218 / 5727;
# 600 -> 147 / 4401 but the reply became factually WRONG ("open 8 AM to 12 AM
# on weekdays and weekends", collapsing distinct weekday and weekend hours).
# 900 is the floor that still answered accurately, so it is the default.
_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "1800"))

_NO_CONTEXT = "No relevant information was found in the knowledge base."

# ── pinned root-section context ──────────────────────────────────────
#
# Ingestion prefixes every chunk with its heading path, e.g.
#   "[Context: QuickBite Express — Restaurant Knowledge Base > MENU 5: Pizza]"
# The chunk whose path has no ">" is the document's *root section* — the
# overview block that carries the venue-level facts (brand name, hours,
# breakfast hours, kitchen close, delivery, parking, wi-fi).
#
# That single block answers almost every question this tool legitimately
# receives, because catalogue questions are refused above and routed to
# `list_products`. Yet it is the block retrieval is worst at finding, since
# it covers ~14 unrelated topics at once and so embeds as a weak match for
# all of them. Measured rank of the root chunk over the QuickBite KB:
#
#   query                        ANN rank   rerank rank
#   "restaurant name"                   4           29
#   "opening and closing hours"         1            2
#   "breakfast timings"                16           25
#   "do you have parking"               2            1
#   "when does the kitchen close"       5            1
#
# The reranker actively demotes it for the two questions where it is the
# only source of truth. With top_k=3 and a char budget smaller than one
# chunk, the correct block was retrieved for "opening hours" at rank 2 and
# then discarded by truncation — the kiosk answered "all day from 11 AM"
# (a snack-counter line) instead of its real hours.
#
# Pinning it removes retrieval from the path for global facts entirely.
# This is deterministic, unlike tuning a corpus-dependent score threshold.
_PIN_ROOT_SECTION = os.getenv("RAG_PIN_ROOT_SECTION", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
_PIN_MAX_CHARS = int(os.getenv("RAG_PIN_MAX_CHARS", "1300"))
_ROOT_SECTION_RE = re.compile(r"^\[Context:\s*([^\]]*)\]")

_pinned_context: str | None = None


def reset_pinned_context() -> None:
    """Drop the cached root-section pin.

    Must be called whenever the vector store changes. The pin is resolved once
    and cached for the process lifetime, so without this a re-ingested or
    cleared knowledge base keeps serving the previous document's global facts
    (restaurant name, opening hours) as authoritative — the customer is told
    the old restaurant's details with full confidence.
    """
    global _pinned_context
    if _pinned_context:
        logger.info("[TOOL:knowledge_lookup] Knowledge base changed — clearing pinned root section")
    _pinned_context = None


def _root_section(pipeline) -> str:
    """Return the document's root-section chunk, cached after first lookup.

    Args:
        pipeline: The shared ``RagPipeline``.

    Returns:
        The root-section text, or an empty string when it cannot be
        identified. Never raises — a missing pin degrades to plain
        retrieval rather than failing the customer's question.
    """
    global _pinned_context
    if _pinned_context is not None:
        return _pinned_context

    _pinned_context = ""
    if not _PIN_ROOT_SECTION:
        return _pinned_context

    try:
        roots = []
        for doc in pipeline.iter_documents():
            text = doc.strip()
            match = _ROOT_SECTION_RE.match(text)
            if match and ">" not in match.group(1):
                roots.append(text)
        if roots:
            # Prefer the densest root block when a corpus has several docs.
            _pinned_context = max(roots, key=len)[:_PIN_MAX_CHARS]
            logger.info(
                "[TOOL:knowledge_lookup] Pinned root section (%d chars)",
                len(_pinned_context),
            )
        else:
            logger.warning(
                "[TOOL:knowledge_lookup] No root section found — global facts "
                "will depend on retrieval ranking",
            )
    except Exception as exc:  # noqa: BLE001 - pin is best-effort
        logger.warning("[TOOL:knowledge_lookup] Root-section pin unavailable: %s", exc)

    return _pinned_context

# Questions about what is sold and what it costs must be answered from the
# product database via ``list_products``, never from knowledge-base prose. The
# knowledge base is marketing/menu narrative: it mentions dish names loosely and
# does not define the catalogue, so composing an answer from it invents products.
# Observed in production — a single "price of chicken burger" turn reported
# seven chicken burgers of which four ("Grilled Chicken Burger", "Chicken BBQ
# Burger", "Chicken Tikka Burger", "Chicken Schnitzel Sandwich") do not exist in
# the catalogue, each with a fabricated price.
#
# Prompt guidance alone did not stop this, so the tool refuses the query itself.
_PRICE_QUERY_RE = re.compile(
    r"\b(?:price|prices|pricing|cost|costs|how much|rate|rates|"
    r"cheapest|expensive|menu items?|what.{0,20}(?:do you (?:have|serve|sell))|"
    r"available|availability)\b",
    re.IGNORECASE,
)
# Terms that indicate a genuine knowledge-base question even when a price-ish
# word appears (e.g. "how much sugar is in the lassi", "is the offer free").
_KB_OVERRIDE_RE = re.compile(
    r"\b(?:hour|hours|open|close|closing|timing|address|location|contact|"
    r"vegan|vegetarian|allerg\w*|gluten|ingredient\w*|calorie\w*|nutrition\w*|"
    r"spicy|policy|refund|delivery|parking|wifi|offer|discount|combo deal)\b",
    re.IGNORECASE,
)

_USE_LIST_PRODUCTS = (
    "This question is about products, prices, or availability. The knowledge "
    "base is not authoritative for the catalogue and using it would invent "
    "items that do not exist. Call `list_products` instead and answer only "
    "from its result. Do not answer this question from the knowledge base or "
    "from memory."
)


def _format_sources(records, budget: int = _MAX_CONTEXT_CHARS) -> str:
    """Render retrieval records into a compact, numbered context block."""
    blocks: list[str] = []
    used = 0
    for index, record in enumerate(records, start=1):
        content = (getattr(record, "content", "") or "").strip()
        if not content:
            continue
        block = f"[{index}] {content}"
        if used + len(block) > budget:
            remaining = budget - used
            # Keep a partial block only when enough room is left to be useful.
            if remaining > 200:
                blocks.append(block[:remaining].rstrip())
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


async def knowledge_lookup(question: str) -> str:
    """Look up facts about hours, ingredients, allergens, policies, or outlet information.

    Use this tool for questions like "What are your opening hours?", "Is the
    Paneer Tikka Burger vegetarian?", or "Do you have parking?".

    Do NOT use this tool for prices, product availability, or menu listings —
    call ``list_products`` for those, it is the only authoritative source for
    product names and prices. Do NOT use it for placing, updating, or
    confirming orders.

    Args:
        question: The user's question about menu or outlet information.

    Returns:
        Knowledge-base excerpts relevant to the question. Compose the customer
        reply from these excerpts; do not read the numbering back to the user.
    """
    logger.info("[TOOL:knowledge_lookup] question=%r", question[:120])

    # Refuse catalogue questions outright — see _PRICE_QUERY_RE.
    if _PRICE_QUERY_RE.search(question) and not _KB_OVERRIDE_RE.search(question):
        logger.info(
            "[TOOL:knowledge_lookup] Redirecting product/price question to list_products: %r",
            question[:120],
        )
        return _USE_LIST_PRODUCTS

    started = time.perf_counter()
    try:
        from pipeline import get_shared_pipeline  # rag-service module

        pipeline = get_shared_pipeline()
        pinned = _root_section(pipeline)
        records = pipeline.retrieve(question)

        # The root section is frequently also returned by retrieval. Paying
        # for it twice would evict the question-specific chunk that retrieval
        # was actually needed for.
        if pinned:
            head = pinned[:80]
            records = [
                record
                for record in records
                if not (getattr(record, "content", "") or "").strip().startswith(head)
            ]

        elapsed_ms = (time.perf_counter() - started) * 1000
        retrieval_latency.record(elapsed_ms)
        llm_metrics.record_retrieval(elapsed_ms)

        remaining = _MAX_CONTEXT_CHARS - (len(pinned) + 2 if pinned else 0)
        retrieved = _format_sources(records, budget=max(remaining, 0))
        blocks = [block for block in (pinned, retrieved) if block]
        context = "\n\n".join(blocks)
        logger.info(
            "[TOOL:knowledge_lookup] retrieved=%d pinned=%d context_chars=%d elapsed_ms=%.0f",
            len(records), len(pinned), len(context), elapsed_ms,
        )
        return context or _NO_CONTEXT
    except Exception as exc:
        logger.error("[TOOL:knowledge_lookup] Retrieval error: %s", exc, exc_info=True)
        return f"I'm unable to look that up right now ({exc})."
