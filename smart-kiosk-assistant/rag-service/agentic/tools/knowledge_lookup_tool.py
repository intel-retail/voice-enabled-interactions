"""knowledge_lookup_tool — wraps the existing RAG pipeline for Q&A.

This gives the ADK agent access to the kiosk knowledge base (menus, FAQs,
policies) so it can answer information questions via the same retrieval +
generation pipeline used by the /api/v1/query endpoint.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def knowledge_lookup(question: str) -> str:
    """Answer a question about menu items, policies, or general information.

    Use this tool to answer questions like "Do you serve pizza?", "What are
    your opening hours?", or "What's the price of a Paneer Tikka Burger?".
    Do NOT use this tool for placing, updating, or confirming orders.

    Args:
        question: The user's question about menu or outlet information.

    Returns:
        A factual answer generated from the knowledge base.
    """
    logger.info("[TOOL:knowledge_lookup] question=%r", question[:120])
    try:
        from pipeline import get_shared_pipeline  # rag-service module

        pipeline = get_shared_pipeline()
        # Collect all streamed tokens into a single string
        tokens: list[str] = []
        for token in pipeline.stream_answer(question, history=None):
            tokens.append(token)
        answer = "".join(tokens).strip()
        logger.debug("[TOOL:knowledge_lookup] answer length=%d", len(answer))
        return answer or "I'm sorry, I couldn't find relevant information."
    except Exception as exc:
        logger.error("[TOOL:knowledge_lookup] Pipeline error: %s", exc)
        return f"I'm unable to look that up right now ({exc})."
