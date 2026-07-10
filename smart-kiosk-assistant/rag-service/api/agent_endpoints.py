"""Agent chat endpoint for the ordering flow.

POST /api/v1/agent/chat
  Request body:
    transcription  str        — user's spoken input (transcribed)
    user_id        str        — customer identifier (default: "anonymous")
    session_id     str        — conversation session ID
    history        list[dict] — optional prior turns [{role, content}, ...]

  Response:
    reply          str        — agent's text response
    tool_calls     list[str]  — tools invoked during this turn
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AgentChatRequest(BaseModel):
    transcription: str = Field(..., description="User's spoken input (transcribed text)")
    user_id: str = Field(default="anonymous", description="Customer identifier")
    session_id: str = Field(..., description="Conversation session ID")
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior conversation turns [{role, content}, ...]",
    )


class AgentChatResponse(BaseModel):
    reply: str
    tool_calls: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=AgentChatResponse, summary="Agent ordering chat")
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    """Run one agent turn for ordering/Q&A.

    The agent decides whether to:
    - Answer from the knowledge base (knowledge_lookup tool → RAG)
    - Place / update / confirm an order (MCP tools → kiosk-core)
    - Suggest upsell items (get_upsell_suggestions MCP tool)
    """
    logger.info(
        "[AGENT-ENDPOINT] session=%s user=%s message=%r",
        request.session_id,
        request.user_id,
        request.transcription[:100],
    )

    try:
        from agentic.ordering_agent import get_ordering_agent

        agent = get_ordering_agent()
        result = await agent.chat(
            message=request.transcription,
            session_id=request.session_id,
            user_id=request.user_id,
            history=request.history,
        )
    except Exception as exc:
        logger.error("[AGENT-ENDPOINT] Unhandled error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        "[AGENT-ENDPOINT] session=%s reply_len=%d tool_calls=%s",
        request.session_id,
        len(result.get("reply", "")),
        result.get("tool_calls", []),
    )
    return AgentChatResponse(
        reply=result["reply"],
        tool_calls=result.get("tool_calls", []),
    )
