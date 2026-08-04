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
    llm_ms         float|None — cumulative genuine LLM time for the turn
    llm_calls      int        — number of LLM round-trips for the turn
    retrieval_ms   float|None — knowledge-base retrieval time for the turn
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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
    llm_ms: float | None = Field(
        default=None,
        description="Cumulative LLM round-trip time for this turn, in milliseconds",
    )
    llm_calls: int = Field(
        default=0,
        description="Number of LLM round-trips made during this turn",
    )
    retrieval_ms: float | None = Field(
        default=None,
        description=(
            "Knowledge-base retrieval time for this turn, in milliseconds. "
            "None when the agent did not call knowledge_lookup."
        ),
    )


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
        llm_ms=result.get("llm_ms"),
        llm_calls=result.get("llm_calls", 0),
        retrieval_ms=result.get("retrieval_ms"),
    )


@router.post("/chat/stream", summary="Agent ordering chat (streaming)")
async def agent_chat_stream(request: AgentChatRequest) -> StreamingResponse:
    """Run one agent turn, emitting speakable sentences as they are produced.

    Emits newline-delimited JSON objects:
      ``{"delta": "<sentence>"}``  zero or more, each cleared for immediate TTS
      ``{"final": {...}}``         exactly one, the authoritative turn result

    ``final.streamed`` repeats everything already sent as deltas, so the caller
    can synthesise only the remainder. When it is empty the caller must speak
    ``final.reply`` in full — either nothing was safe to stream, or a
    post-generation guard rewrote the reply and the stream was discarded.

    Deltas are best-effort: a turn may legitimately produce none. Callers must
    always treat ``final`` as the source of truth.
    """
    logger.info(
        "[AGENT-ENDPOINT] stream session=%s user=%s message=%r",
        request.session_id,
        request.user_id,
        request.transcription[:100],
    )

    queue: asyncio.Queue = asyncio.Queue()

    async def run() -> None:
        try:
            from agentic.ordering_agent import get_ordering_agent

            agent = get_ordering_agent()
            result = await agent.chat(
                message=request.transcription,
                session_id=request.session_id,
                user_id=request.user_id,
                history=request.history,
                on_safe_sentence=lambda s: queue.put_nowait({"delta": s}),
            )
            await queue.put({"final": result})
        except Exception as exc:
            logger.error(
                "[AGENT-ENDPOINT] stream failed: %s", exc, exc_info=True
            )
            # Surface a speakable failure rather than a truncated stream: the
            # caller has no way to retry mid-turn for a voice customer.
            await queue.put({
                "final": {
                    "reply": "Sorry, I encountered an error. Please try again.",
                    "tool_calls": [],
                    "streamed": "",
                }
            })
        finally:
            await queue.put(None)

    async def body():
        task = asyncio.create_task(run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item) + "\n"
        finally:
            # A disconnected client must not leave the turn running: it holds
            # an ADK session and an OVMS slot.
            if not task.done():
                task.cancel()

    return StreamingResponse(body(), media_type="application/x-ndjson")
