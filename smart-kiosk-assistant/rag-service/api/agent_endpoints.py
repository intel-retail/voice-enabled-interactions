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
    llm_ttft_ms    float|None — cumulative prefill/time-to-first-token
    llm_calls      int        — number of LLM round-trips for the turn
    retrieval_ms   float|None — knowledge-base retrieval time for the turn
    mcp_ms         float|None — cumulative MCP tool round-trip time for the turn
    mcp_calls      int        — number of MCP tool round-trips for the turn
    guard_ms       float|None — cumulative truthfulness-guard processing time
    template_ms    float|None — deterministic reply-template render time
    templated      bool       — True when the narration LLM call was skipped
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

# /chat-no-adk is a lab-only A/B benchmark twin of /chat (see its docstring
# below) that can place/confirm real orders through the same MCP tools.
# main.py mounts this whole router in every deployment where
# ORDERING_AGENT_ENABLED is set (the default), so without its own flag this
# route would be reachable in production despite being intended for
# tests/benchmarks/*_no_adk.py only. Off by default; benchmarks must opt in.
BENCHMARK_ENDPOINTS_ENABLED = (
    os.getenv("AGENT_BENCHMARK_ENDPOINTS_ENABLED", "false").lower() == "true"
)


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
    speculative: bool = Field(
        default=False,
        description=(
            "If true, this is a speculative draft turn run ahead of the "
            "customer's final (endpointed) utterance. Every mutating "
            "ordering tool this turn is forced into dry_run mode server-side "
            "— nothing is persisted to the orders database regardless of "
            "what the agent decides to call. Used for cache-warming and "
            "preview drafting; never set this for a real, confirmed turn."
        ),
    )


class AgentChatResponse(BaseModel):
    reply: str
    tool_calls: list[str] = Field(default_factory=list)
    tool_call_detail: list[dict] = Field(
        default_factory=list,
        description=(
            "Exact {tool_name, kwargs, result} dispatched this turn, in call "
            "order. Populated ONLY when the request itself had "
            "speculative=true (its only real consumer: a caller replaying "
            "these calls for real, dry_run=False, instead of re-running the "
            "LLM, if the customer's final utterance still matches the "
            "draft's input). Always [] for normal (non-speculative) turns —"
            " this endpoint has no request authentication and rag-service's "
            "port is published in docker-compose.yml, so kwargs/results "
            "(user IDs, order IDs, cart contents, internal error payloads) "
            "must not be returned to every caller by default."
        ),
    )
    llm_ms: float | None = Field(
        default=None,
        description=(
            "Cumulative LLM round-trip time for this turn, in milliseconds. "
            "Covers prefill AND decode — the full stream, not just first token."
        ),
    )
    llm_ttft_ms: float | None = Field(
        default=None,
        description=(
            "Cumulative time-to-first-token for this turn, in milliseconds. "
            "llm_ms - llm_ttft_ms is the decode (token generation) cost."
        ),
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
    mcp_ms: float | None = Field(
        default=None,
        description=(
            "Cumulative MCP tool round-trip time for this turn, in "
            "milliseconds (network + kiosk-core request handling, including "
            "its SQLite time). None when no MCP tool was called."
        ),
    )
    mcp_calls: int = Field(
        default=0,
        description="Number of MCP tool round-trips made during this turn",
    )
    guard_ms: float | None = Field(
        default=None,
        description=(
            "Cumulative time spent in the truthfulness guards (menu/removal/"
            "confirm result recording and whole-reply validation), in "
            "milliseconds. None when no guard-relevant tool ran this turn."
        ),
    )
    template_ms: float | None = Field(
        default=None,
        description=(
            "Time spent rendering a deterministic reply template, in "
            "milliseconds. None when no template was attempted this turn."
        ),
    )
    templated: bool = Field(
        default=False,
        description=(
            "True when a deterministic template produced the reply, meaning "
            "the second (narration) LLM call was skipped for this turn."
        ),
    )


def _scoped_tool_call_detail(request: AgentChatRequest, result: dict) -> list[dict]:
    """Return ``tool_call_detail`` only for the speculative-replay use case.

    This endpoint has no request authentication and rag-service's port is
    published in docker-compose.yml, so raw tool kwargs/results (user IDs,
    order IDs, cart contents, internal error payloads) must not go out to
    every caller by default -- only to the one flow that actually needs them
    (a speculative turn's caller replaying its dry-run calls for real).
    """
    if not request.speculative:
        return []
    return result.get("tool_call_detail", [])


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
        from agentic.plugin_loader import load_agent_factory

        _agent_factory = load_agent_factory()
        agent = _agent_factory()
        result = await agent.chat(
            message=request.transcription,
            session_id=request.session_id,
            user_id=request.user_id,
            history=request.history,
            speculative=request.speculative,
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
        tool_call_detail=_scoped_tool_call_detail(request, result),
        llm_ms=result.get("llm_ms"),
        llm_ttft_ms=result.get("llm_ttft_ms"),
        llm_calls=result.get("llm_calls", 0),
        retrieval_ms=result.get("retrieval_ms"),
        mcp_ms=result.get("mcp_ms"),
        mcp_calls=result.get("mcp_calls", 0),
        guard_ms=result.get("guard_ms"),
        template_ms=result.get("template_ms"),
        templated=result.get("templated", False),
    )


if BENCHMARK_ENDPOINTS_ENABLED:
    @router.post(
        "/chat-no-adk",
        response_model=AgentChatResponse,
        summary="Agent ordering chat — non-ADK benchmark variant",
    )
    async def agent_chat_no_adk(request: AgentChatRequest) -> AgentChatResponse:
        """Benchmark-only twin of ``POST /chat`` for the ADK vs non-ADK A/B.

        Routes to ``plugins.kiosk.ordering_agent_without_adk.OrderingAgentWithoutADK``
        instead of the production plugin loader — same request/response shape,
        same MCP tools, same guards/templates where reused, but every turn is at
        most one LLM call with no ``tools=`` field (see that module's docstring).
        Exists only so ``tests/benchmarks/agent_latency_benchmark.py`` can hit
        both agents by URL alone. Only registered when
        ``AGENT_BENCHMARK_ENDPOINTS_ENABLED=true`` (default: off); a normal
        deployment never mounts this route.
        """
        logger.info(
            "[AGENT-ENDPOINT-NOADK] session=%s user=%s message=%r",
            request.session_id,
            request.user_id,
            request.transcription[:100],
        )

        # OrderingAgentWithoutADK never implements the dry_run contract (see
        # its chat() docstring): its order-mutation path calls
        # directive_mode.run_turn(), which invokes place_order/
        # remove_from_order/confirm_active_order with no dry_run flag at
        # all. Letting speculative=true reach this agent would silently
        # persist a real, unconfirmed order mutation instead of the no-op
        # preview the AgentChatRequest contract promises. Reject rather than
        # execute it for real.
        if request.speculative:
            raise HTTPException(
                status_code=400,
                detail=(
                    "speculative=true is not supported on /chat-no-adk — "
                    "OrderingAgentWithoutADK does not implement dry-run "
                    "ordering tool calls. Use /chat for speculative turns."
                ),
            )

        try:
            from plugins.kiosk.ordering_agent_without_adk import get_ordering_agent_without_adk

            agent = get_ordering_agent_without_adk()
            result = await agent.chat(
                message=request.transcription,
                session_id=request.session_id,
                user_id=request.user_id,
                history=request.history,
                speculative=request.speculative,
            )
        except Exception as exc:
            logger.error("[AGENT-ENDPOINT-NOADK] Unhandled error: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        logger.info(
            "[AGENT-ENDPOINT-NOADK] session=%s reply_len=%d tool_calls=%s",
            request.session_id,
            len(result.get("reply", "")),
            result.get("tool_calls", []),
        )
        return AgentChatResponse(
            reply=result["reply"],
            tool_calls=result.get("tool_calls", []),
            tool_call_detail=_scoped_tool_call_detail(request, result),
            llm_ms=result.get("llm_ms"),
            llm_ttft_ms=result.get("llm_ttft_ms"),
            llm_calls=result.get("llm_calls", 0),
            retrieval_ms=result.get("retrieval_ms"),
            mcp_ms=result.get("mcp_ms"),
            mcp_calls=result.get("mcp_calls", 0),
            guard_ms=result.get("guard_ms"),
            template_ms=result.get("template_ms"),
            templated=result.get("templated", False),
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
            from agentic.plugin_loader import load_agent_factory

            _agent_factory = load_agent_factory()
            agent = _agent_factory()
            result = await agent.chat(
                message=request.transcription,
                session_id=request.session_id,
                user_id=request.user_id,
                history=request.history,
                on_safe_sentence=lambda s: queue.put_nowait({"delta": s}),
                speculative=request.speculative,
            )
            if not request.speculative:
                # Same gating as _scoped_tool_call_detail() above -- this
                # streaming path bypasses AgentChatResponse entirely and
                # dumps `result` straight into the wire, so it must scrub
                # this key itself instead of relying on the pydantic model.
                result = {k: v for k, v in result.items() if k != "tool_call_detail"}
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
