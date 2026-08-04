"""AgentClient — HTTP client for the rag-service agent endpoint.

Sends kiosk turns to the rag-service agent and yields the reply as text
chunks, so callers can feed TTS incrementally.

Two transports are supported, selected by ``config.AGENT_STREAM_ENABLED``:

* **Buffered** (default) — ``POST /api/v1/agent/chat``. The agent orchestrates
  its tools and post-processing guards over the whole reply, so the response
  arrives as one chunk and is yielded as one chunk.
* **Streaming** — ``POST /api/v1/agent/chat/stream``. The agent releases
  complete sentences as they are generated, but only once it can prove the
  turn is not one its whole-reply guards might rewrite. Sentences are yielded
  as they arrive and any unspoken remainder is yielded at the end, letting TTS
  overlap with generation.

Both paths run the reply through :func:`validate_reply` before the turn ends;
in the streaming path a correction that fires after audio was already spoken
causes the corrected reply to be replayed in full.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Generator

import httpx

from kiosk_core import config
from kiosk_core.ordering.order_claim_guard import validate_reply

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


class AgentClient:
    """HTTP client for the ordering agent endpoint on rag-service."""

    def __init__(self, agent_url: str, timeout_seconds: float | None = None):
        self.agent_url = agent_url
        self.timeout_seconds = timeout_seconds or config.DEFAULT_HTTP_TIMEOUT_SECONDS

    def get_reply(
        self,
        transcription: str,
        session_id: str,
        user_id: str = "anonymous",
        history: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        """Call the agent and yield the reply text.

        The entire reply is returned in a single yield so downstream TTS
        logic receives the full response (the agent needs all tool calls to
        complete before it can compose the final answer).

        Args:
            transcription: User's spoken input (transcribed).
            session_id:    Conversation session identifier.
            user_id:       Customer identifier (default "anonymous").
            history:       Prior turns [{role, content}, ...].

        Yields:
            Reply text (one or more chunks — currently one whole reply).

        Raises:
            httpx.HTTPStatusError: On non-2xx response from the agent.
        """
        payload: dict[str, object] = {
            "transcription": transcription,
            "session_id": session_id,
            "user_id": user_id,
        }
        if history:
            cleaned = [
                {"role": str(t.get("role", "")), "content": str(t.get("content", ""))}
                for t in history
                if t.get("content")
            ]
            if cleaned:
                payload["history"] = cleaned

        logger.info(
            "[AGENT-CLIENT] session=%s user=%s message=%r",
            session_id,
            user_id,
            transcription[:100],
        )

        if config.AGENT_STREAM_ENABLED:
            yield from self._get_reply_streaming(payload, session_id)
            return

        with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
            response = client.post(self.agent_url, json=payload)
            response.raise_for_status()
            data = response.json()

        reply = data.get("reply", "")
        tool_calls = data.get("tool_calls", [])
        llm_ms = data.get("llm_ms")
        llm_calls = data.get("llm_calls", 0)
        retrieval_ms = data.get("retrieval_ms")

        logger.info(
            "[AGENT-CLIENT] session=%s reply_len=%d tool_calls=%s llm_ms=%s llm_calls=%s "
            "retrieval_ms=%s",
            session_id,
            len(reply),
            tool_calls,
            llm_ms,
            llm_calls,
            retrieval_ms,
        )

        # Reconcile the reply against the tools that actually ran before any of
        # it reaches TTS. The agent has been observed announcing a placed and
        # confirmed order without invoking a single ordering tool, so this
        # claim is verified here rather than trusted.
        reply, corrected = validate_reply(reply, tool_calls)
        if corrected:
            logger.error(
                "[AGENT-CLIENT] session=%s reply failed order-claim validation "
                "and was corrected (tool_calls=%s)",
                session_id, tool_calls,
            )

        if reply:
            yield reply
        # Yield tool_calls and LLM timings as metadata so callers can record
        # pipeline traces with genuine LLM time (not whole-agent round-trip).
        yield {
            "_tool_calls": tool_calls,
            "_llm_ms": llm_ms,
            "_llm_calls": llm_calls,
            "_retrieval_ms": retrieval_ms,
        }

    def _get_reply_streaming(
        self,
        payload: dict[str, object],
        session_id: str,
    ) -> Generator[str | dict[str, object], None, None]:
        """Consume the agent's NDJSON stream, speaking safe sentences early.

        The agent releases only sentences that provably cannot be rewritten by
        one of its post-generation guards, and reports them back in
        ``final.streamed``. This method still re-validates the authoritative
        reply here, because kiosk-core applies its own order-claim guard that
        the agent knows nothing about.

        Args:
            payload:    Request body for the agent endpoint.
            session_id: Conversation session identifier, for logging.

        Yields:
            Reply text chunks, then a metadata dict of tool calls and timings.
        """
        spoken: list[str] = []
        final: dict = {}

        with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
            with client.stream(
                "POST", config.DEFAULT_AGENT_STREAM_URL, json=payload
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    delta = event.get("delta")
                    if delta:
                        logger.info(
                            "[AGENT-CLIENT] session=%s streamed sentence (%d chars)",
                            session_id, len(delta),
                        )
                        # Sentences arrive stripped, but the session joins
                        # response parts with "". Re-insert the separator the
                        # model's own spacing would have provided.
                        yield delta if not spoken else " " + delta
                        spoken.append(delta)
                        continue
                    if "final" in event:
                        final = event["final"]

        reply = final.get("reply", "")
        tool_calls = final.get("tool_calls", [])
        streamed = final.get("streamed", "")

        reply, corrected = validate_reply(reply, tool_calls)

        remainder = _remainder_after(reply, streamed) if streamed else reply
        if corrected and spoken:
            # kiosk-core rejected a claim the agent considered safe to stream.
            # The customer already heard part of the answer, so speak the whole
            # corrected reply: repeating a clause is recoverable, leaving a
            # false claim uncorrected is not.
            logger.error(
                "[AGENT-CLIENT] session=%s order-claim validation corrected a reply "
                "whose prefix was already spoken — replaying corrected reply in full "
                "(tool_calls=%s)",
                session_id, tool_calls,
            )
            remainder = reply
        elif corrected:
            logger.error(
                "[AGENT-CLIENT] session=%s reply failed order-claim validation "
                "and was corrected (tool_calls=%s)",
                session_id, tool_calls,
            )

        logger.info(
            "[AGENT-CLIENT] session=%s streaming turn done | sentences_streamed=%d "
            "reply_len=%d remainder_len=%d tool_calls=%s",
            session_id, len(spoken), len(reply), len(remainder), tool_calls,
        )

        if remainder:
            yield remainder if not spoken else " " + remainder
        yield {
            "_tool_calls": tool_calls,
            "_llm_ms": final.get("llm_ms"),
            "_llm_calls": final.get("llm_calls", 0),
            "_retrieval_ms": final.get("retrieval_ms"),
        }


def _remainder_after(reply: str, streamed: str) -> str:
    """Return the part of ``reply`` not already spoken as streamed sentences.

    Whitespace is normalised on both sides because the streamed sentences are
    re-joined with single spaces while the reply keeps the model's spacing.
    Any mismatch is treated as "nothing was spoken" by the caller upstream, so
    this only has to handle the prefix case.
    """
    norm_reply = _WS_RE.sub(" ", reply).strip()
    norm_streamed = _WS_RE.sub(" ", streamed).strip()
    if not norm_streamed or not norm_reply.startswith(norm_streamed):
        return reply
    return norm_reply[len(norm_streamed):].strip()
