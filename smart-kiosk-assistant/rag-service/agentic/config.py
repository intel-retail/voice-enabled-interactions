"""Agent configuration — all settings are env-driven.

Environment variables:
    AGENT_LLM_URL      Base URL of the OVMS OpenAI-compatible endpoint
                       (default: http://ovms-llm:8000/v3)
    AGENT_LLM_MODEL    Model name served by OVMS (default: Qwen3-4B)
    AGENT_MCP_CONFIG   Path to mcp_servers.json
                       (default: ./agentic/resources/mcp_servers.json)
    AGENT_LOG_LEVEL    Logging level for the agentic package (default: INFO)
    AGENT_SESSION_TTL  In-memory session TTL seconds (default: 3600)
    AGENT_ENABLE_THINKING
                       Render the Qwen3 chat template in thinking mode
                       (default: false — thinking tokens are pure decode cost)
    AGENT_TEMPERATURE  Sampling temperature for the agent LLM (default: 0.0)
    AGENT_TOP_P        Nucleus sampling cutoff (default: 1.0)
    AGENT_SEED         Sampling seed forwarded to OVMS (default: 42)
    AGENT_MAX_TOKENS   Max generated tokens per LLM round-trip (default: 320)
    AGENT_COMPACT_TOOL_DESCRIPTIONS
                       Strip "Returns:"/"Raises:"/"Example:" sections from MCP
                       tool docstrings before they are sent as tool schemas
                       (default: true — saves ~428 prefill tokens per call)
    AGENT_RETRY_ON_MISSING_TOOL_CALL
                       Retry a turn once when the model promises a lookup but
                       calls no tool (default: true)
"""

from __future__ import annotations

import os

# OVMS endpoint shared by the RAG LLM refactor and the agent
LLM_URL: str = os.getenv("AGENT_LLM_URL", "http://ovms-llm:8000/v3")
LLM_MODEL: str = os.getenv("AGENT_LLM_MODEL", "Qwen3-4B")

# Path to mcp_servers.json (resolved relative to rag-service root)
_default_mcp_config = os.path.join(os.path.dirname(__file__), "resources", "mcp_servers.json")
MCP_CONFIG_PATH: str = os.getenv("AGENT_MCP_CONFIG", _default_mcp_config)

# Agent behaviour
AGENT_SESSION_TTL: int = int(os.getenv("AGENT_SESSION_TTL", "3600"))
LOG_LEVEL: str = os.getenv("AGENT_LOG_LEVEL", "INFO").upper()

# Qwen3 hybrid-reasoning switch. When false (the default) the chat template is
# rendered in non-thinking mode, so the model emits no <think> block. Thinking
# tokens are pure decode cost: on Panther Lake iGPU decode runs at ~16 tok/s,
# and a thinking turn spends ~160 extra tokens (~10 s across the two LLM calls
# Google ADK makes per tool-calling turn). Set AGENT_ENABLE_THINKING=true to
# restore chain-of-thought if tool-selection accuracy regresses.
ENABLE_THINKING: bool = os.getenv("AGENT_ENABLE_THINKING", "false").lower() in ("true", "1", "yes")

# Sampling parameters for the agent LLM. These are set explicitly because
# LiteLLM omits any parameter left unset, which makes OVMS fall back to the
# OpenAI defaults (temperature=1.0). At temperature 1.0 a 4B int4 model is
# non-deterministic about *whether* it emits a tool call and about how it
# phrases the tool argument, so the same question yields a different answer —
# or a "let me look that up" dead end — on every turn. Greedy decoding costs
# no extra latency; it removes that variance.
TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.0"))
TOP_P: float = float(os.getenv("AGENT_TOP_P", "1.0"))
SEED: int = int(os.getenv("AGENT_SEED", "42"))

# Upper bound on generated tokens per LLM round-trip. Without a cap a turn that
# fails to call a tool free-runs and produces an 800-char invented answer: two
# such turns in a single session took 12.3s and 13.2s of LLM time versus ~5s for
# a grounded tool-calling turn. Kiosk replies are 2-3 spoken sentences, so this
# bounds worst-case latency without truncating legitimate answers (the longest
# valid reply, a full product listing, is ~300 chars).
MAX_TOKENS: int = int(os.getenv("AGENT_MAX_TOKENS", "320"))

# Strip the trailing reference sections ("Returns:", "Raises:", "Example:", …)
# from every MCP tool docstring before it becomes the tool's LLM-facing
# description.
#
# The MCP tool docstrings are Google-style and written for humans reading
# kiosk_core/ordering/mcp_server.py: they document the return payload shape in
# detail. That payload never influences *which* tool the model picks or *how* it
# fills the arguments — the model sees the real result after the call — but the
# text is re-sent as part of the tool schema on every LLM round-trip.
#
# Measured with the Qwen3-4B tokenizer across all 12 kiosk tools: 1,627 →
# 1,199 docstring tokens, a 428-token (26%) cut of prefill on each of the two
# LLM calls an ADK tool-calling turn makes. The summary line and the whole
# "Args:" block — which do steer tool choice and argument filling — are kept
# verbatim.
#
# Set AGENT_COMPACT_TOOL_DESCRIPTIONS=false to send the full docstrings again
# with a restart, no rebuild.
COMPACT_TOOL_DESCRIPTIONS: bool = os.getenv(
    "AGENT_COMPACT_TOOL_DESCRIPTIONS", "true"
).lower() in ("true", "1", "yes")

# Retry a turn once when the model announces a lookup ("let me check that")
# but emits no tool call, leaving the customer with no answer.
RETRY_ON_MISSING_TOOL_CALL: bool = os.getenv(
    "AGENT_RETRY_ON_MISSING_TOOL_CALL", "true"
).lower() in ("true", "1", "yes")

# Emit complete sentences to the caller as the model produces them, instead of
# withholding the reply until the whole turn is post-processed.
#
# Measured on OVMS/Qwen3-4B-int4: the first sentence of a reply is ready at
# ~700 ms while the full reply takes 1.3-4.0 s, so the customer currently
# waits for text that already exists. Streaming it moves time-to-first-audio
# from ~5.0 s to ~2.8 s.
#
# This is OFF by default and deliberately so. chat() applies several
# whole-reply guards that can *replace* or *regenerate* an answer after the
# fact (unbacked order claims, forced knowledge/catalogue recovery, leaked
# tool syntax). A sentence that has already been spoken cannot be recalled,
# so early emission is only safe for sentences that provably cannot trigger
# any of those paths — see _SentenceGate. Set to false to fall back to the
# buffered path with no rebuild.
STREAM_SENTENCES: bool = os.getenv(
    "AGENT_STREAM_SENTENCES", "false"
).lower() in ("true", "1", "yes")

# Retrieve knowledge-base context BEFORE the first LLM call for questions that
# are unambiguously about the outlet (hours, name, parking, allergens...).
#
# Google ADK's tool loop costs two inferences per tool-calling turn, and
# Qwen3-4B-int4 frequently answers an outlet question from "memory" on the
# first pass — which is invented, trips the ungrounded-reply guard, and forces
# a retry, for THREE inferences in total. Measured live: "restaurant name and
# timings" took 8.5 s of LLM time across 3 calls, of which retrieval was 110 ms.
#
# Pre-grounding inverts that. Retrieval is deterministic and cheap, so the
# facts are put in front of the model before it speaks, and the turn needs one
# inference with no tool call and nothing to retry. The model cannot invent
# hours it was already handed.
#
# Set to false to restore the pure tool-calling path.
PREGROUND_KNOWLEDGE: bool = os.getenv(
    "AGENT_PREGROUND_KNOWLEDGE", "true"
).lower() in ("true", "1", "yes")
