"""Google ADK runtime — model and session factory.

Creates a LiteLlm model pointing at OVMS and provides a reusable
Runner + InMemorySessionService, mirroring the pattern from alert-agent-service.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentic import config as agent_cfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService


def create_adk_model() -> "LiteLlm":
    """Build and return a LiteLlm model connected to OVMS directly.

    Uses the ``openai/`` provider prefix so LiteLLM treats OVMS as an
    OpenAI-compatible endpoint (HTTP, no proxy).
    """
    from google.adk.models.lite_llm import LiteLlm

    # Do NOT use litellm_proxy — that routes to a separate LiteLLM proxy server.
    # Use the openai/ provider which calls the base_url directly over HTTP.
    LiteLlm.use_litellm_proxy = False

    model_id = f"openai/{agent_cfg.LLM_MODEL}"
    logger.info("[ADK] Creating LiteLlm model=%s base_url=%s", model_id, agent_cfg.LLM_URL)
    return LiteLlm(
        model=model_id,
        tool_choice="auto",
        api_base=agent_cfg.LLM_URL,
        api_key="local",
    )


def create_session_service() -> "InMemorySessionService":
    """Return a new in-memory session service."""
    from google.adk.sessions import InMemorySessionService

    return InMemorySessionService()


def create_runner(agent, session_service: "InMemorySessionService") -> "Runner":
    """Create an ADK Runner for the given agent + session service."""
    from google.adk.runners import Runner

    return Runner(
        app_name=agent.name,
        agent=agent,
        session_service=session_service,
    )
