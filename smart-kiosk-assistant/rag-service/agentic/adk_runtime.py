"""Google ADK runtime — model and session factory.

Creates a LiteLlm model pointing at OVMS and provides a reusable
Runner + InMemorySessionService, mirroring the pattern from alert-agent-service.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from agentic import config as agent_cfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService


def create_adk_model() -> "LiteLlm":
    """Build and return a LiteLlm model connected to OVMS.

    Configures the LiteLLM proxy to point at the OVMS endpoint so ADK
    routes tool-calling requests through the OVMS-served model.
    """
    from google.adk.models.lite_llm import LiteLlm

    os.environ.setdefault("LITELLM_PROXY_API_KEY", "local")
    os.environ.setdefault("LITELLM_PROXY_API_BASE", agent_cfg.LLM_URL)
    LiteLlm.use_litellm_proxy = True

    model_id = f"litellm_proxy/{agent_cfg.LLM_MODEL}"
    logger.info("[ADK] Creating LiteLlm model=%s base_url=%s", model_id, agent_cfg.LLM_URL)
    return LiteLlm(model=model_id, tool_choice="auto")


def create_session_service() -> "InMemorySessionService":
    """Return a new in-memory session service."""
    from google.adk.sessions import InMemorySessionService

    return InMemorySessionService()


def create_runner(agent, session_service: "InMemorySessionService") -> "Runner":
    """Create an ADK Runner for the given agent + session service."""
    from google.adk.runners import Runner

    return Runner(agent=agent, session_service=session_service)
