"""Google ADK runtime — model and session factory.

Creates a LiteLlm model pointing at OVMS and provides a reusable
Runner + InMemorySessionService, mirroring the pattern from alert-agent-service.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from agentic import config as agent_cfg
from agentic import llm_metrics

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService


def create_adk_model() -> "LiteLlm":
    """Build and return a LiteLlm model connected to OVMS directly.

    Uses the ``openai/`` provider prefix so LiteLLM treats OVMS as an
    OpenAI-compatible endpoint (HTTP, no proxy).

    The returned model is a ``_TimedLiteLlm``, which records the duration of
    every LLM round-trip into :mod:`agentic.llm_metrics` so the pipeline trace
    can report genuine LLM time separately from tool and framework overhead.
    """
    from google.adk.models.lite_llm import LiteLlm

    class _TimedLiteLlm(LiteLlm):
        """LiteLlm that reports each round-trip's duration to llm_metrics."""

        async def generate_content_async(self, llm_request, stream: bool = False):
            started = time.perf_counter()
            recorded = False
            try:
                async for response in super().generate_content_async(llm_request, stream):
                    # Record once, when the first chunk arrives for a streamed
                    # call, otherwise on the single non-streamed response.
                    if not recorded:
                        llm_metrics.record((time.perf_counter() - started) * 1000)
                        recorded = True
                    yield response
            finally:
                if not recorded:
                    # Call failed before producing output — still count the
                    # time spent so the trace does not under-report.
                    llm_metrics.record((time.perf_counter() - started) * 1000)

    # Do NOT use litellm_proxy — that routes to a separate LiteLLM proxy server.
    # Use the openai/ provider which calls the base_url directly over HTTP.
    LiteLlm.use_litellm_proxy = False

    model_id = f"openai/{agent_cfg.LLM_MODEL}"
    logger.info(
        "[ADK] Creating LiteLlm model=%s base_url=%s enable_thinking=%s "
        "temperature=%s top_p=%s seed=%s max_tokens=%s",
        model_id, agent_cfg.LLM_URL, agent_cfg.ENABLE_THINKING,
        agent_cfg.TEMPERATURE, agent_cfg.TOP_P, agent_cfg.SEED, agent_cfg.MAX_TOKENS,
    )
    # ``extra_body`` is forwarded verbatim by LiteLLM's openai provider into the
    # OVMS /v3/chat/completions request. ``chat_template_kwargs.enable_thinking``
    # is how Qwen3 hybrid-reasoning models are switched out of thinking mode —
    # the in-prompt ``/no_think`` marker is a Qwen2.5 convention and is ignored
    # by Qwen3's chat template.
    #
    # temperature/top_p/seed are passed explicitly: LiteLLM drops unset
    # parameters, so OVMS would otherwise apply the OpenAI default of
    # temperature=1.0 and sample tool calls non-deterministically.
    return _TimedLiteLlm(
        model=model_id,
        tool_choice="auto",
        api_base=agent_cfg.LLM_URL,
        api_key="local",
        temperature=agent_cfg.TEMPERATURE,
        top_p=agent_cfg.TOP_P,
        seed=agent_cfg.SEED,
        max_tokens=agent_cfg.MAX_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": agent_cfg.ENABLE_THINKING}},
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
