"""Agent configuration — all settings are env-driven.

Environment variables:
    AGENT_LLM_URL      Base URL of the OVMS OpenAI-compatible endpoint
                       (default: http://ovms-llm:8000/v3)
    AGENT_LLM_MODEL    Model name served by OVMS (default: Qwen3-4B)
    AGENT_MCP_CONFIG   Path to mcp_servers.json
                       (default: ./agentic/resources/mcp_servers.json)
    AGENT_LOG_LEVEL    Logging level for the agentic package (default: INFO)
    AGENT_SESSION_TTL  In-memory session TTL seconds (default: 3600)
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
