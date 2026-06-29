"""Shared pytest setup for rag-service tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


RAG_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_SERVICE_ROOT))


try:
    import aiohttp  # noqa: F401
except ImportError:
    aiohttp_stub = types.ModuleType("aiohttp")

    class ClientError(Exception):
        """Minimal aiohttp.ClientError replacement for tests."""

    class ClientTimeout:
        """Minimal aiohttp.ClientTimeout replacement for tests."""

        def __init__(self, total: float | None = None) -> None:
            self.total = total

    class ClientSession:
        """Placeholder that tests monkeypatch before use."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("aiohttp is not installed; tests must monkeypatch ClientSession")

    aiohttp_stub.ClientError = ClientError
    aiohttp_stub.ClientTimeout = ClientTimeout
    aiohttp_stub.ClientSession = ClientSession
    sys.modules["aiohttp"] = aiohttp_stub


@pytest.fixture
def reset_mcp_state() -> None:
    """Reset module-level MCP registries around each MCP test."""
    from agentic import mcp_client

    mcp_client._servers.clear()
    mcp_client._tools.clear()
    mcp_client._session_ids.clear()
    yield
    mcp_client._servers.clear()
    mcp_client._tools.clear()
    mcp_client._session_ids.clear()
