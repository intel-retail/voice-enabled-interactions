"""Tests for the agentic MCP client without a live MCP server."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from types import SimpleNamespace

from agentic import mcp_client


def _run(coro: Any) -> Any:
    """Run an async test helper without depending on pytest-asyncio."""
    return asyncio.run(coro)


def _install_fake_mcp(monkeypatch, state: dict[str, Any]) -> None:
    """Install fake mcp modules used by mcp_client imports."""

    class _FakeStreamableHTTP:
        def __init__(self, url: str) -> None:
            self._url = url

        async def __aenter__(self):
            state["stream_urls"].append(self._url)
            return ("read", "write", None)

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    class _FakeSession:
        def __init__(self, read: str, write: str) -> None:
            self.read = read
            self.write = write

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def initialize(self) -> None:
            state["initialized"] += 1

        async def list_tools(self):
            return SimpleNamespace(tools=state["tools"])

        async def call_tool(self, tool_name: str, arguments: dict[str, Any]):
            state["tool_calls"].append((tool_name, arguments))
            if state.get("call_tool_error") is not None:
                raise state["call_tool_error"]
            delay = state.get("call_tool_delay", 0.0)
            if delay:
                await asyncio.sleep(delay)
            return SimpleNamespace(content=state["result_content"])

    mcp_mod = types.ModuleType("mcp")
    mcp_mod.ClientSession = _FakeSession
    client_mod = types.ModuleType("mcp.client")
    stream_mod = types.ModuleType("mcp.client.streamable_http")
    stream_mod.streamablehttp_client = lambda url: _FakeStreamableHTTP(url)

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.client", client_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", stream_mod)


def test_discover_tools_lists_mcp_tools(monkeypatch, reset_mcp_state) -> None:
    """discover_tools converts MCP list_tools result into MCPTool objects."""
    state = {
        "stream_urls": [],
        "initialized": 0,
        "tool_calls": [],
        "tools": [
            SimpleNamespace(
                name="place_order",
                description="Create a draft order",
                inputSchema={"type": "object", "properties": {"items": {"type": "array"}}},
            )
        ],
        "result_content": [],
    }
    _install_fake_mcp(monkeypatch, state)

    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local")
    tools = _run(mcp_client.discover_tools(server))

    assert state["stream_urls"] == ["http://mcp.local"]
    assert state["initialized"] == 1
    assert [tool.name for tool in tools] == ["place_order"]
    assert tools[0].server == "core"
    assert tools[0].to_function_schema()["parameters"]["properties"]["items"]["type"] == "array"


def test_call_tool_invokes_registered_tool_with_arguments(monkeypatch, reset_mcp_state) -> None:
    """call_tool uses the registered server and forwards args unchanged."""
    state = {
        "stream_urls": [],
        "initialized": 0,
        "tool_calls": [],
        "tools": [],
        "result_content": [SimpleNamespace(text='{"order_id":"ORD-1"}')],
    }
    _install_fake_mcp(monkeypatch, state)

    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local")
    mcp_client._servers["core"] = server
    mcp_client._tools["place_order"] = mcp_client.MCPTool(name="place_order", server="core")

    result = _run(mcp_client.call_tool("place_order", {"user_id": "u1", "items": []}))

    assert result == {"status": "success", "result": '{"order_id":"ORD-1"}'}
    assert state["stream_urls"] == ["http://mcp.local"]
    assert state["initialized"] == 1
    assert state["tool_calls"] == [("place_order", {"user_id": "u1", "items": []})]


def test_call_tool_times_out(monkeypatch, reset_mcp_state) -> None:
    """call_tool timeout returns an error payload instead of raising."""
    state = {
        "stream_urls": [],
        "initialized": 0,
        "tool_calls": [],
        "tools": [],
        "result_content": [SimpleNamespace(text="late")],
        "call_tool_delay": 0.05,
    }
    _install_fake_mcp(monkeypatch, state)

    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local", timeout=0.01)
    mcp_client._servers["core"] = server
    mcp_client._tools["confirm_order"] = mcp_client.MCPTool(name="confirm_order", server="core")

    assert _run(mcp_client.call_tool("confirm_order", {"order_id": "ORD-1"})) == {
        "error": "Tool confirm_order timed out"
    }


def test_call_tool_returns_error_payload_on_transport_failure(monkeypatch, reset_mcp_state) -> None:
    """MCP transport errors are returned to the agent, not raised."""
    state = {
        "stream_urls": [],
        "initialized": 0,
        "tool_calls": [],
        "tools": [],
        "result_content": [],
        "call_tool_error": RuntimeError("MCP unavailable"),
    }
    _install_fake_mcp(monkeypatch, state)

    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local")
    mcp_client._servers["core"] = server
    mcp_client._tools["confirm_order"] = mcp_client.MCPTool(name="confirm_order", server="core")

    assert _run(mcp_client.call_tool("confirm_order", {"order_id": "ORD-1"})) == {
        "error": "MCP unavailable"
    }


# ---------------------------------------------------------------------------
# Tool description compaction (prefill reduction)
# ---------------------------------------------------------------------------

_REAL_DOCSTRING = """List the menu categories available, with how many items each holds.

    Use this when the customer asks what the restaurant serves in general
    ("what do you have?", "show me the menu") without naming a category.

    Args:
        category: One of: burgers, pizza, wraps, sides, beverages, desserts.

    Returns:
        One ``{category, item_count}`` entry per category, alphabetically.
    """


def test_compact_drops_returns_section_and_keeps_args(monkeypatch) -> None:
    """The Args block steers argument filling and must survive compaction."""
    monkeypatch.setattr(mcp_client.agent_cfg, "COMPACT_TOOL_DESCRIPTIONS", True)
    out = mcp_client.compact_tool_description(_REAL_DOCSTRING)
    assert "Returns:" not in out
    assert "item_count} entry per category" not in out
    assert "Args:" in out
    assert "burgers, pizza, wraps" in out
    assert out.startswith("List the menu categories available")
    assert len(out) < len(_REAL_DOCSTRING)


def test_compact_strips_every_trailing_section_kind(monkeypatch) -> None:
    """Raises/Yields/Example/Note headings are dropped like Returns."""
    monkeypatch.setattr(mcp_client.agent_cfg, "COMPACT_TOOL_DESCRIPTIONS", True)
    for heading in ("Returns", "Return", "Raises", "Yields", "Example",
                    "Examples", "Note", "Notes"):
        doc = f"Do the thing.\n\n    {heading}:\n        Something verbose.\n"
        assert mcp_client.compact_tool_description(doc) == "Do the thing."


def test_compact_is_a_noop_without_a_section_heading(monkeypatch) -> None:
    """A docstring with no trailing section is returned unchanged."""
    monkeypatch.setattr(mcp_client.agent_cfg, "COMPACT_TOOL_DESCRIPTIONS", True)
    doc = "Confirm the customer's active order."
    assert mcp_client.compact_tool_description(doc) == doc


def test_compact_does_not_match_inline_prose(monkeypatch) -> None:
    """'Returns:' only counts as a heading on its own line."""
    monkeypatch.setattr(mcp_client.agent_cfg, "COMPACT_TOOL_DESCRIPTIONS", True)
    doc = "Cancel an order. Returns: nothing useful when already cancelled."
    assert mcp_client.compact_tool_description(doc) == doc


def test_compact_keeps_original_when_truncation_would_empty_it(monkeypatch) -> None:
    """A docstring that is only a Returns block still has to describe the tool."""
    monkeypatch.setattr(mcp_client.agent_cfg, "COMPACT_TOOL_DESCRIPTIONS", True)
    doc = "Returns:\n    The order id.\n"
    assert mcp_client.compact_tool_description(doc) == doc


def test_compact_disabled_returns_full_docstring(monkeypatch) -> None:
    """The flag falls back to the previous behaviour with no rebuild."""
    monkeypatch.setattr(mcp_client.agent_cfg, "COMPACT_TOOL_DESCRIPTIONS", False)
    assert mcp_client.compact_tool_description(_REAL_DOCSTRING) == _REAL_DOCSTRING


def test_compact_handles_empty_description(monkeypatch) -> None:
    """Tools may legitimately carry no description at all."""
    monkeypatch.setattr(mcp_client.agent_cfg, "COMPACT_TOOL_DESCRIPTIONS", True)
    assert mcp_client.compact_tool_description("") == ""


def test_function_schema_uses_the_compacted_description(monkeypatch) -> None:
    """to_function_schema is the path that reaches the LLM tool schema."""
    monkeypatch.setattr(mcp_client.agent_cfg, "COMPACT_TOOL_DESCRIPTIONS", True)
    tool = mcp_client.MCPTool(
        name="list_categories", server="core", description=_REAL_DOCSTRING
    )
    schema = tool.to_function_schema()
    assert schema["description"].startswith("[MCP:core] List the menu categories")
    assert "Returns:" not in schema["description"]
    assert tool.description == _REAL_DOCSTRING  # source of truth untouched
