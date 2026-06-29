"""Tests for the agentic MCP client without a live MCP server."""

from __future__ import annotations

import asyncio
from typing import Any

from agentic import mcp_client


def _run(coro: Any) -> Any:
    """Run an async test helper without depending on pytest-asyncio."""
    return asyncio.run(coro)


def test_discover_tools_lists_mcp_tools(monkeypatch, reset_mcp_state) -> None:
    """discover_tools converts MCP tools/list JSON into MCPTool objects."""
    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local")

    async def fake_http_request(
        requested_server: mcp_client.MCPServerConfig,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        assert requested_server is server
        assert method == "tools/list"
        assert params == {}
        return {
            "tools": [
                {
                    "name": "place_order",
                    "description": "Create a draft order",
                    "inputSchema": {"type": "object", "properties": {"items": {"type": "array"}}},
                }
            ]
        }

    monkeypatch.setattr(mcp_client, "_http_request", fake_http_request)

    tools = _run(mcp_client.discover_tools(server))

    assert [tool.name for tool in tools] == ["place_order"]
    assert tools[0].server == "core"
    assert tools[0].to_function_schema()["parameters"]["properties"]["items"]["type"] == "array"


def test_call_tool_invokes_registered_tool_with_arguments(monkeypatch, reset_mcp_state) -> None:
    """call_tool sends tools/call with the registered server and arguments."""
    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local")
    mcp_client._servers["core"] = server
    mcp_client._tools["place_order"] = mcp_client.MCPTool(
        name="place_order",
        server="core",
        description="Create a draft order",
    )
    calls: list[dict[str, Any]] = []

    async def fake_http_request(
        requested_server: mcp_client.MCPServerConfig,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append({"server": requested_server, "method": method, "params": params})
        return {"content": [{"type": "text", "text": '{"order_id":"ORD-1"}'}]}

    monkeypatch.setattr(mcp_client, "_http_request", fake_http_request)

    result = _run(mcp_client.call_tool("place_order", {"user_id": "u1", "items": []}))

    assert result == {"status": "success", "result": '{"order_id":"ORD-1"}'}
    assert calls == [
        {
            "server": server,
            "method": "tools/call",
            "params": {"name": "place_order", "arguments": {"user_id": "u1", "items": []}},
        }
    ]


def test_http_request_retries_once_after_stale_session(monkeypatch, reset_mcp_state) -> None:
    """A stale 401/404 session ID is cleared and the request retried once."""
    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local")
    mcp_client._session_ids["core"] = "stale-session"
    posts: list[dict[str, Any]] = []
    responses = [
        _FakeResponse(status=401, headers={}, payload={}),
        _FakeResponse(status=200, headers={"mcp-session-id": "fresh-session"}, payload={"result": {"ok": True}}),
    ]

    class FakeClientSession:
        """Fake aiohttp.ClientSession that returns scripted responses."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeClientSession":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> "_FakeResponse":
            posts.append({"url": url, "json": json, "headers": dict(headers)})
            return responses.pop(0)

    monkeypatch.setattr(mcp_client.aiohttp, "ClientSession", FakeClientSession)

    result = _run(mcp_client._http_request(server, "tools/list", {}))

    assert result == {"ok": True}
    assert len(posts) == 2
    assert posts[0]["headers"]["mcp-session-id"] == "stale-session"
    assert "mcp-session-id" not in posts[1]["headers"]
    assert mcp_client._session_ids["core"] == "fresh-session"


def test_call_tool_returns_error_payload_on_transport_failure(monkeypatch, reset_mcp_state) -> None:
    """MCP tool transport errors are returned to the agent, not raised."""
    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local")
    mcp_client._servers["core"] = server
    mcp_client._tools["confirm_order"] = mcp_client.MCPTool(name="confirm_order", server="core")

    async def fake_http_request(
        requested_server: mcp_client.MCPServerConfig,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        raise RuntimeError("MCP unavailable")

    monkeypatch.setattr(mcp_client, "_http_request", fake_http_request)

    assert _run(mcp_client.call_tool("confirm_order", {"order_id": "ORD-1"})) == {
        "error": "MCP unavailable"
    }


class _FakeResponse:
    """Async context manager emulating the aiohttp response subset we use."""

    def __init__(self, status: int, headers: dict[str, str], payload: dict[str, Any]) -> None:
        self.status = status
        self.headers = headers
        self._payload = payload
        self.content: list[bytes] = []

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise mcp_client.aiohttp.ClientError(f"HTTP {self.status}")

    async def json(self) -> dict[str, Any]:
        return self._payload
