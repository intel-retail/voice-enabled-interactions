"""MCP client — discovers tools on the kiosk-core MCP server and invokes them.

Uses the official ``mcp`` Python library with streamable-HTTP transport.
Maintains a persistent per-server session so that tool results delivered
asynchronously over the SSE GET stream are received before the connection
is torn down.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MCPTool:
    name: str
    server: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)

    def to_function_schema(self) -> dict:
        return {
            "name": self.name,
            "description": f"[MCP:{self.server}] {self.description}",
            "parameters": self.input_schema or {"type": "object", "properties": {}},
        }


@dataclass
class MCPServerConfig:
    name: str
    url: str
    enabled: bool = True
    timeout: float = 30.0
    description: str = ""


# ---------------------------------------------------------------------------
# Module-level registries
# ---------------------------------------------------------------------------

_servers: dict[str, MCPServerConfig] = {}
_tools: dict[str, MCPTool] = {}

# Persistent session pool: server_name → (ClientSession, AsyncExitStack)
_sessions: dict[str, tuple[Any, Any]] = {}
_session_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _session_lock
    if _session_lock is None:
        _session_lock = asyncio.Lock()
    return _session_lock


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_mcp_config(config_path: str) -> list[MCPServerConfig]:
    """Parse mcp_servers.json and return enabled server configs."""
    import json
    from pathlib import Path

    path = Path(config_path)
    if not path.exists():
        logger.warning("[MCP] Config not found at %s — no MCP tools loaded", config_path)
        return []

    with path.open() as fh:
        data = json.load(fh)

    configs = []
    for srv in data.get("servers", []):
        if not srv.get("enabled", True):
            logger.debug("[MCP] Server %s is disabled — skipping", srv.get("name"))
            continue
        configs.append(
            MCPServerConfig(
                name=srv["name"],
                url=srv["url"],
                enabled=True,
                timeout=float(srv.get("timeout", 30.0)),
                description=srv.get("description", ""),
            )
        )
    logger.info("[MCP] Loaded %d enabled server(s) from %s", len(configs), config_path)
    return configs


# ---------------------------------------------------------------------------
# Persistent session management
# ---------------------------------------------------------------------------


async def _get_session(server: MCPServerConfig):
    """Return a live ClientSession for the server, creating one if needed."""
    from contextlib import AsyncExitStack
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    async with _get_lock():
        existing = _sessions.get(server.name)
        if existing is not None:
            session, _ = existing
            return session

        stack = AsyncExitStack()
        try:
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(server.url)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            _sessions[server.name] = (session, stack)  # keep stack alive
            logger.info("[MCP] Persistent session opened for server=%s", server.name)
            return session
        except BaseException as exc:
            # Suppress cleanup errors — the MCP library's anyio TaskGroup may
            # raise CancelledError or RuntimeError during teardown after a
            # connection failure; swallowing them here prevents double-exception
            # noise and lets the original error propagate cleanly.
            try:
                await stack.aclose()
            except Exception:
                pass
            logger.error("[MCP] Failed to open session for %s: %s", server.name, exc)
            raise


async def _close_sessions() -> None:
    """Close all persistent sessions (call at shutdown)."""
    async with _get_lock():
        for session, stack in _sessions.values():
            await stack.aclose()
        _sessions.clear()


# ---------------------------------------------------------------------------
# Tool discovery & invocation
# ---------------------------------------------------------------------------


async def discover_tools(server: MCPServerConfig) -> list[MCPTool]:
    """Discover available tools using the persistent session."""
    tools: list[MCPTool] = []
    try:
        session = await _get_session(server)
        result = await session.list_tools()
        for tool in result.tools:
            schema = {}
            if hasattr(tool, "inputSchema"):
                schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
            elif hasattr(tool, "input_schema"):
                schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
            tools.append(MCPTool(
                name=tool.name,
                server=server.name,
                description=tool.description or "",
                input_schema=schema,
            ))
        logger.info("[MCP] Discovered %d tool(s) on server %s: %s",
                    len(tools), server.name, [t.name for t in tools])
    except Exception as exc:
        logger.error("[MCP] Failed to discover tools from %s (%s): %s", server.name, server.url, exc)
    return tools


async def call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Invoke a tool on its registered MCP server using the persistent session."""
    tool = _tools.get(tool_name)
    if tool is None:
        raise ValueError(f"MCP tool not found: {tool_name}")

    server = _servers.get(tool.server)
    if server is None:
        raise ValueError(f"MCP server not found: {tool.server}")

    logger.info("[MCP] Calling tool=%s on server=%s args=%s", tool_name, tool.server, arguments)
    try:
        session = await _get_session(server)
        result = await asyncio.wait_for(
            session.call_tool(tool_name, arguments),
            timeout=server.timeout,
        )
        content = result.content or []
        text = "\n".join(
            c.text for c in content
            if hasattr(c, "text") and c.text
        )
        logger.info("[MCP] Tool=%s result=%s", tool_name, text[:200])
        return {"status": "success", "result": text or str(content)}
    except asyncio.TimeoutError:
        logger.error("[MCP] Tool=%s timed out after %.0fs", tool_name, server.timeout)
        entry = _sessions.pop(server.name, None)
        if entry:
            await entry[1].aclose()
        return {"error": f"Tool {tool_name} timed out"}
    except Exception as exc:
        logger.error("[MCP] Tool=%s call failed: %s", tool_name, exc)
        entry = _sessions.pop(server.name, None)
        if entry:
            await entry[1].aclose()
        return {"error": str(exc)}


async def bootstrap_mcp_tools(config_path: str) -> dict[str, MCPTool]:
    """Load config, discover all tools, populate module-level registries."""
    global _servers, _tools
    server_configs = load_mcp_config(config_path)
    for srv in server_configs:
        _servers[srv.name] = srv
        for tool in await discover_tools(srv):
            _tools[tool.name] = tool

    logger.info("[MCP] Bootstrap complete — %d tool(s) available: %s", len(_tools), list(_tools))
    return _tools


def get_all_tools() -> dict[str, MCPTool]:
    return dict(_tools)

