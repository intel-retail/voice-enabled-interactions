"""MCP client — discovers tools on the kiosk-core MCP server and invokes them.

Adapted from alert-agent-service `src/agentic/mcp_client.py`.
Supports HTTP/SSE transport (the transport kiosk-core exposes via fastmcp).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import aiohttp

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
_session_ids: dict[str, str] = {}  # server_name → mcp-session-id


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
# HTTP / SSE request helper (matches alert-agent transport)
# ---------------------------------------------------------------------------


async def _http_request(server: MCPServerConfig, method: str, params: dict, _retry: bool = True) -> dict:
    """Send a JSON-RPC 2.0 request and handle both plain JSON and SSE responses.

    On failure, clears any stale session ID and retries once so that a
    kiosk-core restart (which issues a new session ID) recovers automatically.
    """
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if server.name in _session_ids:
        headers["mcp-session-id"] = _session_ids[server.name]

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=server.timeout)
        ) as session:
            async with session.post(server.url, json=payload, headers=headers) as resp:
                if resp.status in (401, 404) and _retry and server.name in _session_ids:
                    # Stale session ID — discard and retry once without it
                    logger.warning(
                        "[MCP] %s returned %d for session=%s — discarding stale session ID and retrying",
                        server.name, resp.status, _session_ids[server.name],
                    )
                    del _session_ids[server.name]
                    return await _http_request(server, method, params, _retry=False)

                resp.raise_for_status()

                # Persist mcp-session-id for subsequent calls
                new_sid = resp.headers.get("mcp-session-id")
                if new_sid:
                    _session_ids[server.name] = new_sid

                content_type = resp.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    async for raw_line in resp.content:
                        line = raw_line.decode().strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        data = json.loads(data_str)
                        if "error" in data:
                            raise RuntimeError(str(data["error"]))
                        return data.get("result", {})
                    raise RuntimeError("Empty SSE response from MCP server")
                else:
                    data = await resp.json()
                    if "error" in data:
                        raise RuntimeError(str(data["error"]))
                    return data.get("result", {})
    except (aiohttp.ClientError, RuntimeError):
        # On any connection/protocol error, also clear the session ID so the
        # next call starts fresh (handles kiosk-core restart mid-stream).
        if _retry and server.name in _session_ids:
            logger.warning("[MCP] Clearing stale session ID for %s after error", server.name)
            del _session_ids[server.name]
        raise


# ---------------------------------------------------------------------------
# Tool discovery & invocation
# ---------------------------------------------------------------------------


async def discover_tools(server: MCPServerConfig) -> list[MCPTool]:
    """Call the MCP server's tool-discovery endpoint and return available tools."""
    tools: list[MCPTool] = []
    try:
        result = await _http_request(server, "tools/list", {})
        for tool_def in result.get("tools", []):
            tools.append(
                MCPTool(
                    name=tool_def["name"],
                    server=server.name,
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("inputSchema", {}),
                )
            )
        logger.info("[MCP] Discovered %d tool(s) on server %s", len(tools), server.name)
    except Exception as exc:
        logger.error("[MCP] Failed to discover tools from %s (%s): %s", server.name, server.url, exc)
    return tools


async def call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Invoke a tool on its registered MCP server and return the result."""
    tool = _tools.get(tool_name)
    if tool is None:
        raise ValueError(f"MCP tool not found: {tool_name}")

    server = _servers.get(tool.server)
    if server is None:
        raise ValueError(f"MCP server not found: {tool.server}")

    logger.info("[MCP] Calling tool=%s on server=%s args=%s", tool_name, tool.server, arguments)
    try:
        result = await _http_request(server, "tools/call", {"name": tool_name, "arguments": arguments})
        # Unwrap text content list if present (MCP spec)
        content = result.get("content", [])
        if isinstance(content, list):
            text = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
            return {"status": "success", "result": text or str(content)}
        logger.debug("[MCP] Tool=%s result=%s", tool_name, result)
        return result
    except Exception as exc:
        logger.error("[MCP] Tool=%s call failed: %s", tool_name, exc)
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
