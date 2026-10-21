"""MCP client — discovers tools on the kiosk-core MCP server and invokes them.

Uses the official ``mcp`` Python library with streamable-HTTP transport.
Both ``discover_tools`` and ``call_tool`` open a fresh per-call session so
that the entire ``anyio.TaskGroup`` lifecycle (enter → call → exit) is
contained in a single asyncio task, avoiding
``RuntimeError: Attempted to exit cancel scope in a different task``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from agentic import config as agent_cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool description compaction
# ---------------------------------------------------------------------------

# Google-style docstring sections that describe the *result* of a call rather
# than how to choose or invoke the tool. Everything from the first such heading
# to the end of the docstring is dropped; the summary and the "Args:" block
# (which do steer tool selection and argument filling) are always kept.
_TRAILING_DOC_SECTION_RE = re.compile(
    r"^[ \t]*(?:Returns|Return|Raises|Yields|Example|Examples|Note|Notes)"
    r"[ \t]*:[ \t]*$",
    re.MULTILINE,
)


def compact_tool_description(description: str) -> str:
    """Drop result-describing sections from an MCP tool docstring.

    The LLM only needs to know *when* to call a tool and *how* to fill its
    arguments; it receives the actual payload after the call, so the docstring's
    ``Returns:``/``Raises:`` prose is dead prefill weight re-sent on every LLM
    round-trip.

    Args:
        description: The raw tool description (a Google-style docstring).

    Returns:
        The description truncated at the first result-describing section
        heading, with trailing whitespace removed. Returned unchanged when
        ``AGENT_COMPACT_TOOL_DESCRIPTIONS`` is disabled, when no such heading is
        present, or when truncation would leave nothing behind.
    """
    if not agent_cfg.COMPACT_TOOL_DESCRIPTIONS or not description:
        return description
    match = _TRAILING_DOC_SECTION_RE.search(description)
    if match is None:
        return description
    trimmed = description[: match.start()].rstrip()
    # A docstring that is *only* a "Returns:" block still has to describe the
    # tool somehow; keep the original rather than hand the model an empty string.
    return trimmed or description


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MCPTool:
    name: str
    server: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)

    @property
    def prompt_description(self) -> str:
        """The description as the LLM should see it, minus result prose."""
        return compact_tool_description(self.description)

    def to_function_schema(self) -> dict:
        return {
            "name": self.name,
            "description": f"[MCP:{self.server}] {self.prompt_description}",
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
# Tool discovery & invocation
# ---------------------------------------------------------------------------


async def discover_tools(server: MCPServerConfig) -> list[MCPTool]:
    """Discover available tools using a fresh per-call session.

    Same rationale as ``call_tool``: keep the streamable-HTTP session
    lifecycle contained in a single asyncio task to avoid anyio
    cancel-scope errors.
    """
    from contextlib import AsyncExitStack
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    tools: list[MCPTool] = []
    try:
        async with AsyncExitStack() as stack:
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(server.url)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await asyncio.wait_for(session.list_tools(), timeout=server.timeout)
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
    except asyncio.TimeoutError:
        logger.error("[MCP] Tool discovery from %s timed out after %.0fs", server.name, server.timeout)
    except Exception as exc:
        logger.error("[MCP] Failed to discover tools from %s (%s): %s", server.name, server.url, exc)
    return tools


async def call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Invoke a tool on its registered MCP server.

    Opens a fresh streamable-HTTP session per call so that the entire
    ``anyio.TaskGroup`` lifecycle (enter → call → exit) happens inside a
    single asyncio task. Reusing a persistent session across tasks (e.g.
    from Google-ADK's tool-invocation task) triggers
    ``RuntimeError: Attempted to exit cancel scope in a different task``
    because the underlying MCP streamable-HTTP transport is task-scoped.
    """
    from contextlib import AsyncExitStack
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    from agentic import llm_metrics

    tool = _tools.get(tool_name)
    if tool is None:
        raise ValueError(f"MCP tool not found: {tool_name}")

    server = _servers.get(tool.server)
    if server is None:
        raise ValueError(f"MCP server not found: {tool.server}")

    logger.info("[MCP] Calling tool=%s on server=%s args=%s", tool_name, tool.server, arguments)

    async def _invoke() -> Any:
        async with AsyncExitStack() as stack:
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(server.url)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            content = result.content or []
            text = "\n".join(
                c.text for c in content
                if hasattr(c, "text") and c.text
            )
            logger.info("[MCP] Tool=%s result=%s", tool_name, text[:200])
            return {"status": "success", "result": text or str(content)}

    start = time.monotonic()
    try:
        return await asyncio.wait_for(_invoke(), timeout=server.timeout)
    except asyncio.TimeoutError:
        logger.error("[MCP] Tool=%s timed out after %.0fs", tool_name, server.timeout)
        return {"error": f"Tool {tool_name} timed out"}
    except Exception as exc:
        logger.error("[MCP] Tool=%s call failed: %s", tool_name, exc)
        return {"error": str(exc)}
    finally:
        # Recorded for every outcome (success, timeout, transport error): a
        # slow *failing* call is just as relevant to the "where did the 4s
        # go" question as a slow successful one.
        llm_metrics.record_mcp((time.monotonic() - start) * 1000)


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

