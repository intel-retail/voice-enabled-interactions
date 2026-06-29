"""Tests for OrderingAgent orchestration with mocked ADK/LLM events."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentic import ordering_agent
from agentic.mcp_client import MCPTool
from agentic.ordering_agent import OrderingAgent


def _run(coro: Any) -> Any:
    """Run an async test helper without depending on pytest-asyncio."""
    return asyncio.run(coro)


def test_make_mcp_callable_invokes_mcp_tool(monkeypatch) -> None:
    """The generated ADK callable delegates to mcp_client.call_tool."""
    fake_call_tool = AsyncMock(return_value={"status": "success", "result": "updated"})
    monkeypatch.setattr(ordering_agent, "call_tool", fake_call_tool)
    mcp_tool = MCPTool(name="update_order", server="core", description="Add items")

    fn = OrderingAgent._make_mcp_callable("update_order", mcp_tool)
    result = _run(fn(order_id="ORD-1", items=[{"product_id": "coke", "quantity": 1}]))

    assert result == {"status": "success", "result": "updated"}
    fake_call_tool.assert_awaited_once_with(
        "update_order",
        {"order_id": "ORD-1", "items": [{"product_id": "coke", "quantity": 1}]},
    )
    assert fn.__name__ == "update_order"
    assert fn.__schema__["name"] == "update_order"


@pytest.mark.parametrize(
    ("message", "expected_tools"),
    [
        ("Do you have vegan burgers?", ["knowledge_lookup"]),
        ("Place an order for a paneer burger", ["list_products", "place_order", "get_upsell_suggestions"]),
        ("Add a coke to my order", ["update_order", "get_upsell_suggestions"]),
        ("Confirm my order", ["confirm_order"]),
    ],
)
def test_chat_records_scripted_adk_tool_calls(message: str, expected_tools: list[str]) -> None:
    """Scripted ADK runner events model OVMS tool-call choices without OVMS."""
    pytest.importorskip("google.adk")

    agent = OrderingAgent()
    agent._bootstrapped = True
    agent._agent = SimpleNamespace(name="kiosk_ordering_agent")
    agent._session_service = _FakeSessionService()
    agent._runner = _FakeRunner()

    result = _run(agent.chat(message=message, session_id="session-1", user_id="user-1"))

    assert result["tool_calls"] == expected_tools
    assert "reply" in result


class _FakeSessionService:
    """Session service that makes _ensure_session a no-op."""

    async def get_session(self, app_name: str, user_id: str, session_id: str) -> object:
        return object()


class _FakeRunner:
    """ADK Runner replacement that yields scripted tool-call events."""

    async def run_async(
        self,
        user_id: str,
        session_id: str,
        new_message: object,
    ) -> Any:
        text = new_message.parts[0].text.lower()
        if "vegan" in text:
            tools = ["knowledge_lookup"]
            reply = "Yes, we have vegan options."
        elif "place" in text:
            tools = ["list_products", "place_order", "get_upsell_suggestions"]
            reply = "I created a draft order."
        elif "add a coke" in text:
            tools = ["update_order", "get_upsell_suggestions"]
            reply = "I added a coke."
        elif "confirm" in text:
            tools = ["confirm_order"]
            reply = "Your order is confirmed!"
        else:
            tools = []
            reply = "How can I help?"

        for tool_name in tools:
            yield SimpleNamespace(tool_call=SimpleNamespace(name=tool_name), content=None)
        yield SimpleNamespace(tool_call=None, content=SimpleNamespace(parts=[SimpleNamespace(text=reply)]))
