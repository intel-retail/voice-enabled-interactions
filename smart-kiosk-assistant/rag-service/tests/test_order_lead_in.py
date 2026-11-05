"""Tests for an ordering lead-in that names no item ("I'll have um...").

The reply_templates functions are pure functions over text. The chat() tests
script directive mode and the ADK runner, so no LLM, network or database is
used.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from plugins.kiosk import ordering_agent
from plugins.kiosk import reply_templates as rt
from plugins.kiosk.ordering_agent import OrderingAgent


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class _FakeSessionService:
    """Session service that makes _ensure_session a no-op."""

    async def get_session(self, app_name: str, user_id: str, session_id: str) -> object:
        return object()


class _FakeRunner:
    """ADK Runner replacement that records each message and answers in text."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def run_async(self, user_id: str, session_id: str, new_message: Any) -> Any:
        self.seen.append(new_message.parts[0].text)
        part = SimpleNamespace(function_call=None, text="Got it.")
        yield SimpleNamespace(partial=False, content=SimpleNamespace(parts=[part]))


class TestLeadInOnly:
    """An ordering lead-in with no item is not an order yet."""

    @pytest.mark.parametrize("message", [
        "I'll have um...",
        "I'll have, um,",
        "I'd like",
        "I would like to order",
        "Can I get a",
        "Let me get, uh,",
        "Um, I want the",
        "Give me a",
    ])
    def test_lead_in_without_item(self, message):
        assert rt.is_lead_in_only(message)

    @pytest.mark.parametrize("message", [
        "I'll have the Peri Peri Chicken Wrap.",
        "I'd like a Pepsi",
        "I'll have the same",
        "I'll have one.",
        "I want that",
        "Can I get some help?",
        "I'd like to know the hours",
        "um",
        "",
    ])
    def test_named_item_or_other_request(self, message):
        assert not rt.is_lead_in_only(message)

    @staticmethod
    def _history(previous):
        return [
            {"role": "user", "content": previous},
            {"role": "assistant", "content": "Sure. What would you like?"},
        ]

    def test_item_is_joined_to_the_lead_in(self):
        # Observed: "I'll have, um, [pause] the Peri Peri Chicken Wrap."
        joined = rt.join_lead_in("the Peri Peri Chicken Wrap.", self._history("I'll have um..."))
        assert joined == "I'll have um the Peri Peri Chicken Wrap."

    @pytest.mark.parametrize("message", [
        "What drinks do you have?",
        "which burgers are veg",
        "Add a Pepsi",
    ])
    def test_question_or_own_request_is_not_joined(self, message):
        assert rt.join_lead_in(message, self._history("I'll have um...")) == message

    @pytest.mark.parametrize("message", [
        "No thanks",
        "No, thank you.",
        "That's all.",
        "Nothing",
        "Never mind.",
        "yes",
        "Hmm",
        "Um, that's it.",
    ])
    def test_bare_reply_is_not_joined(self, message):
        assert rt.join_lead_in(message, self._history("I'll have um...")) == message

    @pytest.mark.parametrize("message", [
        "Hold on",
        "Wait.",
        "One second.",
        "Let me think.",
        "Actually nothing.",
        "Forget it.",
        "Never mind, that's all.",
    ])
    def test_stall_or_two_part_decline_is_not_joined(self, message):
        assert rt.join_lead_in(message, self._history("I'll have um...")) == message

    def test_item_after_a_stall_word_is_still_joined(self):
        # Only whole replies are excluded; an item after "actually" is an item.
        joined = rt.join_lead_in("Actually, the Pepsi.", self._history("I'll have um..."))
        assert joined == "I'll have um Actually, the Pepsi."

    @pytest.mark.parametrize("message", [
        "the Pepsi, is that the diet one?",
        "the Pepsi, how much is it",
    ])
    def test_question_after_the_item_is_not_joined(self, message):
        # No leading question word: the first is caught by "?", the second
        # (ASR dropped the "?") by the price check.
        assert rt.join_lead_in(message, self._history("I'll have um...")) == message

    def test_not_joined_after_a_complete_turn(self):
        message = "the Peri Peri Chicken Wrap."
        assert rt.join_lead_in(message, self._history("I'll have a Pepsi.")) == message
        assert rt.join_lead_in(message, None) == message


def test_chat_asks_for_the_item_after_a_bare_lead_in(monkeypatch) -> None:
    """Observed: "I'll have um..." became "Got it. A chicken burger." and a burger in the cart."""
    pytest.importorskip("google.adk")
    monkeypatch.setattr(ordering_agent, "get_all_tools", lambda: {"dummy": object()})
    monkeypatch.setattr(ordering_agent.directive_mode, "DIRECTIVE_MODE", True)
    run_turn = AsyncMock()
    monkeypatch.setattr(ordering_agent.directive_mode, "run_turn", run_turn)
    agent = OrderingAgent()
    agent._bootstrapped = True
    agent._agent = SimpleNamespace(name="kiosk_ordering_agent")
    agent._session_service = _FakeSessionService()
    agent._runner = _FakeRunner()

    result = _run(agent.chat(message="I'll have um...", session_id="session-1", user_id="user-1"))

    assert result["reply"] == ordering_agent._LEAD_IN_ONLY_REPLY
    assert result["tool_calls"] == []
    assert result["llm_calls"] == 0
    run_turn.assert_not_awaited()


def test_chat_joins_the_item_to_the_lead_in(monkeypatch) -> None:
    """The item said after "Sure. What would you like?" ends the same sentence."""
    pytest.importorskip("google.adk")
    monkeypatch.setattr(ordering_agent, "get_all_tools", lambda: {"dummy": object()})
    agent = OrderingAgent()
    agent._bootstrapped = True
    agent._agent = SimpleNamespace(name="kiosk_ordering_agent")
    agent._session_service = _FakeSessionService()
    agent._runner = _FakeRunner()
    history = [
        {"role": "user", "content": "I'll have um..."},
        {"role": "assistant", "content": ordering_agent._LEAD_IN_ONLY_REPLY},
    ]

    _run(agent.chat(
        message="the Peri Peri Chicken Wrap.", session_id="session-1", user_id="user-1",
        history=history,
    ))

    assert agent._runner.seen[0].endswith("I'll have um the Peri Peri Chicken Wrap.")


def test_chat_closes_the_order_on_a_decline_after_a_lead_in(monkeypatch) -> None:
    """"No, thank you." after "Sure. What would you like?" still finalises the order."""
    pytest.importorskip("google.adk")
    monkeypatch.setattr(ordering_agent, "get_all_tools", lambda: {"dummy": object()})
    agent = OrderingAgent()
    agent._bootstrapped = True
    agent._agent = SimpleNamespace(name="kiosk_ordering_agent")
    agent._session_service = _FakeSessionService()
    agent._runner = _FakeRunner()
    cart = ordering_agent._CartState()
    cart.known_items = [{"product_id": "burger", "quantity": 1}]
    agent._cart_states["session-1"] = cart
    confirm = AsyncMock(return_value="Your order is confirmed.")
    monkeypatch.setattr(agent, "_force_confirm", confirm)
    history = [
        {"role": "user", "content": "I'll have um..."},
        {"role": "assistant", "content": ordering_agent._LEAD_IN_ONLY_REPLY},
    ]

    result = _run(agent.chat(
        message="No, thank you.", session_id="session-1", user_id="user-1", history=history,
    ))

    confirm.assert_awaited_once()
    assert result["tool_calls"] == ["confirm_active_order"]
