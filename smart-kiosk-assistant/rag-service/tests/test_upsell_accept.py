"""Tests for adding the offered item when the customer accepts an upsell."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from plugins.kiosk import ordering_agent
from plugins.kiosk import upsell_accept
from plugins.kiosk.ordering_agent import OrderingAgent

_OFFER_TURN = (
    "Got it. A classic chicken burger. Your total is now ₹169. "
    "Would you also like Classic French Fries (Regular) (₹89)?"
)


def _history(*assistant: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for text in assistant:
        out.append({"role": "user", "content": "I'd like to order a classic chicken burger."})
        out.append({"role": "assistant", "content": text})
    return out


class TestOfferedItem:
    def test_name_with_its_own_parentheses(self) -> None:
        assert upsell_accept.offered_item(_history(_OFFER_TURN)) == "Classic French Fries (Regular)"

    def test_other_currency_and_comma_in_name(self) -> None:
        turn = "Got it. Your total is now $4.28. Would you also like Fountain Drink, Medium ($1.79)?"
        assert upsell_accept.offered_item(_history(turn)) == "Fountain Drink, Medium"

    def test_no_offer_in_last_assistant_turn(self) -> None:
        hist = _history(_OFFER_TURN, "Your total is now ₹258. Would you like anything else?")
        assert upsell_accept.offered_item(hist) is None

    def test_empty_history(self) -> None:
        assert upsell_accept.offered_item(None) is None
        assert upsell_accept.offered_item([]) is None


class TestIsAcceptance:
    @pytest.mark.parametrize(
        "message",
        [
            "Yes, please go ahead and add that.",
            "Yes",
            "yes please",
            "Sure, add that",
            "I'll take it",
            "Okay",
            "Go ahead",
            "Yeah, add that too, thanks",
        ],
    )
    def test_whole_utterance_acceptance(self, message: str) -> None:
        assert upsell_accept.is_acceptance(message)

    @pytest.mark.parametrize(
        "message",
        [
            "Thank you",
            "No thanks",
            "No",
            "Yes, and a coke",
            "Yes but make it large",
            "Add a coke",
            "Yes, confirm my order",
            "Go ahead and confirm",
            "What size is it?",
            "",
            " . ",
        ],
    )
    def test_anything_else_does_not_match(self, message: str) -> None:
        assert not upsell_accept.is_acceptance(message)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _agent(monkeypatch) -> OrderingAgent:
    monkeypatch.setattr(ordering_agent, "get_all_tools", lambda: {"dummy": object()})
    agent = OrderingAgent()
    agent._bootstrapped = True
    agent._agent = SimpleNamespace(name="kiosk_ordering_agent")
    return agent


def _place_order_result() -> dict[str, Any]:
    payload = {
        "order_id": 1,
        "status": "draft",
        "total": 258.0,
        "items": [
            {"product_id": "burger", "name": "Classic Chicken Burger", "quantity": 1},
            {"product_id": "fries", "name": "Classic French Fries (Regular)", "quantity": 1},
        ],
        "just_added": [{"name": "Classic French Fries (Regular)", "quantity": 1}],
        "upsell_suggestions": [],
    }
    return {"status": "success", "result": json.dumps(payload)}


def test_chat_accept_adds_offered_item(monkeypatch) -> None:
    """"Yes, add that" after an offer adds that item in code, with no LLM call."""
    pytest.importorskip("google.adk")
    agent = _agent(monkeypatch)
    fake_call_tool = AsyncMock(return_value=_place_order_result())
    monkeypatch.setattr(ordering_agent, "call_tool", fake_call_tool)

    result = _run(agent.chat(
        message="Yes, please go ahead and add that.",
        session_id="s1", user_id="u1", history=_history(_OFFER_TURN),
    ))

    fake_call_tool.assert_awaited_once_with("place_order", {
        "user_id": "u1",
        "items": [{"product_id": "Classic French Fries (Regular)", "quantity": 1}],
        "dry_run": False,
    })
    assert result["tool_calls"] == ["place_order"]
    assert result["llm_calls"] == 0
    assert "Classic French Fries (Regular)" in result["reply"]
    assert agent._cart_states["s1"].known_items[-1]["product_id"] == "fries"


def test_chat_accept_falls_through_when_add_fails(monkeypatch) -> None:
    """A rejected add is not spoken as a success; the turn goes to the normal path."""
    pytest.importorskip("google.adk")
    agent = _agent(monkeypatch)
    rejected = {"status": "success", "result": json.dumps({"error": "not on the menu"})}
    monkeypatch.setattr(ordering_agent, "call_tool", AsyncMock(return_value=rejected))

    spoken = _run(agent._accept_upsell("Classic French Fries (Regular)", "u1", "s1"))

    assert spoken == ""
    assert "s1" not in agent._cart_states


def test_chat_yes_without_offer_does_not_add(monkeypatch) -> None:
    """A "yes" when the last turn made no offer never adds anything by itself."""
    pytest.importorskip("google.adk")
    agent = _agent(monkeypatch)
    accept = AsyncMock(return_value="should not be called")
    monkeypatch.setattr(agent, "_accept_upsell", accept)
    monkeypatch.setattr(ordering_agent.directive_mode, "DIRECTIVE_MODE", False)
    monkeypatch.setattr(agent, "_ensure_session", AsyncMock(side_effect=RuntimeError("normal path")))

    with pytest.raises(RuntimeError, match="normal path"):
        _run(agent.chat(
            message="Yes",
            session_id="s1", user_id="u1",
            history=_history("Your total is now ₹169. Would you like anything else?"),
        ))
    accept.assert_not_awaited()
