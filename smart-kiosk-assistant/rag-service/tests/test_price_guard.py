"""Tests for the order-total truthfulness guard (plugins/kiosk/price_guard.py).

Pure functions over reply text and recorded tool outcomes — no LLM, no
network, no database.
"""

from __future__ import annotations

import json

import pytest

from plugins.kiosk import price_guard


def _mcp_envelope(payload: dict) -> dict:
    """Wrap a tool payload the way mcp_client.call_tool returns it."""
    return {"status": "success", "result": json.dumps(payload)}


@pytest.fixture(autouse=True)
def _fresh_turn():
    """Every test starts with a clean per-turn state, like production does."""
    price_guard.begin_turn()
    yield
    price_guard.begin_turn()


class TestRecordToolResult:
    def test_captures_total_from_place_order(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"order_id": 1, "total": 169.0})
        )
        assert price_guard.current_state().total == 169.0

    def test_captures_total_from_get_current_order(self) -> None:
        price_guard.record_tool_result(
            "get_current_order", _mcp_envelope({"order_id": 1, "total": 258.0})
        )
        assert price_guard.current_state().total == 258.0

    def test_ignores_unrelated_tool(self) -> None:
        price_guard.record_tool_result(
            "list_products", _mcp_envelope({"products": []})
        )
        assert price_guard.current_state().total is None

    def test_ignores_failed_result(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"error": "not on the menu"})
        )
        assert price_guard.current_state().total is None

    def test_non_numeric_total_is_ignored_not_raised(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"total": "n/a"})
        )
        assert price_guard.current_state().total is None


class TestValidateReply:
    def test_no_authoritative_total_leaves_reply_untouched(self) -> None:
        reply = "Your total is now 999 rupees."
        corrected, changed = price_guard.validate_reply(reply)
        assert corrected == reply
        assert changed is False

    def test_matching_total_is_untouched(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"total": 169.0})
        )
        reply = "I've added the burger. Your total is now 169 rupees."
        corrected, changed = price_guard.validate_reply(reply)
        assert corrected == reply
        assert changed is False

    def test_mismatched_total_is_corrected_in_place(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"total": 169.0})
        )
        reply = "I've added the burger. Your total is now 199 rupees."
        corrected, changed = price_guard.validate_reply(reply)
        assert changed is True
        assert "169" in corrected
        assert "199" not in corrected

    def test_matches_currency_symbol_and_comma_grouping(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"total": 1234.0})
        )
        reply = "Your total is now ₹1,000."
        corrected, changed = price_guard.validate_reply(reply)
        assert changed is True
        assert "1234" in corrected

    def test_fractional_total_is_rendered_with_two_decimals(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"total": 169.5})
        )
        reply = "Your total is now 199 rupees."
        corrected, changed = price_guard.validate_reply(reply)
        assert changed is True
        assert "169.50" in corrected

    def test_no_total_mention_is_untouched(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"total": 169.0})
        )
        reply = "I've added the Classic Chicken Burger to your order."
        corrected, changed = price_guard.validate_reply(reply)
        assert corrected == reply
        assert changed is False

    def test_per_item_price_mention_without_total_word_is_untouched(self) -> None:
        # A per-item price is already sourced from the tool result by
        # reply_templates' own formatting; this guard only reasons about the
        # order *total*, so an unrelated 169 elsewhere in the sentence must
        # not be treated as a total claim.
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"total": 258.0})
        )
        reply = "The Classic Chicken Burger is 169 rupees."
        corrected, changed = price_guard.validate_reply(reply)
        assert corrected == reply
        assert changed is False


class TestMismatched:
    def test_true_when_sentence_total_disagrees(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"total": 169.0})
        )
        assert price_guard.mismatched("Your total is now 199 rupees,") is True

    def test_false_when_sentence_total_agrees(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"total": 169.0})
        )
        assert price_guard.mismatched("Your total is now 169 rupees,") is False

    def test_false_when_no_authoritative_total_known(self) -> None:
        assert price_guard.mismatched("Your total is now 199 rupees,") is False

    def test_false_when_sentence_has_no_total_mention(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"total": 169.0})
        )
        assert price_guard.mismatched("Would you like fries with that?") is False


class TestTurnIsolation:
    def test_begin_turn_clears_previous_total(self) -> None:
        price_guard.record_tool_result(
            "place_order", _mcp_envelope({"total": 169.0})
        )
        assert price_guard.current_state().total == 169.0
        price_guard.begin_turn()
        assert price_guard.current_state().total is None
