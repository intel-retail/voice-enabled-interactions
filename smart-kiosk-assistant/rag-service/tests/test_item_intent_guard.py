"""Tests for the stale-item-reference guard (agentic/item_intent_guard.py).

Pure functions over text and tool-call args — no LLM, no network, no
database. They must stay that way.
"""

from __future__ import annotations

import pytest

from plugins.kiosk import item_intent_guard as guard


class TestExtractNamedItem:
    def test_extracts_item_after_add(self):
        assert guard.extract_named_item("Go ahead and add a pizza.") == "pizza"

    def test_extracts_item_after_i_would_like(self):
        assert guard.extract_named_item("I'd like a burger") == "burger"

    def test_extracts_multi_word_item(self):
        assert (
            guard.extract_named_item("add the classic chicken burger")
            == "classic chicken burger"
        )

    def test_bare_confirmation_returns_none(self):
        for utt in ("yes", "go ahead", "sounds good", "please confirm", "correct"):
            assert guard.extract_named_item(utt) is None, utt

    def test_empty_utterance_returns_none(self):
        assert guard.extract_named_item("") is None

    def test_no_add_phrase_returns_none(self):
        assert guard.extract_named_item("What are your opening hours?") is None

    @pytest.mark.parametrize(
        "utterance",
        [
            "Yes, I would like one of those.",
            "I would like to try all of them.",
            "I will have the same.",
            "Add some of those",
            "I'd like both of them",
            "Order the usual",
            "I want everything",
        ],
    )
    def test_anaphoric_reference_returns_none(self, utterance):
        """Anaphora must never override the model's context-resolved product.

        Regression: these all matched the add/order pattern and were
        substituted verbatim into the tool call, producing replies like
        "Sorry, we don't have of those on the menu" and discarding a valid
        multi-item order.
        """
        assert guard.extract_named_item(utterance) is None, utterance

    @pytest.mark.parametrize(
        "utterance,expected",
        [
            ("Add a margherita pizza.", "margherita pizza"),
            ("I want a cold coffee.", "cold coffee"),
            ("Add some fries.", "fries"),
        ],
    )
    def test_concrete_items_still_extracted(self, utterance, expected):
        assert guard.extract_named_item(utterance) == expected

    def test_bare_category_still_extracted(self):
        """A category is kept deliberately.

        It prevents the stale reference (pending fries) from being ordered,
        and mcp_server turns the unresolvable category into a "which pizza?"
        question rather than an off-menu refusal.
        """
        assert guard.extract_named_item("Go ahead and add a pizza.") == "pizza"


class TestMismatches:
    def test_disjoint_tokens_mismatch(self):
        assert guard.mismatches("french fries", "pizza") is True

    def test_overlapping_tokens_not_mismatch(self):
        assert guard.mismatches("classic chicken burger", "chicken burger") is False

    def test_empty_reference_not_mismatch(self):
        assert guard.mismatches("", "pizza") is False

    def test_empty_named_item_not_mismatch(self):
        assert guard.mismatches("french fries", "") is False


class TestCorrectedReference:
    def test_corrects_stale_single_item(self):
        items = [{"product_id": "french fries", "quantity": 1}]
        result = guard.corrected_reference("update_order", items, "Go ahead and add a pizza.")
        assert result == "pizza"

    def test_leaves_matching_reference_untouched(self):
        items = [{"product_id": "classic chicken burger", "quantity": 1}]
        result = guard.corrected_reference(
            "place_order", items, "add the classic chicken burger"
        )
        assert result is None

    def test_leaves_bare_confirmation_untouched(self):
        items = [{"product_id": "french fries", "quantity": 1}]
        result = guard.corrected_reference("update_order", items, "yes go ahead")
        assert result is None

    def test_skips_multi_item_calls(self):
        items = [
            {"product_id": "french fries", "quantity": 1},
            {"product_id": "pepsi", "quantity": 1},
        ]
        result = guard.corrected_reference("update_order", items, "add a pizza too")
        assert result is None

    def test_skips_non_order_tools(self):
        items = [{"product_id": "french fries", "quantity": 1}]
        result = guard.corrected_reference("list_products", items, "add a pizza")
        assert result is None

    def test_skips_when_items_missing(self):
        assert guard.corrected_reference("place_order", None, "add a pizza") is None

    def test_skips_when_reference_empty(self):
        items = [{"product_id": "", "quantity": 1}]
        result = guard.corrected_reference("place_order", items, "add a pizza")
        assert result is None


class TestTentativeIntent:
    """Tentative/exploratory phrases must NOT be treated as item references."""

    @pytest.mark.parametrize("utterance", [
        "I was thinking of ordering a burger.",
        "I was thinking of ordering a classic chicken burger",
        "I'm thinking of having a pizza",
        "I'm considering a coffee",
        "Maybe a fries",
        "Maybe I could get a burger",
        "What about a pizza?",
        "How about a coffee?",
        "I might want fries",
        "I might like a burger",
        "Possibly a pizza",
        "I was going to get a burger",
        "I was planning to order a coffee",
        "Just browsing",
        "I could try a burger",
    ])
    def test_tentative_returns_none(self, utterance: str):
        """Tentative phrases must return None — not an item reference."""
        assert guard.extract_named_item(utterance) is None, (
            f"Expected None for tentative utterance: {utterance!r}"
        )

    @pytest.mark.parametrize("utterance", [
        "Add a burger",
        "I want a pizza",
        "Order me a coffee",
        "Get me fries please",
        "I'd like a burger",
        "Can I get a coffee?",
        "I'll have a pizza",
    ])
    def test_direct_order_not_affected(self, utterance: str):
        """Direct order phrases must still return an item reference."""
        result = guard.extract_named_item(utterance)
        assert result is not None, (
            f"Expected non-None for direct order: {utterance!r}"
        )


class TestPlaceholderReference:
    """A made-up numeric product_id is replaced with what the customer said."""

    def test_bare_item_replaces_placeholder(self):
        # Observed: "One Perry chicken wrap." -> place_order(product_id="12345").
        items = [{"product_id": "12345", "quantity": 1}]
        assert (
            guard.corrected_reference("place_order", items, "One Perry chicken wrap.")
            == "Perry chicken wrap"
        )

    def test_follow_up_item_replaces_placeholder(self):
        # Observed: "and a fresh lime soda, please." -> product_id="67890".
        items = [{"product_id": "67890", "quantity": 1}]
        assert (
            guard.corrected_reference("update_order", items, "and a fresh lime soda, please.")
            == "fresh lime soda"
        )

    def test_named_item_replaces_placeholder(self):
        items = [{"product_id": "12345", "quantity": 1}]
        assert guard.corrected_reference("place_order", items, "Add a Pepsi") == "Pepsi"

    @pytest.mark.parametrize("utterance", ["yes", "one of those", "the second one", "two please"])
    def test_placeholder_kept_when_customer_named_nothing(self, utterance: str):
        items = [{"product_id": "12345", "quantity": 1}]
        assert guard.corrected_reference("place_order", items, utterance) is None

    @pytest.mark.parametrize("utterance", [
        "maybe a pizza",
        "I was thinking of a burger",
        "Just browsing",
    ])
    def test_tentative_utterance_is_not_an_item(self, utterance: str):
        assert guard.extract_bare_item(utterance) is None
        items = [{"product_id": "12345", "quantity": 1}]
        assert guard.corrected_reference("place_order", items, utterance) is None

    def test_number_the_customer_said_is_not_a_placeholder(self):
        assert not guard.is_placeholder_reference("12345", "I want item 12345")
        assert guard.is_placeholder_reference("12345", "One Perry chicken wrap.")
        assert not guard.is_placeholder_reference("BURGER-NV-001", "a burger")

    def test_real_reference_without_verb_is_untouched(self):
        # Only a placeholder is replaced from a verb-less utterance.
        items = [{"product_id": "Peri Peri Chicken Wrap", "quantity": 1}]
        assert guard.corrected_reference("place_order", items, "One Perry chicken wrap.") is None

    def test_multi_item_call_is_untouched(self):
        items = [{"product_id": "12345", "quantity": 1}, {"product_id": "67890", "quantity": 1}]
        assert guard.corrected_reference("place_order", items, "One Perry chicken wrap.") is None
