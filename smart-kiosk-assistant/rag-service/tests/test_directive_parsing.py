"""Tests for parse_directive (plugins/kiosk/directive_mode.py).

Background (PR #112 review, item 25): directive mode is the live ordering
path -- the model emits an ``<act>…</act>`` block that is executed directly
against the cart -- and it shipped without tests.

This parser is where a malformed generation either gets rejected or silently
corrupts somebody's order, so the cases worth pinning are the rejections. Its
contract is all-or-nothing: a directive that is not wholly understood returns
None and the caller falls back, because applying half of
"remove the burger; add the wrap" is worse than applying none of it.
"""
from __future__ import annotations

import pytest

from plugins.kiosk.directive_mode import parse_directive, parse_price_directive


class TestWellFormedDirectives:
    def test_add_with_explicit_quantity(self):
        assert parse_directive("add|Classic Chicken Burger|2") == [
            {"verb": "add", "name": "Classic Chicken Burger", "quantity": 2}
        ]

    def test_quantity_defaults_to_one(self):
        # The model routinely omits the count for a single item.
        assert parse_directive("add|Veg Wrap") == [
            {"verb": "add", "name": "Veg Wrap", "quantity": 1}
        ]

    @pytest.mark.parametrize("quantity_field", ["", " "])
    def test_blank_quantity_field_defaults_to_one(self, quantity_field):
        # A trailing separator is not a malformed count, and a whitespace-only
        # field is indistinguishable from an empty one once stripped.
        assert parse_directive(f"add|Veg Wrap|{quantity_field}") == [
            {"verb": "add", "name": "Veg Wrap", "quantity": 1}
        ]

    def test_confirm_takes_no_operands(self):
        assert parse_directive("confirm") == [{"verb": "confirm"}]

    def test_multiple_clauses_keep_their_order(self):
        # Order matters: remove-then-add and add-then-remove are different
        # carts when the same item appears in both.
        assert parse_directive("remove|Veg Wrap|1;add|Classic Chicken Burger|1") == [
            {"verb": "remove", "name": "Veg Wrap", "quantity": 1},
            {"verb": "add", "name": "Classic Chicken Burger", "quantity": 1},
        ]

    def test_surrounding_whitespace_is_tolerated(self):
        assert parse_directive("  add | Veg Wrap | 2  ; confirm ") == [
            {"verb": "add", "name": "Veg Wrap", "quantity": 2},
            {"verb": "confirm"},
        ]

    def test_verb_is_case_insensitive(self):
        assert parse_directive("ADD|Veg Wrap|1") == [
            {"verb": "add", "name": "Veg Wrap", "quantity": 1}
        ]

    def test_item_name_case_is_preserved(self):
        # The name is matched against the catalogue downstream and is also
        # spoken back to the customer, so it must not be folded.
        parsed = parse_directive("add|Classic CHICKEN Burger|1")
        assert parsed[0]["name"] == "Classic CHICKEN Burger"

    def test_empty_clauses_between_separators_are_skipped(self):
        assert parse_directive("add|Veg Wrap|1;;confirm") == [
            {"verb": "add", "name": "Veg Wrap", "quantity": 1},
            {"verb": "confirm"},
        ]


class TestRejections:
    """Every case here must return None, never a partial action list."""

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            ";",
            None,
        ],
    )
    def test_nothing_to_do_is_none_not_empty_list(self, raw):
        # None and [] are different to the caller: [] would read as "a valid
        # directive that does nothing", silencing the fallback path.
        assert parse_directive(raw) is None

    def test_unknown_verb_rejects(self):
        assert parse_directive("cancel|Veg Wrap|1") is None

    def test_add_without_an_item_name_rejects(self):
        assert parse_directive("add") is None

    def test_add_with_an_empty_item_name_rejects(self):
        assert parse_directive("add||2") is None

    @pytest.mark.parametrize("quantity", ["two", "1.5", "-1", "1x"])
    def test_non_integer_quantity_rejects(self, quantity):
        # A quantity that is not a plain positive integer must not be coerced
        # -- guessing here bills the customer for the wrong number of items.
        assert parse_directive(f"add|Veg Wrap|{quantity}") is None

    def test_zero_quantity_rejects(self):
        # "add zero burgers" is not a removal and not a no-op; it is a
        # generation the model should not have produced.
        assert parse_directive("add|Veg Wrap|0") is None

    def test_one_bad_clause_rejects_the_whole_directive(self):
        # The key property: the valid first clause must NOT be applied on its
        # own. Dropping the second half of this would leave the wrap removed
        # and nothing added back.
        assert parse_directive("remove|Veg Wrap|1;add|Veg Wrap|zero") is None

    def test_a_bad_clause_before_a_good_one_also_rejects(self):
        assert parse_directive("add|Veg Wrap|zero;confirm") is None


class TestPriceDirective:
    def test_returns_the_trimmed_name(self):
        assert parse_price_directive("  Chocolate Brownie  ") == "Chocolate Brownie"

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_empty_is_none(self, raw):
        assert parse_price_directive(raw) is None
