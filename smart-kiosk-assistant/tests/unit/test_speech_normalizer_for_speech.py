"""Tests for speech_normalizer.for_speech (kiosk_core/speech_normalizer.py).

Background (PR #112 review, item 25): for_speech is the last thing to touch a
reply before it is handed to the synthesiser, and it shipped without tests.

Everything here is a rewrite the customer *hears*, so a regression is not a
wrong value in a log -- it is the kiosk reading "three three eight rupees"
for a total, or spelling out an HTML tag. The ordering of the rules matters
as much as the rules themselves: currency has to be consumed before the bare
number handling can reach the digits, which is why the money cases below pin
the whole phrase and not just the number.
"""
import pytest

from kiosk_core.speech_normalizer import for_speech


class TestCurrency:
    def test_rupee_amount_is_spoken_as_a_quantity_not_digits(self):
        assert for_speech("Your total is ₹338.") == (
            "Your total is three hundred thirty eight rupees."
        )

    def test_rupees_and_paise(self):
        assert for_speech("The price is ₹5.49") == (
            "The price is five rupees and forty nine paise"
        )

    def test_rupee_amount_split_by_a_stray_space(self):
        # The model intermittently emits "₹5. 49"; without the repair the
        # sentence splitter would also treat that period as an end of
        # sentence and speak the two halves separately.
        assert for_speech("The price is ₹5. 49") == (
            "The price is five rupees and forty nine paise"
        )

    def test_thousands_separator_is_not_read_as_a_list(self):
        assert "one thousand" in for_speech("That comes to ₹1,200")

    def test_dollar_amount(self):
        assert for_speech("It costs $5.49") == "It costs five forty nine"


class TestNumbersAndUnits:
    def test_percentage(self):
        assert for_speech("We are 20% off today") == "We are twenty percent off today"

    def test_compressed_unit_is_expanded(self):
        assert for_speech("330 ml bottle") == "three hundred thirty milliliters bottle"

    def test_decimal_is_read_as_point(self):
        assert for_speech("4.6 oz pack") == "four point six ounces pack"

    def test_clock_time_keeps_its_meridiem(self):
        assert for_speech("Ready at 3:30 PM") == "Ready at three thirty PM"

    def test_time_range_becomes_to_not_minus(self):
        # A dash between times would otherwise be read as "minus", or
        # swallowed, turning opening hours into nonsense.
        assert for_speech("Open 8 AM-11 PM") == "Open eight AM to eleven PM"

    def test_price_range_becomes_to(self):
        assert " to " in for_speech("Between ₹80-₹120")


class TestMarkupAndLayout:
    def test_stray_tags_are_removed(self):
        # These reach the synthesiser as literal words otherwise.
        assert for_speech("Item one <b>bold</b> tag") == "Item one bold tag"

    def test_bulleted_lines_are_read_as_a_list(self):
        assert for_speech("Menu:\n- Burger\n- Wrap") == "Menu: Burger, Wrap"

    def test_newlines_collapse_to_spaces(self):
        assert for_speech("First line\nsecond line") == "First line second line"


class TestPassThrough:
    @pytest.mark.parametrize("value", ["", None])
    def test_empty_input_is_returned_unchanged(self, value):
        # Callers pass whatever the model produced, including nothing at all;
        # this must not raise on the TTS path.
        assert for_speech(value) == value

    def test_plain_prose_is_untouched(self):
        text = "Got it, one Classic Chicken Burger."
        assert for_speech(text) == text

    def test_is_idempotent_for_currency(self):
        # The normaliser can be reached twice for a sentence that is re-sent
        # (speculative draft then real reply); running it again must not
        # re-expand already-spoken words.
        once = for_speech("Your total is ₹338.")
        assert for_speech(once) == once
