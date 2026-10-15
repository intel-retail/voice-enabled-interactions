"""Tests for phrase-level (comma) TTS streaming in ``_SentenceGate``.

Historically the gate only released a complete *sentence* (terminated by
``.``/``!``/``?``). This releases at comma boundaries too, so TTS can start
speaking an early clause instead of waiting for the whole sentence — mirroring
kiosk-voice-lab-main's ``speak_streaming()`` phrase splitter
(``pipeline/tts.py``).
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "rag-service", _ROOT / "plugins" / "kiosk"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ordering_agent as oa  # noqa: E402
from plugins.kiosk import price_guard  # noqa: E402


def _gate(message: str = "order a burger"):
    released: list[str] = []
    gate = oa._SentenceGate(message, released.append)
    return gate, released


@pytest.fixture(autouse=True)
def _fresh_price_guard_turn():
    """price_guard is consulted by the gate; isolate its state per test."""
    price_guard.begin_turn()
    yield
    price_guard.begin_turn()


class TestPhraseLevelSplitting:
    """The gate must release complete clauses, not just complete sentences."""

    def test_splits_on_comma_not_just_terminal_punctuation(self) -> None:
        gate, released = _gate()
        gate.feed("Sure, ", ["place_order"])
        gate.feed("your total is now 250 rupees, ", ["place_order"])
        gate.feed("would you like fries too? ", ["place_order"])
        assert released == [
            "Sure,",
            "your total is now 250 rupees,",
            "would you like fries too?",
        ]

    def test_does_not_split_on_a_digit_grouping_comma(self) -> None:
        gate, released = _gate()
        gate.feed("Your total is now 1,234 rupees. ", ["place_order"])
        assert released == ["Your total is now 1,234 rupees."]

    def test_trailing_phrase_without_a_boundary_stays_buffered(self) -> None:
        gate, released = _gate()
        gate.feed("Sure, one moment while I check", ["place_order"])
        assert released == ["Sure,"]

    def test_comma_split_phrase_is_still_guard_checked(self) -> None:
        """A phrase failing a safety condition closes the gate exactly like an
        unsafe full sentence would — phrase-level release must not bypass any
        of _is_safe()'s existing checks."""
        gate, released = _gate()
        gate.feed("place_order {\"items\": 1}, ", ["place_order"])
        gate.feed("all set. ", ["place_order"])
        assert released == []

    def test_price_mismatch_withholds_the_comma_phrase(self) -> None:
        price_guard.record_tool_result(
            "place_order", {"status": "success", "result": '{"total": 169.0}'}
        )
        gate, released = _gate()
        gate.feed("Your total is now 199 rupees, ", ["place_order"])
        assert released == []

    def test_price_match_releases_the_comma_phrase(self) -> None:
        price_guard.record_tool_result(
            "place_order", {"status": "success", "result": '{"total": 169.0}'}
        )
        gate, released = _gate()
        gate.feed("Your total is now 169 rupees, ", ["place_order"])
        gate.feed("would you like fries too? ", ["place_order"])
        assert released == ["Your total is now 169 rupees,", "would you like fries too?"]
