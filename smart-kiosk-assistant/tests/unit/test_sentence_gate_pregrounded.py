"""Tests for the pre-grounded relaxation of ``_SentenceGate``.

The gate releases a sentence early only when no whole-reply guard in ``chat()``
can still rewrite it. Every such guard is gated on ``not tool_calls``, which
originally made the gate withhold *any* tool-less turn.

A pre-grounded turn is the documented exception: the authoritative knowledge is
injected into the prompt, so calling no tool is the intended outcome, and each
mirrored guard is additionally gated on ``not pregrounded``. These tests pin
that exemption and — more importantly — pin the riders that keep it sound.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "rag-service", _ROOT / "plugins" / "kiosk"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ordering_agent as oa  # noqa: E402


def _gate(message: str = "what payment methods do you accept", *, pregrounded: bool):
    """Build a gate over a capture list.

    Returns:
        ``(gate, released)`` where ``released`` accumulates emitted sentences.
    """
    released: list[str] = []
    gate = oa._SentenceGate(message, released.append, pregrounded=pregrounded)
    return gate, released


class TestPregroundedExemption:
    """A pre-grounded, tool-less turn may stream; an ungrounded one may not."""

    def test_tool_less_turn_is_withheld_when_not_pregrounded(self):
        gate, released = _gate(pregrounded=False)
        gate.feed("We accept all major credit and debit cards. ", [])
        assert released == []

    def test_tool_less_turn_streams_when_pregrounded(self):
        gate, released = _gate(pregrounded=True)
        gate.feed("We accept all major credit and debit cards. ", [])
        assert released == ["We accept all major credit and debit cards."]

    def test_pregrounded_releases_each_complete_sentence(self):
        gate, released = _gate(pregrounded=True)
        gate.feed("We accept cards. ", [])
        gate.feed("We also accept UPI. ", [])
        assert released == ["We accept cards.", "We also accept UPI."]

    def test_incomplete_trailing_fragment_stays_buffered(self):
        """Speech is unrecallable, so a sentence without a terminator waits."""
        gate, released = _gate(pregrounded=True)
        gate.feed("We accept cards. And we also", [])
        assert released == ["We accept cards."]

    def test_tool_call_still_opens_the_gate_without_pregrounding(self):
        """The pre-existing path must keep working unchanged."""
        gate, released = _gate("do you have parking", pregrounded=False)
        gate.feed("We have a shared parking lot. ", ["knowledge_lookup"])
        assert released == ["We have a shared parking lot."]


class TestRidersRemainEnforced:
    """The relaxation must not disable the checks that keep it sound."""

    def test_parroted_knowledge_marker_is_withheld(self):
        """Rider (d2) is load-bearing now: _strip_knowledge_markers() rewrites
        a parroted block, so it must never be spoken first."""
        gate, released = _gate(pregrounded=True)
        gate.feed("[knowledge] We accept cards. ", [])
        assert released == []

    def test_confirm_intent_turn_is_never_streamed(self):
        """_force_confirm() replaces the whole reply on a confirm turn."""
        gate, released = _gate("yes confirm my order", pregrounded=True)
        gate.feed("Your order is confirmed. ", [])
        assert released == []

    def test_order_claim_without_an_order_tool_is_withheld(self):
        """chat() substitutes _ORDER_CLAIM_FALLBACK when no order tool ran."""
        gate, released = _gate(pregrounded=True)
        gate.feed("Your order has been placed. ", [])
        assert released == []

    def test_leaked_tool_syntax_is_withheld(self):
        gate, released = _gate(pregrounded=True)
        gate.feed('place_order {"items": 1}. ', [])
        assert released == []

    def test_tool_mention_is_withheld(self):
        gate, released = _gate(pregrounded=True)
        gate.feed("Let me use get_current_order to check. ", [])
        assert released == []

    def test_thinking_block_is_withheld(self):
        gate, released = _gate(pregrounded=True)
        gate.feed("<think> the user asked about payment. ", [])
        assert released == []

    def test_gate_is_one_way_after_an_unsafe_sentence(self):
        """Once closed the gate stays closed, even for a later safe sentence."""
        gate, released = _gate(pregrounded=True)
        gate.feed("[knowledge] leaked. ", [])
        gate.feed("We accept cards. ", [])
        assert released == []


class TestPregroundingCannotCollideWithCatalogue:
    """The catalogue-promise guard is the one guard not pregrounded-exempt.

    It cannot collide with this relaxation because ``chat()`` only pre-grounds a
    knowledge question that does *not* match the catalogue regex. This test pins
    that mutual exclusion at the regex level, so a future widening of either
    pattern fails here rather than silently letting a catalogue turn stream.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "what desserts do you have",
            "how much is a burger",
            "what is on the menu",
            "what drinks do you offer",
        ],
    )
    def test_catalogue_queries_are_never_pregrounded(self, message):
        assert oa._CATALOGUE_QUERY_RE.search(message)

    @pytest.mark.parametrize(
        "message",
        [
            "what payment methods do you accept",
            "what are your opening hours",
            "where are you located",
        ],
    )
    def test_knowledge_queries_used_for_pregrounding_are_not_catalogue(self, message):
        assert oa._KNOWLEDGE_QUERY_RE.search(message)
        assert not oa._CATALOGUE_QUERY_RE.search(message)
