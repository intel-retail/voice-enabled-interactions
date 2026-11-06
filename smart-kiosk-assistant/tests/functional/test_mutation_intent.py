"""Unit tests for the cart-mutation intent check in reply_templates.

These are tier1 (no Docker/ML/audio hardware required) — ``is_browse_intent``
is pure string work over the customer's utterance.

Background: the same check gates directive mode in ``ordering_agent.chat``.
An order it misses falls to the tool-calling path, which reached the first
LLM token about 7x later than directive mode on an internal Panther Lake test
system. The everyday ordering phrasings below used to miss; questions must
still read as browse.
"""
import sys
from pathlib import Path

import pytest

# reply_templates imports `agentic`, which lives under rag-service/. Put that
# directory on the path only for this import and take it off again: left at
# the front, it makes later tests' `import main` load rag-service/main.py
# instead of kiosk-core's.
_RAG_SERVICE = str(Path(__file__).resolve().parents[2] / "rag-service")
sys.path.append(_RAG_SERVICE)
try:
    from plugins.kiosk.reply_templates import is_browse_intent  # noqa: E402
finally:
    sys.path.remove(_RAG_SERVICE)

pytestmark = pytest.mark.tier1


@pytest.mark.parametrize(
    "utterance",
    [
        # Phrasings that already matched.
        "Please add one regular Classic French Fries too.",
        "Remove the brownie.",
        "That is everything. Please confirm my order now.",
        "Order me a Pepsi.",
        # Phrasings that used to miss and send the turn to the slow path.
        "I would like one Classic Chicken Burger please.",
        "I'd like a Classic Chicken Burger, please.",
        "I'd like to get two Pepsis.",
        "I will have one Spicy Chicken Crunch Burger.",
        "I'll have two Pepsis.",
        "We'll take the Paneer Tikka Burger.",
        "Can I get a Chicken Tikka Kathi Roll?",
        "Could I have one masala chai?",
        "Give me a large fries.",
        "Let me get a brownie.",
        "I want a cold coffee.",
        "We need two veg burgers.",
    ],
)
def test_ordering_phrasings_are_mutations(utterance):
    assert is_browse_intent(utterance) is False


@pytest.mark.parametrize(
    "utterance",
    [
        "What desserts do you have?",
        "How much is the brownie?",
        "What's in my order?",
        "I'd like to know what's vegetarian.",
        "I would like to see the menu.",
        # rag-service/tests/test_reply_templates.py relies on this one being
        # a browse turn, so list_products can be spoken as the whole reply.
        "I would like to explore burgers.",
        "Do you have anything spicy?",
        "What do you recommend?",
        "Can you tell me your hours?",
    ],
)
def test_questions_stay_browse(utterance):
    assert is_browse_intent(utterance) is True


def test_known_false_positive_is_a_request_not_an_order():
    # "Can I get <quantity word> ..." reads as an order even when the thing is
    # not on the menu. Accepted: directive mode emits no <act> for it and
    # falls back, costing one wasted generation, never a wrong cart write.
    assert is_browse_intent("Can I get some help?") is False
