"""Guards the duplicated confirm-claim regex against silent drift.

``plugins.kiosk.directive_mode`` holds a *copy* of
``ordering_agent._CONFIRM_CLAIM_RE`` rather than importing it, because
``ordering_agent`` already imports ``directive_mode`` and a back-import would
create a cycle. Two independent copies can drift apart over time, so this test
pins them to the same pattern and flags: whenever one is edited, the other must
be updated too (or the pair moved to a shared module).
"""

from __future__ import annotations

from plugins.kiosk import directive_mode, ordering_agent


def test_confirm_claim_regex_matches_ordering_agent() -> None:
    assert (
        directive_mode._CONFIRM_CLAIM_RE.pattern
        == ordering_agent._CONFIRM_CLAIM_RE.pattern
    ), (
        "directive_mode._CONFIRM_CLAIM_RE has drifted from "
        "ordering_agent._CONFIRM_CLAIM_RE — keep the copies in sync "
        "(or extract a shared constant)."
    )
    assert (
        directive_mode._CONFIRM_CLAIM_RE.flags
        == ordering_agent._CONFIRM_CLAIM_RE.flags
    )


def test_confirm_claim_regex_catches_representative_claims() -> None:
    # A smoke set shared by both copies: if the pattern is edited and these
    # break, both guards lose coverage together.
    positives = [
        "Your order is now confirmed.",
        "Got it. Your order is confirmed. Thank you!",
        "I've confirmed your order.",
        "Your order has been placed.",
        "Your order number is 42.",
    ]
    negatives = [
        "Would you like to confirm your order?",
        "Done. I've added Classic Chicken Burger to your order.",
    ]
    for text in positives:
        assert directive_mode._CONFIRM_CLAIM_RE.search(text), text
    for text in negatives:
        assert not directive_mode._CONFIRM_CLAIM_RE.search(text), text
