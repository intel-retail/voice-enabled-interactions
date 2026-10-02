"""Unit tests for the adaptive endpoint's sentence-completeness check.

These are tier1 (no Docker/ML/audio hardware required) — ``_looks_complete``
is pure string work over the transcript assembled so far.

Background: ``KIOSK_CORE_SILENCE_TIMEOUT_SECONDS`` has to cover the worst case
(a customer hesitating mid-sentence), so it was raised to 1.5s after 0.65s cut
people off mid-order. That makes *every* turn pay the hesitation tax, including
the majority that end on an obviously finished sentence.

A loudness detector cannot tell those two apart; reading the words can. This
check only ever SHORTENS the wait, so every uncertain case must return False
and fall back to the full timeout. The tests below pin that fail-closed
behaviour in both directions.
"""
import sys
from unittest.mock import MagicMock

import pytest

# `kiosk_core.audio_session` imports sounddevice, which needs PortAudio at
# import time — unavailable outside the container. Mirrors the mocking done in
# tests/functional/conftest.py. Must run before the kiosk_core import below.
sys.modules.setdefault("sounddevice", MagicMock())

from kiosk_core.audio_session import _looks_complete, _endpoint_stability_key  # noqa: E402

MIN_WORDS = 3


@pytest.mark.parametrize(
    "transcript",
    [
        "I would like to order one classic chicken burger",
        "Can I get a coke",
        "two large fries please",
        "make it a meal",
    ],
)
def test_finished_sentences_allow_early_commit(transcript):
    assert _looks_complete(transcript, MIN_WORDS) is True


@pytest.mark.parametrize(
    "transcript,reason",
    [
        ("I would like a burger and", "dangling conjunction"),
        ("Can I get some, um", "hesitation sound"),
        ("I want a burger with", "preposition needing an object"),
        ("Could I also have the", "dangling article"),
        ("I would like two of those and a", "dangling article after conjunction"),
        ("Give me fries,", "explicit continuation comma"),
        # "what desserts do you have" reads finished, but the customer just as
        # often continues ("do you have ... any milkshakes?"). The asymmetry
        # decides it: a false positive cuts the customer off mid-order, a false
        # negative only costs the 0.4s saving. Ambiguous tails stay incomplete.
        ("what desserts do you have", "auxiliary that often takes an object"),
    ],
)
def test_mid_thought_transcripts_keep_the_full_timeout(transcript, reason):
    assert _looks_complete(transcript, MIN_WORDS) is False, reason


@pytest.mark.parametrize("transcript", ["", "   ", None])
def test_missing_transcript_fails_closed(transcript):
    """ASR has not returned yet: behave exactly like the fixed timeout."""
    assert _looks_complete(transcript, MIN_WORDS) is False


@pytest.mark.parametrize("transcript", ["I want", "one", "yes"])
def test_too_few_words_fails_closed(transcript):
    """Two words or fewer is almost always a fragment mid-utterance."""
    assert _looks_complete(transcript, MIN_WORDS) is False


def test_whisper_question_mark_cannot_rescue_a_dangling_word():
    """Whisper punctuates fragments, so punctuation is not evidence of an
    ending — the final word decides."""
    assert _looks_complete("Do you have any of the?", MIN_WORDS) is False


def test_punctuation_and_case_do_not_break_the_word_scan():
    assert _looks_complete("Order one classic chicken burger.", MIN_WORDS) is True


class TestEndpointStabilityKey:
    """_endpoint_stability_key() backs _endpoint_transcript_stable()'s
    change-detection: it must treat a punctuation-only ASR hallucination on
    the trailing silence tail (e.g. a stray extra ".") as NO change, while
    still treating any real new/changed word as a change — otherwise the
    stability window resets right when it should be confirming, which is
    exactly the observed 0%-80% shortcut firing-rate swing.
    """

    def test_trailing_hallucinated_period_is_ignored(self):
        before = "Hi, I would like to order one classic chicken burger. Please."
        after = "Hi, I would like to order one classic chicken burger. Please. ."
        assert _endpoint_stability_key(before) == _endpoint_stability_key(after)

    def test_repeated_trailing_period_is_ignored(self):
        assert _endpoint_stability_key("One coke please.") == _endpoint_stability_key(
            "One coke please.."
        )

    def test_case_and_whitespace_do_not_count_as_a_change(self):
        assert _endpoint_stability_key("One Coke Please.") == _endpoint_stability_key(
            "  one coke please  "
        )

    def test_a_genuinely_new_word_still_counts_as_a_change(self):
        before = "I would like one classic chicken burger"
        after = "I would like one classic chicken burger and"
        assert _endpoint_stability_key(before) != _endpoint_stability_key(after)

    def test_a_changed_word_still_counts_as_a_change(self):
        before = "Can I get a coke"
        after = "Can I get a sprite"
        assert _endpoint_stability_key(before) != _endpoint_stability_key(after)

    def test_none_and_empty_are_equivalent(self):
        assert _endpoint_stability_key(None) == _endpoint_stability_key("")
