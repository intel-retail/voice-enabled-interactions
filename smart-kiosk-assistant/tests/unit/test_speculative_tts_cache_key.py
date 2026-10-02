"""Tests for _normalize_sentence_for_cache_key (kiosk_core/audio_session.py).

Background: the speculative-draft TTS pre-synthesis cache
(BaseAudioSession._tts_cache) used to be keyed by the exact, raw sentence
string. In practice the real turn's fully-guarded reply almost never
produces a byte-identical sentence to the speculative draft's predicted
reply, even when both are built from the same wording, because of trivial
formatting drift: double spaces, a trailing period the real reply omits,
curly vs straight punctuation, etc. That made the cache's hit rate ~0 in
live testing (see docker-compose.yml's KIOSK_CORE_SPECULATIVE_DRAFT_ENABLED
comment, "2026-09-09 re-test" note).

This normalizer collapses that formatting-only drift so a cache lookup
succeeds when the WORDING is the same, while deliberately still missing
when the wording actually differs (a genuinely different sentence must
never play cached audio for a different one).
"""
from kiosk_core.audio_session import _normalize_sentence_for_cache_key


def test_whitespace_and_case_insensitive():
    a = "Got it.  One Classic Chicken Burger."
    b = "got it. one classic chicken burger."
    assert _normalize_sentence_for_cache_key(a) == _normalize_sentence_for_cache_key(b)


def test_trailing_punctuation_insensitive():
    a = "Your total is now ₹169"
    b = "Your total is now ₹169."
    assert _normalize_sentence_for_cache_key(a) == _normalize_sentence_for_cache_key(b)


def test_currency_symbol_preserved():
    # ₹169 and $169 must NOT collapse to the same key -- that would be a
    # genuinely different (and possibly wrong) amount spoken aloud.
    a = "Your total is now ₹169"
    b = "Your total is now $169"
    assert _normalize_sentence_for_cache_key(a) != _normalize_sentence_for_cache_key(b)


def test_decimal_point_preserved():
    # ₹169.50 and ₹16950 must NOT collapse to the same key: stripping "."
    # unconditionally (as generic punctuation) changes the actual amount.
    a = "Your total is now ₹169.50"
    b = "Your total is now ₹16950"
    assert _normalize_sentence_for_cache_key(a) != _normalize_sentence_for_cache_key(b)


def test_trailing_period_after_decimal_amount_still_stripped():
    # The sentence-ending "." must still be treated as formatting drift even
    # when the sentence also contains a genuine decimal amount earlier.
    a = "Your total is now ₹169.50"
    b = "Your total is now ₹169.50."
    assert _normalize_sentence_for_cache_key(a) == _normalize_sentence_for_cache_key(b)


def test_different_wording_still_misses():
    a = "Would you also like Classic French Fries?"
    b = "Would you like some fries with that?"
    assert _normalize_sentence_for_cache_key(a) != _normalize_sentence_for_cache_key(b)


def test_curly_quotes_normalized_same_as_straight():
    a = "You\u2019re all set."
    b = "You're all set."
    assert _normalize_sentence_for_cache_key(a) == _normalize_sentence_for_cache_key(b)
