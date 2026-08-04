"""Unit tests for BaseAudioSession._filter_target_speaker().

These are tier1 (no Docker/ML required) — the method is pure text/dict
filtering logic operating on segments already tagged with ``is_primary`` by
the audio-analyzer service. See docs/audio-analyzer-diarization-plan.md.
"""
import pytest

from kiosk_core.audio_session import BaseAudioSession


def _make_session(session_id: str = "test-session") -> BaseAudioSession:
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = session_id
    return session


@pytest.mark.tier1
class TestFilterTargetSpeaker:
    def test_empty_segments_returns_empty_string(self):
        session = _make_session()
        assert session._filter_target_speaker([]) == ""

    def test_keeps_only_primary_segments(self):
        session = _make_session()
        segments = [
            {"text": "I'd like a burger", "speaker": "SPEAKER_00", "is_primary": True},
            {"text": "get out of the way", "speaker": "SPEAKER_01", "is_primary": False},
        ]
        assert session._filter_target_speaker(segments) == "I'd like a burger"

    def test_multiple_primary_segments_are_joined(self):
        session = _make_session()
        segments = [
            {"text": "I'd like", "speaker": "SPEAKER_00", "is_primary": True},
            {"text": "a burger please", "speaker": "SPEAKER_00", "is_primary": True},
        ]
        assert session._filter_target_speaker(segments) == "I'd like a burger please"

    def test_analyzer_rejection_is_authoritative_over_domain_keywords(self):
        """An enrolled-voice mismatch must drop the chunk even if it is on-topic.

        Regression: an interloper saying "I can add paneer tikka burger" was
        accepted because the analyzer's explicit ``is_primary=False`` verdict
        fell through to the first-speaker/semantic fallback, letting a
        bystander inject items into the customer's cart.
        """
        session = _make_session()
        segments = [
            {"text": "can I get a burger and fries", "speaker": "SPEAKER_01", "is_primary": False},
        ]
        assert session._filter_target_speaker(segments) == ""

    def test_analyzer_rejection_does_not_lock_on_to_interloper(self):
        """A rejected-only chunk must not arm the first-speaker lock."""
        session = _make_session()
        segments = [
            {"text": "I can add paneer tikka burger", "speaker": "SPEAKER_01", "is_primary": False},
        ]
        assert session._filter_target_speaker(segments) == ""
        assert getattr(session, "_primary_speaker_id", None) is None

    def test_mixed_chunk_keeps_primary_and_drops_interloper(self):
        session = _make_session()
        segments = [
            {"text": "one classic chicken burger", "speaker": "SPEAKER_00", "is_primary": True},
            {"text": "I can add paneer tikka burger", "speaker": "SPEAKER_01", "is_primary": False},
        ]
        assert session._filter_target_speaker(segments) == "one classic chicken burger"

    def test_no_primary_and_no_domain_match_drops_chunk(self):
        session = _make_session()
        segments = [
            {"text": "the weather is nice today", "speaker": "SPEAKER_01", "is_primary": False},
        ]
        assert session._filter_target_speaker(segments) == ""

    def test_missing_is_primary_key_falls_back_to_first_speaker_lock(self):
        session = _make_session()
        segments = [
            {"text": "hello there", "speaker": "SPEAKER_00"},
        ]
        # No is_primary key at all (e.g. audio-analyzer identity disabled or
        # not yet enrolled) — the analyzer has no verdict, so tier 2 applies:
        # lock on to the first speaker label seen and treat it as the customer.
        assert session._filter_target_speaker(segments) == "hello there"
        assert session._primary_speaker_id == "SPEAKER_00"

    def test_first_speaker_lock_drops_later_interloper_without_analyzer_flag(self):
        session = _make_session()
        assert session._filter_target_speaker(
            [{"text": "one veg burger", "speaker": "SPEAKER_00"}]
        ) == "one veg burger"
        # A different label in a later chunk is unconditionally dropped.
        assert session._filter_target_speaker(
            [{"text": "add paneer tikka burger", "speaker": "SPEAKER_01"}]
        ) == ""

    def test_strict_drop_can_be_disabled(self, monkeypatch):
        """Escape hatch: with strict drop off, a rejected but on-topic chunk
        is recovered via the heuristics instead of being discarded."""
        from kiosk_core import config

        monkeypatch.setattr(config, "DEFAULT_SPEAKER_STRICT_DROP", False)
        session = _make_session()
        segments = [
            {"text": "can I get a burger and fries", "speaker": "SPEAKER_01", "is_primary": False},
        ]
        assert "burger" in session._filter_target_speaker(segments)
