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

    def test_no_primary_segment_falls_back_to_domain_keyword_match(self):
        session = _make_session()
        segments = [
            {"text": "can I get a burger and fries", "speaker": "SPEAKER_01", "is_primary": False},
        ]
        result = session._filter_target_speaker(segments)
        assert "burger" in result

    def test_no_primary_and_no_domain_match_drops_chunk(self):
        session = _make_session()
        segments = [
            {"text": "the weather is nice today", "speaker": "SPEAKER_01", "is_primary": False},
        ]
        assert session._filter_target_speaker(segments) == ""

    def test_segments_missing_is_primary_key_are_treated_as_non_primary(self):
        session = _make_session()
        segments = [
            {"text": "hello there", "speaker": "SPEAKER_00"},
        ]
        # No is_primary key at all (e.g. audio-analyzer identity disabled) —
        # should not be kept outright, only surfaced via semantic fallback.
        assert session._filter_target_speaker(segments) == ""
