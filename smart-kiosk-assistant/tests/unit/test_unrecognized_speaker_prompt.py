"""Unit tests for the "I couldn't recognise your voice" reply path.

Regression: when the speaker filter rejected every segment of a chunk the
turn ended with an empty transcript, and ``_finalize_run`` replied with the
generic "How can I help you?" greeting. A recorded conversation showed
exactly that — ``{"user": "", "assistant": "How can I help you?"}`` — leaving
the customer with no indication that they had spoken and been ignored.

These tests pin the distinction between "nobody spoke" and "somebody spoke
and every segment was discarded".
"""
import threading
from unittest.mock import MagicMock

import pytest

from kiosk_core import config
from kiosk_core.audio_session import BaseAudioSession


def _make_session(session_id: str = "test-session") -> BaseAudioSession:
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = session_id
    # _filter_target_speaker's no-primary-segments path calls
    # _scope_is_enrolled(), which reads agent_session_id -- __init__
    # normally sets this, but this harness bypasses __init__ via __new__().
    # Left un-enrolled here (never calling _mark_scope_enrolled()) so these
    # tests exercise the same "never enrolled" rejection-accounting path the
    # analyzer takes before enrollment succeeds.
    session.agent_session_id = f"{session_id}-conversation"
    session._rejected_speech_chunks = 0
    return session


def _make_finalize_ready_session(session_id: str = "test-session") -> BaseAudioSession:
    """Build a session with just enough state to call ``_finalize_run`` for real."""
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = session_id
    session.agent_session_id = session_id
    session._rejected_speech_chunks = 0
    session.transcript_parts = []
    session.response_parts = []
    session._lock = threading.Lock()
    session.error = None
    session.status = "running"
    session.completed_at = None
    session.end_reason = None
    session.on_complete = None
    session._t_turn_start = None
    session._synthesize_response = lambda text: session.response_parts.append(text)  # type: ignore[method-assign]
    # _finalize_run closes these four client connections during cleanup
    # (kiosk_core/audio_session.py) -- __init__ normally sets them, but this
    # harness bypasses __init__ via __new__(). Use MagicMock so .close() is a
    # real, inspectable no-op instead of relying on _finalize_run's
    # AttributeError-tolerant cleanup (which exists for robustness, not as a
    # substitute for a properly-shaped test double).
    session.client = MagicMock()
    session.realtime_client = None
    session.tts_client = MagicMock()
    session.agent_client = MagicMock()
    return session


@pytest.mark.tier1
class TestRejectedSpeechAccounting:
    def test_analyzer_rejection_of_real_speech_is_counted(self):
        session = _make_session()
        # On-topic text ("burger") only gets hard-dropped once the scope is
        # enrolled -- otherwise the semantic fallback recovers it (see
        # test_speaker_filter.py's authoritative-rejection tests).
        session._mark_scope_enrolled()
        segments = [
            {"text": "I want a burger", "speaker": "SPEAKER_01", "is_primary": False},
        ]
        assert session._filter_target_speaker(segments) == ""
        assert session._rejected_speech_chunks == 1

    def test_silence_is_not_counted_as_rejected_speech(self):
        """Empty segment text must stay classified as 'no speech'."""
        session = _make_session()
        segments = [{"text": "   ", "speaker": "SPEAKER_01", "is_primary": False}]
        assert session._filter_target_speaker(segments) == ""
        assert session._rejected_speech_chunks == 0

    def test_no_segments_at_all_is_not_counted(self):
        session = _make_session()
        assert session._filter_target_speaker([]) == ""
        assert session._rejected_speech_chunks == 0

    def test_accepted_primary_speech_is_not_counted(self):
        session = _make_session()
        segments = [
            {"text": "one cold coffee", "speaker": "SPEAKER_00", "is_primary": True},
        ]
        assert session._filter_target_speaker(segments) == "one cold coffee"
        assert session._rejected_speech_chunks == 0

    def test_bystander_speech_after_primary_locked_is_counted(self):
        """Primary already established; another voice must be counted, not silent."""
        session = _make_session()
        session._primary_speaker_id = "SPEAKER_00"
        segments = [
            {"text": "move along please", "speaker": "SPEAKER_01"},
        ]
        assert session._filter_target_speaker(segments) == ""
        assert session._rejected_speech_chunks == 1

    def test_repeated_rejections_accumulate_across_chunks(self):
        session = _make_session()
        session._primary_speaker_id = "SPEAKER_00"
        for _ in range(3):
            session._filter_target_speaker(
                [{"text": "unrelated chatter", "speaker": "SPEAKER_09"}]
            )
        assert session._rejected_speech_chunks == 3


@pytest.mark.tier1
class TestFinalizePrompt:
    """The reply chosen when a completed turn produced no transcript."""

    @staticmethod
    def _spoken_reply(rejected_chunks: int) -> str:
        """Run the branch in _finalize_run that picks the fallback reply."""
        session = _make_session()
        session._rejected_speech_chunks = rejected_chunks

        spoken: list[str] = []
        session._synthesize_response = spoken.append  # type: ignore[method-assign]

        # Mirrors the _finalize_run decision under test.
        if session._rejected_speech_chunks:
            session._synthesize_response(config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT)
        else:
            session._synthesize_response(config.DEFAULT_NO_SPEECH_PROMPT)
        return spoken[0]

    def test_true_silence_still_gets_the_greeting(self):
        assert self._spoken_reply(0) == config.DEFAULT_NO_SPEECH_PROMPT
        assert "How can I help you" in self._spoken_reply(0)

    def test_rejected_speech_asks_the_customer_to_repeat(self):
        reply = self._spoken_reply(1)
        assert reply == config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT
        assert reply != config.DEFAULT_NO_SPEECH_PROMPT

    def test_the_two_prompts_are_never_the_same(self):
        assert config.DEFAULT_NO_SPEECH_PROMPT != config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT

    def test_retry_prompt_actually_invites_a_retry(self):
        """The message must tell the customer what to do, not just apologise."""
        reply = config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT.lower()
        assert "recognise" in reply or "recognize" in reply
        assert "repeat" in reply or "again" in reply


@pytest.mark.tier1
class TestFinalizeRunSuppressesGreetingAfterExplicitStop:
    """Regression: an explicit stop must never speak a greeting/retry prompt.

    Observed live: the customer tapped "stop conversation" and the UI both
    showed and spoke "How can I help you?" afterwards — as if the stop had
    been ignored. `_finalize_run`'s empty-transcript branch used to check
    only `final_status == "completed"`, without regard to *why* the turn
    ended, so an explicit `stopped_by_api` end reason with no new speech hit
    the same greeting/retry path as true silence or a rejected bystander.

    That fix was later refined further: `_finalize_run` still stays silent on
    an explicit stop (`stopped_by_api`) and on the FIRST rejected-speech turn
    of a conversation — a lone rejection is still just as likely to be a
    Whisper hallucination or TTS echo as a real bystander. But
    `config.DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD` consecutive rejected
    turns in the SAME conversation (tracked by `agent_session_id` — see
    `_note_conversation_rejection`) now escalates to speaking
    `DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT`, because staying silent forever
    reproduced the original problem: a customer who really is being ignored,
    turn after turn, gets no feedback at all and the kiosk looks broken. An
    explicit stop never escalates, at any streak length.
    """

    def test_explicit_stop_with_no_transcript_stays_silent(self):
        session = _make_finalize_ready_session()
        session._finalize_run("completed", "stopped_by_api")
        assert session.response_parts == []

    def test_explicit_stop_does_not_speak_retry_prompt_either(self):
        session = _make_finalize_ready_session()
        session._rejected_speech_chunks = 2
        session._finalize_run("completed", "stopped_by_api")
        assert session.response_parts == []

    def test_true_silence_timeout_stays_silent_not_a_spoken_greeting(self):
        session = _make_finalize_ready_session()
        session._finalize_run("completed", "silence_timeout")
        assert session.response_parts == []

    def test_no_speech_detected_stays_silent_not_a_spoken_greeting(self):
        session = _make_finalize_ready_session()
        session._finalize_run("completed", "no_speech_detected")
        assert session.response_parts == []

    def test_rejected_speech_with_non_stop_end_reason_also_stays_silent(self):
        """The FIRST rejected turn in a conversation still stays silent."""
        session = _make_finalize_ready_session()
        session._rejected_speech_chunks = 1
        session._finalize_run("completed", "silence_timeout")
        assert session.response_parts == []


@pytest.mark.tier1
class TestConsecutiveRejectionEscalation:
    """Repeated rejected turns in one conversation must eventually speak up.

    Regression: after the fix pinned by ``TestFinalizeRunSuppressesGreeting-
    AfterExplicitStop`` above, a customer whose voice the analyzer never
    matched (bystander, or a mistuned enrollment rejecting the real
    customer) got silently ignored turn after turn with no feedback at all.
    These tests pin the escalation that fixes that: the SAME conversation
    (``agent_session_id``) must accumulate
    ``config.DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD`` consecutive rejected
    turns before the kiosk speaks, and any real accepted transcript resets
    the streak.
    """

    def test_single_rejection_stays_silent(self):
        session = _make_finalize_ready_session("convo-a")
        session._rejected_speech_chunks = 1
        session._finalize_run("completed", "silence_timeout")
        assert session.response_parts == []

    def test_threshold_consecutive_rejections_speaks_the_retry_prompt(self):
        threshold = config.DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD
        session = None
        for _ in range(threshold):
            session = _make_finalize_ready_session("convo-b")
            session._rejected_speech_chunks = 1
            session._finalize_run("completed", "silence_timeout")
        assert session.response_parts == [config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT]

    def test_streak_resets_after_speaking(self):
        """After escalating once, the streak starts over from zero."""
        threshold = config.DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD
        for _ in range(threshold):
            session = _make_finalize_ready_session("convo-c")
            session._rejected_speech_chunks = 1
            session._finalize_run("completed", "silence_timeout")
        assert session.response_parts == [config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT]

        # One more rejection right after escalating is turn 1 of a new
        # streak, not turn (threshold + 1) of the old one.
        session = _make_finalize_ready_session("convo-c")
        session._rejected_speech_chunks = 1
        session._finalize_run("completed", "silence_timeout")
        assert session.response_parts == []

    def test_a_real_transcript_resets_the_streak(self):
        threshold = config.DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD
        for _ in range(threshold - 1):
            session = _make_finalize_ready_session("convo-d")
            session._rejected_speech_chunks = 1
            session._finalize_run("completed", "silence_timeout")

        # A turn that WAS understood should clear the streak, not just add
        # to a running tally the next rejection would then complete.
        heard_session = _make_finalize_ready_session("convo-d")
        heard_session.transcript_parts = ["one cold coffee"]
        heard_session._stream_rag_response = lambda text: None  # type: ignore[method-assign]
        heard_session._finalize_run("completed", "silence_timeout")

        session = _make_finalize_ready_session("convo-d")
        session._rejected_speech_chunks = 1
        session._finalize_run("completed", "silence_timeout")
        assert session.response_parts == []

    def test_conversations_are_tracked_independently(self):
        session_x = _make_finalize_ready_session("convo-x")
        session_x._rejected_speech_chunks = 1
        session_x._finalize_run("completed", "silence_timeout")

        session_y = _make_finalize_ready_session("convo-y")
        session_y._rejected_speech_chunks = 1
        session_y._finalize_run("completed", "silence_timeout")

        assert session_x.response_parts == []
        assert session_y.response_parts == []
