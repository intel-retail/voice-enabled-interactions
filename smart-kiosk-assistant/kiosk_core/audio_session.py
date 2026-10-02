import logging
import math
import hashlib
import shutil
import tempfile
import threading
import time
import wave
import io
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
import re
from typing import Callable
from uuid import uuid4

import numpy as np
import sounddevice as sd

from kiosk_core import config, conversation_recorder
from kiosk_core.agent_client import AgentClient
from kiosk_core.analyzer_client import AnalyzerClient
from kiosk_core.realtime_analyzer_client import RealtimeAnalyzerClient, derive_realtime_ws_url
from kiosk_core.models import FileSessionStartRequest, SessionStartRequest
from kiosk_core.pipeline_latency import (
    AgentSpan, AsrSpan, GuardSpan, LlmSpan, McpSpan, PipelineLatencyStore,
    RetrievalSpan, TemplateSpan, TtsSpan, TurnTrace, WallTimes, pipeline_store,
)
from kiosk_core.rag_client import RagClient
from kiosk_core.tts_client import TtsClient


logger = logging.getLogger(__name__)

# ── Pre-synthesized opener cache ──────────────────────────────────────────
# Rendered at most once per (text, model, voice, language, instructions) for
# the whole process and reused by every session, so the opener never costs
# TTS time on the hot path. Guarded by a lock because concurrent sessions
# can race on the first turn after startup.
_opener_lock = threading.Lock()
_opener_cache: dict[tuple[str, str, str | None, str | None, str | None], Path | None] = {}


def _render_opener(
    tts_client: TtsClient,
    text: str,
    model: str,
    voice: str | None,
    language: str | None,
    instructions: str | None,
) -> Path | None:
    """Return a cached WAV of ``text``, synthesising it on first use.

    Args:
        tts_client:   Client used for the one-off synthesis.
        text:         Opener phrase. Must be non-committal — see
                      ``config.DEFAULT_OPENER_TEXT``.
        model:        TTS model name.
        voice:        TTS voice, or None for the service default.
        language:     TTS language, or None for the service default.
        instructions: Optional TTS style instructions.

    Returns:
        Path to the rendered WAV, or None when synthesis failed. A failure is
        cached as None so a broken TTS service cannot make every turn pay a
        failed round-trip.
    """
    key = (text, model, voice, language, instructions)
    with _opener_lock:
        if key in _opener_cache:
            return _opener_cache[key]

        cache_dir = Path(config.DEFAULT_OPENER_CACHE_DIR)
        if not cache_dir.is_absolute():
            cache_dir = Path(__file__).resolve().parent.parent / cache_dir
        # hashlib, not hash(): str hashing is salted per process (PYTHONHASHSEED),
        # so hash() would pick a new filename on every restart and re-synthesise
        # the opener instead of reusing the cached WAV on disk.
        digest = hashlib.sha256("\x00".join(str(k) for k in key).encode()).hexdigest()[:16]
        path = cache_dir / f"opener_{digest}.wav"

        if path.exists() and path.stat().st_size > 0:
            _opener_cache[key] = path
            return path

        try:
            t0 = time.monotonic()
            tts_client.synthesize_to_file(
                text=text,
                output_path=str(path),
                model=model,
                voice=voice,
                language=language,
                instructions=instructions,
            )
            logger.info(
                "[OPENER] Rendered %r -> %s in %.0f ms",
                text, path.name, (time.monotonic() - t0) * 1000,
            )
            _opener_cache[key] = path
        except Exception:
            logger.exception("[OPENER] Synthesis failed; opener disabled for this process")
            _opener_cache[key] = None
        return _opener_cache[key]


_SENTENCE_PATTERN = re.compile(r"^(.+?[.!?,:;](?:[\"')\]]+)?)(?:\s+|$)", re.DOTALL)
# Whisper hallucination tokens to strip from transcripts
_WHISPER_JUNK = re.compile(
    r"\[(?:BLANK_AUDIO|Music|Noise|Applause|Laughter|Silence|Background Music|noise|music)\]",
    re.IGNORECASE,
)

# ASR homophone normalization: "cart" (the shopping cart) is routinely
# mis-transcribed as "card" — same vowel sound, no acoustic distinction for
# Whisper. Observed live: "remove item in my card" reached the agent
# verbatim; "card" is a legitimately in-domain word (payment method), so no
# existing guard caught it, and the LLM improvised a "contact customer
# support" refusal instead of removing the item — a pure ASR-homophone
# hallucination, not a code bug reachable by any menu/order guard. This is
# corrected at the same layer as _WHISPER_JUNK: a deterministic transcript
# normalization before the text ever reaches the agent.
#
# Scoped narrowly to avoid corrupting genuine payment-card mentions:
#   - Only fires when "card" appears near an order-cart verb/phrase
#     (remove/delete/clear/empty/what's in/add ... to/in my card).
#   - Never fires if the utterance also contains a payment-context word
#     (pay, payment, credit, debit, swipe, tap, cash, upi) anywhere, since a
#     genuine "pay by card" / "swipe my card" must not be rewritten.
#   - Never fires for "gift card" / "loyalty card" / "membership card",
#     which are real nouns distinct from "cart".
_PAYMENT_CONTEXT_RE = re.compile(
    r"\b(?:pay|paying|payment|credit|debit|swipe|tap|paypal|upi|cash)\b",
    re.IGNORECASE,
)
_CARD_CART_HOMOPHONE_RE = re.compile(
    r"\b(?:remove|removing|delete|deleting|take out|taking out|clear|clearing|"
    r"empty|emptying|what'?s|whats|check|show|view)\b(?:\s+\S+){0,6}?\s+"
    r"(?<!gift\s)(?<!loyalty\s)(?<!membership\s)card\b"
    r"|\badd(?:ing)?\b(?:\s+\S+){0,8}?\s+to\s+(?:my\s+)?"
    r"(?<!gift\s)(?<!loyalty\s)(?<!membership\s)card\b"
    r"|\bin\s+my\s+(?<!gift\s)(?<!loyalty\s)(?<!membership\s)card\b",
    re.IGNORECASE,
)


def _normalize_card_cart_homophone(text: str) -> str:
    """Rewrite an order-context "card" mis-transcription to "cart".

    Args:
        text: Raw (already Whisper-junk-stripped) transcript text.

    Returns:
        ``text`` unchanged unless an order-cart phrase containing "card"
        is found and no payment-context word is present anywhere in the
        utterance, in which case the matched "card" occurrence(s) are
        rewritten to "cart".
    """
    if not text or _PAYMENT_CONTEXT_RE.search(text):
        return text

    def _swap(match: re.Match) -> str:
        return re.sub(r"\bcard\b", "cart", match.group(0), flags=re.IGNORECASE)

    return _CARD_CART_HOMOPHONE_RE.sub(_swap, text)


# Whisper emits a short stock phrase when handed near-silence. The PyTorch
# "openai" provider suppresses these via no_speech_prob/avg_logprob, but the
# OpenVINO GenAI provider exposes no confidence signal at all (its `scores`
# field is a degenerate 1.0), so a silent chunk transcribes as "you" or
# "Thank you." and would otherwise start a spurious agent turn. Matching the
# whole utterance keeps genuine speech containing these words intact.
_WHISPER_FILLER = re.compile(
    r"^\W*(?:you|thank you|thanks(?: for watching)?|bye|okay|ok|good|uh|um|"
    r"thank you\.? bye|please subscribe)\W*$",
    re.IGNORECASE,
)

_DEDUP_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Sentence splitter for speculative TTS pre-synthesis (see
# BaseAudioSession._prewarm_tts_from_draft). Deliberately simple — the exact
# split doesn't need to match the real streaming path's own sentence
# boundaries; a cache hit only requires the real _tts_worker to later
# synthesise a byte-identical sentence string, whatever split produced it.
_SPEC_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Normalizes whitespace/case/punctuation-only differences between a
# speculative draft's predicted sentence and the real turn's actual sentence
# so the _tts_cache lookup in _tts_worker isn't defeated by trivial
# formatting drift (double spaces, curly vs straight quotes, a trailing
# period the real reply omits, ₹ spacing, etc.) that doesn't change what is
# actually spoken. Deliberately NOT true fuzzy/edit-distance matching — two
# sentences that differ in WORDING (the dominant real-world case, since the
# same LLM prompt rarely regenerates byte-identical prose) still miss, by
# design: presynthesised audio for text B must never be played for text A.
#
# "." is deliberately kept out of the strip set here (unlike other
# punctuation) because a period between two digits is a DECIMAL POINT, not
# formatting drift -- ₹169.50 and ₹16950 are different amounts. A period
# that is NOT strictly between two digits (a sentence-ending "." included)
# is stripped separately below, by _CACHE_KEY_NON_DECIMAL_DOT_RE.
_CACHE_KEY_PUNCT_RE = re.compile(r"[^\w\s₹$€£.]", re.UNICODE)
# Matches a "." that is not sandwiched between two digits, i.e. every "."
# except a decimal point.
_CACHE_KEY_NON_DECIMAL_DOT_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")


def _normalize_sentence_for_cache_key(text: str) -> str:
    """Collapse whitespace/case/punctuation-only drift for _tts_cache keys."""
    normalized = _CACHE_KEY_PUNCT_RE.sub("", text.lower())
    normalized = _CACHE_KEY_NON_DECIMAL_DOT_RE.sub("", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


# ── Process-wide opener TTS cache ───────────────────────────────────────────
# Maps a (normalized sentence, model, voice, language, instructions) tuple to
# an already-synthesised WAV that has ALREADY been trimmed and gain-adjusted
# exactly as _tts_worker would. Shared across every session in the process so
# a stock opener ("Got it.") is synthesised once per kiosk boot rather than
# once per turn — see DEFAULT_TTS_OPENER_CACHE_ENABLED in config.py for why
# this is safe where speculative pre-synthesis is not.
#
# Entries are only ever written from a real turn's completed output, so this
# cache never causes a TTS request that would not otherwise have happened.
_OPENER_TTS_CACHE: dict[tuple, str] = {}
_OPENER_TTS_CACHE_LOCK = threading.Lock()
_OPENER_TTS_CACHE_DIR = Path(__file__).resolve().parent.parent / "generated_audio" / "_opener_cache"

# Sentences containing a digit are order-specific ("Your total is now ₹169.",
# "That's 2 burgers.") and, being short enough to pass the length filter, would
# otherwise be admitted. They can never be replayed -- the next order has a
# different total -- so each one permanently burns a slot, and a long kiosk
# uptime would fill the cache with dead total-variants and then stop admitting
# the genuine stock openers this cache exists for. Excluding digits keeps
# admission to phrasing that actually recurs ("Got it.", "Sure.").
_OPENER_VOLATILE_RE = re.compile(r"\d")


def _opener_cache_key(sentence: str, request) -> tuple | None:
    """Build an opener-cache key, or None if this sentence is not cacheable.

    The voice/model/language/instructions are part of the key because two
    sessions may legitimately request different voices; a clip synthesised for
    one must never be replayed for another.
    """
    if not config.DEFAULT_TTS_OPENER_CACHE_ENABLED:
        return None
    if len(sentence) > config.DEFAULT_TTS_OPENER_CACHE_MAX_CHARS:
        return None
    normalized = _normalize_sentence_for_cache_key(sentence)
    if not normalized:
        return None
    if _OPENER_VOLATILE_RE.search(sentence):
        return None
    return (
        normalized,
        request.tts_model,
        request.tts_voice,
        request.tts_language,
        request.tts_instructions,
    )


def _collapse_repeated_phrases(text: str) -> str:
    """Fold consecutive repeated word runs (window 1-8) to a single occurrence.

    Applied to a fixpoint so triples collapse too. Safety net for transcripts
    assembled from multiple ASR chunks where the same utterance can appear
    twice ("How are you today? How are you today?").
    """
    if len(text.split()) < 2:
        return text

    def _one_pass(words: list[str]) -> list[str]:
        result: list[str] = []
        i = 0
        while i < len(words):
            matched = False
            max_window = min(8, (len(words) - i) // 2)
            for w in range(max_window, 0, -1):
                if words[i:i + w] == words[i + w:i + 2 * w]:
                    result.extend(words[i:i + w])
                    i += 2 * w
                    matched = True
                    break
            if not matched:
                result.append(words[i])
                i += 1
        return result

    current = text.split()
    for _ in range(8):
        collapsed = _one_pass(current)
        if collapsed == current:
            break
        current = collapsed
    return " ".join(current)


# ── Adaptive endpoint: does the transcript sound finished? ──────────────────
# A word that ends a transcript but cannot end a sentence. If the customer
# stopped on one of these they are mid-thought, so the turn must keep the full
# silence_timeout_seconds instead of committing early.
_INCOMPLETE_TAIL_WORDS = frozenset(
    # hesitation sounds
    "um uh umm uhh er ah hmm erm".split()
    # contractions that open a clause: "I'd like..." cut after "I'd"
    + "i've i'd i'll i'm we've we'd we'll you've you'd you'll they've they'd "
      "they'll wanna gonna gimme lemme".split()
    # conjunctions, articles and determiners
    + "and or but so like the a an to of my your our their some any those "
      "these that this".split()
    # auxiliaries and pronouns
    + "do does did is are was were can could would will have has had get got "
      "i we you they".split()
    # prepositions that must take an object: "with...", "a burger and fries for..."
    + "for with in on at by from about without".split()
)

_ENDPOINT_WORD_RE = re.compile(r"[^a-z' ]")


def _looks_complete(text: str, min_words: int) -> bool:
    """True when a transcript reads like a finished utterance.

    Used only to SHORTEN the end-of-turn wait, never to extend it, so every
    uncertain case must return False. An empty or not-yet-arrived transcript
    therefore fails closed and the caller keeps the full silence timeout.

    Args:
        text: The transcript assembled so far this turn.
        min_words: Fewest words that may be judged complete.

    Returns:
        Whether the turn may be ended early.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    # A trailing comma is an explicit continuation, whatever the words are.
    if raw.endswith(","):
        return False
    words = _ENDPOINT_WORD_RE.sub("", raw.lower()).split()
    if len(words) < min_words:
        return False
    # Whisper punctuates fragments with "?" and ".", so punctuation cannot
    # rescue a dangling word — judge on the final word itself.
    return words[-1] not in _INCOMPLETE_TAIL_WORDS


def _endpoint_stability_key(text: str) -> str:
    """Normalize a candidate transcript for the endpoint stability check.

    _endpoint_transcript_stable() must only reset its confirmation timer when
    the WORDS actually change, not on punctuation-only noise. Whisper
    frequently hallucinates a stray trailing "." (or repeats one) while
    decoding the customer's own trailing silence tail -- e.g. "...please."
    becoming "...please. ." a tick later with no new word spoken. Compared as
    raw strings that reads as "the transcript changed", which restarts the
    stability window right when it should be confirming, and is exactly why
    the shortcut's measured firing rate swings so widely run to run (see
    _endpoint_transcript_stable's docstring). Reusing _looks_complete's own
    word-extraction regex means this can never treat a genuine new word as
    unchanged -- it only ignores differences _looks_complete itself already
    ignores.
    """
    return " ".join(_ENDPOINT_WORD_RE.sub("", (text or "").strip().lower()).split())


def _assemble_transcript(parts: list[str]) -> str:
    """Join transcript chunk parts, dropping consecutive duplicates.

    A chunk's text that repeats the previous kept part (case/punctuation-
    insensitive) is skipped before joining, then any residual repeated phrase
    inside the joined string is collapsed. This is the single assembly point
    for the user-visible transcript, so it guards against duplicate text no
    matter which upstream ASR path produced it.
    """
    deduped: list[str] = []
    prev_norm: str | None = None
    for part in parts:
        if not part:
            continue
        norm = " ".join(_DEDUP_PUNCT_RE.sub(" ", part.lower()).split())
        if norm and norm == prev_norm:
            continue
        deduped.append(part)
        prev_norm = norm
    return _collapse_repeated_phrases(" ".join(deduped).strip())


# Domain vocabulary for the semantic fallback in _filter_target_speaker.
# When the primary customer is silent for an entire chunk this set is used
# to decide whether a background speaker said something kiosk-relevant enough
# to warrant re-assigning the primary (e.g. a new customer stepped up).
_DOMAIN_KEYWORDS: frozenset[str] = frozenset({
    "order", "orders", "ordering", "menu", "item", "items",
    "burger", "pizza", "sandwich", "wrap", "salad", "combo",
    "fries", "drink", "water", "coffee", "tea", "juice", "soda",
    "price", "cost", "how much", "pay", "payment", "card", "cash",
    "checkout", "bill", "receipt", "change",
    "ticket", "seat", "flight", "hotel", "book", "booking", "reserve",
    "help", "assist", "please", "want", "need", "like", "get",
})

# ── Consecutive speaker-rejection tracking (cross-turn, per conversation) ──
# ``_rejected_speech_chunks`` on a session instance only counts chunks
# rejected WITHIN that one turn. A BaseAudioSession is created fresh per
# voice turn, so distinguishing "this is the first rejected turn" from "this
# conversation has been rejected several turns running" needs state that
# outlives a single instance, keyed by the persistent ``agent_session_id``
# (see its docstring in __init__) shared across every turn of one
# conversation. See config.DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD for the
# rationale on why a streak, not a single rejection, gates the retry prompt.
_consecutive_rejections_lock = threading.Lock()
_consecutive_rejections: dict[str, int] = {}
# Simple bound so a long-running kiosk process doesn't accumulate one entry
# per conversation forever. A kiosk lane never has anywhere near this many
# conversations in flight at once, so clearing on overflow only ever discards
# stale entries from finished conversations.
_MAX_TRACKED_CONVERSATIONS = 500


def _note_conversation_rejection(agent_session_id: str) -> int:
    """Record a rejected turn for ``agent_session_id`` and return the streak.

    Args:
        agent_session_id: The persistent conversation identifier shared
            across every voice turn of one customer's session.

    Returns:
        The number of consecutive rejected turns recorded so far for this
        conversation, including this one.
    """
    with _consecutive_rejections_lock:
        if (
            len(_consecutive_rejections) > _MAX_TRACKED_CONVERSATIONS
            and agent_session_id not in _consecutive_rejections
        ):
            _consecutive_rejections.clear()
        count = _consecutive_rejections.get(agent_session_id, 0) + 1
        _consecutive_rejections[agent_session_id] = count
        return count


def _reset_conversation_rejections(agent_session_id: str) -> None:
    """Clear the rejection streak for ``agent_session_id``.

    Called whenever a turn produces a real, accepted transcript — the
    customer was successfully heard, so any earlier rejection streak no
    longer says anything about whether they are being ignored now.
    """
    with _consecutive_rejections_lock:
        _consecutive_rejections.pop(agent_session_id, None)


def reset_all_rejection_tracking() -> None:
    """Drop all tracked rejection streaks.

    Test-only entry point — prevents state from one test leaking into the
    next when several tests reuse the same default conversation id.
    """
    with _consecutive_rejections_lock:
        _consecutive_rejections.clear()


class BaseAudioSession:
    # Conversation scopes known to have an enrolled reference voice in the
    # audio-analyzer. See DEFAULT_DIARIZATION_ENROLLMENT_PRIMING_ENABLED.
    _enrolled_scopes: set[str] = set()
    _enrolled_scopes_lock = threading.Lock()

    @property
    def _streaming_active(self) -> bool:
        """Whether continuous ASR streaming is live for this session.

        Backed by __streaming_active_flag (set once at connect time) AND a
        liveness check on realtime_client -- a WS that dies mid-session
        (server restart, network blip) previously left __streaming_active_flag
        stuck True forever, so every later send_frame/preview/commit call
        silently no-op'd (RuntimeError on a closed asyncio loop, swallowed)
        and the session was stuck replaying one stale/frozen transcript with
        no fallback. Now this flips False the instant the client reports it
        died, so callers fall back to the file-based AnalyzerClient path the
        same way a failed initial connect is already handled.
        """
        if not self.__streaming_active_flag:
            return False
        if self.realtime_client is None or not self.realtime_client.is_alive():
            if self.__streaming_active_flag:
                logger.warning(
                    "[SESSION] session=%s | realtime analyzer connection lost; "
                    "falling back to file-based ASR for the rest of this session",
                    self.session_id,
                )
            self.__streaming_active_flag = False
            return False
        return True

    @_streaming_active.setter
    def _streaming_active(self, value: bool) -> None:
        self.__streaming_active_flag = value

    def __init__(
        self,
        request: SessionStartRequest,
        on_complete: Callable[[str], None] | None = None,
    ):
        self.session_id = str(uuid4())
        # Persistent agent session ID — reused across all voice turns in the same
        # conversation so the ADK agent retains order state between mic presses.
        # Falls back to the audio session UUID if no conversation_id was supplied.
        self.agent_session_id: str = request.conversation_id or self.session_id
        self.request = request
        self.on_complete = on_complete
        self.client = AnalyzerClient(request.analyzer_url)
        self.rag_client = RagClient(request.rag_url)
        self.tts_client = TtsClient(request.tts_url)
        # Agent client is used when the ordering feature is enabled.
        # All turns go through the agent — it decides Q&A vs ordering.
        if config.ORDERING_ENABLED:
            agent_url = getattr(request, "agent_url", None) or config.DEFAULT_AGENT_URL
            self.agent_client: AgentClient | None = AgentClient(agent_url)
            logger.info("[SESSION] Agent routing enabled → %s", agent_url)
        else:
            self.agent_client = None
        self.created_at = datetime.now(UTC)
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.status = "created"
        self.end_reason: str | None = None
        self.error: str | None = None
        self.transcript_parts: list[str] = []
        self.response_parts: list[str] = []
        self.tts_audio_segments: list[dict[str, object]] = []
        self.tts_errors: list[str] = []
        # Segments can finish synthesis out of order when
        # config.DEFAULT_TTS_WORKER_CONCURRENCY > 1 (e.g. a short sentence 2
        # finishing before a longer sentence 1). tts_audio_segments must still
        # be exposed to clients in index order — gradio_app.py queues newly
        # appended segments for playback in list-append order, so an
        # out-of-order append would play sentence 2 before sentence 1.
        # _tts_pending_publish holds finished-but-not-yet-publishable segments
        # until every lower index has already been published.
        self._tts_next_publish_index = 1
        self._tts_pending_publish: dict[int, dict[str, object]] = {}
        self.stop_requested_at: datetime | None = None

        # ── Primary-speaker lock-on ────────────────────────────────────────────
        # The audio-analyzer sets is_primary when it has an enrolled reference
        # voice for the conversation (see speaker_scope_id). kiosk-core honours
        # that flag when present and otherwise falls back to locking onto the
        # first speaker label seen in the session and treating all subsequent
        # segments from that label as the customer (primary).
        # Segments from any different label are unconditionally dropped — the
        # semantic fallback is only used before the primary is established.
        self._primary_speaker_id: str | None = None
        # Number of chunks in this turn that carried real transcribed speech
        # which the speaker filter then rejected. Distinguishes "nobody said
        # anything" from "somebody spoke and we discarded all of it", so
        # _finalize_run can ask the customer to repeat instead of replying with
        # a generic greeting that hides the rejection.
        self._rejected_speech_chunks: int = 0
        # Analyzer's own session_id — passed on every chunk so per-session
        # state (e.g. pyannote enrolled speaker embedding) persists across
        # the many chunked HTTP requests made in this kiosk session.
        # Initialised from our own session_id; the analyzer echoes/normalises
        # it via the X-Session-ID response header, which we then reuse.
        self._analyzer_session_id: str = self.session_id

        # ── Continuous ASR streaming (optional, feature-flagged) ────────────
        # When enabled, frames are streamed continuously to the analyzer over
        # a persistent WebSocket (see kiosk_core/realtime_analyzer_client.py)
        # instead of being POSTed as WAV files per flush. Falls back to the
        # file-based AnalyzerClient (self.client, above) for this session if
        # the socket fails to connect — never fails the turn.
        self.realtime_client: RealtimeAnalyzerClient | None = None
        self.__streaming_active_flag = False
        # Timer for the non-destructive preview ping cadence (independent of
        # chunk_frames -- see the streaming preview-flush block).
        self._last_preview_ping_at: float = 0.0
        if config.DEFAULT_ANALYZER_STREAMING_ENABLED:
            try:
                self.realtime_client = RealtimeAnalyzerClient(
                    ws_url=derive_realtime_ws_url(request.analyzer_url),
                    session_id=self._analyzer_session_id,
                    speaker_scope_id=self.agent_session_id,
                    diarization=config.DEFAULT_DIARIZATION_ENABLED,
                    language=request.language,
                )
                self.realtime_client.start(
                    timeout=config.DEFAULT_REALTIME_CONNECT_TIMEOUT_SECONDS
                )
                self.__streaming_active_flag = True
                logger.info(
                    "[SESSION] session=%s | continuous ASR streaming ENABLED", self.session_id
                )
            except Exception:
                logger.exception(
                    "[SESSION] session=%s | failed to open realtime analyzer stream; "
                    "falling back to file-based ASR for this session",
                    self.session_id,
                )
                self.realtime_client = None
                self.__streaming_active_flag = False

        # Highest segment end-time already consumed from the analyzer. The
        # analyzer runs in append_to_session mode (session_id is reused) and
        # returns the cumulative segment list on every chunk with offset-
        # adjusted timestamps; without this cursor kiosk-core would re-append
        # every prior primary-speaker segment on each subsequent chunk,
        # duplicating utterances in transcript_parts.
        #
        # Advanced by this chunk's OWN measured duration after every flush —
        # not by the response's reported segment end times — because a chunk
        # flushed without diarization (segments=[], see
        # DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED) would otherwise leave this
        # cursor stale until the next diarized chunk, which then sees the
        # analyzer's full cumulative segment list as entirely "fresh" and
        # re-appends everything already committed via the flat-text path.
        # Client duration and the analyzer's own cumulative timeline advance
        # in lockstep by construction (same audio bytes sent either way), so
        # this is a safe, response-shape-independent substitute.
        self._last_analyzer_segment_end: float = 0.0
        # Same problem, same fix, for the OTHER response shape: when a chunk
        # is flushed WITHOUT diarization (the normal case for every
        # intermediate chunk — see DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED),
        # the analyzer has no per-segment timestamps to offset-dedup by, so
        # its flat "text" field is simply the ENTIRE session's transcript so
        # far (verified in audio-analyzer's pipeline.py: session_state["text"]
        # is prefixed onto every response). Track the last cumulative flat
        # text we've already consumed so only the new suffix is appended.
        self._last_cumulative_flat_text: str = ""
        # ───────────────────────────────────────────────────────────────────────

        # ── Speculative drafting (cache-warm + TTS pre-synth) ───────────────
        # See config.DEFAULT_SPECULATIVE_DRAFT_ENABLED for the full rationale.
        # A background thread runs a dry-run (never-persists) agent turn on
        # each preview-ASR transcript snapshot while the customer is still
        # talking, so the LLM/tool prefix cache and TTS are already warm by
        # the time the real endpoint fires. "Newest-wins": only the result of
        # the LATEST-STARTED draft is ever kept — an older one that happens to
        # finish after a newer one started is discarded, since a longer/
        # corrected transcript update makes it stale.
        self._speculative_lock = threading.Lock()
        self._speculative_generation: int = 0
        self._speculative_draft: dict | None = None  # {"transcript": str, "result": dict}
        # Sentence-text -> already-synthesised WAV path, populated by a
        # speculative draft's TTS pre-synthesis. _tts_worker consults this
        # before calling TTS for real — a hit is only possible when the real,
        # fully-guarded reply produces an IDENTICAL sentence, so nothing is
        # ever spoken that the real pipeline didn't itself decide to say.
        self._tts_cache: dict[str, str] = {}
        self._spec_tts_index = 0  # unique filename counter for cached synth files

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._audio_queue: Queue[np.ndarray] = Queue()
        self._thread = threading.Thread(target=self._run, name=f"mic-session-{self.session_id}", daemon=True)
        # ── ASR chunk-flush worker ──────────────────────────────────────────
        # Chunk transcription is a blocking HTTP round-trip to audio-analyzer
        # (whisper-small on CPU: ~1.5-4.7s depending on chunk duration). It
        # used to run inline in the frame-reading loop (_process_frame_stream),
        # which meant a mid-utterance chunk flush blocked silence-timeout
        # detection and frame ingestion for its entire duration — that
        # latency then serialised on top of the final tail-chunk flush,
        # roughly doubling the delay the customer felt after they stopped
        # talking (observed live: ~5-6s total instead of ~1-1.5s for
        # utterances long enough to trigger a mid-stream flush).
        #
        # A single dedicated worker thread now owns every _flush_chunk() call
        # instead. The main frame-reading loop just enqueues each chunk's
        # frames and keeps consuming new audio / evaluating silence in real
        # time; the worker drains the queue FIFO (one HTTP call at a time,
        # matching audio-analyzer's own one-request-at-a-time model cache),
        # which preserves both transcript_parts ordering and the analyzer's
        # cumulative-segment cursor (_last_analyzer_segment_end) exactly as
        # before. Only the *final* chunk's flush latency remains in the
        # customer-perceived critical path — _finalize_run explicitly waits
        # for the queue to drain (see the join() in _process_frame_stream)
        # before reading the completed transcript.
        self._flush_queue: Queue[list[np.ndarray] | None] = Queue()
        self._flush_thread = threading.Thread(
            target=self._flush_worker, name=f"asr-flush-{self.session_id}", daemon=True,
        )
        self._speech_started = False
        self._captured_samples = 0
        self._source_kind = "audio"
        # True once any speech frame has landed in the *current* chunk_frames
        # buffer since it was last cleared by a flush. Reset alongside
        # chunk_frames on every flush (timed cap, adaptive pause, or final).
        # Used only when KIOSK_CORE_SKIP_EMPTY_FINAL_FLUSH_ENABLED is set —
        # see config.py for the full rationale.
        self._chunk_has_speech = False
        # True whenever speech has been captured this turn that has NOT yet
        # been confirmed transcribed by a successful (non-empty) ASR flush.
        # Unlike _chunk_has_speech (which resets the instant a flush is
        # QUEUED, regardless of what comes back), this only clears once
        # _flush_chunk actually appends real text to transcript_parts — see
        # its assignment there. This exists because _chunk_has_speech alone
        # was unsafe to skip the final flush on: a chunk can be enqueued
        # (resetting _chunk_has_speech) and then have its ASR call come back
        # empty (e.g. a transient audio-analyzer miss on real speech) with no
        # later chunk to catch it, silently losing the whole utterance.
        # Reproduced live on rec1_16k.wav (2026-09-15): a chunk-size-cap
        # flush containing the entire order came back 0-segment, the final
        # flush was then skipped because _chunk_has_speech read False, and
        # the turn ended with an empty transcript. This flag makes the skip
        # safe: it can only fire once real content has actually landed.
        self._unconfirmed_speech_pending = False
        # Count of turns where the final tail-chunk flush was skipped because
        # it held no unflushed speech, purely for observability/metrics.
        self._final_flush_skipped = False

        # ── Adaptive VAD state ─────────────────────────────────────────────
        # `request.silence_threshold` is the seed/fallback gate. When adaptive
        # VAD is enabled the effective gate (`_vad_threshold`) is re-derived
        # from the measured noise floor after the calibration window; until
        # then the seed value is used, so behaviour is unchanged if
        # calibration never completes (e.g. a very short recording).
        self._vad_threshold: float = float(self.request.silence_threshold)
        self._noise_floor: float | None = None
        self._vad_calibrating: bool = config.ADAPTIVE_VAD_ENABLED
        self._vad_calibration_rms: list[float] = []
        # ───────────────────────────────────────────────────────────────────

        # ── Pipeline timing (monotonic clock) ──────────────────────────────────
        # All _t_* fields are set during _finalize_run / _stream_rag_response.
        # Using time.monotonic() for accurate durations; datetime only for display.
        self._t_capture_start: float | None = None  # first speech frame detected
        # File-replay only: monotonic instant the fixture began streaming
        # (set at the very first line of FileAudioSession._run(), before any
        # frame is read). Ground-truth anchor for benchmark v2v — pairs with
        # an offline, VAD-independent measurement of where the fixture's real
        # speech ends (see tests/benchmarks/v2v_fixture_benchmark.py). None
        # for microphone/browser-stream sessions.
        self._t_playback_start: float | None = None
        self._asr_ms_total: float = 0.0             # summed transcribe_file time
        self._asr_chunks: int = 0                   # number of transcribe calls
        self._t_turn_start: float | None = None     # start of _finalize_run
        # True instant the customer stopped talking (endpoint commit time
        # minus the trailing-silence wait). Set in _log_last_word_spoken.
        # This predates _t_turn_start not just by the silence wait but also
        # by the final chunk's ASR round-trip (_flush_queue.join() blocks
        # _finalize_run until that completes) — using this instead of
        # reconstructing "t0 - endpoint_wait_seconds" fixes a real undercount
        # in voice-to-voice latency that previously ignored that round-trip.
        self._t_last_word: float | None = None
        # Directly-observed instant of the LAST audio frame actually
        # containing speech that kiosk-core sent to audio-analyzer (updated
        # every time — see the send_frame() call sites in
        # _process_frame_stream — so only the most recent one survives).
        # _t_last_word above is *derived*: "now minus silence_run_seconds" at
        # the moment the endpoint fires, which assumes the frame-processing
        # loop's own silence_run_seconds counter tracks real elapsed time
        # exactly — true only if frames are read at a perfectly steady
        # cadence. Bursty delivery (GC pause, scheduler jitter, a slow VAD
        # inference tick) can make that counter run ahead of or behind the
        # wall clock, silently skewing every v2v number derived from it. This
        # field is not derived from anything — it is the wall-clock instant
        # send_frame() was actually called for the last speech-containing
        # frame — so it is preferred over the derived value wherever both
        # exist (see _log_last_word_spoken).
        self._t_last_speech_frame_sent: float | None = None
        self._t_agent_start: float | None = None    # just before agent HTTP call
        self._t_agent_end: float | None = None      # agent reply received
        # End of the token stream / trailing-text handling — before
        # _stop_tts_workers() runs in the turn's finally block. Isolates the
        # real agent+tool round-trip from the TTS-synthesis drain that
        # follows it, which _t_turn_end (below) otherwise bundles in.
        self._t_agent_stream_end: float | None = None
        self._t_first_tts: float | None = None      # first TTS sentence queued
        # First audio the customer could actually hear, stamped when the audio
        # EXISTS (opener file copied, or first synthesized segment written to
        # disk) — never when a sentence is merely handed to the TTS worker.
        # Queuing a sentence makes no sound: with the opener disabled this used
        # to fall back to the queue stamp and under-reported voice-to-voice
        # latency by a whole TTS round-trip (~300ms measured).
        self._t_first_audio: float | None = None
        # Trailing silence the endpoint waited through before committing the
        # turn. Needed to report voice-to-voice latency, because every other
        # timestamp in the trace starts after this wait has already elapsed.
        self._endpoint_wait_seconds: float | None = None
        # How long the mandatory drain-and-join after the endpoint decision
        # (self._flush_queue.join(), a few lines above _finalize_run) actually
        # blocked for — i.e. the final chunk's real ASR round-trip. See the
        # comment on _t_last_word above: this is the gap between _t_last_word
        # and _t_turn_start that endpoint_wait_seconds does not cover. Exposed
        # as its own field so voice_to_voice_ms is reconstructable from parts
        # (endpoint_wait_ms + final_flush_wait_ms + time_to_first_audio_ms)
        # instead of silently vanishing into an unexplained gap.
        self._final_flush_wait_seconds: float | None = None
        # Monotonic instant the final-flush drain-and-join sequence began
        # (right after the endpoint decision / mic-release signal was
        # processed). Paired with _t_last_word to compute
        # WallTimes.post_speech_gap_ms -- see that field's docstring.
        self._t_final_flush_start: float | None = None
        # Whether THIS turn committed via the sentence-completeness shortcut
        # (DEFAULT_ENDPOINT_SHORT_SECONDS path) rather than the full
        # silence_timeout_seconds fallback. None until the turn ends via one
        # of the two silence-based commit branches (stays None for
        # "stopped_by_api"/"max_duration_reached"/etc., where neither ran).
        # Added to answer, empirically per-turn rather than by inference from
        # endpoint_wait_ms alone, "is the adaptive shortcut actually firing,
        # or is ASR too slow for it to ever get a chance?" — see
        # docs/performance-improvements-2026-09.md.
        self._endpoint_shortcut_fired: bool | None = None
        # Endpoint completeness stability tracking (see
        # config.DEFAULT_ENDPOINT_STABLE_SECONDS): the transcript text last
        # seen by the completeness shortcut, and when it started reading
        # that way without changing.
        self._endpoint_stable_transcript: str | None = None
        self._endpoint_stable_since: float | None = None
        # First segment carrying the answer, stamped when its WAV is on disk.
        # _t_first_tts marks when a sentence was queued for synthesis, which is
        # ~one TTS call earlier and would understate voice-to-voice latency.
        self._t_first_answer_audio: float | None = None
        self._t_last_tts: float | None = None       # last TTS segment written (in worker thread)
        self._t_turn_end: float | None = None       # after worker.join()
        self._tts_segment_count: int = 0
        # ───────────────────────────────────────────────────────────────────────

        self._frame_samples = max(1, int(self.request.sample_rate * config.DEFAULT_BLOCK_DURATION_SECONDS))
        self._frame_duration_seconds = self._frame_samples / self.request.sample_rate
        self._vad_calibration_frames = max(
            1, int(config.DEFAULT_VAD_CALIBRATION_SECONDS / self._frame_duration_seconds)
        )
        # Preroll must outlast the calibration window: calibration frames are
        # classified as non-speech (they are what defines "non-speech"), so they
        # land in the preroll deque. If the deque were shorter than the window a
        # customer who starts talking immediately would lose their opening word.
        preroll_seconds = config.DEFAULT_PREROLL_SECONDS
        if config.ADAPTIVE_VAD_ENABLED:
            preroll_seconds = max(preroll_seconds, config.DEFAULT_VAD_CALIBRATION_SECONDS + 0.2)
        preroll_frames = max(1, int(preroll_seconds / self._frame_duration_seconds))
        self._preroll_frames: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self._session_output_dir = Path(__file__).resolve().parent.parent / "generated_audio" / self.session_id

        # Silero VAD (optional, feature-flagged — see config.KIOSK_CORE_SILERO_VAD_ENABLED).
        # Only constructed when enabled: it loads an onnxruntime session, which
        # is unnecessary overhead for the default RMS-VAD path.
        self._silero_vad = None
        if config.KIOSK_CORE_SILERO_VAD_ENABLED:
            try:
                from kiosk_core.silero_vad import SileroVAD

                self._silero_vad = SileroVAD(
                    config.DEFAULT_SILERO_VAD_MODEL_PATH,
                    sample_rate=self.request.sample_rate,
                    intra_op_threads=config.DEFAULT_SILERO_VAD_INTRA_OP_THREADS,
                )
            except ValueError as exc:
                # Expected, not exceptional: the session's sample rate isn't
                # one Silero supports (e.g. 24kHz browser/Kokoro audio). Log
                # concisely and use the rate-agnostic RMS VAD instead.
                logger.warning(
                    "session=%s | Silero VAD unavailable (%s); using RMS VAD",
                    self.session_id,
                    exc,
                )
                self._silero_vad = None
            except Exception:
                # Fail open: fall back to the RMS VAD rather than breaking the
                # session if the model file/onnxruntime isn't available.
                logger.exception(
                    "session=%s | Silero VAD enabled but failed to initialize; falling back to RMS VAD",
                    self.session_id,
                )
                self._silero_vad = None

    def start(self) -> None:
        with self._lock:
            if self.status != "created":
                raise ValueError("Session already started")
            self.status = "running"
            self.started_at = datetime.now(UTC)
        self._flush_thread.start()
        self._thread.start()

    def stop(self, reason: str = "stopped_by_api") -> None:
        with self._lock:
            if self.status not in {"running", "stopping"}:
                raise ValueError(f"Session is not running: {self.status}")
            self.status = "stopping"
            self.end_reason = reason
            self.stop_requested_at = datetime.now(UTC)
        self._stop_event.set()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            transcript = _assemble_transcript(self.transcript_parts)
            response_text = "".join(self.response_parts).strip()
            return {
                "session_id": self.session_id,
                "source_kind": self._source_kind,
                "status": self.status,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "stop_requested_at": self.stop_requested_at.isoformat() if self.stop_requested_at else None,
                "end_reason": self.end_reason,
                "error": self.error,
                "speech_started": self._speech_started,
                "noise_floor_rms": round(self._noise_floor, 1) if self._noise_floor is not None else None,
                "vad_threshold": round(self._vad_threshold, 1),
                "vad_calibrated": self._noise_floor is not None,
                "captured_audio_seconds": round(self._captured_samples / self.request.sample_rate, 3),
                "transcript": transcript,
                "partial_transcript": transcript,
                "transcript_parts": list(self.transcript_parts),
                "response": response_text,
                "response_parts": list(self.response_parts),
                "tts_audio_segments": [dict(segment) for segment in self.tts_audio_segments],
                "tts_errors": list(self.tts_errors),
            }

    def _run(self) -> None:
        raise NotImplementedError

    def _process_frame_stream(self, frame_iterator) -> tuple[str, str | None]:
        chunk_frames: list[np.ndarray] = []
        silence_run_seconds = 0.0
        final_status = "completed"
        end_reason = self.end_reason or "completed"

        # Adaptive flush threshold: flush accumulated speech to the background
        # ASR worker as soon as a natural pause appears, so the tail chunk at
        # true endpoint (silence_timeout_seconds) is as short as possible.
        # Only fires when the chunk has genuine speech content (avoids flushing
        # near-empty buffers) and when adaptive_flush_pause_seconds > 0.
        adaptive_pause = self.request.adaptive_flush_pause_seconds
        _adaptive_flushed = False  # guard: only one adaptive flush per silence run

        try:
            for frame in frame_iterator:
                if self._stop_event.is_set():
                    break

                rms = self._rms(frame)
                is_speech = rms >= self._vad_threshold
                # Refine the gate from the measured floor, then re-classify:
                # during the calibration window the seed threshold is still in
                # force, so the first frames after calibration must be judged
                # by the newly derived gate rather than the stale seed.
                # Still computed/updated even when Silero VAD is active: it's
                # cheap and keeps the RMS gate ready as an instant fallback.
                self._update_vad_threshold(rms, is_speech)
                if self._vad_calibrating:
                    # Still measuring the room — hold the frame in preroll
                    # rather than committing to a speech/silence decision.
                    is_speech = False
                else:
                    is_speech = rms >= self._vad_threshold

                if self._silero_vad is not None:
                    # Model-based speech probability replaces the RMS decision
                    # while feature-flagged on. int16-scale frame -> normalized
                    # float32 [-1, 1], as the Silero graph expects.
                    normalized = frame.astype(np.float32) / 32768.0
                    speech_prob = self._silero_vad.prob(normalized)
                    is_speech = speech_prob >= config.DEFAULT_SILERO_VAD_THRESHOLD

                if not self._speech_started:
                    if is_speech:
                        self._speech_started = True
                        # Set here too, not just in the is_speech branch
                        # below: this branch `continue`s before ever reaching
                        # that one, so a short utterance whose speech never
                        # extends past this first frame would otherwise leave
                        # _chunk_has_speech False and get silently dropped by
                        # the empty-final-flush skip.
                        self._chunk_has_speech = True
                        self._unconfirmed_speech_pending = True
                        if self._t_capture_start is None:
                            self._t_capture_start = time.monotonic()
                        while self._preroll_frames:
                            buffered = self._preroll_frames.popleft()
                            chunk_frames.append(buffered)
                            self._captured_samples += len(buffered)
                            if self._streaming_active:
                                self.realtime_client.send_frame(buffered.tobytes())
                        chunk_frames.append(frame)
                        self._captured_samples += len(frame)
                        if self._streaming_active:
                            self.realtime_client.send_frame(frame.tobytes())
                        # Every frame reaching this branch is speech by
                        # construction (the preroll-drain loop above only
                        # ever buffered frames while still deciding, and this
                        # one just triggered is_speech=True itself).
                        self._t_last_speech_frame_sent = time.monotonic()
                    else:
                        self._preroll_frames.append(frame)
                    continue

                chunk_frames.append(frame)
                self._captured_samples += len(frame)
                if self._streaming_active:
                    self.realtime_client.send_frame(frame.tobytes())
                    if is_speech:
                        # Overwritten every time -- only the LAST speech
                        # frame's send instant survives by the time the
                        # endpoint fires. See the field's docstring in
                        # __init__ for why this is preferred over deriving
                        # the same instant from silence_run_seconds.
                        self._t_last_speech_frame_sent = time.monotonic()

                if is_speech:
                    silence_run_seconds = 0.0
                    _adaptive_flushed = False  # new speech: allow adaptive flush again
                    self._chunk_has_speech = True
                    self._unconfirmed_speech_pending = True
                    # New speech resets the silence-domain clock to 0, so any
                    # stability state left over from a PRIOR silence run must
                    # be cleared too -- otherwise the next silence run's
                    # _endpoint_transcript_stable() call would diff its fresh
                    # (small) silence_run_seconds against a stale, much larger
                    # _endpoint_stable_since from before, and could never
                    # read stable again this turn.
                    self._endpoint_stable_transcript = None
                    self._endpoint_stable_since = None
                else:
                    silence_run_seconds += self._frame_duration_seconds
                    # Fire an immediate preview the instant speech is believed
                    # to end, instead of waiting for the next scheduled
                    # DEFAULT_REALTIME_PREVIEW_COMMIT_SECONDS (0.4s) tick
                    # below. Previously the buffer covering the customer's
                    # actual last word only got re-transcribed on whatever
                    # tick happened to land next -- up to 0.4s of pure
                    # scheduling lag before ASR even started on the complete
                    # utterance (measured: this was the largest single
                    # contributor to asr_last_word_to_transcript_ms). Gated
                    # the same way as the periodic ping (silence_run_seconds
                    # == exactly one frame means this is the FIRST silence
                    # frame this run) so it never fires more than once per
                    # utterance boundary.
                    if (
                        self._streaming_active
                        and config.DEFAULT_PREVIEW_FLUSH_ENABLED
                        and self._chunk_has_speech
                        and silence_run_seconds == self._frame_duration_seconds
                    ):
                        self.realtime_client.preview()
                        self._last_preview_ping_at = time.monotonic()

                # ── Adaptive pause flush ────────────────────────────────────
                # When the speaker pauses for adaptive_flush_pause_seconds
                # (default 0.70s) — but hasn't reached the endpoint yet —
                # flush the current speech to the background worker now so ASR
                # starts immediately. The tail chunk at endpoint will then
                # contain only silence frames (effectively empty), keeping
                # critical-path ASR cost near-zero.
                # Only fire once per silence run; reset when speech resumes.
                # Minimum 0.5s chunk: Whisper has a fixed per-call overhead
                # (~1.2s on CPU, ~150ms on GPU) that dominates sub-0.5s inputs
                # — sending near-empty frames wastes more time than it saves.
                #
                # Checked BEFORE the timed chunk-size cap below: both
                # thresholds can be crossed on the same frame (adaptive_pause
                # is often close to chunk_seconds), and when that happens the
                # genuine end-of-speech pause must win. Losing that race to
                # the cap means the real content only gets flushed once the
                # cap's full duration fills — which can land at (or after) the
                # same instant the endpoint timer also fires, leaving the
                # completeness shortcut below with no transcribed content to
                # judge and no choice but to fall through to the full
                # silence_timeout_seconds wait. See docs/performance-
                # improvements-2026-09.md for the measured turn this fixes.
                if (
                    adaptive_pause > 0
                    and not _adaptive_flushed
                    and silence_run_seconds >= adaptive_pause
                    and silence_run_seconds < self.request.silence_timeout_seconds
                ):
                    if not self._chunk_has_speech:
                        # chunk_frames holds only silence: every frame in it
                        # arrived after the LAST flush cleared the buffer, and
                        # no is_speech frame has landed since (chunk_has_speech
                        # is reset on every flush). This happens when an
                        # earlier duration-triggered flush (preview flush or
                        # the chunk-size cap) already cleared the buffer at or
                        # after the true end of speech — measured case: the
                        # preview flush's 1.5s boundary landed 0.59s after the
                        # customer's last word, so the adaptive-pause flush
                        # then had nothing but silence to send and still had
                        # to wait out the 0.5s minimum-content guard below
                        # before it would fire, adding ~0.4-0.5s of pure dead
                        # time with an ASR round-trip on empty audio at the
                        # end of it. Calling audio-analyzer here would only
                        # ever get back "" or a filler, so treat the adaptive
                        # flush as satisfied immediately without enqueuing any
                        # work — there is nothing new to transcribe.
                        logger.debug(
                            "[CHUNK] session=%s | adaptive flush skipped at %.2fs pause "
                            "(no unflushed speech in buffer)",
                            self.session_id,
                            silence_run_seconds,
                        )
                        _adaptive_flushed = True
                        continue
                    # Minimum 0.5s chunk: Whisper has a fixed per-call overhead
                    # (~1.2s on CPU, ~150ms on GPU) that dominates sub-0.5s
                    # inputs — sending near-empty frames wastes more time than
                    # it saves. Only reached when the buffer DOES hold real
                    # speech (the branch above handles the pure-silence case).
                    if chunk_frames and self._chunk_duration_seconds(chunk_frames) >= 0.5:
                        logger.debug(
                            "[CHUNK] session=%s | adaptive flush at %.2fs pause (%.2fs of audio)",
                            self.session_id,
                            silence_run_seconds,
                            self._chunk_duration_seconds(chunk_frames),
                        )
                        # Not the final chunk — see DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED
                        # in config.py for why this one is flagged non-final.
                        self._flush_queue.put(
                            (self._trim_trailing_silence(chunk_frames, silence_run_seconds), False)
                        )
                        chunk_frames = []
                        _adaptive_flushed = True
                        self._chunk_has_speech = False
                        # Do NOT reset silence_run_seconds — we're still in silence,
                        # the endpoint counter keeps running toward silence_timeout_seconds.
                    continue

                # ── Timed chunk flush (max chunk size cap) ──────────────────
                if self._chunk_duration_seconds(chunk_frames) >= self.request.chunk_seconds:
                    # Enqueue for the background flush worker instead of
                    # transcribing inline — see the _flush_queue docstring in
                    # __init__ for why this must not block this loop. Not the
                    # final chunk — speech may still be ongoing.
                    self._flush_queue.put((chunk_frames, False))
                    chunk_frames = []
                    silence_run_seconds = 0.0
                    _adaptive_flushed = False
                    self._chunk_has_speech = False
                    continue

                # ── Preview flush (continuous ASR during active speech) ─────
                # Streaming mode: send a cheap, NON-destructive preview ping
                # over the already-open socket instead of a real (destructive)
                # flush. The analyzer transcribes everything buffered since
                # the last real commit without clearing it, so this always
                # has full utterance context (never a mid-word slice) and
                # never touches the official, persisted transcript -- that
                # is still built exclusively by real commits below (adaptive-
                # pause flush / final flush), at the same safe cadence/spans
                # as the file-based path. See
                # kiosk_core/realtime_analyzer_client.py's module docstring
                # and RealtimeAnalyzerClient.preview() for the full rationale.
                #
                # Timed independently of chunk_frames (which only tracks
                # audio for the next REAL commit) via _last_preview_ping_at,
                # so ticking previews never interfere with adaptive/final
                # flush accounting.
                if self._streaming_active:
                    now_mono = time.monotonic()
                    # Once trailing silence has begun, ping much more often
                    # (matching kiosk-voice-lab-main's tick_quiet_s=0.15
                    # quiet-mode acceleration) so a fresh, complete-looking
                    # snapshot is almost always ready within
                    # DEFAULT_ENDPOINT_SHORT_SECONDS of the customer's last
                    # word -- see DEFAULT_REALTIME_PREVIEW_COMMIT_QUIET_SECONDS's
                    # docstring in config.py for why the flat cadence made the
                    # completeness shortcut a timing coin-flip.
                    _preview_interval = (
                        config.DEFAULT_REALTIME_PREVIEW_COMMIT_QUIET_SECONDS
                        if silence_run_seconds > 0
                        else config.DEFAULT_REALTIME_PREVIEW_COMMIT_SECONDS
                    )
                    if (
                        config.DEFAULT_PREVIEW_FLUSH_ENABLED
                        and silence_run_seconds < adaptive_pause
                        and self._chunk_has_speech
                        and (now_mono - self._last_preview_ping_at) >= _preview_interval
                    ):
                        self.realtime_client.preview()
                        self._last_preview_ping_at = now_mono
                    # Deliberately no `continue` here (unlike the file-based
                    # block below, which owns chunk_frames/_flush_queue and
                    # DOES `continue` when it fires): a preview ping never
                    # consumes chunk_frames, so execution must still reach
                    # the Endpoint check below on every frame.

                # Flush the accumulated chunk to the background ASR worker
                # periodically WHILE the customer is still talking — not just
                # at the chunk-size cap (6.0s) or the end-of-utterance pause
                # below. This is what shortens the chunk the adaptive-pause
                # flush has to transcribe once silence actually begins, which
                # is what was landing too late (~1.23-1.28s) for the
                # completeness shortcut to fire. See config.py for the full
                # rationale.
                #
                # Scoped to silence_run_seconds < adaptive_pause so this never
                # overlaps the adaptive-pause flush's own domain (the trailing
                # pause) — this block only fires during genuinely continuous
                # speech.
                #
                # Gated on unfinished_tasks == 0: audio-analyzer serialises
                # every WhisperPipeline.generate() call behind one global
                # lock, so a second concurrent preview call would just queue
                # up behind the first rather than run in parallel — this
                # check makes each tick self-relaunching (fire again as soon
                # as the previous one lands) instead of piling up requests.
                #
                # Streaming mode handles previews entirely via the
                # non-destructive ping above -- this destructive file-based
                # flush must not also run for those sessions.
                if (
                    not self._streaming_active
                    and config.DEFAULT_PREVIEW_FLUSH_ENABLED
                    and silence_run_seconds < adaptive_pause
                    and self._chunk_duration_seconds(chunk_frames) >= config.DEFAULT_PREVIEW_FLUSH_INTERVAL_SECONDS
                    and self._chunk_has_speech
                    and self._flush_queue.unfinished_tasks == 0
                ):
                    logger.debug(
                        "[CHUNK] session=%s | preview flush at %.2fs of continuous speech",
                        self.session_id,
                        self._chunk_duration_seconds(chunk_frames),
                    )
                    # Not the final chunk — see DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED
                    # in config.py for why this one is flagged non-final.
                    #
                    # This flush can itself land mid-pause (the interval
                    # boundary reached while silence_run_seconds is small but
                    # non-zero) — that dangling trailing silence is exactly
                    # what makes the analyzer's cumulative Whisper buffer
                    # hallucinate a completion word (measured: "Good."
                    # appearing at the THIRD preview flush, well before the
                    # customer's true end of speech). Trim it the same way as
                    # the adaptive-pause/final flushes.
                    self._flush_queue.put(
                        (self._trim_trailing_silence(chunk_frames, silence_run_seconds), False)
                    )
                    chunk_frames = []
                    self._chunk_has_speech = False
                    continue

                # ── Endpoint (trailing silence) ─────────────────────────────
                # Two waits, not one. A transcript that reads as a finished
                # sentence commits at endpoint_short_seconds; anything that
                # looks mid-thought keeps the full silence_timeout_seconds.
                # _looks_complete fails closed, so if the adaptive flush's ASR
                # has not landed yet this is exactly the old fixed behaviour.
                #
                # Streaming mode: judge completeness against the OFFICIAL
                # transcript (real commits only) PLUS the latest non-
                # destructive preview of whatever's been said since the last
                # real commit -- this is what lets the shortcut fire the
                # instant silence starts, without waiting on a real commit's
                # ASR round trip. The preview text is used for this decision
                # ONLY; transcript_parts (the actual turn text) is still
                # built exclusively by real commits below.
                _endpoint_candidate_transcript = _assemble_transcript(self.transcript_parts)
                if self._streaming_active:
                    _preview_text = self.realtime_client.latest_preview_text()
                    if _preview_text:
                        _endpoint_candidate_transcript = (
                            f"{_endpoint_candidate_transcript} {_preview_text}".strip()
                        )
                if (
                    config.DEFAULT_ENDPOINT_COMPLETE_ENABLED
                    and silence_run_seconds >= config.DEFAULT_ENDPOINT_SHORT_SECONDS
                    and silence_run_seconds < self.request.silence_timeout_seconds
                    # The tail has been flushed, so no un-transcribed speech is
                    # held back. chunk_frames is NOT a usable test here: every
                    # frame is appended to it, silence included, so it refills
                    # the moment the adaptive flush clears it.
                    #
                    # Streaming mode doesn't need this gate: the preview ping
                    # above already keeps _endpoint_candidate_transcript fresh
                    # continuously, independent of whether/when the file-based
                    # adaptive flush would have fired.
                    and (self._streaming_active or _adaptive_flushed)
                    # ...and every flushed chunk has been transcribed. Without
                    # this, an in-flight tail could leave a truncated transcript
                    # that still reads as a finished sentence ("Order one
                    # classic"), and the turn would commit on the wrong words.
                    and (self._streaming_active or self._flush_queue.unfinished_tasks == 0)
                    and _looks_complete(
                        _endpoint_candidate_transcript,
                        config.DEFAULT_ENDPOINT_MIN_WORDS,
                    )
                    # See DEFAULT_ENDPOINT_STABLE_SECONDS / kiosk-voice-lab-main
                    # assess_complete(text, stable): "reads complete" must hold
                    # unchanged for a short window, not just on the instant a
                    # chunk lands, before it is trusted to shorten the wait.
                    # Clocked on silence_run_seconds (audio-domain), not
                    # wall-clock -- see _endpoint_transcript_stable's
                    # docstring for why: bursty frame delivery can advance
                    # silence_run_seconds far faster than real time elapses.
                    and self._endpoint_transcript_stable(_endpoint_candidate_transcript, silence_run_seconds)
                ):
                    logger.info(
                        "[ENDPOINT] session=%s | early commit at %.2fs "
                        "(transcript reads complete; saved %.2fs)",
                        self.session_id,
                        silence_run_seconds,
                        self.request.silence_timeout_seconds - silence_run_seconds,
                    )
                    self._endpoint_wait_seconds = silence_run_seconds
                    self._endpoint_shortcut_fired = True
                    end_reason = "silence_timeout"
                    self._log_last_word_spoken(silence_run_seconds)
                    break

                if silence_run_seconds >= self.request.silence_timeout_seconds:
                    self._endpoint_wait_seconds = silence_run_seconds
                    self._endpoint_shortcut_fired = False
                    logger.info(
                        "[ENDPOINT] session=%s | fallback commit at full "
                        "silence_timeout_seconds=%.2fs (completeness shortcut "
                        "did not fire — transcript not stable/complete in time, "
                        "or ASR round-trip had not returned by "
                        "%.2fs)",
                        self.session_id,
                        self.request.silence_timeout_seconds,
                        config.DEFAULT_ENDPOINT_SHORT_SECONDS,
                    )
                    self._log_last_word_spoken(silence_run_seconds)
                    break

                if (self._captured_samples / self.request.sample_rate) >= self.request.max_session_seconds:
                    end_reason = "max_duration_reached"
                    break

        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("Audio session %s failed", self.session_id)

        # The final chunk is enqueued the same way as every mid-stream chunk —
        # the worker's own exception handling (see _flush_worker) already
        # treats any single chunk's ASR failure as non-fatal, so there is
        # nothing left for this call site to catch. This IS the final chunk —
        # always keep full diarization on it regardless of
        # DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED (see config.py).
        #
        # See config.DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED: when set, and
        # nothing in chunk_frames is unflushed speech (the adaptive-pause
        # flush already sent every real word — chunk_frames is trailing
        # silence only), skip this call entirely rather than pay Whisper's
        # fixed per-call round trip to transcribe silence into "".
        #
        # Drain the queue BEFORE reading _unconfirmed_speech_pending: an
        # earlier chunk (adaptive-pause or timed-cap flush) may still be
        # in-flight on the worker thread, and its outcome is exactly what
        # decides whether skipping here is safe. Skipped when the flag is
        # already known-clear, so this is a no-op (immediate return) in the
        # overwhelmingly common case where that earlier flush finished
        # during the trailing-silence wait, as designed — it only actually
        # blocks in the rare case that ASR round-trip was unusually slow.
        # Timed from here (not just the final put+join below) so that rare
        # blocking case is still fully accounted for in final_flush_wait_ms
        # rather than silently absorbed before the clock starts.
        _t_final_flush_start = time.monotonic()
        self._t_final_flush_start = _t_final_flush_start
        if config.DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED and self._unconfirmed_speech_pending:
            self._flush_queue.join()
        skip_final_flush = (
            config.DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED
            and not self._chunk_has_speech
            and not self._unconfirmed_speech_pending
        )
        if (
            chunk_frames
            and self._speech_started
            and self._chunk_has_speech
            and not skip_final_flush
        ):
            # Genuine new speech was captured since the last flush (not just
            # trailing silence) — this chunk must actually be sent.
            self._flush_queue.put(
                (self._trim_trailing_silence(chunk_frames, silence_run_seconds), True)
            )
        elif (
            self._streaming_active
            and self._speech_started
            and self._unconfirmed_speech_pending
            and not skip_final_flush
        ):
            # Nothing NEW to send — chunk_frames (if any) is pure trailing
            # silence, not real speech (see the _chunk_has_speech gate on the
            # branch above). This covers two cases: chunk_frames is empty
            # because every captured frame was already swept into an earlier
            # non-final flush (typically the adaptive-pause flush firing on
            # a short utterance that finishes well inside one pause window),
            # or chunk_frames holds only the trailing silence the endpoint
            # wait accumulated afterward. Either way, that earlier flush's
            # own commit() was fire-and-forget (wait=False), so the
            # analyzer's real transcript for this utterance is still
            # in-flight and must be retrieved somehow.
            #
            # Pass EMPTY frames rather than the leftover silence bytes: see
            # _flush_chunk's streaming branch, which peeks the analyzer's
            # already-cached snapshot before committing anything. A
            # content-free commit can never produce a NEW completion event,
            # so if we instead queued the real (silent) chunk_frames here,
            # the resulting commit(wait=True) would race the earlier
            # commit's async completion arriving RIGHT before this one's own
            # baseline is captured — and lose that race just as often as it
            # wins, burning the full
            # DEFAULT_REALTIME_FINAL_COMMIT_TIMEOUT_SECONDS window for
            # nothing when it does. Measured live: a turn's real transcript
            # landed <25ms before this exact wait began, and the turn still
            # paid the full 2.5s. Empty frames make _flush_chunk skip
            # straight to the already-cached result instead.
            self._flush_queue.put(([], True))
        elif chunk_frames and self._speech_started and skip_final_flush:
            self._final_flush_skipped = True
            logger.info(
                "[CHUNK] session=%s | skipped empty final tail-chunk flush "
                "(%.2fs of silence, no unflushed/unconfirmed speech since last "
                "successful flush)",
                self.session_id,
                self._chunk_duration_seconds(chunk_frames),
            )
        elif chunk_frames and self._speech_started:
            # Neither branch above matched (typically: non-realtime session,
            # e.g. browser push-to-talk over POST or file-replay, so
            # _streaming_active is False) and skip_final_flush is False —
            # either DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED is off, or this
            # chunk still carries unconfirmed speech. Restore the original,
            # unconditional pre-skip-flag behavior: always send the final
            # tail chunk, even if it turns out to be pure trailing silence.
            # A prior refactor (see git history around
            # config.DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED) collapsed this
            # into the "skip" branch above regardless of skip_final_flush,
            # silently dropping every non-realtime session's final chunk.
            self._flush_queue.put(
                (self._trim_trailing_silence(chunk_frames, silence_run_seconds), True)
            )

        # Signal the worker to stop after draining everything queued so far,
        # then block until it has actually finished — _finalize_run (called
        # right after this returns) needs the complete transcript, and the
        # worker is what now owns every transcript_parts append. This is the
        # only wait left in the critical path: just the last chunk's ASR
        # latency, no longer serialised behind an earlier chunk's.
        #
        # Timed explicitly (rather than left as an implicit gap between
        # _t_last_word and _t_turn_start) because it was previously invisible
        # in the trace: endpoint_wait_ms only reports the trailing-silence
        # wait, so a turn's reported wait+compute numbers silently undercounted
        # voice_to_voice_ms by however long this join blocked — on some turns
        # the single largest stage in the whole clock. See
        # WallTimes.final_flush_wait_ms.
        self._flush_queue.put(None)
        self._flush_queue.join()
        self._final_flush_wait_seconds = time.monotonic() - _t_final_flush_start

        return final_status, end_reason

    def _flush_worker(self) -> None:
        """Background worker that transcribes queued chunks one at a time.

        Runs on its own thread so the frame-reading loop in
        ``_process_frame_stream`` is never blocked waiting on an
        audio-analyzer round-trip. A single worker draining a FIFO queue
        guarantees chunks are still flushed in capture order, which
        ``_flush_chunk`` depends on for both ``transcript_parts`` ordering and
        the analyzer's cumulative-segment cursor (``_last_analyzer_segment_end``).

        A ``None`` item is the stop sentinel (queued once, after the last real
        chunk, by ``_process_frame_stream``). Any exception from an individual
        chunk's ``_flush_chunk`` call is caught and logged here rather than
        propagated, so one bad chunk can never lose the rest of an otherwise
        good utterance — the same reasoning the previous synchronous code
        already applied to the final chunk only; this now applies uniformly
        to every chunk.
        """
        while True:
            item = self._flush_queue.get()
            try:
                if item is None:
                    break
                frames, is_final = item
                try:
                    self._flush_chunk(frames, is_final)
                except Exception:
                    logger.exception(
                        "Audio session %s: chunk flush failed (non-fatal)", self.session_id,
                    )
            finally:
                self._flush_queue.task_done()

    def _finalize_run(self, final_status: str, end_reason: str) -> None:
        # Attempt RAG whenever there is a transcript, even if the session
        # ended with an error mid-stream (e.g. a transient ASR failure on one
        # chunk).  Only skip entirely when NO audio was captured at all.
        self._t_turn_start = time.monotonic()
        transcript = _assemble_transcript(self.transcript_parts)
        if transcript:
            # A real, accepted transcript ends any rejection streak for this
            # conversation — the customer was just heard, so a stale streak
            # from earlier turns must not trigger the retry prompt later.
            _reset_conversation_rejections(self.agent_session_id)
            try:
                self._stream_rag_response(transcript)
            except Exception as exc:
                with self._lock:
                    self.error = str(exc)
                logger.exception("RAG query failed for session %s", self.session_id)
        elif final_status == "completed":
            # The transcript is empty — the kiosk usually has nothing
            # meaningful to say. Log why the turn produced no output, then
            # decide whether this specific case still warrants speaking.
            #
            # Rationale for staying silent by default:
            #  - "stopped_by_api" / "no_speech_detected": user explicitly stopped
            #    without speaking — speaking any prompt looks like the button
            #    had no effect.
            #  - Rejected-speech (speaker filter / diarization): often triggered
            #    by Whisper hallucinations ("you", "thank you", etc.) on
            #    background noise or TTS echo from the previous turn. These are
            #    not real utterances, so "I couldn't recognise your voice" is
            #    a false alarm on the FIRST occurrence.
            #  - "silence_timeout": VAD ended the turn with no speech — the
            #    customer is either not there or not ready; prompting can feel
            #    intrusive and the UI already shows "🎧 Listening…" or re-arms
            #    automatically in conversation mode.
            #
            # Exception: rejected speech is escalated once it becomes a
            # STREAK (config.DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD
            # consecutive rejected turns in the same conversation). Staying
            # silent forever there regressed to the exact problem the retry
            # prompt originally existed for — a genuinely ignored customer
            # (real bystander, or a mistuned enrollment rejecting them) gets
            # zero feedback and the kiosk looks unresponsive. An explicit stop
            # never escalates: speaking here would look like the stop button
            # didn't work.
            if self._rejected_speech_chunks and end_reason != "stopped_by_api":
                streak = _note_conversation_rejection(self.agent_session_id)
                threshold = config.DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD
                if streak >= threshold:
                    logger.info(
                        "Session %s: %d chunk(s) rejected AND %d consecutive "
                        "rejected turn(s) for conversation %s (threshold=%d) "
                        "— asking the customer to repeat",
                        self.session_id, self._rejected_speech_chunks, streak,
                        self.agent_session_id, threshold,
                    )
                    self._synthesize_response(config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT)
                    _reset_conversation_rejections(self.agent_session_id)
                else:
                    logger.info(
                        "Session %s: %d chunk(s) of speech were rejected by the speaker "
                        "filter (likely hallucination or echo) — staying silent "
                        "(streak=%d/%d)",
                        self.session_id, self._rejected_speech_chunks, streak, threshold,
                    )
            else:
                logger.info(
                    "Session %s: empty transcript (end_reason=%s) — staying silent",
                    self.session_id, end_reason,
                )

        with self._lock:
            if final_status == "completed" and self.end_reason == "stopped_by_api":
                end_reason = "stopped_by_api"
            self.status = final_status
            self.completed_at = datetime.now(UTC)
            self.end_reason = end_reason

        # Record the completed turn for offline analysis. Controlled entirely
        # by config.CONVERSATION_LOGGING_ENABLED -- a no-op (no file I/O) when
        # the flag is off, and never raises when it's on (see
        # conversation_recorder.record_turn).
        conversation_recorder.record_turn(
            conversation_id=self.agent_session_id,
            turn_id=self.session_id,
            user_text=transcript,
            assistant_text="".join(str(part) for part in self.response_parts),
            end_reason=end_reason,
        )

        logger.info(
            "Session %s ended with reason=%s transcript=%s",
            self.session_id,
            self.end_reason,
            " ".join(self.transcript_parts).strip(),
        )
        if self.on_complete is not None:
            self.on_complete(self.session_id)
        # Release the analyzer's persistent HTTP connection now that the
        # session is done issuing chunk flushes. Never let this raise —
        # cleanup must not turn a successful turn into a failed one.
        try:
            self.client.close()
        except Exception:
            logger.exception("Session %s: failed to close analyzer client", self.session_id)
        # Unlike client/tts_client/agent_client above and below, this
        # attribute lookup was NOT inside its own try/except -- an
        # AttributeError from a partially-built session (e.g. a test double)
        # would escape here, breaking the "cleanup must not raise" invariant
        # this whole method documents. getattr(..., None) makes it consistent
        # with the other three cleanup calls.
        if getattr(self, "realtime_client", None) is not None:
            try:
                self.realtime_client.close()
            except Exception:
                logger.exception("Session %s: failed to close realtime analyzer client", self.session_id)
        # Same cleanup for the TTS and agent clients' persistent connections
        # (see TtsClient/AgentClient __init__ for why these are now
        # session-scoped, reused httpx.Client instances instead of one per
        # call). The attribute lookups sit INSIDE the try blocks on purpose:
        # _finalize_run also runs for sessions that failed during construction
        # (and for partially-built sessions in tests), where these attributes
        # may not exist at all. An AttributeError escaping here would turn a
        # completed turn into a crashed one during pure cleanup.
        try:
            self.tts_client.close()
        except Exception:
            logger.exception("Session %s: failed to close TTS client", self.session_id)
        try:
            if self.agent_client is not None:
                self.agent_client.close()
        except Exception:
            logger.exception("Session %s: failed to close agent client", self.session_id)

    def _synthesize_response(self, text: str) -> None:
        """Speak a fixed response directly via TTS, without calling RAG."""
        with self._lock:
            self.response_parts.append(text)
        sentence_queue: Queue[tuple[int | None, str | None]] = Queue()
        workers = self._start_tts_workers(sentence_queue)
        sentence_queue.put((1, text))
        self._stop_tts_workers(sentence_queue, workers)

    def _log_last_word_spoken(self, silence_run_seconds: float) -> None:
        """Log the wall-clock instant the customer stopped talking.

        This is the true start of the voice-to-voice clock. Two ways to get
        it:

        1. Derived: the endpoint algorithm only *decides* to commit
           ``silence_run_seconds`` later, once the trailing-silence window
           has elapsed, so "now minus that many seconds" should be the last
           real word. This assumes the frame-processing loop's own
           ``silence_run_seconds`` counter tracks wall-clock time exactly —
           true only under perfectly steady frame delivery.
        2. Observed: ``self._t_last_speech_frame_sent``, stamped at the
           actual instant ``send_frame()`` was called for the last
           speech-containing frame (see _process_frame_stream). Not derived
           from anything, so immune to the drift (1) is exposed to (bursty
           delivery, GC pauses, a slow VAD tick).

        Prefer (2) whenever it exists — streaming sessions always have it
        once real speech was captured. The derived value is logged alongside
        it (drift_ms) so a growing gap between the two is visible in the
        trace rather than silently biasing v2v numbers one way or the other.
        """
        derived_t_last_word = time.monotonic() - silence_run_seconds
        drift_ms = 0.0
        if self._t_last_speech_frame_sent is not None:
            self._t_last_word = self._t_last_speech_frame_sent
            drift_ms = (derived_t_last_word - self._t_last_speech_frame_sent) * 1000
        else:
            self._t_last_word = derived_t_last_word
        last_word_ts = datetime.now(UTC) - timedelta(
            seconds=time.monotonic() - self._t_last_word
        )
        logger.info(
            "[VOICE2VOICE] session=%s conversation=%s event=last_word_spoken "
            "ts=%s (endpoint committed after %.2fs trailing silence, "
            "source=%s, derived_vs_observed_drift_ms=%.1f)",
            self.session_id, self.agent_session_id,
            last_word_ts.isoformat(), silence_run_seconds,
            "observed" if self._t_last_speech_frame_sent is not None else "derived",
            drift_ms,
        )

    def _log_first_response_audio(self, latency_ms: float) -> None:
        """Log the wall-clock instant the first REAL answer audio is ready.

        Fires once per turn, the moment the first non-opener TTS segment is
        written to disk (i.e. the earliest point the customer could actually
        hear content). `latency_ms` is last-word-to-here, i.e. the true
        voice-to-voice latency for this turn.
        """
        logger.info(
            "[VOICE2VOICE] session=%s conversation=%s event=first_response_audio "
            "ts=%s voice_to_voice_ms=%.0f",
            self.session_id, self.agent_session_id,
            datetime.now(UTC).isoformat(), latency_ms,
        )

    def _emit_opener(self) -> None:
        """Play a cached, non-committal opener while the agent turn runs.

        Ordering turns emit only a tool call — the spoken reply is templated
        after the tool returns — so there is no model text to stream during the
        ~2 s the LLM spends generating tool-call JSON. This publishes a
        pre-rendered segment at index 0 (model sentences start at 1) so the
        customer hears something immediately.

        The segment is a plain file copy, so it costs no TTS round-trip. Any
        failure is swallowed: the opener is a latency optimisation and must
        never be able to break a turn.
        """
        if not config.DEFAULT_OPENER_ENABLED:
            return
        text = (config.DEFAULT_OPENER_TEXT or "").strip()
        if not text:
            return

        source = _render_opener(
            self.tts_client,
            text,
            self.request.tts_model,
            self.request.tts_voice,
            self.request.tts_language,
            self.request.tts_instructions,
        )
        if source is None:
            return

        try:
            destination = self._session_output_dir / "response_000.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            with self._lock:
                if self._t_first_audio is None:
                    self._t_first_audio = time.monotonic()
                self._t_last_tts = time.monotonic()
                self.tts_audio_segments.append(
                    {
                        "index": 0,
                        "text": text,
                        "audio_file": str(destination),
                    }
                )
            logger.info("[OPENER] session=%s emitted %r", self.session_id, text)
        except Exception:
            logger.exception("[OPENER] Failed to publish opener for session %s", self.session_id)

    def _stream_rag_response(self, transcript: str) -> None:
        pending_text = ""
        self._log_speculative_draft_match(transcript)
        # Emitted before the agent call so the customer hears it while the
        # model is still generating the turn's tool call.
        self._emit_opener()
        sentence_queue: Queue[tuple[int | None, str | None]] = Queue()
        workers = self._start_tts_workers(sentence_queue)

        history = list(getattr(self.request, "history", []) or [])

        # Route through the ordering agent when enabled; fall back to direct RAG.
        if self.agent_client is not None:
            logger.info("[SESSION] Routing turn to agent: session=%s (conv=%s) message=%r",
                        self.session_id, self.agent_session_id, transcript[:80])
        if self.agent_client is not None:
            logger.info("[SESSION] Routing turn to agent: session=%s (conv=%s) message=%r",
                        self.session_id, self.agent_session_id, transcript[:80])
            token_source = self.agent_client.get_reply(
                transcription=transcript,
                session_id=self.agent_session_id,  # persistent across voice turns
                user_id=getattr(self.request, "user_id", None) or config.DEFAULT_ORDERING_USER_ID,
                history=history,
            )
            label = "Agent"
        else:
            token_source = self.rag_client.stream_answer(transcript, history=history)
            label = "RAG"

        print(f"\n{label} response for session {self.session_id}:\n", end="", flush=True)
        sentence_index = 0
        _first_token_seen = False
        _tool_calls: list[str] = []
        _llm_ms: float | None = None
        _llm_ttft_ms: float | None = None
        _llm_calls: int = 0
        _retrieval_ms: float | None = None
        _mcp_ms: float | None = None
        _mcp_calls: int = 0
        _guard_ms: float | None = None
        _template_ms: float | None = None
        # t_agent_start set here — generator body (HTTP call) runs on first iteration
        if self.agent_client is not None:
            self._t_agent_start = time.monotonic()
        try:
            for token in token_source:
                # Handle metadata sentinel from AgentClient BEFORE appending to response_parts
                # to avoid dict items in response_parts (which causes TypeError in snapshot())
                if isinstance(token, dict) and "_tool_calls" in token:
                    _tool_calls = token["_tool_calls"]
                    _llm_ms = token.get("_llm_ms")
                    _llm_ttft_ms = token.get("_llm_ttft_ms")
                    _llm_calls = token.get("_llm_calls", 0)
                    _retrieval_ms = token.get("_retrieval_ms")
                    _mcp_ms = token.get("_mcp_ms")
                    _mcp_calls = token.get("_mcp_calls", 0)
                    _guard_ms = token.get("_guard_ms")
                    _template_ms = token.get("_template_ms")
                    continue

                with self._lock:
                    self.response_parts.append(token)
                print(token, end="", flush=True)

                if not _first_token_seen:
                    _first_token_seen = True
                    self._t_agent_end = time.monotonic()

                pending_text += token
                complete_sentences, pending_text = self._drain_complete_sentences(pending_text)
                if sentence_index == 0 and complete_sentences:
                    complete_sentences = (
                        self._split_first_phrase(complete_sentences[0])
                        + complete_sentences[1:]
                    )
                for sentence in complete_sentences:
                    sentence_index += 1
                    if sentence_index == 1:
                        self._t_first_tts = time.monotonic()
                    sentence_queue.put((sentence_index, sentence))

            trailing_text = pending_text.strip()
            if trailing_text:
                trailing_fragments = (
                    self._split_first_phrase(trailing_text)
                    if sentence_index == 0
                    else [trailing_text]
                )
                for fragment in trailing_fragments:
                    sentence_index += 1
                    if sentence_index == 1:
                        self._t_first_tts = time.monotonic()
                    sentence_queue.put((sentence_index, fragment))

            # If agent_end wasn't set (empty reply), set it now
            if self._t_agent_start is not None and self._t_agent_end is None:
                self._t_agent_end = time.monotonic()

            # Stamped here — BEFORE _stop_tts_workers() below drains any
            # still-synthesising TTS segments — so agent.stream_ms measures
            # only the LLM/tool round-trip a reply actually took, not that
            # plus however long TTS still had left to run. agent.total_ms
            # (existing metric) intentionally keeps including the TTS drain;
            # this is the isolated figure for anyone debugging tool/LLM time.
            self._t_agent_stream_end = time.monotonic()

        finally:
            self._stop_tts_workers(sentence_queue, workers)
            self._t_turn_end = time.monotonic()
            self._tts_segment_count = sentence_index
            print(flush=True)

        # ── Record pipeline turn trace ──────────────────────────────────────
        self._record_turn_trace(
            _tool_calls, _llm_ms, _llm_calls, _retrieval_ms, _llm_ttft_ms,
            _mcp_ms, _mcp_calls, _guard_ms, _template_ms,
        )

    def _record_turn_trace(
        self,
        tool_calls: list[str],
        llm_ms: float | None = None,
        llm_calls: int = 0,
        retrieval_ms: float | None = None,
        llm_ttft_ms: float | None = None,
        mcp_ms: float | None = None,
        mcp_calls: int = 0,
        guard_ms: float | None = None,
        template_ms: float | None = None,
    ) -> None:
        """Build and persist a TurnTrace for the completed voice turn."""
        t0 = self._t_turn_start
        t_agent_s = self._t_agent_start
        t_agent_e = self._t_agent_end
        t_first = self._t_first_tts
        t_last = self._t_last_tts
        t_end = self._t_turn_end

        def _ms(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return round((b - a) * 1000, 1)

        # ASR = summed analyzer round-trips measured in _flush_chunk. It cannot
        # be derived from turn_start, because every chunk is already transcribed
        # by the time _finalize_run stamps t0.
        asr_ms = round(self._asr_ms_total, 1) if self._asr_chunks else None
        # Continuous-streaming mode only: customer's last word -> transcript
        # ready, with the deliberate trailing-silence wait excluded (that's
        # endpoint_wait_ms, a design choice, not ASR compute time). See
        # AsrSpan.last_word_to_transcript_ms's docstring for the rationale.
        asr_last_word_to_transcript_ms = None
        if self.realtime_client is not None and self._t_last_word is not None:
            _landed_at = self.realtime_client.first_content_landed_at_or_after(self._t_last_word)
            if _landed_at is not None:
                asr_last_word_to_transcript_ms = round(
                    max(0.0, (_landed_at - self._t_last_word) * 1000), 1
                )
        # Agent TTFT = from agent_start to when first token (reply) arrived
        ttft_ms = _ms(t_agent_s, t_agent_e)
        # Agent total = from agent_start to last TTS segment done (whole orchestration)
        agent_total_ms = _ms(t_agent_s, t_end)
        # Agent stream = from agent_start to the end of the token stream,
        # BEFORE the TTS-drain wait _stop_tts_workers() blocks on. This is
        # the actual LLM+tool round-trip time — agent_total_ms above
        # includes however long TTS still had left to run afterward, which
        # made it look ~2x larger than the real agent/tool cost on turns
        # with a long reply (confirmed: ~774ms real vs ~2,500ms reported for
        # rec1's place_order turn, the gap matching tts_ms almost exactly).
        agent_stream_ms = _ms(t_agent_s, self._t_agent_stream_end)
        # TTS = from first sentence queued to last segment written
        tts_ms = _ms(t_first, t_last)
        # Time to first audio = from end of speech (t0, stamped by
        # _finalize_run) to the moment the first segment became available.
        #
        # Measured from t0 rather than from agent_start because the opener is
        # published *before* the agent call starts, which would otherwise make
        # this negative. t0 is also the more honest boundary: it is what the
        # customer actually waits through after they stop talking.
        ttfa_ms = _ms(t0, self._t_first_audio or t_first)
        # Voice to voice: the customer's last word to the first sound.
        #
        # Prefer _t_last_word (set in _log_last_word_spoken) when available:
        # it is measured directly from the endpoint decision, backdated by
        # the trailing-silence wait, so it is anchored the instant speech
        # actually stopped. Using it directly (rather than reconstructing
        # "t0 - endpoint_wait_seconds" and adding endpoint_wait_ms back onto a
        # t0-based delta) also naturally includes the final chunk's ASR
        # round-trip that _finalize_run blocks on before stamping t0 — that
        # gap was previously silently missing from this metric.
        #
        # Falls back to the older endpoint_wait_ms-based reconstruction when
        # _t_last_word was never set (e.g. end_reason was "stopped_by_api" or
        # "max_duration_reached", which don't go through the silence-timeout
        # commit path that sets it).
        endpoint_wait_ms = (
            round(self._endpoint_wait_seconds * 1000, 1)
            if self._endpoint_wait_seconds is not None
            else None
        )
        final_flush_wait_ms = (
            round(self._final_flush_wait_seconds * 1000, 1)
            if self._final_flush_wait_seconds is not None
            else None
        )
        t_last_word = self._t_last_word
        # Gap between the customer's true last speech frame and the instant
        # the flush/turn-start sequence began.
        #
        # Only meaningful on the browser mic-release path (endpoint_wait_ms
        # is None there): reports the full, real wall-clock gap -- how long
        # after the customer's last word they took to release the mic
        # button, plus any trailing buffered frames. Both ends are backend
        # monotonic timestamps, no browser clock involved.
        #
        # Forced to 0 on the silence-timeout path (endpoint_wait_ms is set):
        # that field is measured in AUDIO-DOMAIN time (silence_run_seconds,
        # counted from audio samples), not wall-clock time, so it is NOT
        # safe to subtract it from a wall-clock gap here -- whenever frames
        # arrive faster than real-time (bursty delivery, a fixture pushed
        # without exact real-time pacing), the audio-domain duration can be
        # far larger than the true wall-clock gap, which previously produced
        # nonsensical negative values. The silence wait is already fully
        # reported via endpoint_wait_ms; there is nothing left to add here.
        if endpoint_wait_ms is not None:
            post_speech_gap_ms = 0.0
        else:
            post_speech_gap_ms = _ms(t_last_word, self._t_final_flush_start)
        if t_last_word is not None:
            v2v_ms = _ms(t_last_word, self._t_first_audio or t_first)
        else:
            v2v_ms = (
                round(endpoint_wait_ms + ttfa_ms, 1)
                if endpoint_wait_ms is not None and ttfa_ms is not None
                else None
            )
        # v2v with BOTH deliberate/customer-side waiting time subtracted back
        # out: the endpoint's trailing-silence wait AND the post-speech gap
        # (mic-release reaction time). Neither is pipeline compute cost, so
        # neither belongs in the number meant to answer "how fast is our
        # compute pipeline" -- see WallTimes.voice_to_voice_post_endpoint_ms.
        # Left over is exactly final_flush_wait_ms + time_to_first_audio_ms.
        v2v_post_endpoint_ms = (
            round(v2v_ms - (endpoint_wait_ms or 0) - max(post_speech_gap_ms or 0, 0), 1)
            if v2v_ms is not None
            else None
        )
        # Same clock, but to the first sound that actually answers. The opener
        # is deliberately excluded here: it breaks the silence but tells the
        # customer nothing, so counting it as "the reply" would flatter the
        # number. Uses the on-disk stamp, not the queue stamp.
        informative_ms = _ms(t0, self._t_first_answer_audio)
        if t_last_word is not None:
            v2v_informative_ms = _ms(t_last_word, self._t_first_answer_audio)
        else:
            v2v_informative_ms = (
                round(endpoint_wait_ms + informative_ms, 1)
                if endpoint_wait_ms is not None and informative_ms is not None
                else None
            )
        # Wall E2E: genuinely end-to-end — from the first speech frame captured
        # (so audio capture and ASR are included) to the last TTS segment
        # written. Falls back to turn_start when no speech was ever detected.
        t_e2e_start = self._t_capture_start or t0
        wall_total_ms = _ms(t_e2e_start, t_end)

        # File-replay ground-truth anchor (see _t_playback_start comment) —
        # only ever set on FileAudioSession, so this is None on a live mic or
        # browser-stream turn.
        playback_to_first_audio_ms = _ms(self._t_playback_start, self._t_first_audio or t_first)

        retrieval_invoked = any(
            "retrieval" in tc.lower() or "knowledge" in tc.lower() or "lookup" in tc.lower()
            for tc in tool_calls
        )

        trace = TurnTrace(
            turn_id=self.session_id,
            conversation_id=self.agent_session_id,
            started_at=self.started_at.isoformat() if self.started_at else datetime.now(UTC).isoformat(),
            ended_at=datetime.now(UTC).isoformat(),
            wall=WallTimes(
                turn_total_ms=wall_total_ms,
                time_to_first_audio_ms=ttfa_ms,
                endpoint_wait_ms=endpoint_wait_ms,
                final_flush_wait_ms=final_flush_wait_ms,
                voice_to_voice_ms=v2v_ms,
                voice_to_voice_post_endpoint_ms=v2v_post_endpoint_ms,
                post_speech_gap_ms=post_speech_gap_ms,
                voice_to_voice_informative_ms=v2v_informative_ms,
                playback_to_first_audio_ms=playback_to_first_audio_ms,
                endpoint_shortcut_fired=self._endpoint_shortcut_fired,
            ),
            asr=AsrSpan(
                ms=asr_ms,
                chunks=self._asr_chunks,
                final_flush_skipped=self._final_flush_skipped,
                last_word_to_transcript_ms=asr_last_word_to_transcript_ms,
            ),
            agent=AgentSpan(
                ttft_ms=ttft_ms,
                total_ms=agent_total_ms,
                stream_ms=agent_stream_ms,
                retrieval=RetrievalSpan(
                    invoked=retrieval_invoked,
                    ms=retrieval_ms,
                ),
                llm=LlmSpan(
                    ms=llm_ms,
                    ttft_ms=llm_ttft_ms,
                    calls=llm_calls,
                    device="GPU",
                ),
                mcp=McpSpan(ms=mcp_ms, calls=mcp_calls),
                guard=GuardSpan(ms=guard_ms, calls=0 if guard_ms is None else 1),
                template=TemplateSpan(ms=template_ms, calls=0 if template_ms is None else 1),
            ),
            tts=TtsSpan(
                ms=tts_ms,
                segments=self._tts_segment_count,
                overlapped_with_agent=True,
            ),
        )
        pipeline_store.record(trace)
        self._emit_vlm_metrics(t_last_word, self._t_first_audio or t_first)
        logger.info(
            "[PIPELINE] turn=%s wall=%.0fms asr=%.0fms(%d) ttft=%.0fms tts=%.0fms "
            "llm=%.0fms(%d) agent_stream=%.0fms mcp=%.0fms(%d) retrieval=%s/%.0fms tools=%s",
            self.session_id,
            wall_total_ms or 0,
            asr_ms or 0,
            self._asr_chunks,
            ttft_ms or 0,
            tts_ms or 0,
            llm_ms or 0,
            llm_calls,
            agent_stream_ms or 0,
            mcp_ms or 0,
            mcp_calls,
            retrieval_invoked,
            retrieval_ms or 0,
            tool_calls,
        )

    def _emit_vlm_metrics(self, t_start_mono: float | None, t_end_mono: float | None) -> None:
        """Emit a real start/end pair for this turn to vlm_metrics_logger.

        ``t_start_mono``/``t_end_mono`` are ``time.monotonic()`` marks (the
        same customer's-last-word -> first-audio anchors used for
        ``WallTimes.voice_to_voice_ms``), NOT epoch time -- vlm_metrics_logger
        writes epoch milliseconds (``time.time()``), a different clock. An
        anchor pair (one ``time.monotonic()`` + one ``time.time()`` reading,
        taken back-to-back right here) converts each mark to real epoch ms
        with sub-millisecond error, so this stamps the ACTUAL instants the
        turn happened at -- unlike tests/benchmarks/
        v2v_scripted_conversation_benchmark.py's emit_vlm_metrics(), which
        only had voice_to_voice_ms after the fact over HTTP and had to
        synthesize a (end=now, start=end-duration) pair instead.

        Best-effort and silent on failure: this is a side-channel metrics
        write, and must never be able to break a live customer turn (missing
        results dir, submodule not vendored correctly, disk full, etc.).
        """
        if not config.VLM_METRICS_ENABLED:
            return
        if t_start_mono is None or t_end_mono is None:
            return
        try:
            # Installed from performance-tools' own repo (see
            # requirements.txt) as the top-level module "vlm_metrics_logger",
            # not vendored under kiosk_core -- see config.py's comment.
            from vlm_metrics_logger import user_log_end_time, user_log_start_time

            anchor_mono = time.monotonic()
            anchor_epoch_ms = time.time() * 1000
            start_epoch_ms = anchor_epoch_ms - (anchor_mono - t_start_mono) * 1000
            end_epoch_ms = anchor_epoch_ms - (anchor_mono - t_end_mono) * 1000
            user_log_start_time(
                start_epoch_ms, config.VLM_METRICS_USECASE_ENV_VAR, unique_id=self.session_id
            )
            user_log_end_time(
                end_epoch_ms, config.VLM_METRICS_USECASE_ENV_VAR, unique_id=self.session_id
            )
        except Exception:  # noqa: BLE001 - metrics logging must never break a live turn
            logger.debug("[PIPELINE] vlm_metrics_logger emit failed", exc_info=True)

    @staticmethod
    def _split_first_phrase(sentence: str) -> list[str]:
        """Split an over-long first segment so speech can start sooner.

        Only ever applied to the first spoken segment of a turn, which sits
        directly on the voice-to-voice critical path. See
        ``config.DEFAULT_TTS_FIRST_PHRASE_MAX_WORDS`` for the rationale and
        the measured synthesis cost model.

        Args:
            sentence: The first complete sentence drained from the token stream.

        Returns:
            One fragment if the sentence is short enough to leave alone, or two
            fragments whose whitespace-joined concatenation preserves the
            original wording. Never drops or reorders words.
        """
        cap = config.DEFAULT_TTS_FIRST_PHRASE_MAX_WORDS
        words = sentence.split()
        if cap <= 0:
            return [sentence]
        if len(words) < cap + config.DEFAULT_TTS_FIRST_PHRASE_MIN_TAIL_WORDS:
            return [sentence]
        return [" ".join(words[:cap]), " ".join(words[cap:])]

    @staticmethod
    def _drain_complete_sentences(buffer: str) -> tuple[list[str], str]:
        sentences: list[str] = []
        remaining = buffer
        while True:
            match = _SENTENCE_PATTERN.match(remaining.lstrip())
            if match is None:
                break
            sentence = match.group(1).strip()
            if sentence:
                sentences.append(sentence)
            remaining = remaining.lstrip()[match.end() :]
        return sentences, remaining

    def _start_tts_workers(
        self, sentence_queue: Queue[tuple[int | None, str | None]]
    ) -> list[threading.Thread]:
        """Start config.DEFAULT_TTS_WORKER_CONCURRENCY threads draining sentence_queue.

        Multiple threads consuming the same Queue is safe (Queue.get/put are
        internally synchronized); each thread runs the same _tts_worker loop.
        See config.DEFAULT_TTS_WORKER_CONCURRENCY for why this is safe to run
        concurrently (index-keyed lookup downstream, thread-safe TtsClient).
        """
        workers = []
        for _ in range(config.DEFAULT_TTS_WORKER_CONCURRENCY):
            worker = threading.Thread(target=self._tts_worker, args=(sentence_queue,), daemon=True)
            worker.start()
            workers.append(worker)
        return workers

    def _stop_tts_workers(
        self,
        sentence_queue: Queue[tuple[int | None, str | None]],
        workers: list[threading.Thread],
    ) -> None:
        """Signal every worker started by _start_tts_workers to exit, then join them.

        One (None, None) sentinel per worker — each worker's loop returns as
        soon as it dequeues its own sentinel, so N workers need N sentinels
        (a single sentinel would only stop one of them, leaving the rest
        blocked on queue.get() forever).
        """
        for _ in workers:
            sentence_queue.put((None, None))
        for worker in workers:
            worker.join()

    def _tts_worker(self, sentence_queue: Queue[tuple[int | None, str | None]]) -> None:
        while True:
            sentence_index, sentence = sentence_queue.get()
            if sentence_index is None or sentence is None:
                return

            output_path = self._session_output_dir / f"response_{sentence_index:03d}.wav"
            # The session directory is otherwise only created by _emit_opener
            # and the speculative pre-synth path, both of which are disabled by
            # default. Without this, the opener-cache branch below (a plain
            # copyfile, which does not create parents) raises FileNotFoundError
            # on sentence 1 — the first segment of the turn — silently losing
            # the opening phrase and leaving _t_first_audio unset.
            self._session_output_dir.mkdir(parents=True, exist_ok=True)
            try:
                opener_key = _opener_cache_key(sentence, self.request)
                opener_path = None
                if opener_key is not None:
                    with _OPENER_TTS_CACHE_LOCK:
                        opener_path = _OPENER_TTS_CACHE.get(opener_key)
                    if opener_path and not Path(opener_path).exists():
                        # Cache dir wiped underneath us — drop the stale entry
                        # and fall through to a normal synthesis.
                        with _OPENER_TTS_CACHE_LOCK:
                            _OPENER_TTS_CACHE.pop(opener_key, None)
                        opener_path = None
                cached_path = (
                    self._tts_cache.get(_normalize_sentence_for_cache_key(sentence))
                    if config.DEFAULT_SPECULATIVE_TTS_PRESYNTH_ENABLED
                    else None
                )
                if opener_path:
                    # A previous turn (possibly in a different session) already
                    # synthesised this exact short opener in this exact voice.
                    # The cached file is the real pipeline's own output, already
                    # trimmed and gain-adjusted, so it is byte-for-byte what we
                    # would produce now — just without the ~280 ms wait.
                    shutil.copyfile(opener_path, output_path)
                    logger.info(
                        "[TTS] session=%s | opener cache HIT for sentence %d (%r) — "
                        "skipped synthesis",
                        self.session_id, sentence_index, sentence,
                    )
                elif cached_path and Path(cached_path).exists():
                    # A speculative draft already synthesised a sentence
                    # matching this one's normalized wording (whitespace/
                    # case/punctuation differences ignored) — the real,
                    # fully-guarded reply just happened to say the same
                    # thing. Reuse the audio instead of paying for TTS again;
                    # nothing new is ever spoken here that the real pipeline
                    # didn't itself decide.
                    shutil.copyfile(cached_path, output_path)
                    logger.info(
                        "[TTS] session=%s | cache HIT for sentence %d (%d chars) — "
                        "reused speculative synthesis",
                        self.session_id, sentence_index, len(sentence),
                    )
                else:
                    self.tts_client.synthesize_to_file(
                        text=sentence,
                        output_path=str(output_path),
                        model=self.request.tts_model,
                        voice=self.request.tts_voice,
                        language=self.request.tts_language,
                        instructions=self.request.tts_instructions,
                    )
                    if config.DEFAULT_TTS_TRIM_ENABLED:
                        self._trim_tts_segment(output_path, sentence)
                    if config.DEFAULT_TTS_GAIN_ENABLED:
                        self._apply_tts_gain(output_path)
                    if opener_key is not None:
                        self._admit_to_opener_cache(opener_key, output_path, sentence)
                with self._lock:
                    self._t_last_tts = time.monotonic()
                    # First segment that carries the ANSWER (the opener is
                    # index 0 and is written by _emit_opener, never here).
                    # Stamped on write, not on queue: queuing a sentence does
                    # not make a sound, and the synthesis in between is a real
                    # part of what the customer waits through.
                    #
                    # Must be gated on sentence_index == 1 specifically, not
                    # "whichever segment finishes first": with
                    # DEFAULT_TTS_WORKER_CONCURRENCY > 1, multiple sentences
                    # synthesize in parallel and a short sentence 2/3 can
                    # finish before a longer sentence 1 — but the customer
                    # still hears sentence 1 FIRST (playback is index-
                    # ordered, see get_response_audio_path), so the
                    # voice-to-voice clock must stop on sentence 1's
                    # completion regardless of synthesis completion order.
                    if self._t_first_answer_audio is None and sentence_index == 1:
                        self._t_first_answer_audio = self._t_last_tts
                        if self._t_first_audio is None:
                            # No opener was emitted, so this segment is also the
                            # first audio of any kind for this turn.
                            self._t_first_audio = self._t_last_tts
                        # Prefer _t_last_word (see _record_turn_trace) — it is
                        # anchored to the real last-word instant, including
                        # the final chunk's ASR round-trip, and keeps this log
                        # line consistent with the API's v2v_informative_ms.
                        anchor = self._t_last_word
                        if anchor is None and self._t_turn_start is not None:
                            anchor = self._t_turn_start - (self._endpoint_wait_seconds or 0.0)
                        if anchor is not None:
                            latency_ms = (self._t_first_answer_audio - anchor) * 1000
                            self._log_first_response_audio(latency_ms)
                    self._publish_tts_segment(
                        sentence_index,
                        {
                            "index": sentence_index,
                            "text": sentence,
                            "audio_file": str(output_path),
                        },
                        _locked=True,
                    )
            except Exception as exc:
                logger.exception("TTS synthesis failed for session %s sentence %s", self.session_id, sentence_index)
                with self._lock:
                    self.tts_errors.append(f"sentence {sentence_index}: {exc}")
                    # Still resolve this index (with no segment) so higher
                    # indices already buffered in _tts_pending_publish by
                    # other worker threads aren't stuck waiting forever for
                    # a segment that will never arrive.
                    self._publish_tts_segment(sentence_index, None, _locked=True)

    def _publish_tts_segment(
        self,
        sentence_index: int,
        segment: dict[str, object] | None,
        _locked: bool = False,
    ) -> None:
        """Append a finished segment to tts_audio_segments in index order.

        Concurrent worker threads (config.DEFAULT_TTS_WORKER_CONCURRENCY > 1)
        can finish out of index order; buffer out-of-turn arrivals in
        _tts_pending_publish and only flush the contiguous prefix so
        tts_audio_segments (and the session snapshot clients poll) always
        grows in index order, matching actual playback order.
        """

        def _flush() -> None:
            self._tts_pending_publish[sentence_index] = segment
            while self._tts_next_publish_index in self._tts_pending_publish:
                pending = self._tts_pending_publish.pop(self._tts_next_publish_index)
                if pending is not None:
                    self.tts_audio_segments.append(pending)
                self._tts_next_publish_index += 1

        if _locked:
            _flush()
        else:
            with self._lock:
                _flush()

    def _trim_tts_segment(self, path: Path, sentence: str) -> None:
        """Trim baked-in silence from a synthesised segment to a fixed pad.

        Segments are synthesised per clause so playback can start early, which
        leaves each one padded with its own lead-in and lead-out silence. Played
        back to back those pads stack into an audible stall at every comma. This
        rewrites the file in place keeping a short, deliberate pad instead —
        longer at a sentence end than mid-sentence, so prosody still breathes.

        Failures are swallowed: a segment that cannot be parsed is simply left
        as synthesised, since degraded pacing is preferable to a lost reply.

        Args:
            path: WAV file to rewrite in place.
            sentence: Text the segment was synthesised from; its final
                punctuation selects the trailing pad.
        """
        try:
            with wave.open(str(path), "rb") as wav_in:
                n_channels = wav_in.getnchannels()
                sample_width = wav_in.getsampwidth()
                frame_rate = wav_in.getframerate()
                frames = wav_in.readframes(wav_in.getnframes())

            # Only 16-bit mono is handled; anything else is left untouched
            # rather than risking a corrupted rewrite.
            if sample_width != 2 or n_channels != 1 or not frames:
                return

            samples = np.frombuffer(frames, dtype=np.int16)
            if samples.size == 0:
                return

            envelope = np.abs(samples.astype(np.int32))
            peak = int(envelope.max())
            if peak <= 0:
                return

            floor = max(peak * config.DEFAULT_TTS_SILENCE_FLOOR, 1.0)
            voiced = np.flatnonzero(envelope > floor)
            if voiced.size == 0:
                return

            trailing_ms = (
                config.DEFAULT_TTS_SENTENCE_PAD_MS
                if sentence.rstrip()[-1:] in ".!?"
                else config.DEFAULT_TTS_CLAUSE_PAD_MS
            )
            lead_pad = int(frame_rate * config.DEFAULT_TTS_LEAD_PAD_MS / 1000.0)
            trail_pad = int(frame_rate * trailing_ms / 1000.0)

            start = max(0, int(voiced[0]) - lead_pad)
            end = min(samples.size, int(voiced[-1]) + 1 + trail_pad)
            if end - start >= samples.size:
                return

            segment = samples[start:end].astype(np.float64)
            # Ramp both edges to true zero so back-to-back playback of
            # separately-synthesised segments never has a sample-level jump
            # at the seam (see DEFAULT_TTS_FADE_MS docstring in config.py).
            fade_samples = min(
                int(frame_rate * config.DEFAULT_TTS_FADE_MS / 1000.0),
                segment.size // 4,
            )
            if fade_samples > 1:
                ramp = np.linspace(0.0, 1.0, fade_samples)
                segment[:fade_samples] *= ramp
                segment[-fade_samples:] *= ramp[::-1]
            segment = np.clip(segment, -32768, 32767).astype(np.int16)

            with wave.open(str(path), "wb") as wav_out:
                wav_out.setnchannels(n_channels)
                wav_out.setsampwidth(sample_width)
                wav_out.setframerate(frame_rate)
                wav_out.writeframes(segment.tobytes())

            logger.debug(
                "[TTS] session=%s trimmed %s: %.2fs -> %.2fs",
                self.session_id, path.name,
                samples.size / frame_rate, (end - start) / frame_rate,
            )
        except Exception:
            logger.warning(
                "[TTS] session=%s could not trim %s; using untrimmed audio",
                self.session_id, path.name, exc_info=True,
            )

    def _apply_tts_gain(self, path: Path) -> None:
        """Boost a synthesized segment's loudness for kiosk speakers.

        SpeechT5's vocoder outputs a quiet, roughly constant level regardless
        of which speaker embedding is used, so a soft-sounding kiosk is a
        level problem, not a voice-choice problem, and there is no gain
        control in the text-to-speech service itself. This peak-normalizes
        the segment to ``DEFAULT_TTS_TARGET_PEAK`` of full scale, then applies
        an extra flat boost (``DEFAULT_TTS_GAIN_DB``), with the combined gain
        hard-clamped at ``DEFAULT_TTS_GAIN_MAX_DB`` so a near-silent or failed
        synthesis can't be amplified into distortion/noise.

        Failures are swallowed: a segment that cannot be parsed is left as
        synthesised, since a quiet reply is preferable to a corrupted one.

        Args:
            path: WAV file to rewrite in place.
        """
        try:
            with wave.open(str(path), "rb") as wav_in:
                n_channels = wav_in.getnchannels()
                sample_width = wav_in.getsampwidth()
                frame_rate = wav_in.getframerate()
                frames = wav_in.readframes(wav_in.getnframes())

            # Only 16-bit mono is handled; anything else is left untouched
            # rather than risking a corrupted rewrite.
            if sample_width != 2 or n_channels != 1 or not frames:
                return

            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            if samples.size == 0:
                return

            peak = float(np.abs(samples).max())
            if peak <= 0:
                return

            int16_max = 32767.0
            normalize_gain = (int16_max * config.DEFAULT_TTS_TARGET_PEAK) / peak
            extra_gain = 10.0 ** (config.DEFAULT_TTS_GAIN_DB / 20.0)
            max_gain = 10.0 ** (config.DEFAULT_TTS_GAIN_MAX_DB / 20.0)
            # `normalize_gain * extra_gain` alone can push the segment's peak
            # past full scale (extra_gain is a flat boost stacked *on top of*
            # normalization, not a replacement for it), which used to get
            # silently hard-clipped by np.clip below. That is real clipping
            # distortion, not a client-side volume issue: observed live on a
            # real synthesized sentence, 14 separate clipped runs (up to 6
            # consecutive samples each) — audible as a subtle crackle/buzz on
            # every loud syllable. `no_clip_gain` caps the total at the exact
            # gain that brings the peak to (not past) full scale, so this
            # segment is never actually clipped, only ever soft-limited.
            no_clip_gain = int16_max / peak
            total_gain = min(normalize_gain * extra_gain, max_gain, no_clip_gain)

            if abs(total_gain - 1.0) < 1e-3:
                return  # already at target level; skip a no-op rewrite

            boosted = np.clip(samples * total_gain, -int16_max, int16_max).astype(np.int16)

            with wave.open(str(path), "wb") as wav_out:
                wav_out.setnchannels(n_channels)
                wav_out.setsampwidth(sample_width)
                wav_out.setframerate(frame_rate)
                wav_out.writeframes(boosted.tobytes())

            logger.debug(
                "[TTS] session=%s gained %s: peak %d -> target %.0f%% FS (total gain %.1f dB)",
                self.session_id, path.name, int(peak),
                config.DEFAULT_TTS_TARGET_PEAK * 100,
                20.0 * math.log10(total_gain),
            )
        except Exception:
            logger.warning(
                "[TTS] session=%s could not apply gain to %s; using unmodified audio",
                self.session_id, path.name, exc_info=True,
            )

    def _on_audio(self, indata, frames, time, status) -> None:
        del frames, time
        if status:
            logger.warning("Audio callback status for %s: %s", self.session_id, status)
        self._audio_queue.put(indata[:, 0].copy())

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        samples = frame.astype(np.float32)
        return float(np.sqrt(np.mean(samples * samples)))

    def _endpoint_transcript_stable(self, transcript: str, silence_run_seconds: float) -> bool:
        """True once ``transcript`` has read unchanged for the stability window.

        Mirrors kiosk-voice-lab-main's ``assess_complete(text, stable)``,
        which requires two consecutive tick snapshots to agree before trusting
        a "finished-sounding" transcript — see
        config.DEFAULT_ENDPOINT_STABLE_SECONDS for why: a word-list check like
        _looks_complete can read "complete" on a transcript that a moment
        later turns out to have been truncated mid-utterance.

        Clocked on ``silence_run_seconds`` (audio-domain: frames-of-silence
        seen so far), NOT wall-clock time.monotonic(). Audio frequently
        arrives in network bursts (chunked HTTP pushes from the benchmark or
        the browser UI alike) — the frame loop can drain a whole burst of
        already-buffered silent frames in a few milliseconds of real time,
        advancing silence_run_seconds by hundreds of ms almost instantly.
        A wall-clock stability window could then never accumulate its 0.2s
        before the audio-domain silence_timeout_seconds cutoff arrived first,
        so the shortcut would silently never fire. silence_run_seconds
        already IS the correct clock the rest of this endpoint logic uses.

        State (the last-seen text and when it started reading that way)
        persists on the instance across calls, since this is evaluated once
        per frame (every DEFAULT_BLOCK_DURATION_SECONDS) while the endpoint's
        other gates are open. The stored/compared value is the normalized
        word-key from _endpoint_stability_key(), not the raw transcript, so a
        punctuation-only ASR hallucination on the trailing silence tail
        (e.g. "...please." -> "...please. .") cannot restart the window --
        only an actual change in the recognized words can.

        Only ever WITHHOLDS an early commit relative to the old behaviour —
        if the transcript is still changing, the caller falls through to the
        existing full silence_timeout_seconds wait, exactly as it would if
        _looks_complete itself had failed.

        DEFAULT_ENDPOINT_STABLE_SECONDS == 0 is a distinct, intentionally
        looser mode: trust the FIRST snapshot that reads complete, instead of
        requiring it to read the same way again on a second call. The
        two-confirmation design normally needs two ASR round trips (each
        measured 90-220ms) to land inside the ~0.15-0.40s shortcut window —
        on real hardware that is frequently too tight (measured: shortcut
        firing rate swings 0%-80% run to run purely on round-trip luck, see
        docs/performance-improvements-2026-09.md), so most turns fall back to
        the full silence_timeout_seconds wait even though the first
        transcript was already correct. Single-confirmation trades away the
        guard against a transcript that "reads complete" for one tick and
        then turns out to have been truncated mid-utterance a tick later —
        re-validate against the fixture/scripted benchmarks if this is
        lowered further or re-enabled after being disabled.
        """
        now = silence_run_seconds
        if config.DEFAULT_ENDPOINT_STABLE_SECONDS <= 0:
            return True
        # Compared on the normalized word-key, not the raw string, so a
        # punctuation-only hallucination (see _endpoint_stability_key) can't
        # masquerade as new speech and restart this window.
        key = _endpoint_stability_key(transcript)
        if key != self._endpoint_stable_transcript:
            self._endpoint_stable_transcript = key
            self._endpoint_stable_since = now
            return False
        if self._endpoint_stable_since is None:
            self._endpoint_stable_since = now
            return False
        return (now - self._endpoint_stable_since) >= config.DEFAULT_ENDPOINT_STABLE_SECONDS

    def _update_vad_threshold(self, rms: float, is_speech: bool) -> None:
        """Derive the speech gate from the measured background noise level.

        During the calibration window every frame is treated as background and
        collected; once enough frames exist the floor is taken as a low
        percentile of them (robust to a customer who starts talking straight
        away, since the quiet gaps between words still dominate the low
        percentile). Afterwards the floor keeps tracking, but only on frames
        classified as non-speech — adapting during speech would make the gate
        climb mid-utterance and cut the customer off.

        The resulting gate is always clamped to
        ``[DEFAULT_VAD_THRESHOLD_MIN, DEFAULT_VAD_THRESHOLD_MAX]`` so a freak
        measurement can never push it high enough to suppress all speech.

        Args:
            rms: Energy of the current frame.
            is_speech: Current classification of this frame, used to freeze
                floor adaptation while the customer is talking.
        """
        if not config.ADAPTIVE_VAD_ENABLED:
            return

        if self._vad_calibrating:
            self._vad_calibration_rms.append(rms)
            if len(self._vad_calibration_rms) < self._vad_calibration_frames:
                return
            floor = float(
                np.percentile(self._vad_calibration_rms, config.DEFAULT_VAD_FLOOR_PERCENTILE)
            )
            self._vad_calibrating = False
            self._vad_calibration_rms = []
        elif not is_speech:
            current = self._noise_floor if self._noise_floor is not None else rms
            # Asymmetric: track a quietening room quickly, but resist being
            # dragged upward by the quiet gaps inside an utterance.
            alpha = (
                config.DEFAULT_VAD_FLOOR_ADAPT_DOWN
                if rms < current
                else config.DEFAULT_VAD_FLOOR_ADAPT_UP
            )
            floor = (1.0 - alpha) * current + alpha * rms
        else:
            return

        gate = floor * (10.0 ** (config.DEFAULT_VAD_MARGIN_DB / 20.0))
        clamped = min(
            max(gate, float(config.DEFAULT_VAD_THRESHOLD_MIN)),
            float(config.DEFAULT_VAD_THRESHOLD_MAX),
        )

        first_time = self._noise_floor is None
        self._noise_floor = floor
        self._vad_threshold = clamped

        if first_time:
            # Logged at INFO deliberately: this single line is the on-site
            # diagnostic for whether the venue's noise floor is workable.
            logger.info(
                "[VAD] session=%s | noise floor RMS=%.0f (%.1f dBFS) | gate=%.0f%s | seed was %d",
                self.session_id,
                floor,
                20.0 * np.log10(max(floor, 1.0) / 32767.0),
                clamped,
                " (CLAMPED — venue louder than gate ceiling, "
                "falling back to permissive detection)"
                if clamped < gate
                else "",
                self.request.silence_threshold,
            )

    def _chunk_duration_seconds(self, frames: list[np.ndarray]) -> float:
        total_samples = sum(len(frame) for frame in frames)
        return total_samples / self.request.sample_rate

    def _trim_trailing_silence(
        self, frames: list[np.ndarray], silence_run_seconds: float
    ) -> list[np.ndarray]:
        """Drop trailing silence beyond a short decay tail before ASR sees it.

        Whisper hallucinates a sentence-completing word/phrase when fed audio
        that dangles into silence (measured case: a spurious trailing
        "Good." after a real order line). kiosk-voice-lab-main avoids this by
        trimming the audio actually sent to ASR down to
        "last speech sample + ~0.15s decay tail" rather than sending
        everything accumulated since speech stopped.

        This only shrinks what gets WRITTEN TO THE WAV FILE for this flush —
        it does not touch ``silence_run_seconds``/endpoint timing, which the
        caller keeps counting exactly as before.

        Args:
            frames: the chunk's frame buffer, oldest first. The trailing
                ``silence_run_seconds`` worth of it (at the tail) is silence;
                everything before that is presumed real speech.
            silence_run_seconds: how much trailing silence is currently
                sitting at the end of ``frames``.

        Returns:
            ``frames`` unchanged if there is nothing to trim (feature
            disabled, no trailing silence, or the decay tail already covers
            all of it); otherwise a shorter list with the excess trailing
            silence frames dropped.
        """
        if not config.DEFAULT_ASR_TRIM_TRAILING_SILENCE_ENABLED:
            return frames
        if silence_run_seconds <= config.DEFAULT_ASR_TRIM_DECAY_SECONDS:
            return frames  # already within the decay tail — nothing to cut
        if self._frame_duration_seconds <= 0:
            return frames
        silence_frame_count = int(round(silence_run_seconds / self._frame_duration_seconds))
        keep_frame_count = int(round(config.DEFAULT_ASR_TRIM_DECAY_SECONDS / self._frame_duration_seconds))
        frames_to_drop = silence_frame_count - keep_frame_count
        if frames_to_drop <= 0 or frames_to_drop >= len(frames):
            return frames
        logger.debug(
            "[ASR-TRIM] session=%s | trimming %.2fs of trailing silence "
            "(%d frames) to a %.2fs decay tail before ASR",
            self.session_id,
            frames_to_drop * self._frame_duration_seconds,
            frames_to_drop,
            config.DEFAULT_ASR_TRIM_DECAY_SECONDS,
        )
        return frames[:-frames_to_drop]

    def _scope_is_enrolled(self) -> bool:
        """Whether the analyzer already holds an enrolled voice for this conversation.

        Returns:
            True when a previous chunk in this conversation came back with an
            ``is_primary`` flag, which only happens once enrollment succeeded.
        """
        with BaseAudioSession._enrolled_scopes_lock:
            return self.agent_session_id in BaseAudioSession._enrolled_scopes

    def _mark_scope_enrolled(self) -> None:
        """Record that this conversation's reference voice is enrolled."""
        with BaseAudioSession._enrolled_scopes_lock:
            # Same unbounded-growth guard as _consecutive_rejections: one entry
            # per conversation would otherwise accumulate forever in a
            # long-running kiosk process. Dropping entries is safe — a cleared
            # scope just re-primes enrollment on its next long chunk.
            if (
                len(BaseAudioSession._enrolled_scopes) > _MAX_TRACKED_CONVERSATIONS
                and self.agent_session_id not in BaseAudioSession._enrolled_scopes
            ):
                BaseAudioSession._enrolled_scopes.clear()
            if self.agent_session_id not in BaseAudioSession._enrolled_scopes:
                BaseAudioSession._enrolled_scopes.add(self.agent_session_id)
                logger.info(
                    "[SPEAKER-ENROLL] session=%s scope=%s | analyzer has an enrolled "
                    "reference voice; intermediate chunks skip diarization from now on",
                    self.session_id, self.agent_session_id,
                )

    def _flush_chunk(self, frames: list[np.ndarray], is_final: bool = True) -> None:
        # Streaming mode's final commit may legitimately carry NO new frames:
        # every real audio byte was already sent via send_frame() as it was
        # captured, and an earlier non-final flush can have already swept the
        # whole utterance into one fire-and-forget commit (see the empty-
        # chunk_frames final-commit call site in _process_frame_stream). This
        # call's only job then is to block for that already-in-flight
        # analyzer result — there is nothing left to concatenate.
        audio = np.concatenate(frames, axis=0) if frames else np.zeros(0, dtype=np.int16)
        temp_path: str | None = None
        if not self._streaming_active:
            temp_path = self._write_temp_wav(audio)
        try:
            duration = len(audio) / self.request.sample_rate
            # Full diarization always runs on the final tail chunk. On
            # intermediate chunks (max-chunk-size-cap flush, adaptive-pause
            # pre-warm flush) it is gated by DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED
            # — see config.py for the latency/accuracy tradeoff this encodes.
            # Exception: an intermediate chunk long enough to enroll a voice is
            # diarized while the conversation has no enrolled reference yet,
            # otherwise the analyzer can never set is_primary and the speaker
            # filter loses its bystander protection
            # (DEFAULT_DIARIZATION_ENROLLMENT_PRIMING_ENABLED).
            diarization_requested = config.DEFAULT_DIARIZATION_ENABLED and (
                is_final
                or config.DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED
                or (
                    config.DEFAULT_DIARIZATION_ENROLLMENT_PRIMING_ENABLED
                    and duration >= config.DEFAULT_DIARIZATION_ENROLL_MIN_SECONDS
                    and not self._scope_is_enrolled()
                )
            )
            logger.info(
                "[CHUNK] session=%s | flushing %.2fs of audio, is_final=%s, diarization=%s",
                self.session_id, duration, is_final, diarization_requested,
            )
            _t_asr = time.monotonic()
            if self._streaming_active:
                # Audio for this chunk was already streamed continuously as
                # it was captured (see send_frame() in _process_frame_stream).
                # commit() only draws the utterance boundary here -- for
                # preview (non-final) chunks it's fire-and-forget so this
                # never blocks the flush worker; for the final chunk it
                # blocks up to a bounded timeout for a fresh snapshot, then
                # falls back to whatever is already available regardless.
                #
                # Peek the cache BEFORE committing when this call carries no
                # new frames (see the commit-only final-flush call site in
                # _process_frame_stream): if an earlier flush's fire-and-
                # forget commit already landed its real completion, sending
                # yet another commit here would only race that arrival
                # against a FRESH DEFAULT_REALTIME_FINAL_COMMIT_TIMEOUT_SECONDS
                # window -- a content-free commit produces no NEW completion
                # event, so a lost race can only time out, not resolve early.
                # Measured live: the real transcript landed <25ms before this
                # exact wait began and the turn still paid the full 2.5s.
                # A chunk WITH new frames must still always commit and wait
                # -- only a content-free commit is safe to skip this way.
                _cached = self.realtime_client.latest_snapshot()
                if not frames and (_cached.get("text") or _cached.get("segments")):
                    payload = _cached
                else:
                    self.realtime_client.commit(
                        wait=is_final,
                        timeout=config.DEFAULT_REALTIME_FINAL_COMMIT_TIMEOUT_SECONDS if is_final else 0.0,
                    )
                    payload = self.realtime_client.latest_snapshot()
            else:
                payload = self.client.transcribe_file(
                    temp_path,
                    language=self.request.language,
                    temperature=self.request.temperature,
                    diarization=diarization_requested,
                    session_id=self._analyzer_session_id,
                    speaker_scope_id=self.agent_session_id,
                    # Bias Whisper towards the real menu vocabulary. Without it the
                    # decoder spells product names phonetically ("aloo tiki",
                    # "Kin Burger"), which the catalogue's fuzzy resolver then
                    # fails to match, so the item silently never reaches the cart.
                    prompt=config.DEFAULT_ASR_PROMPT,
                )
            # Accumulate genuine ASR time. Chunks are transcribed as they are
            # flushed during capture, long before _finalize_run runs, so this
            # is the only place the cost can be observed.
            self._asr_ms_total += (time.monotonic() - _t_asr) * 1000
            self._asr_chunks += 1
            # Latch the analyzer's assigned session id from the first
            # response so subsequent chunks reuse the same server-side
            # session (and its enrolled primary voice embedding).
            assigned = (
                payload.get("_analyzer_session_id")
                if isinstance(payload, dict)
                else None
            )
            if assigned and assigned != self._analyzer_session_id:
                logger.info(
                    "[CHUNK] session=%s | analyzer session pinned to %s",
                    self.session_id, assigned,
                )
                self._analyzer_session_id = assigned
            segments: list[dict] = payload.get("segments", []) if isinstance(payload, dict) else []
            # A segment resolved as is_primary=True proves the analyzer now holds
            # an enrolled reference voice for this conversation, so later
            # intermediate chunks no longer need to pay for diarization. The key
            # alone is not enough: the analyzer emits is_primary=False on every
            # segment while enrollment is still deferred (no span >= the minimum
            # enrollable duration yet), and treating that as success would end
            # priming before a reference voice ever exists.
            if any(segment.get("is_primary") is True for segment in segments):
                self._mark_scope_enrolled()
            raw_text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else str(payload).strip()
            # Snapshot the cursor BEFORE advancing it for this chunk — the
            # segments dedup below must filter against what was already
            # committed prior to this flush, not after.
            _committed_before = self._last_analyzer_segment_end

            # Did the analyzer report ANY content at all for this chunk
            # (before the dedup filtering below mutates segments/raw_text)?
            # Pipeline.transcribe() only advances its own persisted
            # `duration` when chunk_by_silence actually yields speech --
            # audio that is pure silence leaves the analyzer's internal
            # cumulative offset untouched. If we always advanced our cursor
            # by this chunk's raw (client-measured) audio length regardless,
            # a silent commit would push the cursor PAST where the analyzer
            # itself is, and the next chunk's genuine new segments would be
            # wrongly compared against a cursor that is now ahead of them --
            # falsely deduping real speech away as "already seen".
            _response_had_content = bool(segments) or bool(raw_text)

            logger.info(
                "[CHUNK] session=%s | audio-analyzer response: %d segment(s), flat_text=%r",
                self.session_id, len(segments), raw_text[:120],
            )

            # ── Dedupe against analyzer's cumulative session state ──────────
            # The analyzer reuses our session_id in append_to_session mode and
            # returns EVERY segment ever produced for the session (with
            # timestamps offset by the accumulated duration). Keep only the
            # segments that start after everything already committed by a
            # PRIOR flush — which may have been a non-diarized (flat-text)
            # flush, so this compares against _committed_before (client-
            # tracked cumulative audio time), not a segment-derived value —
            # otherwise every prior utterance gets re-appended to the
            # transcript on each new chunk.
            if segments:
                fresh_segments = [
                    s for s in segments
                    if float(s.get("end", 0.0)) > _committed_before + 1e-3
                ]
                if len(fresh_segments) != len(segments):
                    logger.info(
                        "[CHUNK] session=%s | deduped %d cumulative segment(s) → %d fresh (cursor=%.2fs)",
                        self.session_id,
                        len(segments) - len(fresh_segments),
                        len(fresh_segments),
                        _committed_before,
                    )
                segments = fresh_segments
                if segments:
                    # Rebuild raw_text from fresh segments so the flat-text
                    # fallback (used when diarization is off or returns no
                    # segments) is also free of the cumulative duplicates.
                    raw_text = " ".join(
                        s.get("text", "").strip() for s in segments if s.get("text", "").strip()
                    ).strip()
                else:
                    raw_text = ""
            elif raw_text:
                # No segments at all (diarization not requested on this
                # chunk — the normal case for intermediate flushes). The
                # flat "text" field is still the analyzer's FULL cumulative
                # session transcript, not just this chunk's new audio, so it
                # needs the same kind of cursor-based dedup as segments
                # above — just keyed on string prefix instead of timestamp,
                # since there's no per-segment timing here.
                cumulative = raw_text
                if cumulative.startswith(self._last_cumulative_flat_text):
                    delta = cumulative[len(self._last_cumulative_flat_text):].strip()
                    if delta != raw_text:
                        logger.info(
                            "[CHUNK] session=%s | deduped cumulative flat text "
                            "(%d chars) → %d new char(s)",
                            self.session_id, len(cumulative), len(delta),
                        )
                    raw_text = delta
                    self._last_cumulative_flat_text = cumulative
                else:
                    # Analyzer's cumulative text no longer starts with what
                    # we last saw (e.g. session state reset or an unexpected
                    # response shape) — fail safe by using the full text
                    # rather than silently dropping words, and reset the
                    # cursor to match so future calls dedupe correctly again.
                    logger.warning(
                        "[CHUNK] session=%s | flat-text cursor mismatch — "
                        "using full response text unmodified",
                        self.session_id,
                    )
                    self._last_cumulative_flat_text = cumulative

            # Advance the shared cursor by THIS chunk's own measured
            # duration — so a later chunk's segment-based dedup (which may
            # request diarization even when this one didn't) has an accurate
            # "already committed" boundary to filter against.
            #
            # Only advance when the analyzer actually reported content: a
            # chunk with no segments/text means the analyzer's own
            # cumulative offset didn't move either (see
            # _response_had_content above) — advancing ours anyway would
            # desync the two and falsely dedup the next real chunk's speech.
            if _response_had_content:
                self._last_analyzer_segment_end = _committed_before + duration

            if segments and diarization_requested:
                text = self._filter_target_speaker(segments)
            else:
                if diarization_requested and not segments:
                    logger.info(
                        "[CHUNK] session=%s | diarization enabled but no segments returned — using flat text",
                        self.session_id,
                    )
                text = raw_text

            if text:
                # Strip Whisper hallucination tokens (e.g. [BLANK_AUDIO], [Music])
                text = _WHISPER_JUNK.sub("", text).strip()
            if text:
                # Correct the "cart" -> "card" ASR homophone before the agent
                # ever sees the transcript (see _normalize_card_cart_homophone).
                normalized = _normalize_card_cart_homophone(text)
                if normalized != text:
                    logger.info(
                        "[CHUNK] session=%s | normalized card->cart homophone: %r -> %r",
                        self.session_id, text[:120], normalized[:120],
                    )
                    text = normalized
            if text and _WHISPER_FILLER.fullmatch(text):
                logger.info(
                    "[CHUNK] session=%s | dropping filler-only transcription: %r",
                    self.session_id, text[:60],
                )
                text = ""
            if text:
                # Drop a leading run of words the transcript already ends
                # with. The two dedup paths above use different cursors — the
                # flat-text path tracks committed *text*, the segment path
                # tracks a *timestamp* — so a segment that merely spans the
                # boundary of a previous flat-text flush is judged "fresh"
                # by `end > cursor` and re-appends words already committed.
                # Observed live: a non-diarized 1.50s flush committed "I would
                # like to order one classic", then the final diarized flush
                # returned the same words as a cumulative segment ending at
                # ~1.6s and appended them a second time.
                deduped = self._strip_duplicate_prefix(text)
                if deduped != text:
                    logger.info(
                        "[CHUNK] session=%s | stripped %d duplicate leading word(s): %r -> %r",
                        self.session_id,
                        len(text.split()) - len(deduped.split()),
                        text[:80], deduped[:80],
                    )
                    text = deduped
            if text:
                logger.info(
                    "[CHUNK] session=%s | appending to transcript: %r",
                    self.session_id, text[:120],
                )
                with self._lock:
                    self.transcript_parts.append(text)
                    transcript_snapshot = " ".join(self.transcript_parts)
                # This chunk's speech is now durably confirmed transcribed —
                # any speech captured strictly before it is accounted for, so
                # the final-flush skip (KIOSK_CORE_SKIP_EMPTY_FINAL_FLUSH_ENABLED)
                # is safe again until the next is_speech frame. Do NOT clear
                # this on an empty/no-usable-text result (the `else` branch
                # below) -- that confirms nothing, so any pending speech from
                # this or an earlier chunk must still be treated as unconfirmed.
                self._unconfirmed_speech_pending = False
                # Fire a speculative draft off the growing preview transcript
                # while the customer is still talking — never on the final
                # (is_final=True) tail chunk, since by then the real,
                # endpoint-triggered turn is about to run for real anyway.
                if not is_final:
                    self._maybe_trigger_speculative_draft(transcript_snapshot)
            else:
                logger.info(
                    "[CHUNK] session=%s | chunk produced no usable text (filtered or empty)",
                    self.session_id,
                )
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)

    def _log_speculative_draft_match(self, final_transcript: str) -> None:
        """Log (diagnostic only) whether the last speculative draft's input
        transcript matches the real, endpoint-fired transcript.

        Purely informational — does not affect the real turn in any way.
        Used to measure how often speculative drafting would have a usable
        result if a future round adds a real skip-the-LLM replay path.
        """
        with self._speculative_lock:
            draft = self._speculative_draft
        if draft is None:
            logger.info("[SPEC-DRAFT] session=%s | no speculative draft was ready", self.session_id)
            return
        norm_final = re.sub(r"\s+", " ", final_transcript).strip().lower()
        norm_draft = re.sub(r"\s+", " ", draft["transcript"]).strip().lower()
        match = norm_final == norm_draft
        logger.info(
            "[SPEC-DRAFT] session=%s | draft_match=%s draft_transcript=%r final_transcript=%r",
            self.session_id, match, draft["transcript"][:120], final_transcript[:120],
        )

    def _maybe_trigger_speculative_draft(self, transcript: str) -> None:
        """Fire a background speculative agent draft for the preview transcript.

        Best-effort, fire-and-forget: spawns a daemon thread and returns
        immediately. See config.DEFAULT_SPECULATIVE_DRAFT_ENABLED for the
        full rationale and safety argument (every mutating tool the draft
        might call is forced into dry_run server-side).
        """
        if not config.DEFAULT_SPECULATIVE_DRAFT_ENABLED:
            return
        if self.agent_client is None:
            return
        transcript = transcript.strip()
        if not transcript:
            return
        with self._speculative_lock:
            self._speculative_generation += 1
            generation = self._speculative_generation
        threading.Thread(
            target=self._run_speculative_draft,
            args=(transcript, generation),
            name=f"spec-draft-{self.session_id}-{generation}",
            daemon=True,
        ).start()

    def _run_speculative_draft(self, transcript: str, generation: int) -> None:
        """Run one speculative draft turn and store it if still the newest.

        Runs entirely off the customer-facing critical path — this thread's
        result is only ever consulted (never awaited) by the real turn later.
        """
        history = list(getattr(self.request, "history", []) or [])
        result = self.agent_client.get_speculative_draft(
            transcription=transcript,
            session_id=self.agent_session_id,
            user_id=getattr(self.request, "user_id", None) or config.DEFAULT_ORDERING_USER_ID,
            history=history,
        )
        if result is None:
            return
        with self._speculative_lock:
            if generation != self._speculative_generation:
                # A newer preview transcript has already started a fresher
                # draft — this one is stale, discard it (newest-wins).
                logger.info(
                    "[SPEC-DRAFT] session=%s | draft gen=%d superseded (now gen=%d) — discarded",
                    self.session_id, generation, self._speculative_generation,
                )
                return
            self._speculative_draft = {"transcript": transcript, "result": result}
        logger.info(
            "[SPEC-DRAFT] session=%s | draft gen=%d ready | transcript=%r reply_len=%d tool_calls=%s",
            self.session_id, generation, transcript[:120],
            len(result.get("reply", "")), result.get("tool_calls", []),
        )
        if config.DEFAULT_SPECULATIVE_TTS_PRESYNTH_ENABLED:
            self._prewarm_tts_from_draft(result.get("reply", ""), generation)

    def _admit_to_opener_cache(self, cache_key: tuple, output_path: Path, sentence: str) -> None:
        """Retain a finished short opener segment for reuse by later turns.

        Called only after a REAL synthesis has completed and been trimmed and
        gain-adjusted, so the retained file is exactly what this pipeline
        produces for that sentence — replaying it later is indistinguishable
        from synthesising it again, minus the ~280 ms Kokoro/CPU cost.

        This never triggers a TTS request of its own; it is a pure copy of
        work already done. Failures are swallowed, since losing a cache entry
        only costs a future turn its normal synthesis time.

        Args:
            cache_key: Key from _opener_cache_key (sentence + voice tuple).
            output_path: Finished, post-processed segment to retain.
            sentence: Source text, for logging only.
        """
        try:
            with _OPENER_TTS_CACHE_LOCK:
                if cache_key in _OPENER_TTS_CACHE:
                    return
                if len(_OPENER_TTS_CACHE) >= config.DEFAULT_TTS_OPENER_CACHE_MAX_ENTRIES:
                    return
                # Reserve the slot inside the lock so two workers racing on the
                # same opener cannot both copy into the same destination.
                slot = len(_OPENER_TTS_CACHE)
                _OPENER_TTS_CACHE[cache_key] = ""
            _OPENER_TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cached_path = _OPENER_TTS_CACHE_DIR / f"opener_{slot:03d}.wav"
            shutil.copyfile(output_path, cached_path)
            with _OPENER_TTS_CACHE_LOCK:
                _OPENER_TTS_CACHE[cache_key] = str(cached_path)
            logger.info(
                "[TTS] session=%s | opener cached %r — future turns skip synthesis",
                self.session_id, sentence,
            )
        except Exception:
            with _OPENER_TTS_CACHE_LOCK:
                # Drop the reservation so a later turn can retry.
                if _OPENER_TTS_CACHE.get(cache_key) == "":
                    _OPENER_TTS_CACHE.pop(cache_key, None)
            logger.warning(
                "[TTS] session=%s | could not cache opener %r — harmless, "
                "future turns will synthesise normally",
                self.session_id, sentence, exc_info=True,
            )

    def _prewarm_tts_from_draft(self, reply_text: str, generation: int) -> None:
        """Pre-synthesise a speculative draft's predicted reply, sentence by sentence.

        Populates self._tts_cache keyed by a whitespace/case/punctuation-
        normalized form of the sentence (see _normalize_sentence_for_cache_key)
        rather than the exact raw string — tolerant of trivial formatting
        drift, but still requires the real turn's sentence to be the SAME
        wording. Never served directly — the real turn's _tts_worker only
        uses a cached file when the real, fully-guarded reply happens to
        produce a matching sentence (see _tts_cache docstring in __init__).
        """
        reply_text = reply_text.strip()
        if not reply_text:
            return
        sentences = [s.strip() for s in _SPEC_SENTENCE_SPLIT_RE.split(reply_text) if s.strip()]
        for sentence in sentences:
            cache_key = _normalize_sentence_for_cache_key(sentence)
            if not cache_key:
                continue
            with self._speculative_lock:
                if generation != self._speculative_generation:
                    return  # superseded mid-loop — stop synthesising stale sentences
                if cache_key in self._tts_cache:
                    continue  # already warm from an earlier draft
                self._spec_tts_index += 1
                idx = self._spec_tts_index
            output_path = self._session_output_dir / f"spec_{idx:04d}.wav"
            try:
                # May run before _emit_opener (which normally creates this
                # directory) since speculative synthesis can fire well before
                # the real endpoint/turn.
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.tts_client.synthesize_to_file(
                    text=sentence,
                    output_path=str(output_path),
                    model=self.request.tts_model,
                    voice=self.request.tts_voice,
                    language=self.request.tts_language,
                    instructions=self.request.tts_instructions,
                )
                if config.DEFAULT_TTS_TRIM_ENABLED:
                    self._trim_tts_segment(output_path, sentence)
                if config.DEFAULT_TTS_GAIN_ENABLED:
                    self._apply_tts_gain(output_path)
                with self._speculative_lock:
                    if generation != self._speculative_generation:
                        output_path.unlink(missing_ok=True)
                        return
                    self._tts_cache[cache_key] = str(output_path)
                logger.info(
                    "[SPEC-TTS] session=%s | pre-synthesised sentence (%d chars) gen=%d",
                    self.session_id, len(sentence), generation,
                )
            except Exception:
                logger.warning(
                    "[SPEC-TTS] session=%s | speculative pre-synthesis failed — "
                    "harmless, real turn will synthesise normally",
                    self.session_id, exc_info=True,
                )
                return

    def _note_rejected_speech(self, segments: list[dict], reason: str) -> None:
        """Record that transcribed speech was discarded by the speaker filter.

        Only counts chunks that actually carried words. A chunk of pure
        silence yields empty segment text and must stay classified as "no
        speech", otherwise every silent moment would trigger the "please
        repeat" prompt.

        Args:
            segments: The diarized segments that were rejected.
            reason: Short tag naming which filter rule rejected them.
        """
        spoken = " ".join(s.get("text", "") for s in segments).strip()
        if not spoken:
            return
        # Mirrors the defensive getattr used for _primary_speaker_id: the
        # filter is exercised directly by unit tests against bare instances.
        self._rejected_speech_chunks = getattr(self, "_rejected_speech_chunks", 0) + 1
        logger.info(
            "[SPEAKER-FILTER] session=%s | rejected speech recorded (%s) "
            "| rejected_chunks=%d | text=%r",
            self.session_id, reason, self._rejected_speech_chunks, spoken[:120],
        )

    def _strip_duplicate_prefix(self, text: str) -> str:
        """Remove leading words of ``text`` already committed to the transcript.

        The analyzer runs in cumulative (``append_to_session``) mode, so each
        response restates earlier speech. Two independent dedup cursors guard
        against that — a text prefix for the flat-text path and a timestamp for
        the segment path — but they cannot see each other, so a segment that
        straddles the boundary of a previous flat-text flush slips through and
        duplicates words. This is the final, path-agnostic backstop.

        Only overlaps of at least ``config.DEFAULT_DUPLICATE_PREFIX_MIN_WORDS``
        words are stripped, so genuine short repetitions ("yes yes", "two two")
        survive.

        Args:
            text: Newly transcribed text about to be appended.

        Returns:
            ``text`` with any duplicated leading word run removed.
        """
        with self._lock:
            committed_words = " ".join(self.transcript_parts).split()
        new_words = text.split()
        if not committed_words or not new_words:
            return text

        def _key(word: str) -> str:
            return word.lower().strip(".,!?;:\"'")

        committed_keys = [_key(w) for w in committed_words]
        new_keys = [_key(w) for w in new_words]
        max_overlap = min(len(committed_keys), len(new_keys))
        for n in range(max_overlap, config.DEFAULT_DUPLICATE_PREFIX_MIN_WORDS - 1, -1):
            if committed_keys[-n:] == new_keys[:n]:
                return " ".join(new_words[n:])
        return text

    def _filter_target_speaker(self, segments: list[dict]) -> str:
        """Filter diarized segments to keep only the primary customer's speech.

        Primary speaker is determined in order of precedence:
        1. Analyzer-provided ``is_primary`` flag on the segment — honoured
           directly when present, so the analyzer's own enrollment logic wins.
           Its verdict is authoritative in *both* directions: if it marks every
           segment non-primary, the chunk is dropped outright.
        2. First-speaker lock-on — the first speaker label seen in this session
           is treated as the customer.  Every subsequent segment from that label
           is kept; any other label is unconditionally dropped.

        Semantic fallback is used only before the primary speaker is
        established (i.e. on the very first chunk that has no clear speaker
        label), so a legitimate opening utterance is never lost.
        """
        if not segments:
            return ""

        # Defensive read — guard against subclasses or test stubs that may not
        # have called BaseAudioSession.__init__.
        primary_speaker_id: str | None = getattr(self, "_primary_speaker_id", None)

        # ── Honor analyzer-provided is_primary when present ──────────────────
        # The flag only appears once the analyzer holds an enrolled reference
        # voice for this conversation (scoped by speaker_scope_id), so when it
        # is present its verdict is authoritative and must be honoured in BOTH
        # directions. In particular "no segment is primary" means the analyzer
        # has positively matched this speech against the enrolled customer and
        # rejected it — the chunk must be dropped, never fall through to the
        # first-speaker rule below (which would lock on to the interloper,
        # because _primary_speaker_id resets on every new audio session).
        #
        # Exception: that authority only holds once a reference voice has
        # actually been enrolled for this scope (_scope_is_enrolled()). If
        # enrollment never happened — e.g. an earlier preview chunk that
        # should have primed it came back empty (ASR/hallucination-filter
        # miss on that chunk) — the analyzer's is_primary=False here reflects
        # "no reference to compare against", not a genuine identity mismatch.
        # Treating that as authoritative silently discards the customer's
        # only utterance for the whole turn. In that case fall through to the
        # same semantic-domain fallback used before any primary is known,
        # instead of hard-dropping.
        if any("is_primary" in s for s in segments):
            primary_segments = [s for s in segments if s.get("is_primary")]
            if not primary_segments and not self._scope_is_enrolled():
                logger.warning(
                    "[SPEAKER-FILTER] session=%s | analyzer rejected all %d segment(s) as "
                    "non-primary but scope was never enrolled — treating as low-confidence, "
                    "falling back to semantic heuristics instead of hard drop | text=%r",
                    self.session_id, len(segments),
                    " ".join(s.get("text", "") for s in segments).strip()[:120],
                )
                # Fall through to the first-speaker/semantic-fallback logic
                # below by treating this like a plain (non-is_primary) batch.
            elif not primary_segments:
                if not config.DEFAULT_SPEAKER_STRICT_DROP:
                    logger.warning(
                        "[SPEAKER-FILTER] session=%s | analyzer rejected all %d segment(s) but "
                        "strict drop is DISABLED — falling back to heuristics",
                        self.session_id, len(segments),
                    )
                else:
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | analyzer rejected all %d segment(s) as "
                        "non-primary — chunk DROPPED | text=%r",
                        self.session_id, len(segments),
                        " ".join(s.get("text", "") for s in segments).strip()[:120],
                    )
                    self._note_rejected_speech(segments, "analyzer_non_primary")
                    return ""
            else:
                # Also update / initialise the lock-on label from the first
                # primary segment so label-based filtering stays in sync.
                label = primary_segments[0].get("speaker", "")
                if label and primary_speaker_id is None:
                    self._primary_speaker_id = label
                    logger.info(
                        "[SPEAKER-LOCK] session=%s | primary speaker locked → %s (is_primary flag)",
                        self.session_id, label,
                    )
                final_text = " ".join(seg.get("text", "") for seg in primary_segments).strip()
                logger.info(
                    "[SPEAKER-FILTER] session=%s | is_primary path: kept=%d | final_text=%r",
                    self.session_id, len(primary_segments), final_text[:120],
                )
                return final_text

        # ── Lock on to the first speaker seen in this session ────────────────
        if primary_speaker_id is None:
            for seg in segments:
                label = seg.get("speaker", "")
                if label:
                    self._primary_speaker_id = label
                    primary_speaker_id = label
                    logger.info(
                        "[SPEAKER-LOCK] session=%s | primary speaker locked → %s (first speaker rule)",
                        self.session_id, primary_speaker_id,
                    )
                    break

        # ── Classify segments as primary / non-primary ───────────────────────
        if primary_speaker_id:
            kept_segments = [s for s in segments if s.get("speaker") == primary_speaker_id]
            discarded_segments = [s for s in segments if s.get("speaker") != primary_speaker_id]
        else:
            # No speaker label at all — treat everything as potentially primary
            kept_segments = []
            discarded_segments = list(segments)

        logger.info(
            "[SPEAKER-FILTER] session=%s | processing %d segment(s): %d primary, %d non-primary",
            self.session_id, len(segments), len(kept_segments), len(discarded_segments),
        )
        for i, segment in enumerate(segments):
            is_primary = primary_speaker_id is not None and segment.get("speaker") == primary_speaker_id
            logger.info(
                "[SPEAKER-FILTER] session=%s | seg[%d] speaker=%s is_primary=%s → %s | text=%r",
                self.session_id, i, segment.get("speaker", "UNKNOWN"), is_primary,
                "KEEP" if is_primary else "DISCARD", segment.get("text", "")[:80],
            )

        # ── Semantic fallback — only before primary is established ───────────
        if not kept_segments and discarded_segments:
            if primary_speaker_id:
                # Primary is known — non-primary utterances are noise, drop them.
                logger.info(
                    "[SPEAKER-FILTER] session=%s | non-primary speech only, primary=%s — chunk DROPPED (no fallback)",
                    self.session_id, primary_speaker_id,
                )
            else:
                # Primary not yet established — run semantic fallback so the
                # very first domain utterance (before a clean speaker label
                # arrives) is not silently discarded.
                logger.info(
                    "[SPEAKER-FILTER] session=%s | primary not yet established — running semantic fallback on %d segment(s)",
                    self.session_id, len(discarded_segments),
                )
                best_score = 0.0
                best_segment: dict | None = None
                for segment in discarded_segments:
                    words = segment.get("text", "").lower().split()
                    if not words:
                        continue
                    overlap = sum(1 for w in words if w in _DOMAIN_KEYWORDS)
                    score = overlap / max(len(words), 1)
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | fallback score speaker=%s score=%.2f | text=%r",
                        self.session_id, segment.get("speaker", "UNKNOWN"), score, segment.get("text", "")[:80],
                    )
                    if score > best_score:
                        best_score = score
                        best_segment = segment

                if best_segment is not None and best_score >= config.DEFAULT_SEMANTIC_FALLBACK_THRESHOLD:
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | semantic fallback ACCEPTED speaker=%s score=%.2f | text=%r",
                        self.session_id, best_segment.get("speaker", "UNKNOWN"), best_score, best_segment.get("text", "")[:80],
                    )
                    kept_segments = [best_segment]
                else:
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | semantic fallback found no domain match (best_score=%.2f, threshold=%.2f) → chunk DROPPED",
                        self.session_id, best_score, config.DEFAULT_SEMANTIC_FALLBACK_THRESHOLD,
                    )

        final_text = " ".join(seg.get("text", "") for seg in kept_segments).strip()
        if not final_text:
            # Covers both remaining rejection routes: a known primary was
            # silent while somebody else spoke, and the semantic fallback
            # finding no domain match before a primary was established.
            self._note_rejected_speech(discarded_segments, "no_primary_segment_kept")
        logger.info(
            "[SPEAKER-FILTER] session=%s | RESULT: kept=%d dropped=%d | final_text=%r",
            self.session_id,
            len(kept_segments),
            len(discarded_segments) - (1 if kept_segments and discarded_segments else 0),
            final_text[:120],
        )
        return final_text

    def _write_temp_wav(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(prefix=f"{self.session_id}-", suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.request.sample_rate)
            wav_file.writeframes(audio.astype(np.int16).tobytes())

        return temp_path


class BrowserStreamSession(BaseAudioSession):
    """Session that receives audio chunks pushed from the browser via HTTP.

    Call push_audio(wav_bytes) from the HTTP handler each time a chunk arrives.
    The session applies the same RMS silence detection and chunk-flushing logic
    as MicrophoneSession.  It ends automatically when:
      - silence_timeout_seconds of silence follows detected speech, OR
      - max_session_seconds of captured audio have been processed, OR
      - stop() is called explicitly (e.g. user clicks stop-recording in browser).
    """

    def __init__(
        self,
        request: SessionStartRequest,
        on_complete: Callable[[str], None] | None = None,
    ):
        super().__init__(request=request, on_complete=on_complete)
        self._thread = threading.Thread(target=self._run, name=f"browser-session-{self.session_id}", daemon=True)
        self._source_kind = "browser"
        # Sentinel: None means end-of-stream
        self._push_queue: Queue[np.ndarray | None] = Queue()

    def push_audio(self, wav_bytes: bytes) -> None:
        """Called from the HTTP handler for each incoming audio chunk."""
        # Ground-truth playback anchor (see FileAudioSession._t_playback_start):
        # stamped on the FIRST chunk this session ever receives, so
        # playback_to_first_audio_ms is also populated for browser/streamed
        # turns, not just file-replay ones. A benchmark client (or the real
        # browser UI) that streams chunks as they are captured makes this a
        # reasonable proxy for "when the customer started talking", with only
        # network/client-buffering jitter as error — same caveat file-replay
        # already carries from thread-scheduling jitter.
        if self._t_playback_start is None:
            self._t_playback_start = time.monotonic()
        # Browser chunks arrive as WAV containers. Decode them first so we only
        # enqueue actual PCM frames (not RIFF/WAV headers), which keeps RMS/VAD
        # and ASR input stable across chunk boundaries.
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                raw = wav_file.readframes(wav_file.getnframes())

            if sample_width != 2:
                raise ValueError(f"Unsupported WAV sample width: {sample_width * 8}-bit")

            audio = np.frombuffer(raw, dtype=np.int16)
            if channels > 1:
                audio = audio.reshape(-1, channels)[:, 0]

            if sample_rate != self.request.sample_rate:
                logger.warning(
                    "[CHUNK] session=%s | WAV sample rate %s differs from session sample_rate %s",
                    self.session_id,
                    sample_rate,
                    self.request.sample_rate,
                )
        except (wave.Error, EOFError) as exc:
            logger.debug(
                "[CHUNK] session=%s | WAV decode failed; falling back to raw PCM (%s)",
                self.session_id,
                exc,
            )
            # Backward-compat fallback for legacy clients that might post raw
            # PCM bytes directly instead of a WAV container.
            audio = np.frombuffer(wav_bytes, dtype=np.int16)

        # Split into frame-sized pieces so _process_frame_stream sees uniform frames
        for start in range(0, len(audio), self._frame_samples):
            frame = audio[start : start + self._frame_samples]
            if len(frame) > 0:
                self._push_queue.put(frame.copy())

    def signal_end(self) -> None:
        """Signal that the browser has stopped recording (enqueue sentinel).

        This is also the honest voice-to-voice anchor for browser sessions.
        The silence-timeout endpoint path stamps ``_t_last_word`` itself, but a
        browser turn normally ends because the customer released the mic, which
        never reaches that path — so without this stamp `voice_to_voice_ms`
        came back ``None`` for every turn driven from the UI, which is exactly
        where the metric matters most.
        """
        if self._t_last_word is None:
            # Prefer the actual last-speech-frame-send instant over "now" —
            # the mic-release signal can arrive a little after the last real
            # word (button-release reaction time, any trailing buffered
            # frames still being sent), so it is a slightly late proxy at
            # best. See _t_last_speech_frame_sent's docstring in __init__.
            self._t_last_word = self._t_last_speech_frame_sent or time.monotonic()
            # Convert the monotonic anchor to an approximate wall-clock instant
            # purely for human-readable logging -- previously this logged a
            # fresh datetime.now(UTC) instead, which silently disagreed with
            # the actual anchor by however long ago the last speech frame was
            # (the customer's post-speech pause before releasing the mic).
            # That made the log line look like voice_to_voice_ms didn't add
            # up; it always did, the log was just showing the wrong instant.
            _gap_s = time.monotonic() - self._t_last_word
            anchor_wall_ts = datetime.now(UTC) - timedelta(seconds=_gap_s)
            logger.info(
                "[VOICE2VOICE] session=%s conversation=%s event=last_word_spoken "
                "ts=%s (signal_end received %.0fms later; source=%s)",
                self.session_id, self.agent_session_id,
                anchor_wall_ts.isoformat(),
                _gap_s * 1000,
                "observed" if self._t_last_speech_frame_sent is not None else "signal_end",
            )
        self._push_queue.put(None)

    def _run(self) -> None:
        final_status = "completed"
        end_reason = self.end_reason or "completed"
        try:
            final_status, end_reason = self._process_frame_stream(self._iter_push_frames())
            if final_status == "completed" and not self._speech_started:
                end_reason = "no_speech_detected"
        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("Browser stream session %s failed", self.session_id)
        finally:
            self._finalize_run(final_status, end_reason)

    def _iter_push_frames(self):
        while not self._stop_event.is_set():
            try:
                frame = self._push_queue.get(timeout=0.25)
            except Empty:
                continue
            if frame is None:
                # End-of-stream sentinel from signal_end()
                break
            yield frame


class MicrophoneSession(BaseAudioSession):
    def __init__(
        self,
        request: SessionStartRequest,
        on_complete: Callable[[str], None] | None = None,
    ):
        super().__init__(request=request, on_complete=on_complete)
        self._thread = threading.Thread(target=self._run, name=f"mic-session-{self.session_id}", daemon=True)
        self._source_kind = "microphone"

    def _run(self) -> None:
        final_status = "completed"
        end_reason = self.end_reason or "completed"
        try:
            with sd.InputStream(
                samplerate=self.request.sample_rate,
                blocksize=self._frame_samples,
                channels=1,
                dtype="int16",
                device=self.request.device,
                callback=self._on_audio,
            ):
                def iter_frames():
                    while not self._stop_event.is_set():
                        try:
                            yield self._audio_queue.get(timeout=0.25)
                        except Empty:
                            continue

                final_status, end_reason = self._process_frame_stream(iter_frames())
        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("Microphone session %s failed", self.session_id)
        finally:
            self._finalize_run(final_status, end_reason)


class TextQuerySession(BaseAudioSession):
    """A typed-question turn. Skips audio capture and ASR entirely, feeding the
    text straight through the RAG/agent pipeline. Per the UI's text-input
    design the answer is returned as text only (no speech is synthesized)."""

    def __init__(
        self,
        request: SessionStartRequest,
        query_text: str,
        on_complete: Callable[[str], None] | None = None,
    ):
        super().__init__(request=request, on_complete=on_complete)
        self._query_text = query_text
        self._thread = threading.Thread(target=self._run, name=f"text-session-{self.session_id}", daemon=True)
        self._source_kind = "text"

    def start(self) -> None:
        # No audio capture ??? the ASR flush worker is never needed; only the
        # main thread (which drives RAG) runs.
        with self._lock:
            if self.status != "created":
                raise ValueError("Session already started")
            self.status = "running"
            self.started_at = datetime.now(UTC)
        self._thread.start()

    def _run(self) -> None:
        final_status = "completed"
        end_reason = "text_query"
        try:
            with self._lock:
                self._speech_started = True
                self.transcript_parts.append(self._query_text)
        except Exception as exc:  # noqa: BLE001
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("Text session %s failed", self.session_id)
        finally:
            self._finalize_run(final_status, end_reason)

    def _tts_worker(self, sentence_queue: Queue[tuple[int | None, str | None]]) -> None:
        # Text queries are answered in text only. Drain the sentence queue
        # without synthesizing so _stream_rag_response still accumulates the
        # response text and completes normally.
        while True:
            sentence_index, sentence = sentence_queue.get()
            if sentence_index is None or sentence is None:
                return

    def _emit_opener(self) -> None:
        # Text queries produce no audio at all, so there is no silent gap to
        # fill and an opener segment would be a spurious audio artefact.
        return


class FileAudioSession(BaseAudioSession):
    def __init__(
        self,
        request: FileSessionStartRequest,
        audio_file_path: str,
        on_complete: Callable[[str], None] | None = None,
    ):
        super().__init__(request=request, on_complete=on_complete)
        self.request = request
        self.audio_file_path = audio_file_path
        self._thread = threading.Thread(target=self._run, name=f"file-session-{self.session_id}", daemon=True)
        self._source_kind = "file"

    def _run(self) -> None:
        # Ground-truth playback anchor: stamped before a single frame is read,
        # so it corresponds to file offset 0 with only thread-scheduling jitter
        # (sub-millisecond in practice) — not an approximation from any VAD.
        self._t_playback_start = time.monotonic()
        final_status = "completed"
        end_reason = self.end_reason or "completed"
        try:
            final_status, end_reason = self._process_frame_stream(self._iter_file_frames())
            if final_status == "completed" and not self._speech_started:
                end_reason = "no_speech_detected"
        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("File session %s failed", self.session_id)
        finally:
            Path(self.audio_file_path).unlink(missing_ok=True)
            self._finalize_run(final_status, end_reason)

    def _iter_file_frames(self):
        with wave.open(self.audio_file_path, "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()

            if sample_width != 2:
                raise ValueError("Only 16-bit PCM WAV files are supported for file-based testing")
            if sample_rate != self.request.sample_rate:
                raise ValueError(
                    f"Uploaded WAV sample rate {sample_rate} Hz does not match the "
                    f"session sample_rate parameter ({self.request.sample_rate} Hz). "
                    f"Either convert the file to {self.request.sample_rate} Hz 16-bit mono PCM "
                    f"(e.g. ffmpeg -i input.wav -ar {self.request.sample_rate} -ac 1 -sample_fmt s16 output.wav) "
                    f"or pass sample_rate={sample_rate} when starting the session."
                )

            while not self._stop_event.is_set():
                raw = wav_file.readframes(self._frame_samples)
                if not raw:
                    break

                frame = np.frombuffer(raw, dtype=np.int16)
                if channels > 1:
                    frame = frame.reshape(-1, channels)[:, 0]

                if len(frame) == 0:
                    continue

                yield frame.copy()

                if self.request.realtime_factor > 0:
                    time.sleep((len(frame) / self.request.sample_rate) / self.request.realtime_factor)
