"""
Turn-level AI pipeline latency tracker.

Captures one TurnTrace per completed voice turn and holds the last N in a
thread-safe ring buffer.  Exposed via GET /api/v1/pipeline/latest and
GET /api/v1/pipeline/recent on kiosk-core.

Design principles
─────────────────
* Wall-clock E2E is MEASURED (ended_at − started_at), never summed from stages,
  so the TTS-overlap with LLM generation is handled correctly.
* Spans are NESTED: retrieval and llm live under agent (matching runtime reality).
* An ``invoked`` flag on retrieval prevents stale latency leaking across turns.
* All durations use monotonic clock (time.monotonic); ISO timestamps use datetime.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class RetrievalSpan:
    invoked: bool = False
    ms: float | None = None


@dataclass
class McpSpan:
    ms: float | None = None        # cumulative MCP tool round-trip time (network + kiosk-core handling)
    calls: int = 0


@dataclass
class GuardSpan:
    ms: float | None = None        # cumulative truthfulness-guard processing time
    calls: int = 0


@dataclass
class TemplateSpan:
    ms: float | None = None        # deterministic reply-template render time
    calls: int = 0


@dataclass
class LlmSpan:
    ms: float | None = None        # total cumulative LLM time this turn
    ttft_ms: float | None = None   # cumulative prefill; ms - ttft_ms = decode
    calls: int = 0
    device: str = "GPU"


@dataclass
class AgentSpan:
    ttft_ms: float | None = None   # time-to-first-token (perceived latency)
    total_ms: float | None = None  # agent_start to last TTS segment done —
                                    # includes TTS-synthesis drain time, NOT
                                    # pure agent/LLM/tool time. Kept as-is
                                    # (existing consumers rely on this whole-
                                    # orchestration semantics); use stream_ms
                                    # below for the isolated figure.
    stream_ms: float | None = None  # agent_start to the end of the token
                                     # stream (before _stop_tts_workers runs)
                                     # — the actual LLM+tool round-trip a
                                     # customer's reply waits on, with TTS
                                     # drain time excluded.
    retrieval: RetrievalSpan = field(default_factory=RetrievalSpan)
    llm: LlmSpan = field(default_factory=LlmSpan)
    mcp: McpSpan = field(default_factory=McpSpan)
    guard: GuardSpan = field(default_factory=GuardSpan)
    template: TemplateSpan = field(default_factory=TemplateSpan)


@dataclass
class AsrSpan:
    ms: float | None = None
    device: str = "CPU"
    chunks: int = 0                # number of transcribe calls summed into ms
    final_flush_skipped: bool = False  # see config.DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED
    # Continuous-streaming mode only (KIOSK_CORE_ANALYZER_STREAMING_ENABLED):
    # the customer's actual last word (backdated from the silence run, same
    # anchor as wall.voice_to_voice_ms) to the moment a transcript covering
    # it actually landed from the analyzer. This is the genuine ASR compute
    # latency contributing to the critical path -- it deliberately EXCLUDES
    # the trailing-silence wait the endpoint separately sits through before
    # acting on that transcript (endpoint_wait_ms already reports that), so
    # this number is comparable to a bare "utterance -> transcript" figure
    # rather than inflated by a design choice unrelated to ASR speed. None
    # when streaming mode never ran this turn, or no post-last-word
    # transcript update was observed (already had everything before the
    # customer finished speaking).
    last_word_to_transcript_ms: float | None = None


@dataclass
class TtsSpan:
    ms: float | None = None
    device: str = "CPU"
    segments: int = 0
    overlapped_with_agent: bool = True   # always true — TTS runs concurrently


@dataclass
class WallTimes:
    turn_total_ms: float | None = None
    time_to_first_audio_ms: float | None = None
    # How long the endpoint waited in trailing silence before committing the
    # turn (1.5s fixed, or KIOSK_CORE_ENDPOINT_SHORT_SECONDS when the
    # transcript already read as a finished sentence). The customer sits
    # through this, so any voice-to-voice figure has to include it.
    endpoint_wait_ms: float | None = None
    # The final chunk's real ASR round-trip: the mandatory drain-and-join of
    # the flush queue (self._flush_queue.join() in _process_frame_stream)
    # that _finalize_run blocks on right after the endpoint decision, before
    # the turn is considered "started" (t_turn_start). This is real wall
    # time the customer waits through that neither endpoint_wait_ms (the
    # trailing-silence run only) nor time_to_first_audio_ms (measured from
    # t_turn_start onward) accounts for — without this field,
    # voice_to_voice_ms could be larger than
    # endpoint_wait_ms + time_to_first_audio_ms with no visible explanation.
    # None when _t_last_word was never set (see WallTimes.voice_to_voice_ms).
    final_flush_wait_ms: float | None = None
    # Customer's last word -> first sound out of the speaker. This is the
    # "voice to voice" clock, the one a customer actually feels, and the only
    # one comparable to external voice-kiosk figures. The other timings in this
    # trace start at the endpoint decision instead, which excludes the wait.
    voice_to_voice_ms: float | None = None
    # voice_to_voice_ms with BOTH non-compute waiting windows subtracted back
    # out: endpoint_wait_ms (the deliberate trailing-silence wait) AND
    # post_speech_gap_ms (mic-release reaction time / trailing buffered
    # frames). Neither is a pipeline/hardware cost, so neither belongs in a
    # number meant to answer "how fast is our compute pipeline". Equal to
    # final_flush_wait_ms + time_to_first_audio_ms. None only when
    # voice_to_voice_ms itself is None.
    voice_to_voice_post_endpoint_ms: float | None = None
    # Browser-mic-release turns only (no endpoint_wait_ms): the gap between
    # the customer's actual last speech frame (_t_last_word, backend-observed)
    # and the moment the flush/turn-start sequence began (right after the
    # mic-release signal was received and processed). This is real elapsed
    # time -- customer reaction time releasing the button, plus any trailing
    # audio still queued -- not network/browser-clock skew (both ends of this
    # gap are backend monotonic timestamps). Previously this silently
    # inflated voice_to_voice_ms with no visible line item; now broken out so
    # voice_to_voice_ms = endpoint_wait_ms + post_speech_gap_ms +
    # final_flush_wait_ms + time_to_first_audio_ms is fully reconstructable.
    # On the silence-timeout path this has endpoint_wait_ms already
    # subtracted out (the two windows overlap there -- flush only starts once
    # the silence wait elapses), so it should be near-zero, not the raw
    # (much larger) elapsed time since last word.
    post_speech_gap_ms: float | None = None
    # Customer's last word -> first sound that carries the ANSWER. The opener
    # ("One moment.") is real audio and legitimately stops the silence, but it
    # is not informative, so it is reported separately rather than allowed to
    # flatter voice_to_voice_ms.
    voice_to_voice_informative_ms: float | None = None
    # File-replay only (FileAudioSession): time from the first byte of the
    # fixture being fed into the pipeline to first audio out. voice_to_voice_ms
    # above estimates "last word" from OUR OWN detector's measured silence run,
    # which is a live-mic necessity — there is no ground truth on a live mic.
    # A fixture, unlike a live mic, HAS an independent ground truth: the exact
    # sample where the recording's real speech ends, discoverable once via a
    # silence detector run directly on the file (outside this pipeline, so it
    # is not circular). Benchmarks combine this field with that offset to get
    # a v2v number with zero dependency on this system's own VAD timing,
    # mirroring kiosk-voice-lab's fixture-manifest t_eos design. None for
    # microphone/browser-stream sessions, where no such anchor exists.
    playback_to_first_audio_ms: float | None = None
    # True: this turn committed via the sentence-completeness shortcut
    # (endpoint_wait_ms ~= KIOSK_CORE_ENDPOINT_SHORT_SECONDS). False: it fell
    # through to the full silence_timeout_seconds wait because the transcript
    # was not yet stable/complete (usually because ASR had not returned in
    # time). None: neither silence-based commit path ran this turn (e.g.
    # stopped_by_api, max_duration_reached). Added to make "is the adaptive
    # shortcut actually firing?" directly observable per turn instead of
    # inferred from endpoint_wait_ms clustering near one value or the other.
    endpoint_shortcut_fired: bool | None = None


@dataclass
class TurnTrace:
    turn_id: str
    conversation_id: str
    started_at: str       # ISO8601 UTC
    ended_at: str | None
    wall: WallTimes = field(default_factory=WallTimes)
    asr: AsrSpan = field(default_factory=AsrSpan)
    agent: AgentSpan = field(default_factory=AgentSpan)
    tts: TtsSpan = field(default_factory=TtsSpan)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineLatencyStore:
    """Thread-safe ring buffer of the last ``maxlen`` TurnTrace records."""

    def __init__(self, maxlen: int = 20) -> None:
        self._lock = threading.Lock()
        self._buffer: deque[TurnTrace] = deque(maxlen=maxlen)

    def record(self, trace: TurnTrace) -> None:
        with self._lock:
            self._buffer.append(trace)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].to_dict()

    def recent(self, n: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._buffer)
        return [t.to_dict() for t in items[-n:]]


# Module-level singleton — imported by audio_session and main.py
pipeline_store = PipelineLatencyStore()
