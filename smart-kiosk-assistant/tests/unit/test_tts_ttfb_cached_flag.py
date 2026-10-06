"""Tests for TtsSpan.first_segment_cached (kiosk_core/pipeline_latency.py).

Background (PR #112 review, item 26): the dashboard's TTS card reported a
time-to-first-byte of well under a millisecond, which reads as a broken
metric. It isn't -- sentence 1 is frequently served straight from the
speculative-draft TTS cache or the opener cache, so ``ttfb_ms`` is timing a
file copy, not the synthesiser.

The figure is honest but meaningless on its own, so every consumer of
``ttfb_ms`` (dashboard card, benchmark KPI table) has to be able to tell the
two cases apart. This flag is that signal, and it is part of the published
trace schema -- dropping it would silently return the card to showing an
unexplained "0 ms".
"""
from dataclasses import asdict

from kiosk_core.pipeline_latency import TtsSpan


def test_flag_defaults_to_false():
    # Callers that never populate the field must not be reported as cached:
    # an unknown provenance has to read as "this was synthesised", because
    # claiming a cache hit would explain away a genuinely fast measurement.
    span = TtsSpan(ms=120.0, segments=2, overlapped_with_agent=True)
    assert span.first_segment_cached is False


def test_flag_survives_serialization():
    # The dashboard and the benchmark harness both read this off the
    # serialised trace, not off the object, so the field has to make it
    # through asdict() under exactly this name.
    span = TtsSpan(
        ms=0.9,
        segments=1,
        overlapped_with_agent=True,
        ttfb_ms=0.8,
        first_segment_cached=True,
    )
    dumped = asdict(span)
    assert dumped["first_segment_cached"] is True
    assert dumped["ttfb_ms"] == 0.8
