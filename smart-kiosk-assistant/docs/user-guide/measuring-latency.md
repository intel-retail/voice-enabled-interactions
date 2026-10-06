# Measuring Latency

This page defines the latency terms used across the kiosk UI, the
`GET /api/v1/pipeline/latest` API, and `make benchmark`, and explains how
to read a benchmark run.

Use these terms as written. Do not introduce synonyms — the same span
having two names in two places is what made earlier numbers
incomparable.

## Terms

| Term | Definition |
| --- | --- |
| **Voice-to-voice latency** | Customer's last word → first sound at the speaker |
| **Endpointing delay** | Customer's last word → turn-end decision |
| **Processing latency** | Turn-end decision → first sound at the speaker |
| **Transcription latency** | Last word → final transcript available |
| **LLM TTFT** | Time to first token |
| **TTS TTFB** | Time to first byte of synthesized audio |

The two parts reconcile exactly:

```text
voice_to_voice_ms == endpointing_delay_ms + processing_latency_ms
```

The clock stops at the **speaker** — when audio is first available for
playback — not when a WAV is written to disk.

> **"Time to first audio" is retired.** It named different spans in
> different places. Use voice-to-voice latency, and state which of its two
> parts you mean.

## The opener

By default the kiosk plays a short cached acknowledgement ("One moment.")
as soon as it decides the turn has ended, before the answer is ready. When
it does, the first sound at the speaker is the opener, so
`voice_to_voice_ms` measures the opener — which is a real thing the
customer hears, but it is not a measure of the pipeline.

Every turn therefore reports both, and says which it was:

| Field | Meaning |
| --- | --- |
| `voice_to_voice_ms` | To the first sound at the speaker, whatever it was |
| `voice_to_voice_answer_ms` | To the first sound **of the answer** |
| `first_audio_was_opener` | Whether the first sound was the opener |
| `opener_failed` | Opener was enabled but could not be rendered, so the customer heard nothing until the answer |

Quote `voice_to_voice_answer_ms` when comparing pipeline changes. Quote
`voice_to_voice_ms` when describing what the customer experiences.

Disable the opener with `KIOSK_CORE_OPENER_ENABLED=false` to measure the
pipeline alone.

## Reading a benchmark run

`make benchmark` writes per-run artifacts to `results/`:

| File | Contents |
| --- | --- |
| `smart_kiosk_v2v_summary_<ts>.csv` | Per-turn latency rows |
| `smart_kiosk_v2v_results_<ts>.json` | Full per-turn traces and percentiles |
| `run_manifest.json` | What actually ran, and whether the run completed |
| `measured_window.json` | First and last measured turn |
| `consolidated_metrics.csv` | Hardware counters plus aggregate latency |

Report the **median and p95**, never the mean — the distribution has a
long tail and the mean hides it.

### `run_manifest.json`

Records what the run actually used, not what was requested. Where the
orchestrator overrides a setting, both the effective value and a
`*_requested` value appear. `status` is `"started"` until the run
completes successfully, so a failed run cannot be mistaken for a good one
on the strength of leftover numbers from the previous run.

### `early_commit_count` — check this first

In a benchmark run the customer's last word is taken from the **test clip
itself** (where speech ends in the WAV), not from the kiosk's own voice
activity detector. Endpointing is part of what is being measured, so
measuring it against its own decision would always report roughly zero.

Because that anchor is independent of the kiosk, the endpointing delay can
come out negative. That means the turn was committed *before the customer
finished speaking* — the transcript is truncated and the order was taken
from a partial sentence.

| Field | Meaning |
| --- | --- |
| `early_commit_count` | Turns committed before the customer finished |
| `early_commit_eligible_count` | Turns with a usable clip anchor |

**`early_commit_count` must be zero.** A non-zero value is a failed run no
matter how good the latency percentiles look — cutting customers off
shortens every measured turn, so the numbers improve as the behaviour gets
worse.

Two settings keep it at zero:

- `KIOSK_CORE_ENDPOINT_STABLE_SECONDS` — the turn is only committed once
  the transcript has stopped changing. While more speech is arriving the
  transcript keeps changing and the window resets, which is what
  distinguishes a mid-sentence pause from the end of a turn. A fixed
  silence threshold cannot tell those apart.
- `KIOSK_CORE_ADAPTIVE_FLUSH_PAUSE_SECONDS` — must not be set below
  `0.50`; shorter pauses are normal mid-sentence.

### Run size

A p95 over four turns is just the slowest of the four. The defaults run
enough varied turns — single items, multi-item orders, changes, removals,
confirmations, mid-sentence pauses, several voices — for the median to be
stable and the p95 not to be a single outlier. Warm-up turns are pruned
before the percentiles are computed.

## Hardware counters

Counters are trimmed to the measured window before consolidation, so
utilization reflects the turns rather than being diluted by the health
wait, model warm-up and teardown.

kiosk-core runs as uid 1000. If `results/` is not writable by that uid,
latency events cannot be recorded and the benchmark reports zero
transactions even though every turn succeeded. A warning is logged at
startup if this is the case.

## Related

- [Configuration](./get-started/configuration.md)
- [How It Works](./how-it-works.md)
- [Troubleshooting](./troubleshooting.md)
