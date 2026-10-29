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
| **Voice-to-voice latency** | Customer's last word → first sound ready for the speaker |
| **Endpointing delay** | Customer's last word → turn-end decision |
| **Processing latency** | Turn-end decision → first sound ready for the speaker |
| **Transcription latency** | Last word → final transcript available |
| **LLM TTFT** | Time to first token |
| **TTS time to first sentence** | Time until sentence 1's synthesized audio WAV is on disk (not the first audio byte) |

The two parts reconcile exactly:

```text
voice_to_voice_ms == endpointing_delay_ms + processing_latency_ms
```

The clock stops when the first answer audio is **ready for the speaker** —
the WAV is on disk and queued for playback. `make benchmark` is headless, so
this is the furthest point it can observe. In the live hands-free UI the
browser additionally reports when playback actually began, recorded
separately as `voice_to_voice_speaker_ms`.

> **"Time to first audio" is retired.** It named different spans in
> different places. Use voice-to-voice latency, and state which of its two
> parts you mean.

## The opener

By default the kiosk plays a short cached acknowledgement ("One moment.")
as soon as it decides the turn has ended, before the answer is ready. When
it does, the first sound ready for the speaker is the opener, so
`voice_to_voice_ms` measures the opener — which is a real thing the
customer hears, but it is not a measure of the pipeline.

Every turn therefore reports both, and says which it was:

| Field | Meaning |
| --- | --- |
| `voice_to_voice_ms` | To the first sound ready for the speaker, whatever it was |
| `voice_to_voice_answer_ms` | To the first sound **of the answer** |
| `first_audio_was_opener` | Whether the first sound was the opener |
| `opener_failed` | Opener was enabled but could not be rendered, so the customer heard nothing until the answer |

Quote `voice_to_voice_answer_ms` when comparing pipeline changes. Quote
`voice_to_voice_ms` when describing what the customer experiences.

Disable the opener with `KIOSK_CORE_OPENER_ENABLED=false` to measure the
pipeline alone.

## Host prerequisites

Everything in this stack runs in Docker **except the measurement harness**,
which runs on the host: the orchestrator re-invokes the in-application harness
with `sys.executable`, so the host interpreter needs its own dependencies.

`make benchmark` handles this automatically. It builds a small venv
(`.venv-benchmark`, from `tests/benchmarks/requirements.txt`) and launches the
orchestrator with it, so the run no longer depends on what the system
`python3` happens to have installed. Two things are checked before the stack
is started, by `make check-benchmark-env`:

| Requirement | Why | If missing |
| --- | --- | --- |
| `websockets` | Default `--uplink ws` (the browser's hands-free path) | Installed into `.venv-benchmark` automatically |
| `ffmpeg` | Resamples every synthesised prompt to 16 kHz mono PCM | **Install it yourself** — not a PyPI package: `sudo apt-get install -y ffmpeg` |

Both are verified up front deliberately. A missing host dependency is not a
property of the pipeline being measured, and discovering it mid-run used to
cost a full stack bring-up and a discarded warmup before surfacing as a
report in which every turn carried the same error — which reads like a broken
stack rather than a broken host.

Useful targets:

```bash
make check-benchmark-env   # verify the host, start nothing
make benchmark-venv        # (re)build .venv-benchmark
```

On a host with no PyPI access, `V2V_UPLINK=rest make benchmark` replays over
the legacy chunked-WAV transport, which is pure stdlib. It still needs
`ffmpeg`, and it measures a different uplink — never mix its numbers with
`ws`-uplink numbers in one comparison.

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

The headline voice-to-voice number the benchmark prints is **clip-anchored**:
the customer's last word is taken from the test clip itself (where speech ends
in the WAV), not from the kiosk's own voice activity detector. Endpointing is
part of what is being measured, so the clip anchor is the honest customer-facing
figure. The detector-anchored number (`voice_to_voice_ms`, started at the
kiosk's VAD) is printed below it only as a secondary cross-check. Live-mic runs
have no clip ground truth and fall back to the detector number as the headline.

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

A p95 over four turns is just the slowest of the four. The default
benchmark runs `order-fastpath`: six single-intent ordering turns (one
clean mutation each, ending in a confirm), all synthesised with one TTS
voice, repeated
`V2V_RUNS` times — 48 measured turns at the default `V2V_RUNS=8`, enough
for a stable median and a p95 that is not a single outlier. This is the
clean-ordering voice-to-voice number; **quote it with the scenario named
next to it** (`order-fastpath`), because it deliberately stays on the fast
path and is faster than a mixed workload.

For the broader workload — single items, multi-item orders, changes,
removals, confirmations, mid-sentence pauses and a price-question + upsell
journey — run `make benchmark V2V_SCRIPT=all` (88 measured turns per run)
and report that number under its own name (`all`). Warm-up turns are pruned
before the percentiles are computed in either case.

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
