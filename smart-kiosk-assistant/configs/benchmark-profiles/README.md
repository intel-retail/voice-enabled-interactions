# Benchmark Device Profiles

A **profile** is a plain env file describing one device combination. It is the
only thing a benchmarker needs to edit or create.

```bash
make list-profiles                          # what's available
make validate-profile PROFILE=asr-npu_tts-cpu   # pre-flight check, no bring-up
make benchmark        PROFILE=asr-npu_tts-cpu   # run it
make benchmark-matrix                       # run every profile, then compare
make compare-profiles                       # rebuild results/matrix_summary.csv
```

Results land in `results/<PROFILE_NAME>/`, hardware counters in
`metrics/<PROFILE_NAME>/`. Nothing is shared between profiles, so runs never
overwrite each other.

---

## The matrix

LLM (`ovms-llm`) is pinned to **GPU** in every profile, as required.

| Profile | ASR | TTS | LLM |
|---|---|---|---|
| `asr-cpu_tts-cpu` | CPU | CPU | GPU |
| `asr-cpu_tts-gpu` | CPU | GPU | GPU |
| `asr-cpu_tts-npu` | CPU | NPU | GPU |
| `asr-gpu_tts-cpu` | GPU | CPU | GPU |
| `asr-gpu_tts-gpu` | GPU | GPU | GPU |
| `asr-gpu_tts-npu` | GPU | NPU | GPU |
| `asr-npu_tts-cpu` | NPU | CPU | GPU |
| `asr-npu_tts-gpu` | NPU | GPU | GPU |
| `asr-npu_tts-npu` | NPU | NPU | GPU |
| `baseline-production` | GPU | CPU | GPU | *(not a matrix cell — see below)* |

---

## Why the matrix pins non-default models

This is the single most important thing to understand before reading any
number this harness produces.

**A device comparison is only valid if the device is the only thing that
changed.** The production defaults cannot satisfy that:

| Stage | Production default | Problem for a device matrix |
|---|---|---|
| ASR | `distil-whisper/distil-small.en` | **Corrupts on NPU** — output goes stale/frozen, repeating the first call's text regardless of new input (confirmed twice independently). It physically cannot run the NPU leg. |
| TTS | Kokoro (`runtime: kokoro`) | **CPU-only.** kokoro-onnx uses the CPU execution provider and is not wired for GPU/NPU. It cannot run the GPU or NPU legs. |

So the matrix pins the two engines that *do* run everywhere:

- **ASR → `whisper-base`** — the largest Whisper checkpoint that compiles on
  NPU, and it runs on CPU and GPU too, so device really is the only variable.
  `whisper-small` and larger fail NPU compilation with
  `Check '!self_attn_nodes.empty()' failed`: OpenVINO NPU requires static
  shapes, and GenAI's NPUW attention-block heuristic only succeeds for
  `whisper-tiny`/`whisper-base`.
- **TTS → `microsoft/speecht5_tts` on the `openvino` runtime** — the only TTS
  engine in this service that supports CPU, GPU *and* NPU.

`baseline-production.env` then exists purely to anchor the matrix to reality.
Compare it on **absolute latency only**. It is *not* the "ASR=GPU, TTS=CPU"
cell — it runs different models.

---

## Engine / device compatibility matrix

The TTS factory selects an implementation from `(name, runtime)` and then hands
it `device`. An inconsistent triple fails at **model load**, not at request
time, so `make validate-profile` checks it up front.

| Engine | `TTS_MODEL` | `TTS_RUNTIME` | Devices | `TTS_SPEAKER` | `TTS_MODEL_VARIANT` |
|---|---|---|---|---|---|
| Kokoro | `kokoro` | `kokoro` | CPU only | `am_michael` (and 27 others) | `custom_voice` (unused) |
| SpeechT5 | `microsoft/speecht5_tts` | `openvino` | CPU, GPU, NPU | `Ryan`, `Miles`, `Aaron`, `Nora`, `Elena`, `Kabir`, `Angus` | `default` |
| Qwen | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `openvino` | CPU, GPU | `Ryan` | `custom_voice` \| `voice_design` |

Voice vocabularies **do not overlap**. Switching the model without switching
the speaker makes every synthesis request fail validation.

| ASR provider | Devices |
|---|---|
| `openvino` | CPU, GPU, NPU |
| `openai` | CPU only |
| `whispercpp` | CPU only |

Diarization is CPU-only: it loads via `torch.device(<value>)`, and PyTorch has
no `gpu` string (Intel GPU would need `xpu`, which the component does not wire).

---

## Hard requirements

**NPU** — `ACCEL_MOUNT_PATH` in `.env` must point at a real host NPU node
(commonly `/dev/accel/accel0`). The portable default is `/dev/null`, which
means *NPU disabled*. With `/dev/null` mounted, OpenVINO **silently falls back
to CPU** — you get plausible numbers under an NPU label. `make validate-profile`
fails the run rather than let that happen.

**GPU** — a `/dev/dri/renderD*` render node must exist, and `RENDER_GID` must
match its group.

**TTS on GPU** — `TTS_DTYPE=fp16` is mandatory. `int8` on this iGPU measured
**281 s first call / 8.7 s steady-state** versus **0.92 s on CPU**. `int4`
produces noise. Validation rejects `GPU`+`int8`.

---

## Writing your own profile

Copy any matrix file and edit. Only `PROFILE_NAME` is required, and it must
match the filename (minus `.env`) — it is what names the output directory.

```bash
cp configs/benchmark-profiles/asr-gpu_tts-cpu.env \
   configs/benchmark-profiles/my-experiment.env
$EDITOR configs/benchmark-profiles/my-experiment.env   # set PROFILE_NAME=my-experiment
make validate-profile PROFILE=my-experiment
make benchmark PROFILE=my-experiment
```

Layering is **base `.env` → profile → make/CLI override**, so a profile only
needs to state what it varies:

```bash
make benchmark PROFILE=asr-npu_tts-cpu V2V_RUNS=3 QUEUE=true
```

### Available keys

| Key | Default | Notes |
|---|---|---|
| `PROFILE_NAME` | — | **Required.** Names `results/<name>/`. |
| `ASR_DEVICE` | `CPU` | `CPU` \| `GPU` \| `NPU` |
| `ASR_PREVIEW_DEVICE` | `$ASR_DEVICE` | Streaming/partial pool, separate from the final-commit pool |
| `ASR_MODEL` | `distil-small.en` | Any HF id or `whisper-{tiny,base,small,medium,large}` |
| `DIARIZATION_DEVICE` | `CPU` | CPU only |
| `TTS_MODEL` / `TTS_RUNTIME` | `kokoro` / `kokoro` | Must be a valid pair |
| `TTS_DEVICE` | `CPU` | Engine-dependent |
| `TTS_DTYPE` | `int8` | `int8` \| `fp16` \| `fp32`; **fp16 required on GPU** |
| `TTS_SPEAKER` | `am_michael` | Engine-dependent vocabulary |
| `TTS_MODEL_VARIANT` | `custom_voice` | Engine-dependent |
| `TARGET_DEVICE` | `GPU` | OVMS LLM — pinned to GPU in all shipped profiles |
| `RAG_EMBEDDING_DEVICE` / `RAG_RERANKER_DEVICE` | `GPU` | Pinned, not under test |
| `QUEUE` | `false` in profiles | `true` re-enables YOLO queue detection |
| `KIOSK_CORE_DIARIZATION_ENABLED` | `false` in profiles | Removes diarization from the critical path |

---

## Reading the output

```
results/<profile>/
├── profile.env               exact resolved config (provenance)
├── run_manifest.json         devices, image tags, git SHA, host capabilities
├── smart_kiosk_v2v_results_*.json
├── consolidated_metrics.csv  hardware counters for this profile only
└── plot_metrics.png
results/matrix_summary.csv    one row per profile — the deliverable
```

`matrix_summary.csv` is what you actually compare:

```
profile, asr_device, tts_device, llm_device, v2v_p50_ms, v2v_p95_ms,
asr_ms, agent_ttft_ms, tts_ms, cpu_util_pct, gpu_util_pct, npu_util_pct, power_w
```

Always judge on **p95**, not median — a single bad tail turn is what a customer
notices, and the median hides it.

### Verify the device actually took effect

Device labels in this table come from the profile. Confirm the *observed*
device independently after a run:

```bash
curl -s localhost:8012/api/v1/pipeline/latest | jq '{asr:.asr.device, llm:.llm.device, tts:.tts.device}'
```

---

## Gotchas

- **Warm-up is not optional.** Changing a device forces an OpenVINO
  recompile/export; first call can be minutes. The harness discards
  `V2V_WARMUP_RUNS` (default 1) before measuring.
- **Profiles run sequentially, never in parallel** — they contend for the same
  silicon.
- **Disk**: qmassa accrues ~596 MB/hour *per profile*. `make clean-metrics
  PROFILE=x` clears one; `make clean-metrics` clears all.
- A full 10-profile matrix is **hours**, not minutes.
