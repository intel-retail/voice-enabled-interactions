# Configuration

`kiosk-core` and `kiosk-ui` are configured through environment variables
(see [Environment Variables](#environment-variables)).

The three model-hosting services (`audio-analyzer`, `text-to-speech`,
`rag-service`) are configured through YAML files that the kiosk pins
and mounts into the containers. The most common changes are the
[model](#model-selection) and the [inference device](#inference-device).

## Model Selection

Each model-hosting service reads the model identifier from the same
pinned config file used for device selection:

| Service | File | Model fields |
|---|---|---|
| `audio-analyzer` | [`configs/audio-analyzer/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/configs/audio-analyzer/config.yaml) | `models.asr.name` (e.g. `whisper-tiny`, `whisper-base`); `sentiment.model` (optional) |
| `text-to-speech` | [`configs/text-to-speech/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/configs/text-to-speech/config.yaml) | `models.tts.name` (e.g. `microsoft/speecht5_tts`, Qwen-TTS variant); `model_variant` |
| `rag-service` | [`rag-service/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/rag-service/config.yaml) | `models.llm.hf_id`, `models.embedding.hf_id`, `retrieval.reranker.hf_id`; per-model `weight_format` (`int4`, `int8`, `fp16`) |

Use Hugging Face IDs where the field name is `hf_id`. Models are
downloaded and exported on first start into the per-service `models/`
directory; subsequent starts reuse the cache.

### Supported / validated models

The kiosk ships with the following defaults. These are the models the
stack has been validated with — they are the recommended starting point.
The **Devices** column lists the supported inference devices for each:

| Service | Field | Default (validated) | Other examples | Devices |
|---|---|---|---|---|
| `audio-analyzer` ASR | `models.asr.name` | `whisper-base` | `whisper-tiny`, `whisper-small`, `whisper-medium`, `whisper-large` | `CPU`, `GPU` (`provider: openvino` required for `GPU`); `NPU` works only for `whisper-tiny`/`whisper-base` — see [ASR Support Matrix](#asr-support-matrix) |
| `audio-analyzer` sentiment | `sentiment.model` | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | other SpeechBrain emotion-recognition models | `CPU`, `GPU` (disabled by default) |
| `text-to-speech` | `models.tts.name` | `microsoft/speecht5_tts` (SpeechT5) | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (Qwen-TTS) | `CPU`, `GPU` (`int4` on iGPU produces noise; use `fp16` or `int8` on GPU) |
| `rag-service` LLM | `models.llm.hf_id` | `Qwen/Qwen3-4B-Instruct-2507` | other OpenVINO-exportable instruct LLMs | `CPU`, `GPU` (`GPU` recommended for acceptable latency); `NPU` is **not supported** — see [`TARGET_DEVICE=NPU`](../troubleshooting.md#target_devicenpu--llm-turns-fail-with-sorry-i-encountered-an-error) |
| `rag-service` embedding | `models.embedding.hf_id` | `BAAI/bge-large-en-v1.5` | `BAAI/bge-base-en-v1.5`, `BAAI/bge-small-en-v1.5` | `CPU`, `GPU` (`CPU` is usually fast enough) |
| `rag-service` reranker | `retrieval.reranker.hf_id` | `BAAI/bge-reranker-base` | `BAAI/bge-reranker-large` | `CPU`, `GPU` (optional) |

> [!IMPORTANT]
> **Changing models is at your own discretion.** The defaults above are
> the only combinations validated with this stack. Configuring models,
> variants, devices, or precisions other than the defaults may negatively
> affect the functionality, accuracy, latency, or stability of the
> application. You are responsible for ensuring the configuration you
> choose is correct and works for your use case — make changes only if you
> understand the implications.
>
> In particular:
> - Some models do not function properly at aggressive quantization. If a
>   model produces garbled, empty, or low-quality output at `int4`, switch
>   that model's `weight_format`/`dtype` to `int8` or `fp16`.
> - A model must be exportable to OpenVINO IR for the OpenVINO backend; not
>   every Hugging Face model is supported.
> - Larger models increase first-run download/export time, memory use, and
>   per-request latency, and may not fit on the selected device.
> - After any change, restart the affected service and verify it loads and
>   responds correctly before relying on it.

## Inference Device

Each model-hosting service reads its device from a pinned config file:

| Service | File | Fields |
|---|---|---|
| `audio-analyzer` | [`configs/audio-analyzer/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/configs/audio-analyzer/config.yaml) | `models.asr.device`, `sentiment.device` |
| `text-to-speech` | [`configs/text-to-speech/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/configs/text-to-speech/config.yaml) | `models.tts.device` |
| `rag-service` | [`rag-service/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/rag-service/config.yaml) | `models.llm.device`, `models.embedding.device`, `retrieval.reranker.device` |

The supported devices for each model are listed in the
[Supported / validated models](#supported--validated-models) table above.

Use uppercase device names (`CPU`, `GPU`, and — for `audio-analyzer` ASR and
`queue-service` only — `NPU`). `rag-service` expects them as quoted strings;
`audio-analyzer` and `text-to-speech` unquoted.

> [!IMPORTANT]
> **`text-to-speech` does not support `NPU`.** `models.tts.device` only
> accepts `CPU`/`GPU` (see `configs/text-to-speech/config.yaml`); there is no
> NPU device mapping for this service in `docker-compose.yml`. Do not set
> `models.tts.device: NPU` — it is not a supported configuration.

After editing, restart the affected service and confirm OpenVINO picked
the device:

```bash
docker compose up -d --build --force-recreate <service-name>
docker compose logs <service-name> | grep -i -E "device|compiling|GPU|CPU"
```

OpenVINO prints a `Compiling model on <DEVICE>` line on first load.

> GPU execution is delegated to the OpenVINO backend used by each
> service. Whether a given model actually runs on GPU and how it
> performs depends on the OpenVINO version and operator coverage for
> that model.

## Audio Analyzer ASR Provider/Device (`config.yaml`)

Configure ASR in `configs/audio-analyzer/config.yaml` (single source of truth):

- `models.asr.provider`
- `models.asr.device`

`make check-env` validates this before startup and rejects unavailable
hardware early. For NPU, `ACCEL_MOUNT_PATH` must be set manually (in
`.env` or the shell) to the host NPU device node (`/dev/accel/accel0`)
before running `make up` or `docker compose up` — neither `make up` nor
`make check-env` auto-detects it. The checked-in `docker-compose.yml`
defaults `ACCEL_MOUNT_PATH` to `/dev/null` so CPU/GPU-only hosts stay
unaffected.

### ASR on NPU: `whisper-tiny`/`whisper-base` only

> [!IMPORTANT]
> NPU works for ASR **only** with `models.asr.name: whisper-tiny` or
> `whisper-base`. `whisper-small`/`medium`/`large` fail to compile on NPU
> with `Check '!self_attn_nodes.empty()' failed`.
>
> **Why:** OpenVINO NPU only supports static-shape models (see
> [OpenVINO NPU docs](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)).
> Whisper's IR has dynamic shapes, so OpenVINO GenAI's NPUW plugin
> pattern-matches attention blocks to make it static — a heuristic that
> succeeds for `tiny`/`base` and fails for larger models. Not fixable via
> config; re-test if you upgrade OpenVINO/GenAI or the audio-analyzer image.

```yaml
models:
  asr:
    provider: openvino
    device: NPU
    name: whisper-base   # or whisper-tiny — whisper-small+ fails to compile
    weight_format: null
```

### Other ASR Configurations

```yaml
# OpenAI + CPU
models: { asr: { provider: openai, device: CPU } }
# OpenVINO + GPU (any model size)
models: { asr: { provider: openvino, device: GPU } }
# OpenVINO + CPU (any model size)
models: { asr: { provider: openvino, device: CPU } }
```

`openai` supports `CPU` only — `openai + GPU` and `openai + NPU` are not supported.

### ASR Support Matrix

| Provider | CPU | GPU | NPU |
|---|---|---|---|
| `openai` | Yes | No | No |
| `whispercpp` | Yes | No | No |
| `openvino` | Yes | Yes (Intel GPU required) | `whisper-tiny`/`whisper-base` only |

If `GPU` is configured and unavailable on the host, `make check-env` fails before startup — no silent fallback.

## Audio Analyzer Diarization Device (`config.yaml`)

> [!IMPORTANT]
> **Diarization requires three gated HuggingFace models.** Before enabling it,
> set `HF_TOKEN` in `.env` and accept the licence on **all three** pages with
> the same account that owns the token:
> [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1),
> [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1),
> and [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0).
> The diarization pipeline pulls `segmentation-3.0` and
> `speaker-diarization-community-1` in addition to the configured
> `speaker-diarization-3.1`, so accepting only the configured model still
> fails with a `401`. (`pyannote/wespeaker-voxceleb-resnet34-LM` is also
> downloaded but is **not** gated — no licence acceptance needed.) Set
> `KIOSK_CORE_DIARIZATION_ENABLED=false` to run without diarization.

Diarization (`models.diarization.device`) is a **separate component from ASR**
(see [ASR Support Matrix](#asr-support-matrix) above) with its own,
more limited device support. Do not assume ASR's `CPU`/`GPU`/`NPU` support
applies to diarization — it does not.

> [!IMPORTANT]
> **In the currently released Kiosk image, diarization only supports `CPU`.**
> The diarizer (`pyannote/speaker-diarization-3.1`, a PyTorch/SpeechBrain
> model) is loaded with `torch.device(<configured value>)`. PyTorch has no
> `"gpu"` device string (Intel GPU support in PyTorch requires `"xpu"`, which
> this component does not use) and no `"npu"` device string at all. Setting
> `device: GPU` or `device: NPU` is **accepted by the config schema** but
> fails at diarizer-load time with an error like:
> ```
> Expected one of cpu, cuda, ipu, xpu, ... device type at start of device string: gpu
> ```
> This is **non-fatal**: the failure is caught, logged as a warning, and
> diarization is disabled for that session — the container stays healthy and
> ASR keeps working, but speaker labels are not produced.
>
> **Use `device: CPU` for diarization.** Do not configure `GPU` or `NPU` for
> `models.diarization.device` — they do not work in this image and will
> silently disable diarization rather than accelerate it.

An OpenVINO-backed diarization path that genuinely supports `GPU`/`NPU`
exists in a newer upstream `edge-ai-libraries` `audio-analyzer` checkout,
but **is not part of the currently released Kiosk image** covered by this
document. Do not configure `GPU`/`NPU` for diarization based on that
upstream code until a Kiosk image that includes it is released.

## Queue Service Device (`QUEUE_DEVICE`)

`queue-service` runs the YOLO26 person detector through a DLStreamer
(`gvainference`) pipeline. The inference device is controlled by
`model.device` in `queue-service/conf/queue-config.yaml`, and can be
overridden without editing that file via `QUEUE_DEVICE` in `.env`
(mapped by `docker-compose.yml` to `QUEUE_SERVICE__MODEL__DEVICE`, read by
`queue-service/src/config_loader.py`).

- Default: `QUEUE_DEVICE=CPU` — always available, no extra device mapping
  needed.
- `QUEUE_DEVICE=GPU` / `QUEUE_DEVICE=NPU` — supported, using the same
  `/dev/dri` and `ACCEL_MOUNT_PATH`-driven `/dev/accel` mapping described
  under [Audio Analyzer ASR Provider/Device](#audio-analyzer-asr-providerdevice-configyaml)
  above. For NPU, set `ACCEL_MOUNT_PATH` yourself to the host NPU device
  node before starting `queue-service` — it is not auto-detected.

Verify the configured device actually reached the pipeline:

```bash
docker logs queue-service 2>&1 | grep "gvainference model"
# Expected: ... gvainference model=... device=NPU ...  (or CPU/GPU, matching QUEUE_DEVICE)
```

> Editing `queue-config.yaml`'s `model.device` directly also works and takes
> precedence in the same way as any other YAML default — `QUEUE_DEVICE` only
> needs to be set when you want to override it without touching the file.

## Identity Service Device (`IDENTITY_DEVICE`)

`identity-service` performs face detection/re-identification
(`face-detection-retail-0005`, `face-reidentification-retail-0095`) and
voice-print embedding (`ecapa-tdnn-voice`) — all three are **OpenVINO IR
models**, loaded via `openvino.Core().compile_model(model, device)`, and
`IDENTITY_DEVICE` is correctly wired end-to-end from `.env` through
`docker-compose.yml` to the OpenVINO compile call. The device-selection
code itself has no bug and no model-format limitation.

> [!NOTE]
> **Face detection/re-identification support `CPU`, `GPU`, and `NPU`.**
> The `identity-service` container has NPU device passthrough via the same
> `ACCEL_MOUNT_PATH`/`/dev/accel` mechanism used by `audio-analyzer` and
> `queue-service`. Set `IDENTITY_DEVICE=NPU` in `.env` together with
> `ACCEL_MOUNT_PATH` pointing to the host NPU device node — this is not
> auto-detected and must be set manually — to run face
> detection/re-identification on the NPU.
>
> **Voice-print embedding (`ecapa-tdnn-voice`) does not support `NPU`.**
> Its OpenVINO IR contains an internal STFT reshape with an unbounded
> dynamic dimension (`aten::view/Reshape`), which the NPU compiler rejects
> at compile time (`Got negative shape dim bound`). With
> `IDENTITY_DEVICE=NPU`, the service starts with face engine enabled and
> voice engine disabled (`inference_ready=false`, since voice verification
> requires both). Use `IDENTITY_DEVICE=CPU` or `IDENTITY_DEVICE=GPU` if
> voice authentication is required.
>
> The face/voice model files are **not downloaded by default** — run
> `./setup_models.sh --identity` first. Without them, the face/voice engines
> stay disabled (`inference_ready=false`) regardless of the configured device.

## OVMS-LLM Device (`TARGET_DEVICE`)

`TARGET_DEVICE` controls the inference device for the `ovms-llm` container
only (the LLM served by OpenVINO Model Server for the ordering agent).
`rag-service`'s own embedding/reranker components have their own,
independent device configuration — see
[RAG Service Embedding/Reranker Device](#rag-service-embeddingreranker-device-rag_embedding_device-rag_reranker_device)
below.

- **Supported:** `TARGET_DEVICE=CPU`, `TARGET_DEVICE=GPU`.
- **Not supported:** `TARGET_DEVICE=NPU` — see below.

> [!IMPORTANT]
> **`TARGET_DEVICE=NPU` is not supported for `ovms-llm`.**
> Device passthrough and model compilation both work — the `ovms-llm`
> container has NPU passthrough via the same `ACCEL_MOUNT_PATH`/`/dev/accel`
> mechanism used by `audio-analyzer` and `queue-service`, OVMS logs
> `Available devices for Open VINO: CPU, GPU, NPU`, and `Qwen3-4B-int8-ov`
> compiles successfully. Generation is nevertheless unusable:
>
> - **INT8** is numerically correct but takes 191–801 s per call, and a single
>   voice turn issues several. That is roughly 100× too slow.
> - **INT4** (`OpenVINO/Qwen3-4B-int4-ov`) is fast (~8 s) but numerically
>   broken on NPU — it emits repeated tokens instead of text and never forms a
>   tool call.
> - The NPU-optimised channel-wise tier (`int4-cw`) is not published for
>   Qwen3-4B, only Qwen3-8B.
>
> Out of the box you will instead hit `HTTP 400 — Input length exceeds the
> maximum allowed length`, because the NPU plugin caps prompts at 1024 tokens.
> Full measurements and analysis:
> [`TARGET_DEVICE=NPU` troubleshooting](../troubleshooting.md#target_devicenpu--llm-turns-fail-with-sorry-i-encountered-an-error)
> (ITEP-96031). `setup_models.sh --device NPU` refuses to run for this reason;
> use `--device NPU --skip-ovms` to target the NPU with the other services.
>
> **Use `TARGET_DEVICE=GPU`** (recommended) **or `TARGET_DEVICE=CPU`.**
> This restriction applies only to the OVMS-served LLM — `queue-service`,
> `identity-service` and `whisper-tiny`/`whisper-base` ASR still use the NPU.

## RAG Service Embedding/Reranker Device (`RAG_EMBEDDING_DEVICE`, `RAG_RERANKER_DEVICE`)

`rag-service`'s embedding (`BAAI/bge-large-en-v1.5`) and reranker
(`BAAI/bge-reranker-base`) components are OpenVINO IR models exported
in-process by `optimum-intel` (`rag-service/utils/ensure_model.py`) and
loaded via `OVModelForFeatureExtraction`/equivalent
(`rag-service/components/embedding_component.py`,
`reranker_component.py`). Their device is set independently of
`TARGET_DEVICE` via `RAG_EMBEDDING_DEVICE` / `RAG_RERANKER_DEVICE` in
`.env` (mapped by `docker-compose.yml` to
`SMART_KIOSK_RAG__MODELS__EMBEDDING__DEVICE` /
`SMART_KIOSK_RAG__RETRIEVAL__RERANKER__DEVICE`).

- **Currently supported:** `RAG_EMBEDDING_DEVICE`/`RAG_RERANKER_DEVICE` =
  `CPU` or `GPU`. Default: `GPU`.
- **Currently unsupported:** `NPU`.

> [!IMPORTANT]
> **`NPU` is not supported for the embedding/reranker models.**
> `optimum-intel`'s default export produces OpenVINO IR with dynamic
> (unbounded) sequence-length and batch shapes — required because queries
> and knowledge-base documents vary in length and the reranker batches
> multiple candidates per call (`rag-service/config.yaml`'s
> `models.embedding.batch_size`). The NPU compiler rejects this IR
> (`Missing upper bound for one or more nodes`); setting either variable to
> `NPU` will crash `rag-service` at startup.
>
> Forcing static/bounded shapes to work around this is not recommended:
> it would require padding every input to a fixed max length (wasting
> compute on short queries) and serializing what is currently a batched
> reranker call into one NPU invocation per candidate — likely slower
> overall than `GPU`/`CPU`, for a component that is not the latency
> bottleneck (the LLM is). `CPU` is normally fast enough for
> embedding/reranking; `GPU` is the default to match prior behavior.

## Environment Variables

kiosk-core has no config file. All settings are controlled through environment variables.

### kiosk-core API (`main:app`)

| Variable | Default | Description |
|---|---|---|
| `KIOSK_CORE_ANALYZER_URL` | `http://127.0.0.1:8010/v1/audio/transcriptions` | audio-analyzer transcription endpoint |
| `KIOSK_CORE_RAG_URL` | `http://127.0.0.1:8020/api/v1/query` | RAG query endpoint |
| `KIOSK_CORE_TTS_URL` | `http://127.0.0.1:8011/v1/audio/speech` | TTS speech synthesis endpoint |
| `KIOSK_CORE_TTS_MODEL` | `qwen-tts` | Model name sent to the TTS service |
| `KIOSK_CORE_TTS_VOICE` | *(unset)* | Voice name sent to the TTS service |
| `KIOSK_CORE_TTS_LANGUAGE` | `English` | Language sent to the TTS service |
| `KIOSK_CORE_TTS_INSTRUCTIONS` | *(unset)* | Optional style instructions for TTS |
| `KIOSK_CORE_SAMPLE_RATE` | `16000` | Default audio sample rate in Hz |
| `KIOSK_CORE_CHUNK_SECONDS` | `4.0` | Length of each audio chunk sent to audio-analyzer |
| `KIOSK_CORE_SILENCE_TIMEOUT_SECONDS` | `1.5` | Silence duration after speech that ends a session |
| `KIOSK_CORE_MAX_SESSION_SECONDS` | `20.0` | Hard cap on session duration |
| `KIOSK_CORE_SILENCE_THRESHOLD` | `900` | RMS threshold below which audio is treated as silence |
| `KIOSK_CORE_BLOCK_DURATION_SECONDS` | `0.1` | PortAudio capture block size |
| `KIOSK_CORE_PREROLL_SECONDS` | `0.3` | Audio buffered before speech starts |
| `KIOSK_CORE_HTTP_TIMEOUT_SECONDS` | `120.0` | HTTP client timeout for downstream calls |

### Kiosk UI runtime mode {#kiosk_ui_mode}

The React kiosk UI (`kiosk-ui/`) ships as a single image that can serve
either of two screens, selected at container start — no rebuild.
`KIOSK_UI_MODE` is the only variable the UI container reads; the browser
reaches `kiosk-core` and the other services through the nginx reverse
proxy baked into the image:

| Variable | Default | Description |
|---|---|---|
| `KIOSK_UI_MODE` | `operator` | `operator` — chat transcript + performance dashboard (existing behaviour), served on port 7860. `customer` — single-view kiosk screen with a queue-aware menu, live cart, and a voice-only "Ask" button, intended for the physical kiosk touchscreen. |

The value is written to `/usr/share/nginx/html/config.js` by
`docker-entrypoint.sh` (installed as an nginx `docker-entrypoint.d`
script) and read by the SPA before the React bundle loads. In
`docker-compose.yml` the two screens are separate containers
(`kiosk-ui` and `kiosk-ui-customer`) built from the *same* image/context,
published on different host ports (`7860` and `7861` respectively) so
they can be shown on two separate monitors during a demo.

## Compose Defaults

When running with the top-level [docker-compose.yml](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/docker-compose.yml), the defaults are wired to the internal Compose network:

- `KIOSK_CORE_ANALYZER_URL=http://audio-analyzer:8010/v1/audio/transcriptions`
- `KIOSK_CORE_RAG_URL=http://rag-service:8020/api/v1/query`
- `KIOSK_CORE_TTS_URL=http://text-to-speech:8011/v1/audio/speech`

Most deployments should leave these values unchanged. Override them only when `kiosk-core` or `kiosk-ui` must call services outside the local Compose stack.

## Session Parameters

Session parameters (chunk duration, silence threshold, etc.) can also be provided per-request in the POST body for `/api/v1/sessions/start` and `/api/v1/sessions/start-file`. Per-request values take precedence over the environment variable defaults.

---

## NPU Deployment Workflow

Step-by-step workflow to run this stack with Intel NPU acceleration where supported.

**NPU support by component:**

| Component | NPU |
|---|---|
| Queue Service (`QUEUE_DEVICE`) | ✅ Yes |
| Identity Service face/re-id (`IDENTITY_DEVICE`) | ✅ Yes |
| Identity Service voice/ECAPA-TDNN (`IDENTITY_DEVICE`) | ❌ No — dynamic shape rejected by NPU compiler |
| Audio Analyzer ASR (`models.asr.device`) | ⚠️ `whisper-tiny`/`whisper-base` only — see [ASR Support Matrix](#asr-support-matrix) |
| Audio Analyzer Diarization (`models.diarization.device`) | ❌ No — CPU only |
| OVMS-LLM (`TARGET_DEVICE`) | ❌ No — compiles, but INT8 is ~100× too slow and INT4 is numerically broken — see [`TARGET_DEVICE`](#ovms-llm-device-target_device) |
| RAG Service embedding/reranker (`RAG_EMBEDDING_DEVICE`/`RAG_RERANKER_DEVICE`) | ❌ No — dynamic shape rejected by NPU compiler |
| Text-to-Speech (`models.tts.device`) | ❌ No |

Do not set NPU for a ❌ component — it will fail to start or silently disable the feature.
The steps below configure NPU for `queue-service`; adapt the device variable for other ✅/⚠️ components.

### 1 — System requirements

| Requirement | Details |
|---|---|
| Hardware | Intel Core Ultra (Meteor Lake or later) with integrated NPU |
| Host driver | Intel NPU driver (`intel-npu-driver`) installed and loaded |
| User-space runtime | `intel-level-zero-npu` package |
| Host device | `/dev/accel/accel0` (or similar) present and accessible |
| OpenVINO | Container image already bundles the correct runtime |

Verify the NPU device node is present before proceeding:
```bash
ls /dev/accel/
# Expected: accel0   accelmon0
```

### 2 — Install the Intel NPU driver (if not already installed)

Refer to the Intel NPU driver repository: <https://github.com/intel/linux-npu-driver/releases>. Installation varies by distribution. After installation:

```bash
# Verify kernel driver is loaded
lsmod | grep intel_vpu
# Verify device node exists
ls -la /dev/accel/accel0
```

### 3 — Set NPU device for queue-service (and optionally audio-analyzer ASR)

```bash
# .env
QUEUE_DEVICE=NPU
```

Optionally, also enable NPU for ASR (`whisper-tiny`/`whisper-base` only) in `configs/audio-analyzer/config.yaml`:

```yaml
models:
  asr:
    provider: openvino
    device: NPU
    name: whisper-base  # whisper-tiny also works; whisper-small+ fails to compile
    weight_format: null
```

### 4 — Set `ACCEL_MOUNT_PATH` and start the stack

`ACCEL_MOUNT_PATH` is **not** auto-detected by `make up` or
`make check-env` — export it yourself before starting the stack:

```bash
export ACCEL_MOUNT_PATH=/dev/accel/accel0
cd smart-kiosk-assistant
make check-env
make up
```

Alternatively, set `ACCEL_MOUNT_PATH=/dev/accel/accel0` directly in
`.env` so it's picked up automatically on every `make up` /
`docker compose up` without exporting it each time.

### 5 — Verify NPU is active

```bash
docker ps --filter "name=queue-service" --format "{{.Names}}\t{{.Status}}"
docker exec queue-service python3 -c "import openvino as ov; print(ov.Core().available_devices)"  # expect: includes NPU
docker logs queue-service 2>&1 | grep -i "device=NPU\|npu"
```

If ASR is on NPU (`whisper-tiny`/`whisper-base` only):

```bash
docker logs audio-analyzer 2>&1 | grep -i "Loading Model"  # expect: device=NPU
curl -s -X POST http://localhost:8010/v1/audio/transcriptions -F "file=@your_sample.wav"
```

If identity-service face/re-id is on NPU:

```bash
IDENTITY_DEVICE=NPU make up IDENTITY=true
docker logs identity-service 2>&1 | grep -i "face\|voice\|inference_ready"
```

### 6 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Container unhealthy, `NPU not in available_devices` | NPU driver not loaded or `/dev/accel/accel0` not mapped | Verify host driver and set `ACCEL_MOUNT_PATH` |
| `libopenvino_intel_npu_compiler_loader.so` missing | NPU compiler not in image | Rebuild the affected image with NPU user-space packages |
| Slow first inference (20–60 s) | NPU compiler cache is empty (cold start) | Normal on first run; subsequent requests will be fast |
| `audio-analyzer` crash-loops with `Check '!self_attn_nodes.empty()' failed` after setting `models.asr.device=NPU` | Model is `whisper-small` or larger — NPUW's self-attention pattern-matching fails to statically-shape the graph for that model size | Use `whisper-tiny` or `whisper-base` on NPU instead, or switch `models.asr.device` to `CPU`/`GPU` |
| Non-NPU containers unhealthy after NPU config change | NPU-unrelated services picking up wrong env | Only modify the specific component's config (e.g. `QUEUE_DEVICE`, `IDENTITY_DEVICE`, `models.asr.device`) |

> **Cold-start note:** The OpenVINO NPU compiler caches compiled kernels inside the container under `/tmp/ov_cache/`. The first inference after a container restart takes significantly longer (20–60 s) while the cache warms up. This is expected behavior.



