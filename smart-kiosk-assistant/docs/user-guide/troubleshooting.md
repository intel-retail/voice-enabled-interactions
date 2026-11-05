# Troubleshooting

## Stack Will Not Start

- Confirm the published host ports are free:

  ```bash
  ss -ltnp | grep -E "7860|8010|8011|8012|8020"
  ```

- Confirm Docker Compose can build:

  ```bash
  docker compose config
  docker compose build
  ```

- Tail individual services to find the first failure:

  ```bash
  docker compose logs -f audio-analyzer
  docker compose logs -f text-to-speech
  docker compose logs -f rag-service
  docker compose logs -f kiosk-core
  docker compose logs -f kiosk-ui
  ```

## `dependency failed to start: container <name> is unhealthy`

Symptom — `make up` (or `docker compose up`) aborts with one or more of:

```text
✘ Container audio-analyzer  Error dependency audio-analyzer failed to start
✘ Container text-to-speech  Error dependency text-to-speech failed to start
dependency failed to start: container text-to-speech is unhealthy
```

`rag-service` and `kiosk-core` gate on `depends_on: service_healthy`, so a
single service that never reports healthy takes the whole stack down with it.
Find the real error in the failing service's own log:

```bash
docker compose logs -f text-to-speech
docker compose logs -f audio-analyzer
```

**Cause 1 — no route to Hugging Face.** The log ends in:

```text
httpx.ConnectError: [Errno 101] Network is unreachable
```

raised from `huggingface_hub` during FastAPI startup. The service tried a live
Hub call (model resolution, or an OpenVINO IR export) and the host has no
direct internet access.

Note that `setup_models.sh` does **not** pre-fetch these models — it only
handles the OVMS LLM, the queue-analytics YOLO model and Silero VAD. The
`audio-analyzer`, `text-to-speech` and `rag-service` models are downloaded
inside their containers on first run, which is why `HF_HUB_OFFLINE=0`
(online) is the shipped default in `.env.example`. Fix with either:

- **Behind a proxy (first run on a proxied host).** Keep `HF_HUB_OFFLINE=0`
  so the models can be fetched, and make sure the containers can see your
  proxy. In order of preference:

  1. **Proxy already in `~/.docker/config.json`** under `proxies.default` —
     Docker injects it into every container and there is nothing to do.

  2. **Proxy exported in your shell** — `docker-compose.yml` declares
     `HTTP_PROXY`/`HTTPS_PROXY` (both letter cases) as value-less keys, so
     Compose passes your shell values straight through:

     ```bash
     export HTTP_PROXY=http://proxy.example.com:3128
     export HTTPS_PROXY=http://proxy.example.com:3128
     make up
     ```

  3. **Proxy pinned in `.env`** — uncomment and fill in:

     ```bash
     # .env
     HTTP_PROXY=http://proxy.example.com:3128
     HTTPS_PROXY=http://proxy.example.com:3128
     ```

  Because the keys are value-less rather than `${HTTP_PROXY:-}`, an unset
  proxy is *omitted* from the container instead of injected as an empty
  string — so option 1 keeps working even when you set nothing, and no
  override file or extra `make` flag is required.

  Use an address routable from inside a container. A loopback proxy
  (`127.0.0.1`/`localhost`) is the container itself, not your host, and
  downloads will hang; `make` warns if it detects one.

  The same values are passed as **build args** to every locally built
  service, so `make build REGISTRY=false` can reach the package mirrors —
  `apt-get`/`pip`/`npm` run inside the builder, where the service-level
  environment does not apply. `HTTP_PROXY`/`HTTPS_PROXY` are predefined build
  args, so they need no `ARG` line and are not baked into the image.

  `make` mirrors each proxy variable across letter cases, so setting only
  `HTTP_PROXY` is enough: `curl`, `wget` and `apt-get` read only the
  lower-case names and would otherwise ignore it. Mirroring only ever fills a
  *missing* counterpart. If you set both cases to different values, `make`
  warns and tells you which one actually takes effect — the lower-case one,
  for Python, `curl` and `wget` alike.

  > **Deprecated:** `KIOSK_HTTP_PROXY` / `KIOSK_HTTPS_PROXY` (read by the
  > removed `docker-compose.proxy.yml`, together with `make up PROXY=true`)
  > are still applied, with a warning, if the standard names are unset.
  > Rename them to `HTTP_PROXY` / `HTTPS_PROXY`; `PROXY=true` is now a no-op.

  Never write `HTTP_PROXY=` with an empty value in `.env`. Empty is *not* the
  same as unset: Compose injects the empty string, which suppresses the proxy
  Docker would otherwise take from `~/.docker/config.json` — for **both**
  letter cases. `export HTTP_PROXY=` behaves identically, and
  `HTTP_PROXY= # optional` is worse still, because Compose does not strip the
  comment and sets the proxy to the literal string `# optional`.
  `make check-env` fails on all three.

- **No egress and no proxy at all.** `make up` no longer starts a run that
  cannot succeed. It probes huggingface.co from a container first and then
  either switches to offline mode or stops:

  | Probe | Caches | Result |
  |---|---|---|
  | reachable | — | starts online (unchanged) |
  | unreachable | all 3 hold real weight files | starts with `HF_HUB_OFFLINE=1` automatically |
  | unreachable | cold or incomplete | refuses to start, prints the probe result and the fix |
  | not determinable | — | warns and continues (never blocks) |

  Previously the stack came up and stalled for the full 600s dependency
  timeout. The probe reports *why* it failed — a proxy returning `403`
  (huggingface.co blocked by policy) is reported differently from no route at
  all. A cache counts only if it contains an actual weight file over 1 MiB;
  an empty `models--*/` directory or an interrupted download does not.

  The probe runs in a container because a host `curl` says nothing about what
  a container reaches through Docker's `proxies.default`. It uses **`curl`**,
  and picks the probe image by whether it actually has `curl` or `wget` rather
  than by name — two things matter here:

  - huggingface.co's edge returns **`403 Forbidden` to busybox `wget`** even
    when the network path is fine (verified on a host where `wget` got `403`
    and `curl` got `200` over the same proxy). A wget-only `403` is therefore
    reported as *inconclusive*, never as "unreachable".
  - some service images ship neither `curl` nor `wget`, so probing with them
    fails with `wget: not found`, which is indistinguishable from no egress.

  An inconclusive probe drives **no** decision — it neither blocks startup nor
  switches to offline mode, and `HF_HUB_OFFLINE` is left exactly as you set
  it. Guessing "offline" is not a safe default: `optimum-cli export openvino`
  must reach the Hub API to infer a model's task, so forcing offline on a host
  that is actually online makes the first-run IR export fail and
  audio-analyzer crash-loop.

  This is also why the **converted IR volume** (`audio_analyzer_models`) is
  what gets inspected, not just the Hugging Face source cache: cached source
  weights are not sufficient for an offline start, because the conversion step
  itself needs the Hub.

  `HF_HUB_OFFLINE=1` is always honoured and never overridden; set
  `HF_AUTO_OFFLINE=false` to restore the old warn-and-continue behaviour.

  Intra-cluster traffic always bypasses the proxy via the shared `NO_PROXY`
  anchor in `docker-compose.yml`.

- **Warm cache, then go offline.** Start the stack once on a connected host
  so each service populates its Hugging Face cache volume, then set offline
  mode so no Hub call is attempted at all:

  ```bash
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  ```

  Do **not** set these on a host that has never completed a successful online
  run — with an empty cache the services will fail to find their models.

Then recreate the affected services:

```bash
docker compose up -d --force-recreate text-to-speech audio-analyzer
```

**Cause 2 — stale image.** If the log shows a configuration or missing-file
error rather than a network error, the `:latest` tag is resolving to an old
cached image. How you refresh it depends on `REGISTRY` in your `.env`:

```bash
# REGISTRY=true (the shipped default) — images are pulled, not built,
# so --no-cache has no effect. Re-pull instead:
docker compose pull <service-name>

# REGISTRY=false — images are built locally:
docker compose build --no-cache <service-name>
# or, for the whole stack:
make build
```

Then recreate the service:

```bash
docker compose up -d --force-recreate <service-name>
```

## First Startup Is Slow

This is expected. On first run each model-hosting service downloads or
exports model assets to its `models/` directory and Hugging Face cache.
Subsequent starts reuse the cached artifacts. The default
`audio-analyzer` healthcheck allows up to ~240 seconds for warmup; the
RAG LLM compile on GPU can also take a few minutes the first time.

## A `health` Endpoint Fails

- Run `docker compose ps` and check the `STATUS` column for `unhealthy`.
- If you are behind a corporate proxy, pass `--noproxy '*'` to `curl`
  when hitting `127.0.0.1`.
- Confirm the service container actually started:

  ```bash
  docker compose logs <service-name>
  ```

## `rtsp-streamer` Never Becomes Healthy

`queue-service` reports `dependency failed to start: container rtsp-streamer
is unhealthy`, and `rtsp-streamer` logs the same block every ~12 s:

```
[rtsp-streamer] Starting mediamtx RTSP server on port 8554 ...
[rtsp-streamer] mediamtx started (PID=18)
[rtsp-streamer] Waiting for mediamtx to accept connections on port 8554 ...
[rtsp-streamer] ERROR RTSP server failed to start on port 8554
```

The embedded MediaMTX server exited instead of binding port 8554. `start.sh`
now prints MediaMTX's own output immediately after that error:

```
[rtsp-streamer] ERROR mediamtx (PID=14) exited before accepting connections on port 8554
[rtsp-streamer] ----- mediamtx log (last 50 lines) -----
ERR: yaml: mapping values are not allowed in this context
[rtsp-streamer] ----- end mediamtx log -----
```

Read that block for the real cause — typically an unreadable or malformed
`rtsp-streamer/mediamtx.yml`, or port `8554` already in use on the host.

```bash
docker compose logs --no-color rtsp-streamer | tail -40
ss -ltnp | grep 8554          # is something else holding the port?
```

If you see no `mediamtx log` block at all, you are running an older image —
rebuild with `docker compose build rtsp-streamer`. `mediamtx.yml` and
`start.sh` are bind-mounted, so a rebuild is usually unnecessary for those
two files alone. Not needed at all when `QUEUE=false`: run `make up
QUEUE=false` to skip `rtsp-streamer` and `queue-service` entirely.

## A Service Crash-Loops with `[GPU] Context was not initialized for 0 device`

A service (most often `audio-analyzer`, but also `rag-service` or
`identity-service`) restarts forever, and its logs show:

```
Configured OpenVINO ASR device 'GPU' is not visible in this runtime.
OpenVINO available_devices=['CPU']
...
Check 'contexts.count(device_id)' failed ... [GPU] Context was not initialized for 0 device
ERROR:    Application startup failed. Exiting.
```

The service is pinned to `GPU` on a host where OpenVINO™ cannot use one. This
happens even when `/dev/dri` exists — the node being present does not mean the
GPU compute runtime is usable.

**Fix:** run `make setup-models TARGET_DEVICE=CPU`. This switches the whole
stack to CPU in one step — it sets `TARGET_DEVICE` and comments out any
per-service `*_DEVICE` variable still pinned to `GPU`:

```
✓ TARGET_DEVICE updated  → CPU
✓ ASR_DEVICE=GPU commented out (it would have overridden TARGET_DEVICE=CPU)
✓ RAG_EMBEDDING_DEVICE=GPU commented out (it would have overridden TARGET_DEVICE=CPU)
```

Confirm with `make check-env`, which prints the effective device per service
and names anything still pinned. On a host with no GPU render node this is a
hard error, so `make up` cannot proceed into a crash-loop:

```
✗ TARGET_DEVICE=CPU but these override(s) still pin GPU: ASR_DEVICE RAG_EMBEDDING_DEVICE
  A per-service *_DEVICE value takes precedence over TARGET_DEVICE, and this
  host has no GPU render node, so each pinned service will crash-loop with
  '[GPU] Context was not initialized for 0 device'.
```

On a host that *does* have a usable GPU the same situation is only a warning,
since pinning one service to GPU while the rest run on CPU is a legitimate
mixed setup.

A healthy CPU-only stack instead reports:

```
✓ effective inference devices  asr=CPU rag-embed=CPU rag-rerank=CPU identity=CPU ovms-llm=CPU
```

Alternatively, keep `GPU` and install the Intel GPU runtime:

```bash
sudo apt-get install -y intel-opencl-icd level-zero
sudo usermod -aG render $USER   # log out and back in
```

> **Why editing `TARGET_DEVICE` alone is not enough.** A per-service
> `*_DEVICE` value takes precedence over `TARGET_DEVICE`, and `.env` is not a
> shell — a variable can only inherit `TARGET_DEVICE` by being **absent or
> commented out**, never by being set to `${TARGET_DEVICE}`. `.env` files
> created before this behaviour existed hard-code these four variables to
> `GPU`; current `.env.example` ships them commented out, but that does not
> change a `.env` you already have. `make setup-models TARGET_DEVICE=CPU`
> reconciles the existing file for you.
>
> Switching back with `make setup-models TARGET_DEVICE=GPU` restores GPU
> everywhere — the commented-out pins simply inherit `TARGET_DEVICE` again.
> A deliberate non-GPU override such as `ASR_DEVICE=NPU` is never rewritten.

## Selected Device Is Not Used

The device field lives in the per-service pinned config (see
[Configuration](./get-started/configuration.md#inference-device)). If the device
does not appear in the logs:

- Check the value is supported for that model (e.g., `audio-analyzer`
  ASR supports `CPU` for `provider: openai`, and `CPU|GPU|NPU` for
  `provider: openvino` — `NPU` only works with `whisper-tiny`/`whisper-base`,
  see [ASR Support Matrix](./get-started/configuration.md#asr-support-matrix)).
- For `GPU`: confirm `/dev/dri` exists and the OpenVINO™ GPU
  runtime is installed.
- Restart the affected service after the change:

  ```bash
  docker compose up -d --build --force-recreate <service-name>
  ```

- Confirm OpenVINO™ picked the device:

  ```bash
  docker compose logs <service-name> | grep -i -E "device|compiling|GPU|CPU"
  ```

For `audio-analyzer` specifically, check the effective provider/device
selection from `configs/audio-analyzer/config.yaml`:

```bash
grep -nE "provider:|device:" configs/audio-analyzer/config.yaml
```

and verify startup behavior:

```bash
docker logs audio-analyzer
```

`docker-compose.yml` is the single Compose file and should not override
ASR provider/device. The ASR selection is read only from
`configs/audio-analyzer/config.yaml`.

> **Important:**
> `audio-analyzer` ASR on NPU works only for `whisper-tiny`/`whisper-base`;
> `whisper-small`+ fails to compile. `make check-env` does not check model
> name, so a `whisper-small`+ NPU config passes `check-env` but crash-loops
> at container startup. See
> [ASR Support Matrix](./get-started/configuration.md#asr-support-matrix).

NPU-capable services (`queue-service`, `identity-service` face/re-id,
`audio-analyzer` ASR with `whisper-tiny`/`whisper-base`) get the host NPU
device via `ACCEL_MOUNT_PATH` (defaults to `/dev/null` so CPU/GPU-only
hosts are unaffected). Neither `make up` nor `make check-env`
auto-detects this — export `ACCEL_MOUNT_PATH` yourself (or set it in
`.env`) before starting the stack.

Before startup, run:

```bash
make check-env
```

### Error: OpenVINO™ does not report an NPU device

```bash
ls -l /dev/accel/
make check-env
```

For direct Compose runs, set the mapping explicitly:

```bash
ACCEL_MOUNT_PATH=/dev/accel/accel0 docker compose up -d queue-service
```

### `models.asr.device=NPU` Fails to Compile for `audio-analyzer`

| Provider + Device | Model | Result |
| --- | --- | --- |
| `openvino` + `NPU` | `whisper-tiny`/`whisper-base` | ✅ Works |
| `openvino` + `NPU` | `whisper-small`+ | ❌ `Check '!self_attn_nodes.empty()' failed` |
| `openvino` + `GPU`/`CPU` | any | ✅ Works |
| `openai` + `GPU`/`NPU` | any | ❌ Not supported (CPU only) |

Fix: use `whisper-tiny`/`whisper-base` on NPU, or switch to `GPU`/`CPU` for larger models, then recreate:

```bash
docker compose up -d --force-recreate audio-analyzer
docker logs audio-analyzer
curl http://localhost:8010/health
```

### `TARGET_DEVICE=NPU` — LLM Turns Fail with "Sorry, I encountered an error"

**Symptom.** The stack starts, `ovms-llm` reports `AVAILABLE`, ASR and TTS
work, but *every* agent turn — both knowledge questions and ordering
requests — replies `Sorry, I encountered an error. Please try again.`

**Cause.** OVMS serves NPU through a **Stateful** servable (Continuous
Batching is CPU/GPU only). The NPU plugin caps prompts at **1024 tokens** by
default. This agent's prompt is far larger:

| Prompt | Tokens | vs 1024 |
| --- | ---: | ---: |
| System instruction only | ~1,530 | 1.5x |
| \+ 12 MCP tool schemas (a normal turn) | ~3,900 | 3.8x |
| \+ one tool result (2nd round-trip) | ~4,200 | 4.1x |

Every prompt exceeds the cap, so OVMS rejects the request with
`HTTP 400 — Input length exceeds the maximum allowed length`. The agent
endpoint swallows the error and surfaces the canned reply.

> **Warning:**
> `MAX_PROMPT_LEN` must be a **top-level** key of `plugin_config`. Written as
> `{"DEVICE_PROPERTIES":{"NPU":{"MAX_PROMPT_LEN":8192}}}` it is accepted at load
> time — the servable still reports `AVAILABLE` — but is **silently ignored**,
> leaving the 1024 default in force. Verified on MTL with Qwen3-4B: the nested
> form rejects a 1,824-token prompt, the top-level form serves 3,624. This is
> recorded only so the next person does not lose a day to it — raising the cap
> makes NPU *work*, not *usable*, for the reasons in the table above.

**Recommended fix — use GPU (or CPU).** NPU is **not supported** for the served
LLM. `setup_models.sh --device NPU` now refuses to run for this reason:

```bash
# .env: TARGET_DEVICE=GPU
./setup_models.sh --device GPU
docker compose up -d --force-recreate ovms-llm
```

The NPU is still used by `queue-service`, `identity-service` and
`whisper-tiny`/`whisper-base` ASR — pass `--skip-ovms` to set those up on NPU
while leaving the LLM on GPU.

**Why NPU is not offered as an option.** Measured on an MTL NPU with Qwen3-4B
and OVMS 2026.3, using a graph with the prompt cap correctly raised:

| Weight format | Output quality | 2,559-token tool-calling turn |
| --- | --- | ---: |
| INT8 (default) | ✅ correct tool call | **801 s** |
| INT4 (`Qwen3-4B-int4-ov`) | ❌ garbage — `"the the the…"`, `"ômeôme…"` | 8 s |
| INT4 channel-wise (`int4-cw`) | — | not published for Qwen3-4B (only 8B) |

INT8 latency scales as 191 s (51-token prompt) → 217 s (1,824) → 245 s (3,624)
→ 801 s once 48 output tokens are generated; decode, not prefill, dominates. A
single agent turn issues several such calls, so the only weight format that is
correct on NPU is roughly two orders of magnitude too slow for voice.

Other Stateful-servable constraints that apply if this is ever revisited:

- NPU uses **static shapes**, so the full `MAX_PROMPT_LEN` window is compiled
  into the model — raising the limit costs compile time, memory and latency
  rather than saving it.
- Requests are handled **strictly one at a time**; two concurrent kiosk sessions
  serialize.
- Continuous-Batching flags (`cache_size`, `max_num_seqs`,
  `max_num_batched_tokens`) are **ignored**. Prefix caching is available only
  via `NPUW_LLM_ENABLE_PREFIX_CACHING`.
- A call can outlast `rag-service`'s 90 s generation ceiling
  (`answering.generation_timeout_secs`), which produces the same canned error
  even when OVMS itself would eventually answer.
- `finish_reason=length` is unsupported, as are beam search, `n > 1` and logprobs.

> **Important:**
> `rag-service` embedding/reranker (`RAG_EMBEDDING_DEVICE` /
> `RAG_RERANKER_DEVICE`, independent of `TARGET_DEVICE`) **cannot** compile on
> NPU at all — they are exported with dynamic sequence-length shapes, which the
> NPU compiler rejects. Leave both on `CPU` or `GPU`.

### `IDENTITY_DEVICE=NPU` — Face/Re-ID Works, Voice Does Not

Face/re-id models run on NPU. The voice model (ECAPA-TDNN) fails to
compile (`Upper bounds are not specified for node ... compute_STFT`), so
voice auth stays disabled (`inference_ready=false`) — this is expected.
See [Identity Service Device](./get-started/configuration.md#identity-service-device-identity_device).

## Permission Errors on Mounted Folders

Every container runs as UID/GID `1000:1000` (baked into each image).
Model files and caches for `audio-analyzer` and `text-to-speech` live
in Docker named volumes (`audio_analyzer_models`,
`audio_analyzer_cache`, `text_to_speech_models`, etc.) initialized with
that ownership, so the usual host-side ownership errors do not apply.
If you still see:

```text
PermissionError: [Errno 13] Permission denied: '...'
```

on a path inside the container, a named volume was likely created
earlier with the wrong ownership (for example by an older root-only
run). Reset it:

```bash
docker compose down
docker volume rm \
  smart-kiosk-assistant_audio_analyzer_models \
  smart-kiosk-assistant_audio_analyzer_cache \
  smart-kiosk-assistant_text_to_speech_models \
  smart-kiosk-assistant_text_to_speech_cache
docker compose up -d
```

Replace `smart-kiosk-assistant_` with whatever Compose project prefix
`docker volume ls` shows on your host. Resetting a volume forces the
services to re-download model assets on next startup.

## Microphone Does Not Work Over a Remote IP (Insecure Origin)

Browsers only expose `navigator.mediaDevices` on a **secure context** —
HTTPS, or a `localhost`/`127.0.0.1` loopback address. When the kiosk stack
runs on a remote or headless machine and you open
`http://<remote-ip>:7860` (operator) or `http://<remote-ip>:7861`
(customer) directly, the page itself loads and renders normally, but every
microphone action fails with:

```text
Microphone access requires HTTPS or localhost.
```

The UI is not broken and the containers are healthy — the browser is
withholding the microphone API because the origin is not trusted. Use
either workaround below.

### Workaround 1 — SSH port forwarding (recommended)

Tunnel the UI port so the browser sees a `localhost` origin, which is
always treated as secure. No browser configuration is needed and it works
in every browser:

```bash
# operator UI
ssh -L 7860:localhost:7860 <user>@<remote-ip>

# customer UI (add -L per port, or combine in one command)
ssh -L 7860:localhost:7860 -L 7861:localhost:7861 <user>@<remote-ip>
```

Leave the SSH session open, then browse to `http://127.0.0.1:7860` (or
`http://127.0.0.1:7861`) on your **local** machine.

> **Note:** Use `127.0.0.1`, not the machine's LAN IP. Only the loopback
> address qualifies as a secure origin.

### Workaround 2 — Chrome insecure-origin flag

Tell Chrome to treat the remote origin as secure. This is per-browser and
must be repeated on every client machine, so prefer the SSH tunnel for
anything beyond a quick demo:

1. Open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.
2. Add the exact origin, including the scheme and port — for example
   `http://10.223.23.34:7860`. Add a second comma-separated entry for
   `http://10.223.23.34:7861` if you also need the customer screen.
3. Set the flag to **Enabled** and relaunch Chrome when prompted.

> **Warning:** This flag disables an origin-security protection for the
> listed addresses. Use it only on trusted networks, and remove the entry
> when you are finished.

## Browser UI Does Not Capture Audio

- If you are reaching the UI over a remote IP, see
  [Microphone Does Not Work Over a Remote IP](#microphone-does-not-work-over-a-remote-ip-insecure-origin)
  first — this is the most common cause.
- Confirm the browser granted microphone permission for the origin you are
  using. Reset the permission and reload if needed.
- Check the `kiosk-ui` logs for upload errors:

  ```bash
  docker compose logs -f kiosk-ui
  ```

## Speaker Labels Are Missing / Diarization Fails to Download

Speaker diarization pulls **three gated** Pyannote models from HuggingFace:

| Model | Gated | Why it is fetched |
| ----- | ----- | ----------------- |
| [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) | Yes | Configured pipeline (`models.diarization.name`) |
| [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) | Yes | Segmentation dependency of the pipeline |
| [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) | Yes | Pulled by `pyannote.audio` during pipeline setup |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | No | Embedding dependency — no licence needed |

All three gated repos must be accepted; accepting only the configured
`speaker-diarization-3.1` still fails.

Typical `audio-analyzer` log signature:

```text
401 Client Error ... Cannot access gated repo for url
https://huggingface.co/pyannote/segmentation-3.0/resolve/main/config.yaml
```

To fix:

1. Confirm `HF_TOKEN` is set in `.env` and the container picked it up:

   ```bash
   docker compose exec audio-analyzer printenv HF_TOKEN
   ```

2. While signed in with the **same** HuggingFace account that owns the
   token, accept the licence on all three pages:
   - <https://huggingface.co/pyannote/speaker-diarization-community-1>
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>

3. Recreate the service so it retries the download:

   ```bash
   docker compose up -d --force-recreate audio-analyzer
   ```

If you do not need per-speaker attribution, disable diarization instead (the
default) — transcription continues to work normally:

```bash
# .env
KIOSK_CORE_DIARIZATION_ENABLED=false
```

Note that a diarizer load failure is **non-fatal**: it is logged as a
warning and diarization is disabled for that session, so `audio-analyzer`
stays healthy and `/health` still passes even when speaker labels are
missing. Always check the logs rather than the health endpoint.

## Answer Is Empty or Off-Topic

- Confirm the knowledge base was ingested. The operator UI exposes a
  Knowledge Base panel; see also
  [rag-service/README.md](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/rag-service/README.md).
- Check `rag-service` logs for retrieval scores and reranker output.
- Try the same question from the API to rule out the UI:

  ```bash
  curl --noproxy '*' -X POST http://127.0.0.1:8020/api/v1/query \
    -H 'Content-Type: application/json' \
    -d '{"query":"What are the store hours?"}'
  ```

## TTS Plays No Audio in the Browser

- Confirm the session snapshot has non-empty `tts_audio_segments` and
  no `tts_errors`. See [API Reference](./api-reference.md).
- The `kiosk-core` container and the `kiosk-ui` container share the
  `generated_audio` Docker volume. If you removed the volume, recreate
  the stack:

  ```bash
  docker compose down
  docker compose up -d --build
  ```

## kiosk-core Cannot Reach a Downstream Service

The compose defaults wire `kiosk-core` and `kiosk-ui` to the internal
service names (`audio-analyzer`, `text-to-speech`, `rag-service`). If
you override these URLs for a host-run setup, confirm:

- The downstream service is reachable from `kiosk-core` (try `curl`
  against the override URL from inside the `kiosk-core` container or
  from the host).
- For host-run downstreams reached from a container, use
  `host.docker.internal` (see the alternative compose snippets in
  [Run With Docker Compose](./get-started/run-container.md)).

## See Also

- [Configuration](./get-started/configuration.md)
- [Get Started](./get-started.md)
