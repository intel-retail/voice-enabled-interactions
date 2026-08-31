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

## Selected Device Is Not Used

The device field lives in the per-service pinned config (see
[Configuration](./get-started/configuration.md#inference-device)). If the device
does not appear in the logs:

- Check the value is supported for that model (e.g. `audio-analyzer`
  ASR supports `CPU` for `provider: openai`, and `CPU|GPU|NPU` for
  `provider: openvino` — `NPU` only works with `whisper-tiny`/`whisper-base`,
  see [ASR Support Matrix](./get-started/configuration.md#asr-support-matrix)).
- For `GPU`: confirm `/dev/dri` exists and the Intel OpenVINO GPU
  runtime is installed.
- Restart the affected service after the change:

  ```bash
  docker compose up -d --build --force-recreate <service-name>
  ```

- Confirm OpenVINO picked the device:

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

> [!IMPORTANT]
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

### Error: OpenVINO does not report an NPU device

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
|---|---|---|
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

### `TARGET_DEVICE=NPU` Causes OVMS/RAG to Restart-Loop

`ovms-llm` compiles on NPU but chat-completion requests fail at runtime
(`Input length exceeds the maximum allowed length`). `rag-service`
embedding/reranker (`RAG_EMBEDDING_DEVICE`/`RAG_RERANKER_DEVICE`, independent
of `TARGET_DEVICE`) fails to compile on NPU entirely — don't set either to `NPU`.

```bash
# .env: TARGET_DEVICE=CPU or GPU
docker compose up -d --force-recreate ovms-llm
```

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

```
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

```
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
|---|---|---|
| [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) | Yes | Configured pipeline (`models.diarization.name`) |
| [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) | Yes | Segmentation dependency of the pipeline |
| [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) | Yes | Pulled by `pyannote.audio` during pipeline setup |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | No | Embedding dependency — no licence needed |

All three gated repos must be accepted; accepting only the configured
`speaker-diarization-3.1` still fails.

Typical `audio-analyzer` log signature:

```
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
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

3. Recreate the service so it retries the download:

   ```bash
   docker compose up -d --force-recreate audio-analyzer
   ```

If you do not need per-speaker attribution, disable diarization instead —
transcription continues to work normally:

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
