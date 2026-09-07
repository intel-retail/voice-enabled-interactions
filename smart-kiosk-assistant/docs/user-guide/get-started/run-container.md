# Run With Docker Compose

This guide covers building the images from source and running the full
stack with `docker compose`. For source code changes to any service, see
[Build from Source](./build-from-source.md).

Microphone audio is captured by the browser and uploaded to `kiosk-core`
as a WAV file. No host audio device is passed into the containers.

## Prerequisites

1. Complete Steps 1–4 of [Get Started](../get-started.md) (Docker, GPU
   drivers, repo clone, HuggingFace token).
2. Run `./setup_models.sh` to download the OVMS LLM model (Step 5 of
   Get Started). The stack will not start correctly without it.

## Build and Start

From `smart-kiosk-assistant/`:

```bash
make check-env
make up
```

For NPU-accelerated services (`queue-service`, `identity-service` face/re-id,
`audio-analyzer` ASR — `whisper-tiny`/`whisper-base` only, see
[ASR Support Matrix](./configuration.md#asr-support-matrix)), export
`ACCEL_MOUNT_PATH` yourself first — it is **not** auto-detected by
`make up` or `make check-env`:

```bash
export ACCEL_MOUNT_PATH=/dev/accel/accel0
make up
```

For direct `docker compose up` (no Makefile detection), set it yourself:

```bash
ACCEL_MOUNT_PATH=/dev/accel/accel0 docker compose up -d queue-service
```

`ACCEL_MOUNT_PATH` is the host NPU device node. The value above is a common
example; actual host path may vary. Compose maps it into the container at
`/dev/accel/accel0`.

| Container | Port | Purpose |
|---|---|---|
| `ovms-llm` | 8000 | Serves Qwen3-4B via OpenAI-compatible API |
| `metrics-collector` | 9000 | Hardware utilization metrics |
| `audio-analyzer` | 8010 | Whisper ASR + speaker diarization |
| `text-to-speech` | 8011 | SpeechT5 TTS synthesis |
| `rag-service` | 8020 | RAG pipeline + ordering agent |
| `kiosk-core` | 8012 | Session API + product ordering |
| `kiosk-ui` | 7860 | React kiosk UI — **operator** screen (chat, performance dashboard) |
| `kiosk-ui-customer` | 7861 | React kiosk UI — **customer** screen (queue-aware menu, cart, voice Ask) |

Both `kiosk-ui` and `kiosk-ui-customer` are the exact same image/build;
only the `KIOSK_UI_MODE` environment variable and published port differ
(see [Configuration](./configuration.md#kiosk_ui_mode)). There is no
separate Dockerfile or build context for the customer screen.

Containers run as non-root; every image is built with UID/GID
`1000:1000` and the named volumes are initialized with that ownership,
so no host UID/GID configuration is required.

## Verify

```bash
docker compose ps
curl --noproxy '*' http://127.0.0.1:8000/v3/models   # ovms-llm
curl --noproxy '*' http://127.0.0.1:8010/health       # audio-analyzer
curl --noproxy '*' http://127.0.0.1:8011/health       # text-to-speech
curl --noproxy '*' http://127.0.0.1:8020/health       # rag-service
curl --noproxy '*' http://127.0.0.1:8012/health       # kiosk-core
```

Open `http://127.0.0.1:7860` for the **operator** screen (chat +
performance dashboard) — click the microphone and speak your question.

Open `http://127.0.0.1:7861` for the **customer** kiosk screen — a single
view with the queue-aware menu, live cart, and a full-width **Ask**
button. Intended for the physical kiosk touchscreen (tested at 1920×1080
landscape) while the operator screen runs on a separate monitor.

> **Running on a remote or headless host?** Browsers only grant microphone
> access on a secure origin, so opening `http://<remote-ip>:7860` directly
> loads the page but fails with *"Microphone access requires HTTPS or
> localhost."* Forward the ports over SSH and use `127.0.0.1`:
>
> ```bash
> ssh -L 7860:localhost:7860 -L 7861:localhost:7861 <user>@<remote-ip>
> ```
>
> See [Microphone Does Not Work Over a Remote IP](../troubleshooting.md#microphone-does-not-work-over-a-remote-ip-insecure-origin)
> for the alternative Chrome flag workaround.

## Logs

```bash
docker compose logs -f kiosk-core
docker compose logs -f kiosk-ui
docker compose logs -f kiosk-ui-customer
```

## Restart / Stop

```bash
docker compose restart            # after env var change
docker compose build && docker compose up -d   # after a source code change
docker compose down               # teardown
```

## Notes

- The default Compose wiring connects `kiosk-core` and `kiosk-ui` to the
  internal `audio-analyzer`, `rag-service`, and `text-to-speech`
  containers. Override these URLs only when this stack must call
  services outside the local Compose network.
- See [Configuration](./configuration.md) for environment variables,
  model selection, and inference device, and
  [API Reference](../api-reference.md) for endpoint details.
