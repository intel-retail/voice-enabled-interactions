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
docker compose build
docker compose up -d
```

| Container | Port | Purpose |
|---|---|---|
| `ovms-llm` | 8000 | Serves Qwen3-4B via OpenAI-compatible API |
| `metrics-collector` | 9000 | Hardware utilization metrics |
| `audio-analyzer` | 8010 | Whisper ASR + speaker diarization |
| `text-to-speech` | 8011 | SpeechT5 TTS synthesis |
| `rag-service` | 8020 | RAG pipeline + ordering agent |
| `kiosk-core` | 8012 | Session API + product ordering |
| `kiosk-ui` | 7860 | React voice kiosk UI |

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

Open `http://127.0.0.1:7860` in a browser, click the microphone, and
speak your question.

## Logs

```bash
docker compose logs -f kiosk-core
docker compose logs -f kiosk-ui
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
