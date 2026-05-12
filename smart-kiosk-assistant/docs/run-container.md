# Run With Docker Compose (Gradio UI Mode)

`docker compose up` starts **kiosk-core** (REST API) and **kiosk-ui** (Gradio browser interface) as containers.

Mic audio is captured by the **browser** via the Web Audio API and uploaded to kiosk-core as a WAV file. No host mic hardware is passed into the containers.

For the terminal-based key-press mic loop instead, see [run-standalone.md](run-standalone.md).

## Before You Start

The three downstream services must be running on the host before starting this stack:

| Service | Default port | Start command |
|---|---|---|
| audio-analyzer | `8010` | `cd audio_analyzer && docker compose up -d` |
| text-to-speech | `8011` | `cd text-to-speech && docker compose up -d` |
| RAG service | `8020` | `cd rag-service && docker compose up -d` |

The Compose file uses `host.docker.internal` (mapped to the host gateway) so containers can reach host-side services by port.

## Start

From the `smart-kiosk-assistant/` directory:

```bash
docker compose up -d --build
```

This starts two containers:

| Container | Port | Purpose |
|---|---|---|
| `kiosk-core` | `8012` | FastAPI session API |
| `kiosk-ui` | `7860` | Gradio voice UI |

## Verify

```bash
docker compose ps
curl --noproxy '*' http://127.0.0.1:8012/health   # {"status":"ok"}
```

Open the Gradio UI in a browser:

```
http://127.0.0.1:7860
```

Click the microphone button, speak your question, and the assistant responds with text and audio.

## Follow Logs

```bash
docker compose logs -f kiosk-core
docker compose logs -f kiosk-ui
```

## Restart / Stop

```bash
# After env var change
docker compose restart

# After code or dependency change
docker compose up -d --build

# Full teardown
docker compose down
```

## Override Downstream URLs

Edit the `environment:` block in `docker-compose.yml`, or pass variables on the command line:

```bash
KIOSK_CORE_ANALYZER_URL=http://192.168.1.10:8010/v1/audio/transcriptions \
KIOSK_CORE_TTS_URL=http://192.168.1.10:8011/v1/audio/speech \
KIOSK_CORE_RAG_URL=http://192.168.1.10:8020/api/v1/query \
docker compose up -d --build
```

See [configuration.md](configuration.md) for all variables.

## Notes

- TTS audio files are written by kiosk-core into `generated_audio/` (volume-mounted into both containers) so the Gradio UI can serve them for playback.
- For API use cases and endpoint details, see [api.md](api.md).
