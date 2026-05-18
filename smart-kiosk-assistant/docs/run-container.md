# Run With Docker Compose (Gradio UI Mode)

`docker compose up` starts **audio-analyzer**, **text-to-speech**, **rag-service**, **kiosk-core** (REST API), and **kiosk-ui** (Gradio browser interface) as containers.

Mic audio is captured by the **browser** via the Web Audio API and uploaded to kiosk-core as a WAV file. No host mic hardware is passed into the containers.

For the terminal-based key-press mic loop instead, see [run-standalone.md](run-standalone.md).

Clone the repo with its dependency submodule before starting:

```bash
git clone --recurse-submodules https://github.com/unarayan/voice-enabled-interactions.git
cd voice-enabled-interactions/smart-kiosk-assistant
```

If the repo is already cloned, run `git submodule update --init --recursive` from the repo root.

## Before You Start

Initialize the submodule once so the edge service sources exist locally:

```bash
git submodule update --init --recursive
```

## Start

From the `smart-kiosk-assistant/` directory:

```bash
docker compose up -d --build
```

This starts five containers:

| Container | Port | Purpose |
|---|---|---|
| `audio-analyzer` | `8010` | Speech-to-text |
| `text-to-speech` | `8011` | Speech synthesis |
| `rag-service` | `8020` | Knowledge-base retrieval |
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

By default the internal service URLs are already wired through the Compose network. Only override them if you want kiosk-core or kiosk-ui to talk to services outside this stack:

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
