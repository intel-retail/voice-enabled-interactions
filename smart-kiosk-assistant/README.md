# kiosk-core

`kiosk-core` is an orchestration service that wires together the audio-analyzer, RAG, and TTS microservices into a single voice assistant pipeline. It exposes a REST API for session control and ships a Gradio browser UI for interactive use.

## Clone With Dependencies

The audio-analyzer and text-to-speech services are linked through the repository-level `edge-ai-libraries/` Git submodule.

Clone with:

```bash
git clone --recurse-submodules https://github.com/unarayan/voice-enabled-interactions.git
cd voice-enabled-interactions
```

If you already cloned the repo without submodules:

```bash
git submodule update --init --recursive
```

## Two Ways to Run

### Mode A — Gradio UI (browser, fully containerised)

Mic audio is captured by the **browser** (Web Audio API) and uploaded to kiosk-core as a file. No host mic passthrough required.

```bash
git submodule update --init --recursive
docker compose up -d --build
# → kiosk-core API   http://127.0.0.1:8012
# → Gradio UI        http://127.0.0.1:7860  ← open in browser, speak
```

Full details: [docs/run-container.md](docs/run-container.md)

### Mode B — CLI mic loop (terminal, host-side)

kiosk-core runs on the host so `sounddevice` opens the mic directly. Press **Enter** to speak, see transcript + RAG response + TTS paths in the terminal.

```bash
uvicorn main:app --host 0.0.0.0 --port 8012 &
python mic_session.py --device "default"
```

Full details: [docs/run-standalone.md](docs/run-standalone.md)

---

- Configuration reference: [docs/configuration.md](docs/configuration.md)
- API reference: [docs/api.md](docs/api.md)

## What It Does

- Accepts a live microphone session or an uploaded audio file.
- Buffers and chunks the audio, sending each chunk to the audio-analyzer ASR service.
- Assembles the final transcript and forwards it to the RAG query service.
- Detects sentence boundaries in the streamed RAG answer and sends each sentence to the TTS service.
- Saves each TTS WAV clip locally for browser playback via the Gradio UI.

## Services Required

kiosk-core calls these downstream services:

| Service | Default URL | Provides |
|---|---|---|
| audio-analyzer | `http://127.0.0.1:8010/v1/audio/transcriptions` | Speech-to-text |
| RAG service | `http://127.0.0.1:8020/api/v1/query` | Knowledge-base Q&A |
| text-to-speech | `http://127.0.0.1:8011/v1/audio/speech` | Speech synthesis |

In container mode, `docker compose up` from `smart-kiosk-assistant/` now starts all five services together:

- `audio-analyzer`
- `text-to-speech`
- `rag-service`
- `kiosk-core`
- `kiosk-ui`

All URLs are overridable via environment variables. See [docs/configuration.md](docs/configuration.md).

## Endpoints

- `GET /health`
- `GET /api/v1/devices`
- `GET /api/v1/sessions`
- `GET /api/v1/sessions/{session_id}`
- `POST /api/v1/sessions/start`
- `POST /api/v1/sessions/start-file`
- `POST /api/v1/sessions/{session_id}/stop`

## Notes

- Do not use this page as the run guide; use the linked docs above.
- The Gradio UI (`gradio_app.py`) captures microphone audio in the browser and sends it to the API via the `start-file` endpoint — no server-side mic hardware is needed for the UI flow.
- TTS audio files are stored under `generated_audio/<session_id>/` relative to this directory.
