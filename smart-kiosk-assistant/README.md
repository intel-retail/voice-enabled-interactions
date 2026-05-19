# Smart Kiosk Assistant

A voice-driven, retrieval-augmented kiosk assistant for retail, QSR, and similar customer-facing deployments. Speak a question in the browser; the assistant transcribes it, answers from a domain knowledge base, and speaks the reply back. Runs locally on Intel CPU/GPU via OpenVINO.

## What It Offers

- Browser mic capture and sequential TTS playback (no host mic passthrough needed).
- Streaming retrieval-augmented answers, domain-aware (QSR / retail / generic).
- Document ingestion from the UI (`.txt` / `.md` or raw text) with a configurable token cap.
- Five services wired together by one `docker-compose.yml`; pinned configs for reproducible behaviour.
- Sample knowledge bases in [knowledge-base-samples/](knowledge-base-samples/).

## Run

Clone with the `edge-ai-libraries/` submodule:

```bash
git clone --recurse-submodules https://github.com/unarayan/voice-enabled-interactions.git
```

Move into the kiosk directory:

```bash
cd voice-enabled-interactions/smart-kiosk-assistant
```

Build and start the stack:

```bash
docker compose build
docker compose up -d
```

Open the UI at [http://127.0.0.1:7860](http://127.0.0.1:7860).

For terminal-only mic mode, see [docs/run-standalone.md](docs/run-standalone.md). Full container instructions: [docs/run-container.md](docs/run-container.md).

## Services

| Service | Port | Source |
|---|---|---|
| audio-analyzer | `8010` | [edge-ai-libraries/microservices/audio-analyzer](../edge-ai-libraries/microservices/audio-analyzer) (submodule, unmodified) |
| text-to-speech | `8011` | [edge-ai-libraries/microservices/text-to-speech](../edge-ai-libraries/microservices/text-to-speech) (submodule, unmodified) |
| rag-service | `8020` | [rag-service/](rag-service/README.md) |
| kiosk-core | `8012` | [main.py](main.py) |
| kiosk-ui | `7860` | [gradio_app.py](gradio_app.py) |

## Configuration

- kiosk-core / kiosk-ui env vars: [docs/configuration.md](docs/configuration.md)
- kiosk-core REST API: [docs/api.md](docs/api.md)
- RAG service config, ingest API, token cap: [rag-service/README.md](rag-service/README.md)
- Pinned `config.container.yaml` for `audio-analyzer` and `text-to-speech` (override upstream without patching it): [configs/README.md](configs/README.md)
