# Smart Kiosk Assistant

Smart Kiosk Assistant is a voice-first retrieval-augmented kiosk stack for retail, QSR, and other customer-facing deployments. A browser session captures microphone audio, streams it through speech-to-text, retrieves grounded answers from a local knowledge base, and plays a generated spoken response back to the user. The full stack is designed to run locally on Intel CPU/GPU with OpenVINO-backed services.

## What This Repository Contains

- A browser-based Gradio kiosk UI with microphone capture and sequential audio playback.
- A FastAPI kiosk orchestrator that coordinates speech-to-text, retrieval, and text-to-speech.
- An internal RAG microservice with ingestion, vector storage, and streamed answers.
- Pinned runtime configuration for upstream `audio-analyzer` and `text-to-speech` microservices.
- Sample knowledge-base content in [knowledge-base-samples/](knowledge-base-samples/).

## Architecture

![Smart Kiosk Assistant architecture](docs/assets/architecture.png)

### Request Flow

1. The browser UI captures microphone audio and sends it to `kiosk-core`.
2. `kiosk-core` forwards audio to `audio-analyzer` for transcription.
3. `kiosk-core` sends the transcription, plus any runtime context, to `rag-service`.
4. `rag-service` retrieves relevant chunks from Chroma, prompts the OpenVINO LLM, and streams the answer back.
5. `kiosk-core` forwards the answer text to `text-to-speech`, stores the generated audio, and returns both text and audio metadata to the UI.

## Service Topology

All five services are started by the top-level [docker-compose.yml](docker-compose.yml).

| Service | Port | Role | Source |
|---|---|---|---|
| `audio-analyzer` | `8010` | Speech-to-text | [../edge-ai-libraries/microservices/audio-analyzer](../edge-ai-libraries/microservices/audio-analyzer) |
| `text-to-speech` | `8011` | Speech synthesis | [../edge-ai-libraries/microservices/text-to-speech](../edge-ai-libraries/microservices/text-to-speech) |
| `rag-service` | `8020` | Retrieval, ingestion, answer generation | [rag-service/README.md](rag-service/README.md) |
| `kiosk-core` | `8012` | Session API and service orchestration | [main.py](main.py) |
| `kiosk-ui` | `7860` | Gradio browser interface | [gradio_app.py](gradio_app.py) |

## Quick Start

Clone the repository and populate only the two upstream microservices this stack needs from the `edge-ai-libraries` submodule:

```bash
git clone https://github.com/intel-retail/voice-enabled-interactions.git
cd voice-enabled-interactions
git submodule update --init --depth 1 edge-ai-libraries
git -C edge-ai-libraries sparse-checkout set --cone microservices/audio-analyzer microservices/text-to-speech
cd smart-kiosk-assistant
```

If the repository is already present, you can apply the same sparse checkout from the repo root:

```bash
git submodule update --init --depth 1 edge-ai-libraries
git -C edge-ai-libraries sparse-checkout set --cone microservices/audio-analyzer microservices/text-to-speech
```

Build and start the full stack:

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
docker compose build
docker compose up -d
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) for the browser UI.

The compose stack runs most containers as your host user. Exporting `LOCAL_UID` and `LOCAL_GID` keeps bind-mounted files writable from the account that launched the stack.

## Documentation Map

- Start here: [docs/overview.md](docs/overview.md) and [docs/get-started.md](docs/get-started.md)
- Architecture and request flow: [docs/how-it-works.md](docs/how-it-works.md)
- Hardware and OS prerequisites: [docs/system-requirements.md](docs/system-requirements.md)
- Build details: [docs/build-from-source.md](docs/build-from-source.md)
- Container startup and verification: [docs/run-container.md](docs/run-container.md)
- Host-run kiosk-core and Gradio UI: [docs/run-standalone.md](docs/run-standalone.md)
- `kiosk-core` API: [docs/api-reference.md](docs/api-reference.md)
- Environment variables and inference device: [docs/configuration.md](docs/configuration.md)
- Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md)
- Release notes: [docs/release-notes.md](docs/release-notes.md)
- RAG service: [rag-service/README.md](rag-service/README.md)
- Pinned configs for upstream services: [configs/README.md](configs/README.md)

## Operational Notes

- Browser audio capture means no host microphone device needs to be passed into the containers.
- The compose stack mounts pinned `config.yaml` files for `audio-analyzer` and `text-to-speech` without modifying their upstream sources.
- `rag-service` persists model caches and vector storage in Docker volumes, so knowledge-base state survives container restarts.
- The browser UI supports knowledge-base ingestion with raw text or `.txt` and `.md` uploads routed through the internal RAG service.
