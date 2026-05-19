# RAG Service

FastAPI retrieval-augmented Q&A service for the [Smart Kiosk Assistant](../README.md). Ingests a domain knowledge base, embeds it into Chroma, and answers streamed questions with an OpenVINO LLM. Listens on port `8020` and is started by the top-level `docker-compose.yml`.

## What It Offers

- Streaming kiosk query API: `POST /api/v1/query` (SSE tokens, optional `sources` event).
- OpenAI-compatible chat: `POST /v1/chat/completions`.
- Document ingestion via raw text (`POST /api/v1/context`) or `.txt`/`.md` upload (`POST /api/v1/context/file`).
- Configurable ingest token cap (default 25 000, set `RAG_MAX_INGEST_TOKENS`).
- Domain-aware system prompt (QSR / retail / generic) with concise, answer-first replies.
- OpenVINO LLM (CPU or GPU), pluggable embeddings and chunking strategies.

## Run & Configure

- Run in Docker: [docs/run-container.md](docs/run-container.md)
- Run on the host: [docs/run-standalone.md](docs/run-standalone.md)
- Configuration reference: [docs/configuration.md](docs/configuration.md)
- API examples: [docs/api.md](docs/api.md)

When started through the kiosk stack, this service's [`config.container.yaml`](config.container.yaml) is mounted in by the top-level compose file and layered over `config.yaml`. The same override pattern is used for the `audio-analyzer` and `text-to-speech` services — see [`../configs/`](../configs/README.md).
