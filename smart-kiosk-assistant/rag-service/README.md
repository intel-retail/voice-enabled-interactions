# RAG Service

FastAPI service for retrieval-augmented question answering in smart-kiosk-assistant.

## Start Here

Use the linked docs for actual run steps and API examples.

- Run in Docker: [docs/run-container.md](docs/run-container.md)
- Run on the host: [docs/run-standalone.md](docs/run-standalone.md)
- Change configuration: [docs/configuration.md](docs/configuration.md)
- API examples: [docs/api.md](docs/api.md)

## What It Does

The service ingests retail context, semantically chunks it, stores embeddings in Chroma, and answers streamed customer questions with an OpenVINO LLM.

It supports:

- Streaming kiosk query API at `POST /api/v1/query`
- OpenAI-compatible chat completions at `POST /v1/chat/completions`
- Context ingestion by raw text or file upload
- OpenVINO LLM export or download on startup
- Configurable semantic chunking using the same LLM family used for answering

## Notes

- Do not use this page as the run guide; use the linked docs above.
- First startup can be slow because model download or OpenVINO export may happen during startup.
- The default API contract keeps kiosk-core compatibility at `http://127.0.0.1:8020/api/v1/query`.
