# Release Notes

## v2026.1.0-rc1

Initial release of Smart Kiosk Assistant.

- Voice loop: browser microphone capture → speech-to-text →
  retrieval-augmented answer → speech synthesis → audio playback.
- Five-service Docker Compose stack: `audio-analyzer`, `text-to-speech`,
  `rag-service`, `kiosk-core`, `kiosk-ui`.
- `kiosk-core` FastAPI session API with snapshot polling and a
  start-from-file endpoint for testing without a microphone.
- `kiosk-ui` Gradio interface with sequential audio playback and a
  knowledge-base ingestion panel (raw text, `.txt`, `.md`).
- `rag-service` with semantic chunking, ANN retrieval, optional
  cross-encoder reranking, and OpenVINO-backed Qwen LLM generation.
- Pinned, kiosk-owned configuration for upstream `audio-analyzer` and
  `text-to-speech` via the `configs/` indirection (no upstream forks).
- Sample QSR and retail knowledge-base content under
  `knowledge-base-samples/`.
- Per-service inference device selection (`CPU`, `GPU`, `NPU`).
  Actual execution depends on the OpenVINO backend's support for the
  chosen model.
- `RELEASE_TAG` in `.env` as the single source of truth for image tags.
