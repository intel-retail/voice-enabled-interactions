# Release Notes: Smart Kiosk Assistant

## 2026.2.0

This release delivers a unified Smart Kiosk Assistant platform with dual React-based user interfaces, OpenVINO-powered AI inference, agentic MCP tool calling, queue-aware ordering, enhanced voice interaction, optional multimodal identity, configurable inference devices, and streamlined deployment.

### Key Updates

- **Dual Kiosk Experiences:** Replaced the previous Gradio interface with a React, Vite, and TypeScript application. A single `kiosk-ui` image supports both operator and customer-facing modes through `KIOSK_UI_MODE`.

  - **Operator mode:** Voice/text chat, live queue monitoring, knowledge-base ingestion, audio configuration, and performance dashboards.
  - **Customer mode:** Queue-aware menu, category navigation, live cart, upsell prompts, queue status, and voice interaction.

- **Agentic Ordering with MCP:** Integrated an MCP tool server into `kiosk-core` to manage catalog browsing, cart operations, order confirmation, and upsell recommendations. Guardrails handle ambiguous item references and invalid quantities.

- **AI Inference with OVMS:** Migrated the Qwen3-4B ordering model from in-process OpenVINO inference to OpenVINO Model Server (OVMS), providing a dedicated OpenAI-compatible endpoint for agentic tool calling.

- **Queue-Aware Ordering:** Added YOLO-based person counting and RTSP streaming with OpenVINO to monitor queue length, provide a live MJPEG overlay, and dynamically adapt menu recommendations during peak periods.

- **Contextual Upselling:** Added a rule-based upsell engine that generates relevant add-on recommendations and surfaces them in both the customer cart and voice responses.

- **Enhanced Speech-to-Text:** Upgraded `audio-analyzer` with OpenAI-compatible SSE streaming transcription, real-time WebSocket transcription, VSS response compatibility, improved multi-speaker segmentation, and persistent speaker enrollment.

- **Speaker Diarization:** Enabled speaker attribution across the `audio-analyzer` and `kiosk-core` pipeline to improve turn identification during multi-speaker conversations.

- **Improved Text-to-Speech:** Added named voice support with faster response generation and more natural speech prosody.

- **Multimodal Identity:** Added an optional identity service supporting Face ID and voiceprint authentication using OpenVINO face inference, ECAPA voice embeddings, FAISS indexing, and SQLite-based loyalty profiles. The UI includes login, registration, and enrollment workflows.

- **Flexible Hardware Acceleration:** Added per-service inference-device configuration for CPU, GPU, and NPU. NPU passthrough is supported for `identity-service`, `ovms-llm`, and audio-analyzer ASR, with independent device configuration for RAG embedding and reranker models.

- **Unified Versioning:** All kiosk services and container images are aligned to `2026.2.0`:

  - `kiosk-core`
  - `kiosk-ui`
  - `queue-service`
  - `identity-service`
  - `rag-service`
  - `rtsp-streamer`
  - `audio-analyzer`
  - `text-to-speech`
  - `metrics-collector`

- **Simplified Deployment and Operations:** Added a Makefile-based workflow for environment initialization, configuration validation, model and sample-video downloads, image builds, service startup, health checks, log monitoring, individual service rebuilds, and cleanup.

- **Sample RTSP Video Provisioning:** Added configurable tooling to download and provision sample video clips required by the queue analytics pipeline.


## 2026.1.0

The initial release of Smart Kiosk Assistant marks the launch of a voice-enabled
interactive application for retail, QSR, Airlines and other customer-facing
environments. The application has the following features:

- Designed as a conversational AI experience, it enables users to engage
  naturally through speech and receive intelligent, spoken responses
  in real time.
- The platform brings together speech recognition, retrieval-augmented
  generation, and text-to-speech in a seamless, end-to-end voice
  interaction flow.
- With browser-based voice capture and natural audio playback, the experience
  feels intuitive, responsive, and ready for real-world engagement.
- Smart Kiosk Assistant grounds every response in an ingestible local knowledge
  base, helping deliver more relevant, context-aware, and business-specific
  answers.
- Its integrated AI stack combines kiosk UI, orchestration, speech-to-text,
  retrieval, and speech synthesis into a unified deployment-ready application.
- The experience is further enhanced by built-in visibility into model KPIs and
  live performance data, including runtime model details and latency metrics.
- Optimized for local and edge deployment, the application leverages OpenVINO
  acceleration on Intel hardware for efficient AI inference.
- Docker Compose packaging and flexible configuration make the solution easy to
  deploy, adapt, and scale across enterprise environments.
- This launch establishes Smart Kiosk Assistant as a strong foundation for
  immersive, intelligent, and voice-first digital engagement experiences.

