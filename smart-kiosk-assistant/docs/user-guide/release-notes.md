# Release Notes: Smart Kiosk Assistant

## 2026.2.0.1

This release expands Smart Kiosk Assistant with queue-aware ordering, a
refreshed web experience, and a streamlined build workflow. This update
includes the following changes:

- The kiosk front end has been rebuilt as a React (Vite + TypeScript)
  single-page application, replacing the previous Gradio interface for a
  faster and more customizable web experience.
- The ordering agent now runs its Qwen3-4B language model through OpenVINO
  Model Server (OVMS) instead of an in-process OpenVINO model, providing a
  dedicated, OpenAI-compatible inference endpoint for tool-calling.
- A new queue analytics capability adds a person-counting service and an
  RTSP streamer, using YOLO detection with OpenVINO to track queue length
  from a video feed and expose a live MJPEG overlay stream.
- The ordering flow can now adapt to real-time queue conditions, surfacing a
  dynamic peak-hour menu driven by the queue-service integration.
- Speaker diarization has been enabled across the audio-analyzer and
  kiosk-core pipeline, improving turn attribution during multi-speaker
  interactions.
- An optional multimodal identity service adds Face ID and voiceprint
  authentication, combining OpenVINO face and ECAPA voice inference with a
  FAISS index and SQLite loyalty profiles, enabled through a dedicated
  deployment profile.
- A Makefile-based workflow simplifies setup and operations with targets for
  environment initialization, model download, sample-video retrieval, image
  build, service startup, health checks, and cleanup.
- Sample-video tooling downloads and provisions the RTSP feed clips used by
  the queue analytics pipeline, configurable through the environment file.



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

