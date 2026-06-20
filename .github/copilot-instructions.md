# GitHub Copilot Instructions — Smart Kiosk Assistant

This file provides persistent project context and skill definitions for AI assistants (GitHub Copilot, Claude, etc.) working in this repository.

---

## Project Overview

**Smart Kiosk Assistant** is a voice-enabled, multimodal AI assistant designed for kiosk deployments. It combines Automatic Speech Recognition (ASR), Speaker Diarization, Speaker Verification, Face Detection/Recognition, and a RAG-based knowledge pipeline into a unified real-time service.

---

## Tech Stack & Skills

### Languages & Runtimes
- **Python 3.11+** — primary language; use modern syntax (match/case, walrus operator, `|` union types)
- **Async Python** — prefer `asyncio`, `async/await`, and `asyncio.gather` for concurrent I/O; avoid blocking calls in async contexts

### Frameworks & APIs
- **FastAPI** — all HTTP endpoints; use `APIRouter`, dependency injection via `Depends`, `BackgroundTasks`, and Pydantic v2 models
- **REST APIs** — RESTful design; use proper HTTP verbs, status codes, and versioned routes (`/api/v1/...`)

### AI / ML Inference
- **OpenVINO** — preferred inference backend for Intel hardware; use `openvino.runtime` Core/CompiledModel API
- **ONNX Runtime** — fallback inference for cross-platform portability; use `onnxruntime.InferenceSession`
- **PyTorch** — model loading, preprocessing, and export; avoid training-time dependencies in production paths

### Computer Vision
- **OpenCV** — image capture, preprocessing (resize, normalize, BGR↔RGB conversion), drawing utilities
- **Face Detection** — detect faces in frames using bounding boxes; handle multi-face scenes
- **Face Recognition** — extract 512-d face embeddings; compare with stored gallery using cosine similarity
- **Face Embeddings** — L2-normalized float32 vectors; store as numpy arrays or in FAISS index

### Audio & Speech
- **FFmpeg** — audio transcoding, resampling to 16kHz mono PCM for ASR pipelines
- **Automatic Speech Recognition (ASR)** — real-time or batch transcription; support Whisper-style models
- **Speaker Diarization** — segment audio by speaker using ONNX/OpenVINO speaker segmentation models
- **Speaker Verification** — extract speaker embeddings (d-vectors/x-vectors); verify identity via cosine similarity threshold
- **Voice Embeddings** — 256/512-d float32 vectors; normalize before storage and comparison
- **Audio Signal Processing** — work with raw PCM, apply VAD, normalize loudness, handle noise robustly
- **Biometrics** — treat face and voice embeddings as sensitive data; never log raw embeddings

### Data & Storage
- **SQLite** — lightweight persistent storage for user profiles, session metadata, and enrollment records; use `aiosqlite` for async access
- **FAISS** — vector similarity search for face and speaker embedding galleries; prefer `IndexFlatIP` with pre-normalized vectors
- **NumPy** — array operations, embedding math (dot products, norms), image batching
- **Vector Similarity Search** — cosine similarity = dot product of L2-normalized vectors; use thresholds (e.g., 0.75 for faces, 0.80 for voice)
- **YAML Configuration** — all runtime config in YAML (`configs/`); load with `PyYAML`; never hardcode paths or thresholds

### Infrastructure
- **Docker** — containerize the service; base image `python:3.11-slim` or OpenVINO runtime image
- **Docker Compose** — orchestrate multi-service deployments (ASR service, RAG service, kiosk core, metrics collector)

### Software Design
- **Low Level Design (LLD)** — design classes and modules before coding; include attribute types and method signatures
- **SOLID Principles** — Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
- **Clean Architecture** — separate `domain`, `application`, `infrastructure`, and `interface` layers; inner layers must not depend on outer layers
- **Repository Pattern** — abstract all DB/FAISS access behind repository interfaces; swap implementations without touching business logic
- **Factory Pattern** — use factories for model loading so the caller never imports concrete backends directly
- **Strategy Pattern** — swap inference backends (OpenVINO ↔ ONNX ↔ PyTorch) via a strategy interface
- **Service Layer** — orchestration logic lives in `*Service` classes; controllers/routers only call services
- **Dependency Injection** — inject repositories and services via FastAPI `Depends` or constructor injection; never instantiate dependencies inside business logic
- **Configuration Management** — load config once at startup; pass a typed `Settings` / `Config` dataclass through DI

### Quality & Operations
- **Error Handling** — raise domain-specific exceptions; never swallow exceptions silently; map to HTTP status codes at the router boundary
- **Logging** — use Python `logging` with structured fields; log at `DEBUG` for inference timings, `INFO` for lifecycle events, `WARNING`/`ERROR` for failures
- **Thread-safe programming** — use `asyncio.Lock` or `threading.Lock` when sharing mutable state (e.g., FAISS index, in-memory caches) across concurrent requests
- **Unit Testing** — `pytest` + `pytest-asyncio`; mock external dependencies (models, DB) with `unittest.mock`
- **Integration Testing** — test full request/response cycles against a real (in-memory SQLite) database; use `httpx.AsyncClient` with FastAPI `TestClient`
- **API Design** — document all endpoints with OpenAPI descriptions; validate input with Pydantic; return consistent error envelopes `{"error": ..., "detail": ...}`

---

## Coding Conventions

1. **File layout**: `kiosk_core/` holds domain/application layers; `rag-service/`, `metrics-collector/` are standalone services.
2. **Imports**: absolute imports only; group as stdlib → third-party → local.
3. **Type hints**: required on all public functions and class attributes.
4. **Docstrings**: Google-style for public classes and methods.
5. **Config keys**: always reference via typed config objects, not raw `dict["key"]` access.
6. **Embedding storage**: always store as `float32`; normalize before writing to FAISS.
7. **Model paths**: resolve relative to project root using `pathlib.Path`.
8. **No secrets in code**: use environment variables or a secrets manager.

---

## Directory Structure (Reference)

```
voice-enabled-interactions/
├── smart-kiosk-assistant/
│   ├── configs/          # YAML config files
│   ├── kiosk_core/       # Core domain & application logic
│   ├── rag-service/      # RAG knowledge pipeline
│   ├── metrics-collector/# Metrics aggregation service
│   ├── docs/             # Architecture & design docs
│   ├── main.py           # FastAPI app entrypoint
│   ├── Dockerfile
│   └── docker-compose.yml
└── .github/
    └── copilot-instructions.md  # ← this file
```
