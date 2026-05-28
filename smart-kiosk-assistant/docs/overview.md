# Overview

Smart Kiosk Assistant is a voice-first, retrieval-augmented kiosk stack
for retail, QSR, and similar customer-facing deployments. The browser
captures microphone audio, the stack transcribes it, retrieves a grounded
answer from a local knowledge base, and plays a synthesized reply. All
inference runs locally on Intel CPU, GPU, or NPU via OpenVINO.

## Services

| Service | Port | Role |
|---|---|---|
| `audio-analyzer` | 8010 | Speech-to-text (Whisper) |
| `text-to-speech` | 8011 | Speech synthesis (SpeechT5 / Qwen-TTS) |
| `rag-service` | 8020 | Knowledge-base retrieval and answer generation |
| `kiosk-core` | 8012 | FastAPI session orchestrator |
| `kiosk-ui` | 7860 | Gradio browser interface |

`audio-analyzer`, `text-to-speech`, and `rag-service` host the inference
models. `kiosk-core` and `kiosk-ui` are I/O-only.

Start with [get-started.md](get-started.md).
