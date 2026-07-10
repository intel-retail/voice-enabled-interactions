# Get Started

Clone the repository, build the stack, and confirm a working voice
ordering session end to end.

Confirm your machine meets the
[System Requirements](./get-started/system-requirements.md) before starting.

## Prerequisites

- Ubuntu 22.04 or newer (Linux, Desktop edition — a browser with microphone
  access is required for voice ordering)
- [Docker](https://docs.docker.com/engine/install/) 24.0+
- [Docker Compose](https://docs.docker.com/compose/install/) V2+
- [Make](https://www.gnu.org/software/make/) (`sudo apt install make`)
- Intel hardware (CPU, iGPU, or NPU)
- Intel GPU drivers (recommended) — see the
  [Intel GPU driver guide](https://dgpu-docs.intel.com)
- A [HuggingFace account and access token](https://huggingface.co/settings/tokens)
- At least 30 GB free disk space for models and Docker images

> **Note:** First-time setup downloads the OVMS LLM model (~4 GB) plus the
> in-container ASR, TTS, embedding, and reranker models. Allow extra time on
> the first run depending on your connection.

## Application Overview

The Smart Kiosk Assistant is a voice-driven quick-service-restaurant (QSR)
ordering kiosk. The browser captures microphone audio and uploads it to
`audio-analyzer` (Whisper ASR + speaker diarization); `rag-service` runs the
RAG pipeline and ordering agent on a Qwen3-4B model served by `ovms-llm`;
`text-to-speech` synthesizes the spoken reply. `kiosk-core` orchestrates the
session and product ordering, `kiosk-ui` is the React front end, and
`rtsp-streamer` + `queue-service` provide person-counting queue analytics.
`metrics-collector` reports hardware utilization, and the optional
`identity-service` adds face + voice authentication.

## Step 1: Install Docker and Intel GPU Drivers

Skip if Docker is already installed (`docker --version` and
`docker compose version` both succeed).

```bash
# Docker Engine + Compose v2
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker
```

For Intel iGPU acceleration (recommended) install the GPU compute runtime:

```bash
sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero
sudo usermod -aG render $USER && newgrp render
```

Verify the GPU device node exists: `ls /dev/dri/renderD*`.
If the packages are not found, follow the [Intel GPU driver guide](https://dgpu-docs.intel.com).
Skip this block and set `TARGET_DEVICE=CPU` in `.env` if no Intel GPU is available.

## Step 2: Clone the Repository

```bash
git clone https://github.com/intel-retail/voice-enabled-interactions.git
cd voice-enabled-interactions/smart-kiosk-assistant
```

## Step 3: Clone `edge-ai-libraries` (Audio and TTS Source)

The `audio-analyzer` and `text-to-speech` images are built from the
[edge-ai-libraries](https://github.com/open-edge-platform/edge-ai-libraries)
monorepo. The compose file expects them at `../edge-ai-libraries/microservices/`,
so both repositories must sit side by side:

```bash
cd ..   # move to the parent directory that contains voice-enabled-interactions
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/open-edge-platform/edge-ai-libraries.git
git -C edge-ai-libraries sparse-checkout set \
    microservices/audio-analyzer microservices/text-to-speech
cd voice-enabled-interactions/smart-kiosk-assistant
```

## Step 4:Create `.env` and Set Required Variables

Create your environment file from the template:

```bash
make init-env        # copies .env.example → .env
```

Open `.env` and set your HuggingFace token (free account at
https://huggingface.co/settings/tokens). You must also accept the Pyannote
licence at https://huggingface.co/pyannote/speaker-diarization-3.1 (required
for speaker diarization):

```bash
# .env
HF_TOKEN=hf_your_token_here
```

For Intel GPU inference (the default `TARGET_DEVICE=GPU`), set `RENDER_GID`
so containers can access `/dev/dri`:

```bash
# find the render group id on your host
getent group render | cut -d: -f3
# then set it in .env, e.g. RENDER_GID=992
```

Set `TARGET_DEVICE=CPU` in `.env` if no Intel GPU is available.

## Step 5: Download the LLM Model for OVMS

The ordering agent uses a Qwen3-4B model served by OVMS. Download it
before starting the stack:

```bash
# GPU (recommended, INT8 ~4 GB)
./setup_models.sh

# CPU only (no Intel GPU)
./setup_models.sh --device CPU

# INT4 model (smaller/faster, ~2 GB, slightly lower accuracy)
./setup_models.sh --int4
```

The script downloads the pre-converted OpenVINO model into `./models/`
and updates `OVMS_MODEL_NAME`, `TARGET_DEVICE`, and `RENDER_GID` in `.env`.
The download happens once — subsequent starts reuse the cached model.

## Step 6: Download Sample Videos for Queue Analytics

The `queue-service` person-counting pipeline consumes an RTSP stream published
by `rtsp-streamer` from clips under `Sample_data/`. Set the download URLs in
`.env` (`SAMPLE_VIDEO_1_URL`, `SAMPLE_VIDEO_2_URL`), then fetch them:

```bash
make download-sample-video
```

The clips are saved as `Sample_data/sample_1.mp4` and `Sample_data/sample_2.mp4`
to match the `MEDIA_FILES` entry in `docker-compose.yml`. You can also copy your
own MP4 files to those paths manually.

## Step 7: Build Images and Start the Stack

```bash
make build           # build local images (or `make build REGISTRY=true` to pull)
make up              # start the stack (runs check-env first)
```

The equivalent raw Compose commands are:

```bash
docker compose build
docker compose up -d
```

`make build` compiles the local service images from source; `ovms-llm` uses a
pre-built upstream image and is pulled automatically. First build takes
10–20 minutes.

The default stack starts nine containers:

| Container | Port | Role |
|---|---|---|
| `ovms-llm` | 8000 (REST), 9001 (gRPC) | Serves Qwen3-4B via OpenAI-compatible API |
| `metrics-collector` | 9000 | Hardware utilization metrics |
| `audio-analyzer` | 8010 | Whisper ASR + speaker diarization |
| `text-to-speech` | 8011 | SpeechT5 TTS synthesis |
| `rag-service` | 8020 | RAG pipeline + ordering agent |
| `kiosk-core` | 8012 | Session API + product ordering |
| `kiosk-ui` | 7860 | Voice kiosk React UI |
| `rtsp-streamer` | 8554 | Publishes sample clips as an RTSP stream |
| `queue-service` | 8090 | Person detection + queue counting |

The optional `identity-service` (port `8013`, face + voice authentication)
starts only when identity is enabled:

```bash
make up IDENTITY=true
```

## Step 8: Verify the Stack Is Healthy

Services start in dependency order. Allow 2–5 minutes on first run for
model assets to download into Docker volumes.

```bash
make status          # container status (or: docker compose ps)
make test            # health-check ovms, audio, tts, rag, kiosk-core, kiosk-ui
make logs            # follow logs for all services (or make logs-core, make logs-rag, ...)
```

You can also probe individual endpoints directly:

```bash
curl --noproxy '*' http://127.0.0.1:8000/v3/models   # ovms-llm
curl --noproxy '*' http://127.0.0.1:8010/health       # audio-analyzer
curl --noproxy '*' http://127.0.0.1:8011/health       # text-to-speech
curl --noproxy '*' http://127.0.0.1:8020/health       # rag-service
curl --noproxy '*' http://127.0.0.1:8012/health       # kiosk-core
curl --noproxy '*' http://127.0.0.1:8090/health       # queue-service
```

Every health endpoint should return `{"status": "ok"}`. The OVMS endpoint
returns the active model name (`OpenVINO/Qwen3-4B-int8-ov`).

## Step 9: Open the Kiosk and Try Voice Ordering

Open a browser on the same machine:

```
http://127.0.0.1:7860
```

1. Click **Allow** when the browser asks for microphone permission.
   (Use `127.0.0.1`, not the machine hostname — browsers block microphone
   on non-HTTPS origins except `localhost`.)
2. Click the **🎤 microphone** button and speak. Try:
   - *"What's on the menu?"*
   - *"Show me your burgers"*
   - *"I'd like to order a Classic Chicken Burger"*
   - *"Confirm my order"*
3. Watch the **🍔 QSR** tab in the right panel — the **Menu** sub-tab
   shows the full catalogue; the **Cart** sub-tab shows your live order
   and the confirmed receipt after you say *"Confirm"*.

## Stop the Services

```bash
make down            # stop all containers (or: docker compose down)
make clean           # stop and remove named volumes for a clean restart
```

## Quick Start Reference

| Task | Command | Description |
|---|---|---|
| Create env file | `make init-env` | Copy `.env.example` → `.env` |
| Download LLM model | `make setup-models` | Fetch the OVMS Qwen3-4B model |
| Download sample videos | `make download-sample-video` | Fetch queue-service RTSP clips |
| Build images | `make build` | Build locally (`REGISTRY=true` to pull) |
| Start services | `make up` | Start the full stack |
| Verify health | `make test` | Check all service endpoints |
| View logs | `make logs` | Tail logs for all services |
| Stop services | `make down` | Stop all containers |
| Clean restart | `make clean` | Stop and remove named volumes |

## Next Steps

- [How It Works](./how-it-works.md)
- [Configuration](./get-started/configuration.md)
- [Build From Source](./get-started/build-from-source.md)
- [API Reference](./api-reference.md)
- [Troubleshooting](./troubleshooting.md)

<!--hide_directive
:::{toctree}
:hidden:

./get-started/system-requirements.md
./get-started/build-from-source.md
./get-started/run-container.md
./get-started/run-standalone.md
./get-started/configuration.md

:::
hide_directive-->
