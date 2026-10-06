# Build From Source

Build Smart Kiosk Assistant from source. Use this path when you need a
code change in any of the kiosk services. To run the prebuilt images
from Docker Hub without rebuilding, see
[Run With Docker Compose](./run-container.md).

## Prerequisites

Verify the [System Requirements](./system-requirements.md).

Beyond Docker, the build and benchmark flows need these host packages:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs python3-venv ffmpeg
```

- `python3-venv` — `make benchmark` creates a virtualenv for the
  performance-tools orchestrator. On Debian/Ubuntu the `venv` module is
  packaged separately from `python3`, and without it the benchmark fails
  with `ensurepip is not available`.
- `ffmpeg` — used to transcode and resample benchmark/replay audio to the
  16 kHz mono PCM the analyzer expects.

## Pinned Upstream Sources

This release builds against **specific commits** of two other
repositories. Both are required; neither is on its project's `main`
branch yet.

| Repository | Pinned commit | Why |
| --- | --- | --- |
| [`intel-retail/performance-tools`](https://github.com/intel-retail/performance-tools) | `c561c9e` | Provides `benchmark_smart_kiosk_v2v.py`, the voice-to-voice benchmark orchestrator, and the natural-endpointing measurement mode. Tracked as a Git **submodule**, so the commit is recorded in this repository — you do not pick it manually. |
| [`open-edge-platform/edge-ai-libraries`](https://github.com/open-edge-platform/edge-ai-libraries) | see below | Supplies `audio-analyzer` and `text-to-speech`. The settings this release relies on (`TEXT_TO_SPEECH_WORKERS`, the ASR preview pool, and forwarding an empty ASR language to English-only checkpoints) are not yet on upstream `main`. |

Building `edge-ai-libraries` from upstream `main` **will fail**:
`text-to-speech` stops with `models.tts.runtime must be 'openvino' or
'pytorch'`, and `audio-analyzer` fails to export
`distil-whisper/distil-small.en`. Use the pinned source below until the
upstream change lands.

> **Note for reviewers / early adopters:** once the upstream pull request
> is merged, replace the clone below with upstream `main` and delete this
> section.

## Clone and Prepare

The kiosk compose builds `audio-analyzer` and `text-to-speech` from the
upstream [edge-ai-libraries](https://github.com/open-edge-platform/edge-ai-libraries)
monorepo. The compose file references those sources at
`../edge-ai-libraries/microservices/{audio-analyzer,text-to-speech}`,
so the two repositories must sit side by side:

```text
<parent>/
├── voice-enabled-interactions/
│   └── smart-kiosk-assistant/   # run docker compose from here
└── edge-ai-libraries/
    └── microservices/
        ├── audio-analyzer/
        └── text-to-speech/
```

From whatever parent directory you keep the source in, run:

```bash
# --recurse-submodules checks out performance-tools at the commit this
# repository pins. Without it the submodule directory is empty and
# `make benchmark` cannot find the benchmark orchestrator.
git clone --recurse-submodules -b main --single-branch \
  https://github.com/intel-retail/voice-enabled-interactions.git
cd voice-enabled-interactions/

# Already cloned without --recurse-submodules? Run this instead:
#   make -C smart-kiosk-assistant update-submodules

git clone --filter=blob:none --sparse \
  https://github.com/sainijit/edge-ai-libraries.git
git -C edge-ai-libraries sparse-checkout set \
  microservices/audio-analyzer microservices/text-to-speech
git -C edge-ai-libraries checkout fc89569

cd smart-kiosk-assistant/
```

`fc89569` is the pinned `edge-ai-libraries` commit described in
[Pinned Upstream Sources](#pinned-upstream-sources). Note the clone
deliberately omits `--depth 1`: a shallow clone cannot check out an
arbitrary commit.

Confirm both pins before building:

```bash
git -C ../performance-tools rev-parse --short HEAD   # expect c561c9e
git -C ../edge-ai-libraries rev-parse --short HEAD   # expect fc89569
```

The sparse checkout pulls only the two microservices the kiosk build
needs; everything else in `edge-ai-libraries` stays unchecked out. A
plain `git clone` of `edge-ai-libraries` also works if you do not mind
the extra files. Only the build flow needs `edge-ai-libraries` on disk
— the pull flow (see [Run With Docker Compose](./run-container.md)) does not.

## Create the Environment File

`.env` is not committed to the repo. `docker-compose.yml` reads
`REGISTRY`/`RELEASE_TAG` from it and falls back to `latest` if it's
missing, so create it before building:

```bash
make init-env        # copies .env.example → .env
```

`.env.example` ships `REGISTRY=true`, which makes `make build` **pull the
released images instead of building your source**. That is the right
default for the pull flow, but it is not what you want here. For a
build-from-source run, either pass the override each time:

```bash
make build REGISTRY=false
```

or edit `.env` once and set `REGISTRY=false`. `docker compose build`
(used later on this page) is unaffected — it always builds.

## Download the LLM Model for OVMS

Before building or starting the stack, download the Qwen3-4B model that
OVMS serves. This runs once and caches into `./models/`:

```bash
# GPU (recommended)
./setup_models.sh

# CPU only
./setup_models.sh --device CPU

# Smaller INT4 model
./setup_models.sh --int4
```

`setup_models.sh` downloads the pre-converted OpenVINO™ model from
HuggingFace Hub and updates `OVMS_MODEL_NAME`, `TARGET_DEVICE`, and
`RENDER_GID` in `.env`. See `./setup_models.sh --help` for all options.

## Build All Images With Compose

The top-level [docker-compose.yml](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/docker-compose.yml)
declares both `image:` and `build:` for each of the five services: `audio-analyzer`,
`text-to-speech`, `rag-service`, `kiosk-core`, and `kiosk-ui`. Both
`REGISTRY` and `RELEASE_TAG` are read from `.env` (created in
[Create the Environment File](#create-the-environment-file) above;
defaults: `REGISTRY=intel`, `RELEASE_TAG` pins the current release).

`docker compose build` rebuilds each service from source and tags the
result as the same `${REGISTRY}/<svc>:${RELEASE_TAG}` reference used by
the pull flow, so subsequent `docker compose up` calls reuse the local
build until you `docker compose pull` again.

```bash
docker compose build
docker compose up -d
```

All five services run as UID/GID `1000:1000` (baked into each image),
and runtime data lives in named Docker volumes initialized with that
ownership, so no host UID/GID configuration is needed.

## Rebuild A Single Service

Rebuild only the service whose source you changed:

```bash
docker compose build audio-analyzer
docker compose up -d audio-analyzer
```

## Build A Single Service Image Directly

Each service can be built directly with `docker build`. From the
repository root:

```bash
# audio-analyzer
docker build -t intel/audio-analyzer:local \
  ../edge-ai-libraries/microservices/audio-analyzer

# text-to-speech
docker build -t intel/text-to-speech:local \
  ../edge-ai-libraries/microservices/text-to-speech

# rag-service
docker build -t intel/rag-service:local ./rag-service

# kiosk-core
docker build -t intel/kiosk-core:local .

# kiosk-ui (React SPA served by nginx)
docker build -t intel/kiosk-ui:local ./kiosk-ui
```

The `kiosk-ui` image serves both the operator screen and the customer
screen; the mode is selected at container start via `KIOSK_UI_MODE`.

## Verifying the Build

Once the stack is started, confirm every service is healthy:

```bash
curl --noproxy '*' http://127.0.0.1:8010/health   # audio-analyzer
curl --noproxy '*' http://127.0.0.1:8011/health   # text-to-speech
curl --noproxy '*' http://127.0.0.1:8020/health   # rag-service
curl --noproxy '*' http://127.0.0.1:8012/health   # kiosk-core
```

A `{"status": "ok"}` response from each endpoint confirms the build is
functional. Open `http://127.0.0.1:7860` to use the browser UI.
