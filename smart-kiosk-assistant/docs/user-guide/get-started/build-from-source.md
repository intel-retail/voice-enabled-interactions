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

This release depends on **specific commits** of two other repositories.

| Repository | Pinned commit | Why |
| --- | --- | --- |
| [`intel-retail/performance-tools`](https://github.com/intel-retail/performance-tools) | `c561c9e` | Provides `benchmark_smart_kiosk_v2v.py`, the voice-to-voice benchmark orchestrator, and the natural-endpointing measurement mode. Tracked as a Git **submodule**, so the commit is recorded in this repository — you do not pick it manually. |
| [`open-edge-platform/edge-ai-libraries`](https://github.com/open-edge-platform/edge-ai-libraries) | `fc89569` | Source for the `audio-analyzer` and `text-to-speech` images. You do **not** build these two services here — they are always pulled pre-built from the registry (Docker Hub `intel/audio-analyzer` and `intel/text-to-speech` by default). The settings this release relies on (`TEXT_TO_SPEECH_WORKERS`, the ASR preview pool, and forwarding an empty ASR language to English-only checkpoints) are not yet on upstream `main`, so the published images are built from the pinned commit above. |

`audio-analyzer` and `text-to-speech` have **no `build:` section** in
`docker-compose.yml` — neither `docker compose build` nor
`make build REGISTRY=false` ever rebuilds them from source. To change
either service, build and push its image from an `edge-ai-libraries`
checkout at `fc89569` (see
[Rebuilding the External TTS/ASR Images](#rebuilding-the-external-ttsasr-images)),
then pull it here.

> **Note for reviewers / early adopters:** once the upstream
> `edge-ai-libraries` change lands and official images are published,
> repoint `REGISTRY` to the upstream namespace and delete this note.

## Clone and Prepare

Only this repository needs to be on disk to build the kiosk services.
`audio-analyzer` and `text-to-speech` are pulled as pre-built images (see
[Pinned Upstream Sources](#pinned-upstream-sources)), so no
`edge-ai-libraries` checkout is required for a normal build-from-source
run.

From whatever parent directory you keep the source in, run:

```bash
# --recurse-submodules checks out performance-tools at the commit this
# repository pins. Without it the submodule directory is empty and
# `make benchmark` cannot find the benchmark orchestrator.
git clone --recurse-submodules -b main --single-branch \
  https://github.com/intel-retail/voice-enabled-interactions.git
cd voice-enabled-interactions/smart-kiosk-assistant/

# Already cloned without --recurse-submodules? Run this instead:
#   make update-submodules
```

Confirm the submodule pin before building:

```bash
git -C ../performance-tools rev-parse --short HEAD   # expect c561c9e
```

You only need an `edge-ai-libraries` checkout if you intend to rebuild
the external ASR/TTS images yourself — see
[Rebuilding the External TTS/ASR Images](#rebuilding-the-external-ttsasr-images).

## Create the Environment File

`.env` is not committed to the repo. `docker-compose.yml` reads
`REGISTRY_REPO`/`RELEASE_TAG` from it and falls back to `latest` if
it's missing, so create it before building:

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
declares both `image:` and `build:` for the three services whose source lives in
this repository: `rag-service`, `kiosk-core`, and `kiosk-ui`. `audio-analyzer`
and `text-to-speech` are declared `image:`-only — their source lives in the
sibling `edge-ai-libraries` repo, so `docker compose build` in this repo does
**not** build them (see
[Build the speech-service images](#build-a-single-service-image-directly)
below). Both `REGISTRY_REPO` and `RELEASE_TAG` are read from `.env` (created in
[Create the Environment File](#create-the-environment-file) above;
defaults: `REGISTRY_REPO=intel`, `RELEASE_TAG` pins the current release).

`docker compose build` rebuilds the three in-repo services from source and tags
each result as the same `${REGISTRY_REPO}/<svc>:${RELEASE_TAG}` reference used by
the pull flow, so subsequent `docker compose up` calls reuse the local
build until you `docker compose pull` again. Build the two speech-service images
separately (next-but-one section) before the first `up`.

```bash
docker compose build          # rag-service, kiosk-core, kiosk-ui
docker compose up -d
```

All services run as UID/GID `1000:1000` (baked into each image),
and runtime data lives in named Docker volumes initialized with that
ownership, so no host UID/GID configuration is needed.

## Rebuild A Single Service

Rebuild only the in-repo service whose source you changed (`rag-service`,
`kiosk-core`, or `kiosk-ui`):

```bash
docker compose build kiosk-core
docker compose up -d kiosk-core
```

`docker compose build audio-analyzer` / `text-to-speech` prints
`No services to build` and exits 0 — those images are built directly from the
`edge-ai-libraries` sources (next section), not from this compose file.

## Build A Single Service Image Directly

Each in-repo service can be built directly with `docker build`. From the
`smart-kiosk-assistant` directory:

```bash
# rag-service
docker build -t intel/rag-service:local ./rag-service

# kiosk-core
docker build -t intel/kiosk-core:local .

# kiosk-ui (React SPA served by nginx)
docker build -t intel/kiosk-ui:local ./kiosk-ui
```

The `kiosk-ui` image serves both the operator screen and the customer
screen; the mode is selected at container start via `KIOSK_UI_MODE`.

## Rebuilding the External TTS/ASR Images

`audio-analyzer` and `text-to-speech` are **not** part of this
repository and are not built by the kiosk flows above. You only need
this if you are changing the ASR or TTS service itself. Check out
`edge-ai-libraries` at the pinned commit, build the image, tag it to
match the `REGISTRY`/`RELEASE_TAG` this stack pulls, and push (or just
build locally and leave it cached):

```bash
git clone --filter=blob:none --sparse \
  https://github.com/sainijit/edge-ai-libraries.git
git -C edge-ai-libraries sparse-checkout set \
  microservices/audio-analyzer microservices/text-to-speech
git -C edge-ai-libraries checkout fc89569

# audio-analyzer
docker build -t intel/audio-analyzer:latest \
  edge-ai-libraries/microservices/audio-analyzer

# text-to-speech (includes Kokoro TTS, absent from upstream main)
docker build -t intel/text-to-speech:latest \
  edge-ai-libraries/microservices/text-to-speech
```

A shallow `--depth 1` clone cannot check out an arbitrary commit, so it
is omitted above. After building, `docker compose up -d` picks up the
freshly tagged local images without a pull.

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
