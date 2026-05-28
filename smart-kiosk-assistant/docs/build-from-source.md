# Build From Source

Build Smart Kiosk Assistant from source for either Docker or host
execution.

## Prerequisites

- Verify the [system requirements](system-requirements.md).
- Clone the repository and initialize the submodule as described in
  [get-started.md](get-started.md#step-1-clone-and-prepare-sources).

## Build All Images With Compose

The top-level [docker-compose.yml](../docker-compose.yml) builds five
images: `audio-analyzer`, `text-to-speech`, `rag-service`, `kiosk-core`,
and `kiosk-ui`. The image tag is read from the `RELEASE_TAG` variable in
[.env](../.env) and defaults to `latest` if unset.

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
docker compose build
docker compose up -d
```

Exporting `LOCAL_UID` and `LOCAL_GID` keeps bind-mounted files writable
from the host user that launched the stack.

## Build a Single Service Image

Each service can be built directly with `docker build`. From the
repository root:

```bash
# audio-analyzer
docker build -t audio-analyzer:local \
  ../edge-ai-libraries/microservices/audio-analyzer

# text-to-speech
docker build -t text-to-speech:local \
  ../edge-ai-libraries/microservices/text-to-speech

# rag-service
docker build -t rag-service:local ./rag-service

# kiosk-core / kiosk-ui (same Dockerfile, different entrypoints)
docker build -t kiosk-core:local .
docker build -t kiosk-ui:local   .
```

The `kiosk-ui` container reuses the `kiosk-core` image and runs
`python3 gradio_app.py` as its command.

## Build a Python Environment (Standalone kiosk-core + UI)

`kiosk-core` and `kiosk-ui` can run directly on the host while the three
model-hosting services run in containers. Install host packages, then
create a virtual environment and install dependencies:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg alsa-utils libsndfile1

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

See [run-standalone.md](run-standalone.md) for the launch commands.

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
