# Run With Docker Compose (Gradio UI Mode)

`docker compose up` starts **audio-analyzer**, **text-to-speech**, **rag-service**, **kiosk-core** (REST API), and **kiosk-ui** (Gradio browser interface) as containers.

Mic audio is captured by the **browser** via the Web Audio API and uploaded to kiosk-core as a WAV file. No host mic hardware is passed into the containers.

If you want to run `kiosk-core` and the Gradio UI directly on the host while keeping the same browser experience, see [run-standalone.md](run-standalone.md).

## Clone And Prepare

Clone the repo and materialize only the two upstream microservices this stack needs:

```bash
git clone https://github.com/intel-retail/voice-enabled-interactions.git
cd voice-enabled-interactions
git submodule update --init --depth 1 edge-ai-libraries
git -C edge-ai-libraries sparse-checkout set --cone microservices/audio-analyzer microservices/text-to-speech
cd smart-kiosk-assistant
```

If the repo is already cloned, run the two `git` commands above from the repo root.

## Start

From the `smart-kiosk-assistant/` directory:

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
docker compose build
docker compose up -d
```

Images are tagged with `RELEASE_TAG` read from `.env` (defaults to `latest` if unset): `audio-analyzer:${RELEASE_TAG}`, `text-to-speech:${RELEASE_TAG}`, `rag-service:${RELEASE_TAG}`, `kiosk-core:${RELEASE_TAG}`, `kiosk-ui:${RELEASE_TAG}`. The committed `.env` pins the current release tag; override it for local builds by exporting `RELEASE_TAG` or editing `.env`.

This starts five containers:

| Container | Port | Purpose |
|---|---|---|
| `audio-analyzer` | `8010` | Speech-to-text |
| `text-to-speech` | `8011` | Speech synthesis |
| `rag-service` | `8020` | Knowledge-base retrieval |
| `kiosk-core` | `8012` | FastAPI session API |
| `kiosk-ui` | `7860` | Gradio voice UI |

All containers run as non-root. Exporting `LOCAL_UID` and `LOCAL_GID`
before `docker compose up` keeps bind-mounted files writable from the host
account that launched the stack.

## Verify

```bash
docker compose ps
curl --noproxy '*' http://127.0.0.1:8012/health   # {"status":"ok"}
```

Open the Gradio UI in a browser:

```
http://127.0.0.1:7860
```

Click the microphone button, speak your question, and the assistant responds with text and audio.

## Follow Logs

```bash
docker compose logs -f kiosk-core
docker compose logs -f kiosk-ui
```

## Restart / Stop

```bash
# After env var change
docker compose restart

# After code or dependency change
docker compose build
docker compose up -d

# Full teardown
docker compose down
```

## Troubleshooting

- The default Compose wiring already connects `kiosk-core` and `kiosk-ui` to the internal `audio-analyzer`, `rag-service`, and `text-to-speech` containers. Most deployments should not override these URLs.
- Only change downstream service URLs when this stack must call services running outside the local Compose network, such as a remote host or a separately managed service tier.
- See [configuration.md](configuration.md) for the advanced environment variables if you need that non-default routing.
- For endpoint details, see [api-reference.md](api-reference.md).
