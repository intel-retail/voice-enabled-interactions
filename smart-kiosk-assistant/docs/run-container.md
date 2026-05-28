# Run With Docker Compose

`docker compose up` starts `audio-analyzer`, `text-to-speech`,
`rag-service`, `kiosk-core` (REST API), and `kiosk-ui` (Gradio interface)
as containers.

Microphone audio is captured by the browser and uploaded to `kiosk-core`
as a WAV file. No host audio device is passed into the containers.

To run `kiosk-core` and the UI directly on the host, see
[run-standalone.md](run-standalone.md).

## Prerequisites

Clone the repository and initialize the submodule as described in
[get-started.md](get-started.md#step-1-clone-and-prepare-sources).

## Start

From `smart-kiosk-assistant/`:

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
docker compose build
docker compose up -d
```

Images are tagged with `RELEASE_TAG` from `.env` (defaults to `latest`).
Override by exporting `RELEASE_TAG` or editing `.env`.

This starts five containers:

| Container | Port | Purpose |
|---|---|---|
| `audio-analyzer` | 8010 | Speech-to-text |
| `text-to-speech` | 8011 | Speech synthesis |
| `rag-service` | 8020 | Knowledge-base retrieval |
| `kiosk-core` | 8012 | FastAPI session API |
| `kiosk-ui` | 7860 | Gradio voice UI |

Containers run as non-root; `LOCAL_UID` / `LOCAL_GID` keep bind-mounted
files writable from the host account.

## Verify

```bash
docker compose ps
curl --noproxy '*' http://127.0.0.1:8012/health   # {"status":"ok"}
```

Open `http://127.0.0.1:7860` in a browser, click the microphone, and
speak your question.

## Logs

```bash
docker compose logs -f kiosk-core
docker compose logs -f kiosk-ui
```

## Restart / Stop

```bash
docker compose restart            # after env var change
docker compose build && docker compose up -d   # after code change
docker compose down               # teardown
```

## Notes

- The default Compose wiring connects `kiosk-core` and `kiosk-ui` to the
  internal `audio-analyzer`, `rag-service`, and `text-to-speech`
  containers. Override these URLs only when this stack must call
  services outside the local Compose network.
- See [configuration.md](configuration.md) for environment variables and
  [api-reference.md](api-reference.md) for endpoint details.
