# Run On The Host

Use this path to run `kiosk-core` and the Gradio UI directly on the host
instead of inside the top-level Compose stack. The microphone is still
captured by the browser and uploaded to `kiosk-core`.

## Prerequisites

Clone the repository and initialize the submodule as described in
[get-started.md](get-started.md#step-1-clone-and-prepare-sources).

## Start Downstream Services

Before starting `kiosk-core` and the UI on the host, make sure these downstream services are available:

- `audio-analyzer` at `http://127.0.0.1:8010/v1/audio/transcriptions`
- `text-to-speech` at `http://127.0.0.1:8011/v1/audio/speech`
- `rag-service` at `http://127.0.0.1:8020/api/v1/query`

One practical setup is to run the two upstream microservices in their own Compose projects and run `rag-service` on the host:

```bash
cd ../edge-ai-libraries/microservices/audio-analyzer && docker compose up -d && cd -
cd ../edge-ai-libraries/microservices/text-to-speech && docker compose up -d && cd -
cd rag-service && python main.py && cd -
```

If you prefer the full all-in-one container deployment, use [run-container.md](run-container.md) instead.

## Python Setup

From the `smart-kiosk-assistant/` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Start kiosk-core

Run the API on the host:

```bash
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8012
```

Default URLs used by `kiosk-core` in this host-run mode:

- `KIOSK_CORE_ANALYZER_URL=http://127.0.0.1:8010/v1/audio/transcriptions`
- `KIOSK_CORE_RAG_URL=http://127.0.0.1:8020/api/v1/query`
- `KIOSK_CORE_TTS_URL=http://127.0.0.1:8011/v1/audio/speech`

## Start The Gradio UI

In a second terminal, from the same `smart-kiosk-assistant/` directory:

```bash
source .venv/bin/activate
python gradio_app.py
```

Default UI URL:

```text
http://127.0.0.1:7860
```

Open that address in a browser, allow microphone access, and use the same browser-based voice flow as the Compose deployment.

## Verify

Check the API and UI separately:

```bash
curl --noproxy '*' http://127.0.0.1:8012/health
```

Then open:

```text
http://127.0.0.1:7860
```

## Advanced Routing

Most host-run setups should keep the default localhost URLs. Override them only if `kiosk-core` or the Gradio UI needs to call services on another host or in another deployment.

See [configuration.md](configuration.md) for the environment variables.

## Notes

- TTS audio clips are written under `generated_audio/` in the project directory.
- Browser capture means no host microphone device needs to be managed by Python directly.
- For endpoint details, see [api-reference.md](api-reference.md).