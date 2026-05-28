# Get Started

Clone the repository, start the full stack, and confirm a working voice
request end to end.

Confirm your machine meets the
[system requirements](system-requirements.md) before starting.

## Step 1: Clone and Prepare Sources

```bash
git clone https://github.com/intel-retail/voice-enabled-interactions.git
cd voice-enabled-interactions
git submodule update --init --depth 1 edge-ai-libraries
git -C edge-ai-libraries sparse-checkout set --cone \
  microservices/audio-analyzer microservices/text-to-speech
cd smart-kiosk-assistant
```

If already cloned, run the two `git` commands above from the repository
root.

## Step 2: Start the Stack

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
docker compose build
docker compose up -d
```

`LOCAL_UID` / `LOCAL_GID` keep bind-mounted files writable from the host
user. First startup downloads and exports model assets and can take a
few minutes.

## Step 3: Verify the Stack Is Healthy

```bash
docker compose ps
curl --noproxy '*' http://127.0.0.1:8010/health   # audio-analyzer
curl --noproxy '*' http://127.0.0.1:8011/health   # text-to-speech
curl --noproxy '*' http://127.0.0.1:8020/health   # rag-service
curl --noproxy '*' http://127.0.0.1:8012/health   # kiosk-core
```

Every response should be `{"status": "ok"}`.

## Step 4: Run the Voice Flow

Open the Gradio UI in a browser:

```text
http://127.0.0.1:7860
```

Then:

1. When the browser prompts for microphone permission, click **Allow**.
2. Use the **knowledge base** panel to ingest the bundled sample content,
   or upload your own `.txt` / `.md` files. Sample content is in
   [knowledge-base-samples/](../knowledge-base-samples/).
3. Click the microphone button and ask a question that the knowledge base
   can answer (for example, "What are the store hours?").
4. Wait for the assistant card to update with:
   - the transcript of what you said,
   - the streamed answer text,
   - a spoken response that plays back automatically.

## Next Steps

- [How It Works](how-it-works.md)
- [Configuration](configuration.md)
- [Run With Docker Compose](run-container.md) /
  [Run On The Host](run-standalone.md)
- [API Reference](api-reference.md)
- [Troubleshooting](troubleshooting.md)
