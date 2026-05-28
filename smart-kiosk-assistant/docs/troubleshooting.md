# Troubleshooting

## Stack Will Not Start

- Confirm the published host ports are free:

  ```bash
  ss -ltnp | grep -E "7860|8010|8011|8012|8020"
  ```

- Confirm Docker Compose can build:

  ```bash
  docker compose config
  docker compose build
  ```

- Tail individual services to find the first failure:

  ```bash
  docker compose logs -f audio-analyzer
  docker compose logs -f text-to-speech
  docker compose logs -f rag-service
  docker compose logs -f kiosk-core
  docker compose logs -f kiosk-ui
  ```

## First Startup Is Slow

This is expected. On first run each model-hosting service downloads or
exports model assets to its `models/` directory and Hugging Face cache.
Subsequent starts reuse the cached artifacts. The default
`audio-analyzer` healthcheck allows up to ~240 seconds for warmup; the
RAG LLM compile on GPU can also take a few minutes the first time.

## A `health` Endpoint Fails

- Run `docker compose ps` and check the `STATUS` column for `unhealthy`.
- If you are behind a corporate proxy, pass `--noproxy '*'` to `curl`
  when hitting `127.0.0.1`.
- Confirm the service container actually started:

  ```bash
  docker compose logs <service-name>
  ```

## Selected Device Is Not Used

The device field lives in the per-service pinned config (see
[configuration.md](configuration.md#inference-device)). If the device
does not appear in the logs:

- Check the value is supported for that model (e.g. `NPU` is only
  supported by the Qwen-TTS variant; `audio-analyzer` ASR supports
  `CPU` and `GPU` only).
- For `GPU`: confirm `/dev/dri` exists and the Intel OpenVINO GPU
  runtime is installed.
- For `NPU`: confirm the Intel NPU driver and matching `level-zero`
  loader are installed; `rag-service` also needs the right `group_add`
  entry for `/dev/accel/accel0`.
- Restart the affected service after the change:

  ```bash
  docker compose up -d --build --force-recreate <service-name>
  ```

- Confirm OpenVINO picked the device:

  ```bash
  docker compose logs <service-name> | grep -i -E "device|compiling|GPU|NPU|CPU"
  ```

## Permission Errors on Mounted Folders

The compose file runs most containers as
`user: "${LOCAL_UID:-1000}:${LOCAL_GID:-1000}"`. If your host user has a
different UID/GID, bind-mounted folders such as
`../edge-ai-libraries/microservices/audio-analyzer/models/`,
`../edge-ai-libraries/microservices/text-to-speech/models/`, or the
service `.cache/` directories may show errors like:

```
PermissionError: [Errno 13] Permission denied: '...'
```

Start the stack with your host user's IDs:

```bash
LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose up -d --build
```

If the directories already exist as `root`, repair them once:

```bash
sudo chown -R "$(id -u):$(id -g)" \
  ../edge-ai-libraries/microservices/audio-analyzer/{models,chunks,storage,.cache} \
  ../edge-ai-libraries/microservices/text-to-speech/{models,storage,.cache}
```

## Browser UI Does Not Capture Audio

- Confirm the browser granted microphone permission for
  `http://127.0.0.1:7860`. Reset the permission and reload if needed.
- Modern browsers restrict microphone access on insecure origins. Use
  `http://127.0.0.1` (loopback) or serve the UI behind HTTPS.
- Check the `kiosk-ui` logs for upload errors:

  ```bash
  docker compose logs -f kiosk-ui
  ```

## Answer Is Empty or Off-Topic

- Confirm the knowledge base was ingested. The Gradio UI exposes an
  ingestion panel; see also
  [rag-service/README.md](../rag-service/README.md).
- Check `rag-service` logs for retrieval scores and reranker output.
- Try the same question from the API to rule out the UI:

  ```bash
  curl --noproxy '*' -X POST http://127.0.0.1:8020/api/v1/query \
    -H 'Content-Type: application/json' \
    -d '{"query":"What are the store hours?"}'
  ```

## TTS Plays No Audio in the Browser

- Confirm the session snapshot has non-empty `tts_audio_segments` and
  no `tts_errors`. See [api-reference.md](api-reference.md).
- The `kiosk-core` container and the `kiosk-ui` container share the
  `generated_audio` Docker volume. If you removed the volume, recreate
  the stack:

  ```bash
  docker compose down
  docker compose up -d --build
  ```

## kiosk-core Cannot Reach a Downstream Service

The compose defaults wire `kiosk-core` and `kiosk-ui` to the internal
service names (`audio-analyzer`, `text-to-speech`, `rag-service`). If
you override these URLs for a host-run setup, confirm:

- The downstream service is reachable from `kiosk-core` (try `curl`
  against the override URL from inside the `kiosk-core` container or
  from the host).
- For host-run downstreams reached from a container, use
  `host.docker.internal` (see the alternative compose snippets in
  [run-container.md](run-container.md)).

## See Also

- [Configuration](configuration.md)
- [Get Started](get-started.md)
- [Run On The Host](run-standalone.md)
