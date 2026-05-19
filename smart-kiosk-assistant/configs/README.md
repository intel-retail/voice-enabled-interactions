# Kiosk service configs

This folder holds **kiosk-pinned** configuration for the upstream services we
spin up from `docker-compose.yml`:

- `audio-analyzer/config.container.yaml` — mounted into the `audio-analyzer`
  container at `/app/audio_analyzer/config.container.yaml` and selected via
  the `AUDIO_ANALYZER_CONFIG_OVERRIDE_PATHS` env var. Pins the ASR model,
  device, denoise/chunking settings, allowed audio formats, and sentiment
  options.
- `text-to-speech/config.container.yaml` — mounted into the `text-to-speech`
  container at `/app/text-to-speech/config.container.yaml` and selected via
  the `TEXT_TO_SPEECH_CONFIG_OVERRIDE_PATHS` env var. Pins the TTS model,
  voice, language, device, dtype, and output format.

> Note: The `rag-service` uses the same override mechanism on its own
> `config.container.yaml` (mounted directly from `rag-service/`). Only the
> two services that come from `edge-ai-libraries/` need the `configs/`
> indirection.

## Why this exists

Both services live in `edge-ai-libraries/microservices/…` and ship their own
default `config.container.yaml`. Those defaults can change over time, and the
kiosk needs reproducible behaviour (specific ASR model, specific TTS voice,
denoise off, etc.).

By keeping a pinned copy here and mounting it over the in-image config, we get:

1. The kiosk always boots the exact service configuration it was tested with,
   regardless of changes made in `edge-ai-libraries`.
2. **No fork / no patches** of the upstream services — they're consumed as-is.
3. Anyone tweaking kiosk behaviour edits files in this folder, not in
   `edge-ai-libraries/`.

## How it works

`docker-compose.yml` mounts these files read-only into each container, and
the service reads them because of the `*_CONFIG_OVERRIDE_PATHS` env var the
compose file sets. No code changes in the services are needed — both
already support a config-override path out of the box.

## Updating

1. Edit the YAML here.
2. `docker compose up -d --build --force-recreate audio-analyzer` (or
   `text-to-speech`).
3. The service restarts with the new pinned config.
