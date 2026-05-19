# Run Standalone (CLI Mic Mode)

Run kiosk-core directly on the host when you want the **key-press terminal mic loop**: press Enter, speak, see transcript + RAG response + TTS paths printed to the terminal — no browser needed.

Because kiosk-core runs on the host, `sounddevice` opens the mic directly. No Docker audio passthrough required.

For the Gradio browser UI instead, see [run-container.md](run-container.md).

Clone the repo with its dependency submodule before starting:

```bash
git clone --recurse-submodules https://github.com/unarayan/voice-enabled-interactions.git
cd voice-enabled-interactions/smart-kiosk-assistant
```

If the repo is already cloned, run `git submodule update --init --recursive` from the repo root.

## Prerequisites

The three downstream services must be running before starting kiosk-core:

```bash
# From smart-kiosk-assistant/
cd ../edge-ai-libraries/microservices/audio-analyzer && docker compose up -d && cd -
cd ../edge-ai-libraries/microservices/text-to-speech && docker compose up -d && cd -
cd rag-service    && python main.py &
```

If you want one command to start everything, use the container flow in [run-container.md](run-container.md) instead.

| Service | Default URL | Port |
|---|---|---|
| audio-analyzer | `http://127.0.0.1:8010/v1/audio/transcriptions` | 8010 |
| text-to-speech | `http://127.0.0.1:8011/v1/audio/speech` | 8011 |
| RAG service | `http://127.0.0.1:8020/api/v1/query` | 8020 |

These URLs can be overridden with environment variables. See [configuration.md](configuration.md).

## System Packages

Install the PortAudio runtime required by `sounddevice`:

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 portaudio19-dev
```

## Python Setup

From the `kiosk_core/` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Start the API

```bash
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8012
```

Default bind address:

- host: `0.0.0.0`
- port: `8012`

## Verify

```bash
curl --noproxy '*' http://127.0.0.1:8012/health
```

## Start the Gradio UI

In a separate terminal, from the same `kiosk_core/` directory:

```bash
source .venv/bin/activate
python gradio_app.py
```

The UI listens on `http://0.0.0.0:7860` by default and is accessible at:

```
http://127.0.0.1:7860
```

Open this address in a browser to use the voice assistant interface.

## Override Downstream URLs

```bash
KIOSK_CORE_ANALYZER_URL=http://127.0.0.1:8010/v1/audio/transcriptions \
KIOSK_CORE_TTS_URL=http://127.0.0.1:8011/v1/audio/speech \
KIOSK_CORE_RAG_URL=http://127.0.0.1:8020/api/v1/query \
uvicorn main:app --host 0.0.0.0 --port 8012
```

For the Gradio UI:

```bash
KIOSK_CORE_UI_BASE_URL=http://127.0.0.1:8012 \
KIOSK_CORE_UI_ANALYZER_URL=http://127.0.0.1:8010/v1/audio/transcriptions \
KIOSK_CORE_UI_RAG_URL=http://127.0.0.1:8020/api/v1/query \
KIOSK_CORE_UI_TTS_URL=http://127.0.0.1:8011/v1/audio/speech \
python gradio_app.py
```

## Live Microphone Session (CLI)

`mic_session.py` runs an **interactive key-press loop** against the kiosk-core API. Each press of Enter opens the mic, the session auto-stops on silence, and the transcript + RAG response + TTS clip paths are printed to the terminal. Press Ctrl+C at the prompt to exit the loop.

### List available input devices first

```bash
python mic_session.py --list-devices
```

### Start the interactive loop

```bash
python mic_session.py --device "default"
```

```
╔══════════════════════════════════════════════╗
║       Smart Kiosk Assistant — Mic CLI        ║
╚══════════════════════════════════════════════╝
  kiosk-core : http://127.0.0.1:8012
  device     : default  |  16000Hz
  silence    : 1.5s timeout  |  threshold 900

  ┌─ Turn 1 ──────────────────────────────────────┐
  │  Press ENTER to speak  (Ctrl+C to quit)       │
  └──────────────────────────────────────────────┘
<Enter>
  Mic is open. Speak now.
  Auto-stops after 1.5s silence or 20.0s total. Ctrl+C to stop early.

  [running] captured 4.1s | chunks: 1

  Assistant:

  The store opens at 9:00 AM on Sunday. ...
  [TTS] generated_audio/<session_id>/response_001.wav
        "The store opens at 9:00 AM on Sunday."

────────────────────────────────────────────────────────────
  You said  : What time does the store open on Sunday?

  Play response with aplay:
    aplay "generated_audio/<session_id>/response_001.wav"
────────────────────────────────────────────────────────────

  ┌─ Turn 2 ──────────────────────────────────────┐
  │  Press ENTER to speak  (Ctrl+C to quit)       │
  └──────────────────────────────────────────────┘
```

**Device selection note:** Raw ALSA hardware devices (`hw:X,Y`) often only support their native sample rate (e.g. 48000 Hz) and will fail with `Invalid sample rate` at 16000 Hz. Use `default` or `pipewire` for software resampling:

```bash
python mic_session.py --device "default"
```

### Common options

```bash
# Specific device index (from --list-devices)
python mic_session.py --device 2

# Lower threshold = more sensitive (picks up quieter speech)
python mic_session.py --device "default" --silence-threshold 500

# Longer session window
python mic_session.py --device "default" --max-session-seconds 30 --silence-timeout-seconds 2.5

# Single session then exit (no loop)
python mic_session.py --device "default" --one-shot
```

## Notes

- TTS audio clips are written to `generated_audio/<session_id>/` relative to the `kiosk_core/` directory.
- Live microphone sessions require a working PortAudio input device on the host. File-driven sessions (`start-file`) have no hardware requirement.
- For API use cases and endpoint details, see [api.md](api.md).
