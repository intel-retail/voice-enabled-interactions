# Identity Service — Multimodal Biometric Authentication (Face ID + Voiceprint)

A **flag-gated** microservice that adds loyalty-grade identity verification to the
Smart AI Kiosk. It recognises a returning customer from **both** a camera frame
(Face ID) **and** a short spoken phrase (Voiceprint), then hands their loyalty
profile (name, favourites, dietary restrictions) back to the kiosk so the agent
can personalise the conversation.

The service only starts when explicitly enabled — see [Feature Flags](#feature-flags).

---

## Why two flags?

| Layer | Flag | Effect |
|---|---|---|
| **Container** | docker-compose `profiles: ["identity"]` | The container is not created at all unless the `identity` profile is active. |
| **Application** | `KIOSK_CORE_IDENTITY_ENABLED` (kiosk-core env) | Tells **kiosk-core** whether to call the identity-service. |

Run the full stack with identity enabled:

```bash
# from smart-kiosk-assistant/
make build up IDENTITY=true
# or, directly:
KIOSK_CORE_IDENTITY_ENABLED=true docker compose --profile identity up -d
```

When the flag is off, neither the container nor any kiosk-core call exists — the
kiosk behaves exactly as before.

---

## Architecture

```
            ┌──────────────┐   challenge / verify    ┌────────────────────┐
  kiosk-ui  │  kiosk-core  │ ─────────────────────▶  │  identity-service  │
  (camera + │ (orchestr.)  │ ◀─────────────────────  │   (this service)   │
   mic)     └──────────────┘   profile / verdict     └─────────┬──────────┘
                                                               │
                       ┌───────────────────────────────────────┼───────────────────┐
                       ▼                       ▼                ▼                    ▼
              OpenVINO face engine   OpenVINO voice engine   FAISS indices    SQLite (shared)
              detect → reid (256-d)  ECAPA-TDNN (192-d)      face + voice     loyalty_profiles
```

- **Service layer** (`identity_core/service.py`) — the only object routers call;
  owns the challenge provider, repository, FAISS managers and inference engines.
- **Inference** (`identity_core/inference/`) — Factory builds two Strategy engines
  (face, voice); both return **L2-normalized `float32`** vectors.
- **Persistence** (`identity_core/persistence/`) — SQLite repository
  (`loyalty_profiles`) + FAISS `IndexFlatIP` managers. **Embeddings live only in
  FAISS**; SQLite stores the integer FAISS offsets, never the vectors.
- **Shared DB** — mounts the same `kiosk_db` volume as kiosk-core, so
  `loyalty_profiles` lives in `kiosk.db` alongside the ordering tables.

### AI models

| Modality | Model | Output | Source |
|---|---|---|---|
| Face detect | `face-detection-retail-0005` | bounding boxes | Open Model Zoo (OpenVINO IR) |
| Face embed | `face-reidentification-retail-0095` | 256-d | Open Model Zoo (OpenVINO IR) |
| Voice embed | `ecapa-tdnn-voice` | 192-d | SpeechBrain ECAPA → OpenVINO IR (converted at setup) |

Verification requires **both** modalities. The fused score is
`0.6·face + 0.4·voice` and must clear `combined_threshold` (default `0.78`).

> The challenge prompt is random (anti-replay only). ECAPA-TDNN is
> text-independent, so the spoken words are **not** matched against the prompt.

---

## REST API

Base URL inside the cluster: `http://identity-service:8013`

| Method | Path | Purpose | Status |
|---|---|---|---|
| GET  | `/health` | Liveness probe | ✅ |
| GET  | `/api/v1/identity/challenge` | Random voice challenge prompt | ✅ |
| GET  | `/api/v1/identity/stats` | Profile + index counts, `inference_ready` | ✅ |
| POST | `/api/v1/identity/verify` | Multimodal (face+voice) verification | ✅ |
| POST | `/api/v1/identity/register` | Admin / manual enrolment | ✅ |

kiosk-core (`:8012`) proxies this contract 1:1 when `KIOSK_CORE_IDENTITY_ENABLED=true`
(see [kiosk-core proxy](#kiosk-core-proxy--kiosk-ui-gate) below), plus one always-on
capability endpoint kiosk-ui uses to decide whether to show the auth gate at all:

| Method | Path (kiosk-core) | Purpose |
|---|---|---|
| GET  | `/api/v1/identity/enabled` | Runtime flag mirroring `KIOSK_CORE_IDENTITY_ENABLED`; **always reachable**, even when the identity feature is off. |
| GET  | `/api/v1/identity/challenge` | Proxies identity-service `/challenge` (gated by the flag). |
| POST | `/api/v1/identity/verify` | Proxies identity-service `/verify`; both `image_base64` and `audio_base64` required. |
| POST | `/api/v1/identity/register` | Proxies identity-service `/register` (self-service enrolment); both `image_base64` and `audio_base64` required. |

---

## kiosk-core proxy + kiosk-ui gate

kiosk-ui wraps the existing chat home page in an `AuthGate` (mounted once in
`main.tsx`, so `App.tsx`/the chat experience itself is untouched):

1. On load, it calls `GET /api/v1/identity/enabled`. If the identity feature is
   disabled or the backend is unreachable, the gate **bypasses** and renders the
   chat home page exactly as before — zero behavioural change.
2. If enabled, it shows a **Login** screen: live camera preview + an on-screen
   challenge phrase (read aloud). On "Authenticate" it captures one JPEG frame
   and a ~3s WAV clip and calls `verify()`. A verified user is redirected to the
   existing chat home page; an unverified user sees an on-screen
   *"User not authenticated"* error and can retry or register.
3. A **Register** screen (linked from Login) collects a display name, generates
   a `user_id` slug (`name` + random suffix), captures face + voice the same
   way, and calls `register()` — reusing the same identity-service enrolment
   pipeline the video-file bootstrap path uses, so newly registered users are
   written to the same FAISS indices + `loyalty_profiles` SQLite table.

Relevant kiosk-ui source: `src/components/Auth/{AuthGate,LoginScreen,RegisterScreen}.tsx`,
`src/hooks/{useCamera,useVoiceCapture}.ts`, `src/api/identityApi.ts`.

---

## Configuration

All settings come from `configs/identity/identity_config.yaml` (mounted read-only)
overlaid with environment overrides. Business code references the typed
`Settings` object — never raw `os.getenv` / `dict["key"]`.

Key env overrides (see `docker-compose.yml`):

| Variable | Default | Meaning |
|---|---|---|
| `IDENTITY_DEVICE` | `GPU` | OpenVINO device (`CPU`/`GPU`/`NPU`) |
| `IDENTITY_DB_PATH` | `/app/data/kiosk.db` | Shared SQLite file |
| `IDENTITY_FAISS_DIR` | `/app/data/identity` | FAISS index directory |
| `IDENTITY_MODELS_DIR` | `/app/models` | Mounted model root |
| `IDENTITY_PRECISION` | `FP16` | Face IR precision |
| `IDENTITY_VOICE_BACKEND` | `openvino` | Voice Strategy (`openvino` \| `speechbrain`) |
| `BOOTSTRAP_ON_START` | `true` | Auto-register YAML profiles on boot (Phase 5) |

---

## Models

Models are **not** committed (`smart-kiosk-assistant/models/` is gitignored).
Download / convert them once with the kiosk setup script:

```bash
# from smart-kiosk-assistant/
./setup_models.sh --identity        # LLM + identity models
./setup_models.sh --identity-only   # identity models only
```

This places models under `models/identity/` (mounted at `/app/models`):

```
models/identity/
├── face-detection-retail-0005/FP16/face-detection-retail-0005.{xml,bin}
├── face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.{xml,bin}
└── ecapa-tdnn-voice/openvino/ecapa-tdnn-voice.{xml,bin}      # converted from SpeechBrain
```

The ECAPA conversion is a one-time, setup-only step
(`tools/convert_ecapa_to_openvino.py`, run inside `.setup-venv`). It is never
imported by the running service, so torch/speechbrain stay out of the container
image.

If any model file is missing the corresponding engine is disabled and the service
still boots (`inference_ready=false`); `verify`/`register` report *not ready*.

---

## Project layout

```
identity-service/
├── main.py                         # FastAPI app + endpoints + lifespan
├── Dockerfile                      # FROM intel/dlstreamer:2026.1.0-ubuntu24
├── requirements.txt
├── identity_core/
│   ├── config.py                   # typed Settings (YAML + env)
│   ├── challenge.py                # random anti-replay prompts
│   ├── models.py                   # Pydantic v2 DTOs
│   ├── service.py                  # IdentityService orchestrator
│   ├── inference/                  # Phase 4 — OpenVINO engines (Factory + Strategy)
│   │   ├── base.py                 #   ABCs + l2_normalize + FaceDetection
│   │   ├── openvino_face.py        #   detect + reid → 256-d
│   │   ├── openvino_voice.py       #   ECAPA IR → 192-d
│   │   └── factory.py              #   build engines, graceful degradation
│   └── persistence/                # Phase 3 — storage
│       ├── db.py                   #   shared kiosk.db, loyalty_profiles schema
│       ├── repository.py           #   AbstractProfileRepository + SQLite impl
│       └── faiss_index.py          #   FaissIndexManager (IndexFlatIP)
└── tools/
    └── convert_ecapa_to_openvino.py  # setup-only ECAPA → OpenVINO IR converter
```

---

## Status

### ✅ Done

- **Phase 1 — kiosk-core orchestration:** feature flag, `IdentityClient`,
  flag-guarded proxy router, `main.py` wiring.
- **Phase 2 — service skeleton:** FastAPI app, typed config, random challenge
  provider, Dockerfile (dlstreamer base), compose profile, shared `kiosk_db`.
- **Phase 3 — storage:** `loyalty_profiles` repository + FAISS index managers;
  `GET /stats`; embeddings persisted only in FAISS.
- **Phase 4 — inference:** OpenVINO face (detect + reid, 256-d) and voice
  (ECAPA-TDNN, 192-d) engines via Factory + Strategy; `setup_models.sh --identity`
  to fetch/convert models; engines degrade gracefully when models are absent.
  Verified: image builds, face engine emits a 256-d L2-normalized embedding.

### ⏳ Pending

- **Phase 5 — bootstrap registration** *(picked up next)*:
  - Implement `IdentityService.register()`: decode image/audio → face/voice
    embeddings → FAISS `add` → `loyalty_profiles` insert.
  - `BOOTSTRAP_ON_START` flow: read `profiles:` from `identity_config.yaml`,
    skip existing `user_id`s, sample video every Nth frame (OpenCV), read WAV.
  - **Needs sample audio/video** from the user; paths go in
    `identity_config.yaml → profiles:` and a `./identity-service/sample-data`
    mount (commented in `docker-compose.yml`).
  - Convert the ECAPA voice IR (`./setup_models.sh --identity`) so
    `inference_ready` flips to `true`.
- **Phase 6 — verification:** fusion scoring + profile retrieval in `verify()`;
  inject favourites/restrictions into the kiosk-core LLM session context.
- **Phase 7 — UI:** ✅ done — `AuthGate` in kiosk-ui gates the chat home page
  behind face+voice login, with a self-service Register screen. Gate presence
  is driven entirely by the backend `KIOSK_CORE_IDENTITY_ENABLED` flag (via
  `GET /api/v1/identity/enabled`), so it is a pure add-on with no impact on the
  chat experience when the feature is off.
- **Phase 8 — tests & docs:** pytest suite under `tests/`; LLD, sequence/class
  diagrams, schema and API documentation under `docs/`.

---

## Local quick check

```bash
# Build just this service
docker compose build identity-service

# Run standalone on CPU with host models mounted
docker run --rm -e IDENTITY_DEVICE=CPU -e BOOTSTRAP_ON_START=false \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/configs/identity:/app/configs/identity:ro" \
  -p 8013:8013 intel/identity-service:latest

curl localhost:8013/health
curl localhost:8013/api/v1/identity/challenge
curl localhost:8013/api/v1/identity/stats
```
