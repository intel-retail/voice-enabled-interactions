# Smart AI Kiosk — High-Level Design

**Version:** 2026.2.0  
**Date:** 2026-06-24  
**Author:** System Architecture Team

---

## 1. Overview

The Smart AI Kiosk is a voice-enabled, edge-deployable AI assistant platform
for public-facing environments — QSR ordering stations, retail help-points,
airports, and banks. Users interact entirely through natural speech; the system
transcribes audio, retrieves contextual knowledge, generates a response, places
orders when requested, and speaks the reply back.

A **Queue-Aware Menu Adaptation** feature extends the platform with real-time
computer-vision-driven queue management. An RTSP video stream (simulating a
camera pointed at the ordering queue) is processed by a DL Streamer pipeline that
counts people. kiosk-core's Queue Manager module receives the count and adjusts
menu recommendations in real time — promoting faster-to-prepare items when the
queue is long, and surfacing full menus when it is short.

The architecture follows a **modular microservices design** running on Intel
hardware via Docker Compose. Every component is independently deployable,
observable, and replaceable.

---

## 2. System Architecture Diagram

```mermaid
graph TB
    %% ── External Actor ──────────────────────────────────────────────
    User(["👤 User\n(Browser / Kiosk Screen)"])

    %% ── Presentation Layer ──────────────────────────────────────────
    subgraph UI["Presentation Layer"]
        REACT["🖥️ Kiosk UI\n(React · TypeScript · Vite)\nport 7860\nnginx reverse-proxy"]
    end

    %% ── Queue + Identity Vision Layer ───────────────────────────────
    subgraph VISION["Vision Layer"]
        RTSP["📹 RTSP Streamer\n(ffmpeg · port 8554)\nmimics queue + identity camera\nrtsp://rtsp-streamer:8554/stream"]
        DLS_QUEUE["⚙️ DL Streamer — Queue Counter\n(GStreamer + OpenVINO)\nrtspsrc → gvadetect(person)\n→ count aggregator\n→ POST /api/v1/queue/update"]
        DLS_IDENT["⚙️ DL Streamer — Identity\n(GStreamer + OpenVINO GPU)\nVideo: rtspsrc → gvadetect\n  → gvaclassify(face-reid)\nAudio: rtspsrc → gvaclassify\n  (ECAPA-TDNN) → appsink\n→ POST /api/v1/identity/verify"]
    end

    %% ── Identity Service ────────────────────────────────────────────
    subgraph IDENTSVC["Identity Service  :8030"]
        IDENT_API["Identity API\nPOST /api/v1/identity/verify\nPOST /api/v1/identity/enroll\nGET  /api/v1/identity/profile/{user_id}"]
        FACE_MODEL["Face Pipeline\n(OpenVINO GPU)\nface-detection-retail-0005\nface-reid-retail-0095\n→ 256-d L2-norm embedding"]
        VOICE_MODEL["Voice Pipeline\n(OpenVINO CPU)\nECAPA-TDNN\n→ 192-d L2-norm embedding"]
        FUSION["Fusion Engine\nScore = 0.6·Sim_face\n       + 0.4·Sim_voice\nFace threshold: 0.80\nVoice threshold: 0.75"]
        FAISS_FACE[("FAISS Index\nface_index.bin\nIndexFlatIP · 256-d")]
        FAISS_VOICE[("FAISS Index\nvoice_index.bin\nIndexFlatIP · 192-d")]
        IDENT_SQLITE[("SQLite — identity.db\nloyalty_profiles\n  user_id PK\n  name · favorites\n  restrictions\n  face_faiss_id\n  voice_faiss_id")]
        BOOTSTRAP["Bootstrap Loader\n(BOOTSTRAP_ON_START)\nidentity_config.yaml\n→ seed FAISS + SQLite")]
    end

    %% ── Orchestration Layer ─────────────────────────────────────────
    subgraph CORE["Orchestration Layer — kiosk-core  :8012"]
        SESSION["Audio Session Manager\n(start-stream / push-chunk / eos / poll)"]
        ORDERING_API["Ordering REST API\n/api/v1/products\n/api/v1/orders\n/api/v1/upsell"]
        MCP_SERVER["MCP Tool Server\n(fastmcp · /mcp)\nlist_products · place_order\nupdate_order · confirm_order\nget_upsell · get_order\nget_queue_status"]
        ORDERING_SVC["Ordering Service\n(business logic · upsell rules)"]
        UPSELL_ENGINE["Upsell Engine\n(YAML rule matching)"]
        subgraph QUEUE_LIB["Queue Manager Library\n(kiosk_core/queue_manager/)"]
            QUEUE_API["Queue API\nPOST /api/v1/queue/update\nGET  /api/v1/queue/status"]
            QUEUE_SVC["QueueManagerService\n(count state · thresholds)\nlow / medium / high / critical"]
            MENU_ADAPT["MenuAdapter\n(adjusts product weights\nbased on queue tier)"]
        end
    end

    %% ── AI Services Layer ───────────────────────────────────────────
    subgraph AI["AI Services Layer"]
        subgraph ASR["Audio Analyzer  :8010"]
            WHISPER["Whisper ASR\n(OpenVINO · INT8)\n/v1/audio/transcriptions"]
            DIARIZE["Speaker Diarization\n(Pyannote · primary / secondary)"]
        end

        subgraph RAG["RAG Service  :8020"]
            RAG_API["RAG Query API\n/api/v1/query"]
            AGENT_EP["Agent Endpoint\n/api/v1/agent/chat"]
            RAG_PIPE["RAG Pipeline\n(embed → retrieve → rerank → generate)"]
            EMBEDDER["Embedding Model\n(OpenVINO)"]
            RERANKER["Reranker Model\n(OpenVINO)"]
            LLM["LLM — Qwen2.5-4B\n(OVIRTextGenPipeline · OpenVINO)\nin-process generation + streaming"]
            CHROMADB[("ChromaDB Vector Store\ncollection: smart-kiosk-assistant-bge-large\npersist: ./storage/vector_db")]
            AGENT["Ordering Agent\n(Google ADK · LlmAgent)\nknowledge_lookup tool\n+ MCP tools via MCP client"]
        end

        subgraph TTS["Text-to-Speech  :8011"]
            TTS_API["TTS API\n(OpenAI-compatible)\n/v1/audio/speech"]
            TTS_MODEL["SpeechT5 / Qwen-TTS\n(OpenVINO)"]
        end
    end

    %% ── Observability Layer ─────────────────────────────────────────
    subgraph OBS["Observability Layer"]
        METRICS["Metrics Collector  :9000\n(CPU · GPU · NPU · Memory)"]
    end

    %% ── Persistence Layer ───────────────────────────────────────────
    subgraph DB["Persistence Layer"]
        SQLITE[("SQLite — kiosk.db\n─────────────────\nproducts · orders\norder_items · users")]
        YAML_SEED["YAML Config\n(products.yaml\n upsell_rules.yaml\n queue_thresholds.yaml\n identity_config.yaml)"]
        KB_STORE[("Knowledge Base\n(vector chunks\nin ChromaDB)")]
        AUDIO_FILES[("Generated Audio\n/app/generated_audio\n*.wav TTS segments")]
    end

    %% ── User → UI ───────────────────────────────────────────────────
    User <-->|"HTTPS / WebSocket\n(mic capture + TTS playback)"| REACT

    %% ── UI → Core ───────────────────────────────────────────────────
    REACT -->|"POST /api/v1/sessions/start-stream\nPOST /api/v1/sessions/{id}/audio\nGET  /api/v1/sessions/{id} (poll)"| SESSION
    REACT -->|"GET  /api/v1/products\nPOST /api/v1/orders"| ORDERING_API
    REACT -->|"GET  /api/v1/queue/status"| QUEUE_API
    REACT -->|"camera frame + mic audio\n(base64) → identity verify"| SESSION

    %% ── Identity flow ───────────────────────────────────────────────
    SESSION -->|"POST /api/v1/identity/verify\n{image_b64, audio_b64}"| IDENT_API
    IDENT_API --> FACE_MODEL
    IDENT_API --> VOICE_MODEL
    FACE_MODEL -->|"256-d vector\nIndexFlatIP search"| FAISS_FACE
    VOICE_MODEL -->|"192-d vector\nIndexFlatIP search"| FAISS_VOICE
    FAISS_FACE --> FUSION
    FAISS_VOICE --> FUSION
    FUSION -->|"profile lookup\nby faiss_id"| IDENT_SQLITE
    FUSION -->|"profile: favorites\n+ restrictions"| SESSION
    SESSION -->|"inject loyalty profile\ninto LLM prompt context"| AGENT_EP
    BOOTSTRAP -.->|"seed on startup"| FAISS_FACE
    BOOTSTRAP -.->|"seed on startup"| FAISS_VOICE
    BOOTSTRAP -.->|"seed on startup"| IDENT_SQLITE
    YAML_SEED -.->|"identity_config.yaml"| BOOTSTRAP

    %% ── DL Streamer → Identity (real-time RTSP path) ────────────────
    RTSP -->|"RTSP stream"| DLS_QUEUE
    RTSP -->|"RTSP stream"| DLS_IDENT
    DLS_QUEUE -->|"POST /api/v1/queue/update\n{ count, timestamp }"| QUEUE_API
    DLS_IDENT -->|"POST /api/v1/identity/verify\n(face crop + voice segment)"| IDENT_API

    %% ── Queue flow ───────────────────────────────────────────────────
    QUEUE_API --> QUEUE_SVC
    QUEUE_SVC --> MENU_ADAPT
    MENU_ADAPT -->|"adjusts recommendation weights"| ORDERING_SVC

    %% ── Core internal ───────────────────────────────────────────────
    SESSION -->|"WAV chunks 16kHz mono"| WHISPER
    SESSION -->|"text query"| RAG_API
    SESSION -->|"text query (ordering turns)"| AGENT_EP
    SESSION -->|"response text"| TTS_API
    ORDERING_API --> ORDERING_SVC
    ORDERING_SVC --> UPSELL_ENGINE
    ORDERING_SVC <-->|"async CRUD"| SQLITE
    YAML_SEED -.->|"products / upsell / queue"| SQLITE
    MCP_SERVER --> ORDERING_SVC
    MCP_SERVER -->|"get_queue_status"| QUEUE_SVC

    %% ── Core → Observability ────────────────────────────────────────
    SESSION -->|"GET /metrics"| METRICS

    %% ── ASR internals ───────────────────────────────────────────────
    WHISPER --> DIARIZE

    %% ── RAG internals ───────────────────────────────────────────────
    RAG_API --> RAG_PIPE
    RAG_PIPE --> EMBEDDER
    RAG_PIPE --> RERANKER
    RAG_PIPE --> LLM
    EMBEDDER <-->|"similarity search"| CHROMADB
    CHROMADB --- KB_STORE
    AGENT_EP --> AGENT
    AGENT -->|"knowledge_lookup"| RAG_PIPE
    AGENT -->|"HTTP/SSE (fastmcp)"| MCP_SERVER

    %% ── TTS persistence ─────────────────────────────────────────────
    TTS_API --> TTS_MODEL
    TTS_MODEL -->|"WAV file write"| AUDIO_FILES
    AUDIO_FILES -->|"FileResponse"| SESSION

    %% ── Styles ──────────────────────────────────────────────────────
    classDef service  fill:#EBF2FA,stroke:#0068B5,color:#000,rx:6
    classDef store    fill:#FFF8E1,stroke:#F9A825,color:#000
    classDef infra    fill:#F3E5F5,stroke:#7B1FA2,color:#000
    classDef ui       fill:#E8F5E9,stroke:#2E7D32,color:#000
    classDef actor    fill:#FFF,stroke:#555,color:#000
    classDef vision   fill:#FFF3E0,stroke:#E65100,color:#000,rx:6
    classDef queuelib fill:#F9FBE7,stroke:#827717,color:#000,rx:6
    classDef identity fill:#FCE4EC,stroke:#C62828,color:#000,rx:6

    class REACT,SESSION,ORDERING_API,MCP_SERVER,ORDERING_SVC,UPSELL_ENGINE service
    class WHISPER,DIARIZE,RAG_API,RAG_PIPE,EMBEDDER,RERANKER,LLM,AGENT_EP,AGENT,TTS_API,TTS_MODEL service
    class METRICS infra
    class SQLITE,CHROMADB,KB_STORE,AUDIO_FILES,YAML_SEED store
    class User actor
    class RTSP,DLS_QUEUE,DLS_IDENT vision
    class QUEUE_API,QUEUE_SVC,MENU_ADAPT queuelib
    class IDENT_API,FACE_MODEL,VOICE_MODEL,FUSION,BOOTSTRAP identity
    class FAISS_FACE,FAISS_VOICE,IDENT_SQLITE store
```

---

## 3. Component Descriptions

### 3.1 Kiosk UI (port 7860)
React + TypeScript + Vite SPA served by nginx inside Docker.
Captures microphone audio via the Web Audio API (AudioWorklet → 16kHz mono WAV),
streams chunks to kiosk-core, polls for partial transcripts and responses, and
plays TTS audio segments via a sequential HTMLAudioElement queue. Right-hand
panels surface the live order cart, device settings, knowledge-base ingestion,
performance charts, model KPIs, and a live queue indicator — all backed by REST
calls proxied through nginx.

### 3.2 kiosk-core (port 8012)
FastAPI orchestration hub. Owns four vertical concerns:

| Sub-module | Responsibility |
|---|---|
| **Audio Session Manager** | Opens browser-streaming sessions; accepts WAV chunks; fans out to ASR → RAG/Agent → TTS; serves TTS WAV files back to the UI |
| **Ordering REST API** | CRUD for products, orders, and upsell suggestions; backed by SQLite |
| **MCP Tool Server** | Exposes ordering + queue operations as MCP tools (fastmcp/SSE) so the RAG-service agent can invoke them without HTTP coupling |
| **Queue Manager Library** | Receives person-count pushes from DL Streamer; maintains current queue tier; adjusts menu recommendation weights via `MenuAdapter` |

### 3.2a Queue Manager Library (inside kiosk-core)

The Queue Manager is a **Python module** (`kiosk_core/queue_manager/`) that acts
as a self-contained library within kiosk-core. It has no dependency on GStreamer
or any CV framework — it is pure business logic that receives a count and applies
policy.

```
kiosk_core/queue_manager/
  ├── __init__.py
  ├── service.py         # QueueManagerService — holds current count + tier state
  ├── models.py          # QueueUpdate, QueueStatus, QueueTier (low/medium/high/critical)
  ├── menu_adapter.py    # MenuAdapter — adjusts product recommendation weights per tier
  └── router.py          # FastAPI router: POST /api/v1/queue/update, GET /api/v1/queue/status
```

**Queue tiers (configured via `queue_thresholds.yaml`):**

| Tier | Person count | Menu behaviour |
|---|---|---|
| `low` | 0–2 | Full menu — surface premium / complex items |
| `medium` | 3–5 | Balanced — no suppression |
| `high` | 6–10 | Promote fast-prep items (drinks, sides, combos) |
| `critical` | 11+ | Suppress slow-prep items; highlight express combos |

The `get_queue_status` MCP tool exposes queue tier to the Ordering Agent so the
LLM can proactively inform customers ("The queue is busy — our express combo is
ready in 3 minutes!").

### 3.3 RTSP Streamer (port 8554)
An `ffmpeg`-based container that reads a pre-recorded video file (or a real USB
camera) and re-streams it over RTSP at `rtsp://rtsp-streamer:8554/stream`. In a
sample/demo deployment this mimics a real camera pointed at the ordering queue.
Swapping to a real camera requires only changing the `RTSP_SOURCE` environment
variable — no code changes.

### 3.4 DL Streamer Pipeline (queue-counter service)
A GStreamer pipeline using Intel DL Streamer (`gvadetect`) to detect and count
people in each video frame. Runs as a **dedicated Docker container** — keeping
the native GStreamer/OpenVINO runtime isolated from the Python services.

**Why a separate container (not embedded in kiosk-core):**
- GStreamer uses a GLib main loop that conflicts with Python's asyncio event loop
- A GStreamer segfault must not kill the ordering API
- kiosk-core Docker image stays lightweight (~300 MB vs. ~3 GB with GStreamer)
- The CV pipeline can be replaced or scaled independently

**Pipeline — Queue Counter:**
```
rtspsrc location=rtsp://rtsp-streamer:8554/stream
  → decodebin → videoconvert
  → gvadetect model=person-detection-0202.xml inference-interval=30
  → gvametaconvert
  → [count aggregation every N seconds]
  → POST http://kiosk-core:8012/api/v1/queue/update {"count": 7}
```

**Pipeline — Identity (real-time RTSP path):**
```
# Video: face detection + re-identification
rtspsrc location=rtsp://rtsp-streamer:8554/stream
  ! rtph264depay ! h264parse ! vaapidecodebin
  ! gvadetect model=models/face-detection-retail-0005.xml device=GPU
  ! gvaclassify model=models/face-reidentification-retail-0095.xml device=GPU
  ! appsink name=video_sink
  → POST http://identity:8030/api/v1/identity/verify {image_b64}

# Audio: speaker voice embedding
rtspsrc location=rtsp://rtsp-streamer:8554/stream
  ! rtpmp4gdepay ! aacparse ! decodebin
  ! audioconvert ! audioresample ! audio/x-raw,rate=16000,channels=1
  ! gvaclassify model=models/ecapa-tdnn-voice.xml device=CPU
  ! appsink name=audio_sink
  → POST http://identity:8030/api/v1/identity/verify {audio_b64}
```

### 3.4a Identity Service (port 8030)

The Identity Service provides **multimodal biometric authentication** — Face ID
and Voiceprint verification — enabling loyalty profile injection into the LLM
conversation context. It runs as a dedicated FastAPI service with its own
isolated SQLite database and FAISS vector indices.

#### Internal Architecture

```
identity-service/
  ├── api/router.py            POST /verify, POST /enroll, GET /profile/{id}
  ├── pipeline/
  │     ├── face_pipeline.py   OpenVINO face-detection + face-reid → 256-d embedding
  │     └── voice_pipeline.py  OpenVINO ECAPA-TDNN → 192-d embedding
  ├── index/
  │     ├── face_index.bin     FAISS IndexFlatIP (256-d, GPU-built)
  │     └── voice_index.bin    FAISS IndexFlatIP (192-d)
  ├── db/                      aiosqlite — identity.db (loyalty_profiles table)
  ├── fusion.py                Weighted score fusion + threshold logic
  └── bootstrap.py             Startup seed from identity_config.yaml
```

#### Bootstrap Flow (Automatic on Startup)

When `BOOTSTRAP_ON_START=true`:

1. Load `identity_config.yaml` — list of loyalty profiles with `video_path` and `audio_path`
2. For each profile: `SELECT 1 FROM loyalty_profiles WHERE user_id = ?` — skip if exists (idempotent across restarts)
3. **Face extraction**: OpenCV samples every 10th frame → face-detection-retail-0005 → landmark alignment → face-reid-retail-0095 → 256-d L2-normalised embedding → `faiss_add()` → `face_faiss_id`
4. **Voice extraction**: Read WAV → resample to 16 kHz → log-mel filterbank → ECAPA-TDNN → 192-d L2-normalised embedding → `faiss_add()` → `voice_faiss_id`
5. Write `loyalty_profiles` row with `{user_id, name, favorites, restrictions, face_faiss_id, voice_faiss_id}`

#### Verification Flow (Runtime)

`POST /api/v1/identity/verify` accepts `{image_b64?, audio_b64?}`:

1. **Face inference** (if image present): decode base64 → face detection → landmark crop → face-reid → 256-d L2-norm vector
2. **Voice inference** (if audio present): decode PCM WAV → 16 kHz resample → spectrogram → ECAPA-TDNN → 192-d L2-norm vector
3. **FAISS search**: `IndexFlatIP.search(query, k=1)` on each index → `(faiss_id, similarity_score)`
4. **Fusion scoring**:

| Modalities present | Score formula | Accept threshold |
|---|---|---|
| Face + Voice | `0.6 × Sim_face + 0.4 × Sim_voice` | Combined ≥ 0.78 |
| Face only | `Sim_face` | ≥ 0.80 |
| Voice only | `Sim_voice` | ≥ 0.75 |

5. On match: `SELECT * FROM loyalty_profiles WHERE face_faiss_id = ?` → return profile JSON
6. **kiosk-core injects** `{favorites, restrictions}` into the LLM agent's system prompt context → personalised ordering session begins

### 3.5 Audio Analyzer — ASR (port 8010)
OpenAI-compatible transcription service. Runs Whisper via OpenVINO (INT8
precision) for low-latency speech-to-text. Integrates Pyannote speaker
diarization to distinguish primary speaker (kiosk operator / customer) from
secondary speakers; kiosk-core drops secondary-speaker segments before
forwarding to the RAG pipeline.

### 3.6 RAG Service (port 8020)
Retrieval-Augmented Generation service with two query paths:

| Path | Endpoint | Used for |
|---|---|---|
| **Direct RAG** | `POST /api/v1/query` | Menu FAQ, informational Q&A |
| **Agent chat** | `POST /api/v1/agent/chat` | Ordering turns requiring tool execution |

The **RAG pipeline** embeds the query (OpenVINO embedding model — BGE-Large),
retrieves top-k chunks from **ChromaDB** (collection: `smart-kiosk-assistant-bge-large`,
persisted at `./storage/vector_db`), reranks, then generates a response with an
in-process Qwen2.5-4B OpenVINO LLM.

The **Ordering Agent** (Google ADK `LlmAgent`) handles turns that require taking
actions — placing orders, updating items, confirming. See Section 3.4a below for
a detailed walk-through of exactly how it works end-to-end.

### 3.6a Agentic Ordering — How It Works

This section explains the Ordering Agent, the MCP protocol, and why the
architecture is structured this way.

#### The Problem It Solves

When a customer says _"Add a Coke and confirm my order"_, that is not a question
to answer — it is a set of **actions** to perform against the database. The plain
RAG pipeline (embed → retrieve → generate) cannot do this; it only produces text.
The Ordering Agent bridges natural language to real HTTP API calls on kiosk-core.

#### The Two Services Involved

```
rag-service (port 8020)            kiosk-core (port 8012)
─────────────────────────          ─────────────────────────────────────
OrderingAgent                      Ordering REST API  (/api/v1/orders …)
  └─ Google ADK LlmAgent      ←──► MCP Tool Server    (/mcp)
       (Qwen3-4B via OVMS)         OrderingService
       + knowledge_lookup          SQLite kiosk.db
```

- **rag-service** contains the LLM reasoning logic (the "brain").
- **kiosk-core** owns the data and business rules (the "hands").

#### What is MCP?

MCP (**Model Context Protocol**) is an open standard that lets an AI agent
discover and call tools exposed by an external server using a uniform JSON-RPC
protocol over HTTP.

Think of it like this: instead of hard-coding `requests.post("http://kiosk-core:8012/api/v1/orders", …)`
inside the agent code every time you add a tool, you point the agent at
`http://kiosk-core:8012/mcp` and it automatically asks _"what tools do you have?"_.
kiosk-core replies with a list of tool names, descriptions, and parameter schemas.
The agent uses that information to decide when and how to call each tool.

In this project **fastmcp** is the library that implements the MCP server inside
kiosk-core. The agent's `MCPClient` is the matching client inside rag-service.

#### Step-by-Step Flow for an Ordering Turn

```
1. USER SPEAKS
   "I'd like a Paneer Tikka Burger please"

2. kiosk-core RECEIVES the transcript and routes to:
   POST http://rag-service:8020/api/v1/agent/chat
   { "message": "I'd like a Paneer Tikka Burger please",
     "session_id": "abc123", "user_id": "kiosk-user" }

3. AGENT BOOTSTRAPS (once at startup)
   rag-service MCPClient → GET http://kiosk-core:8012/mcp
   kiosk-core replies: "I have 6 tools:
     list_products, place_order, update_order,
     get_order, confirm_order, get_upsell_suggestions"
   Each tool gets wrapped as a Python async function the LLM can call.

4. LLM REASONS (Qwen3-4B on OVMS)
   Given the user message + tool list, the LLM decides:
   → first call list_products(category="burgers") to find the product_id
   → then call place_order(user_id="kiosk-user", items=[{product_id, qty}])

5. AGENT EXECUTES TOOL CALLS (HTTP to kiosk-core REST API)
   MCPClient → POST http://kiosk-core:8012/mcp
   { "method": "tools/call",
     "params": { "name": "place_order",
                 "arguments": { "user_id": "kiosk-user",
                                "items": [{"product_id": "BURGER-PANEER-001", "quantity": 1}] } } }

   kiosk-core MCP server receives this → calls OrderingService.place_order()
   → writes to SQLite (orders + order_items tables)
   → returns Order { order_id: 1, total: 199.0, status: "draft" }

6. AGENT AUTO-UPSELLS
   LLM sees the instruction: "after placing an order, call get_upsell_suggestions"
   → MCPClient → tools/call get_upsell_suggestions(product_ids=["BURGER-PANEER-001"])
   → kiosk-core checks upsell_rules.yaml → returns ["Add a Cold Drink?"]

7. LLM COMPOSES REPLY
   "I've added a Paneer Tikka Burger (₹199) to your order!
    You might also like a Cold Drink — would you like to add one?"

8. rag-service RETURNS to kiosk-core
   { "reply": "I've added ...", "tool_calls": ["list_products", "place_order", "get_upsell_suggestions"] }

9. kiosk-core sends reply text to TTS → WAV → UI plays audio
```

#### Key Points

| Concept | Concrete reality in this codebase |
|---|---|
| **Ordering Agent** | `rag-service/agentic/ordering_agent.py` — `OrderingAgent` class wrapping a Google ADK `LlmAgent` |
| **LLM for reasoning** | Qwen3-4B served by OVMS at `http://ovms-llm:8000/v3` (OpenAI-compatible) |
| **MCP server** | `kiosk_core/ordering/mcp_server.py` — `FastMCP("kiosk-ordering")`, mounted at `/mcp` on port 8012 |
| **MCP client** | `rag-service/agentic/mcp_client.py` — discovers tools at startup from `mcp_servers.json` |
| **Tool config** | `rag-service/agentic/resources/mcp_servers.json` — points to `http://kiosk-core:8012/mcp` |
| **Tools exposed** | `list_products`, `place_order`, `update_order`, `get_order`, `confirm_order`, `get_upsell_suggestions` |
| **knowledge_lookup** | Local Python function — calls the RAG pipeline directly (no HTTP hop) |
| **Session memory** | ADK `InMemorySessionService` — maintains multi-turn context per `session_id` |

#### Why MCP Instead of Direct HTTP Calls?

The agent could directly call `POST /api/v1/orders`. MCP adds one level of
indirection but gives three benefits:

1. **Auto-discovery** — add a new tool to kiosk-core's MCP server and the agent
   picks it up on next startup with no code change in rag-service.
2. **Schema-driven** — the LLM receives the parameter schema automatically, so it
   knows what arguments each tool expects.
3. **Protocol separation** — the agent doesn't need to know about REST verbs,
   URL paths, or auth headers. It only knows: _"call tool X with args Y"_.

### 3.7 Text-to-Speech (port 8011)
OpenAI-compatible synthesis service (`POST /v1/audio/speech`). Runs SpeechT5 or
Qwen-TTS via OpenVINO. Writes generated WAV files to the `generated_audio`
shared Docker volume; kiosk-core reads them back and serves them to the UI via
`GET /api/v1/sessions/{id}/audio/{filename}`.

### 3.8 Metrics Collector (port 9000)
Lightweight HTTP service that polls hardware telemetry (CPU, GPU, NPU, memory
utilisation) and exposes it at `GET /metrics`. The React UI displays live
performance charts; kiosk-core can query it for observability context.

---

## 4. SQLite Database Schema

The system uses **two SQLite databases** with distinct responsibilities:

### 4.1 kiosk.db — Ordering Domain (kiosk-core)

**ChromaDB** is used exclusively for RAG vector embeddings; **FAISS** is used
exclusively for biometric embeddings. Neither is mixed into these relational tables.

```
┌─────────────┐       ┌──────────────────┐       ┌───────────────────┐
│   products  │       │      orders      │       │    order_items    │
│─────────────│       │──────────────────│       │───────────────────│
│ product_id  │◄──FK──│ user_id (FK→users│  ┌───►│ id               │
│ name        │       │ order_id (PK)    │──┤    │ order_id (FK)    │
│ category    │       │ status           │  └───►│ product_id (FK)  │
│ price       │       │ total            │       │ quantity          │
└─────────────┘       │ created_at       │       │ price             │
        ▲             └──────────────────┘       └───────────────────┘
  Seeded from
  products.yaml
  (idempotent)

┌───────────┐
│   users   │
│───────────│
│ user_id   │◄──────── orders.user_id FK
│ name      │
└───────────┘
```

### 4.2 identity.db — Loyalty Profiles (identity-service)

Loyalty profiles are stored in a **separate SQLite database** inside the
identity-service container, isolated from the ordering domain. FAISS index
offsets (`face_faiss_id`, `voice_faiss_id`) serve as the bridge between
the relational store and the vector indices.

```
┌────────────────────────────────────┐
│         loyalty_profiles           │
│────────────────────────────────────│
│ user_id       TEXT  PRIMARY KEY    │
│ name          TEXT                 │
│ favorites     TEXT  (JSON array)   │  ← injected into LLM prompt
│ restrictions  TEXT  (JSON array)   │  ← dietary / allergy restrictions
│ face_faiss_id INTEGER              │──► face_index.bin offset
│ voice_faiss_id INTEGER             │──► voice_index.bin offset
│ created_at    DATETIME             │
└────────────────────────────────────┘
           │
           │  FAISS IndexFlatIP (cosine / Inner Product)
           ▼
┌──────────────────────┐   ┌──────────────────────┐
│  face_index.bin      │   │  voice_index.bin      │
│  dim=256 · L2-norm   │   │  dim=192 · L2-norm    │
│  face-reid-retail    │   │  ECAPA-TDNN           │
│  threshold ≥ 0.80    │   │  threshold ≥ 0.75     │
└──────────────────────┘   └──────────────────────┘
         Fusion: Score = 0.6 × Sim_face + 0.4 × Sim_voice
```

**PRAGMA settings (both databases):** `journal_mode=WAL`, `foreign_keys=ON`.

---

## 5. Data Flow — Voice Ordering Turn

```
Browser mic → WAV chunk → kiosk-core (Audio Session)
    │
    ├─→ Audio Analyzer  (Whisper ASR + Diarization)
    │       └─→ transcript (primary speaker only)
    │
    ├─→ RAG Service Agent (/api/v1/agent/chat)
    │       ├─→ LLM (OVMS / Qwen3-4B) selects tool
    │       ├─→ knowledge_lookup → RAG pipeline → ChromaDB → LLM
    │       └─→ MCP tools → kiosk-core MCP Server → SQLite
    │               e.g. place_order → orders + order_items
    │                    get_upsell_suggestions → rule match
    │                    get_queue_status → current queue tier
    │
    ├─→ Text-to-Speech  (SpeechT5 / Qwen-TTS → WAV file)
    │
    └─→ UI poll response:  transcript + response + TTS audio URLs
            └─→ Browser plays WAV segments sequentially
```

---

## 5a. Data Flow — Queue-Aware Menu Adaptation

```
[RTSP Streamer :8554]
  ffmpeg loops video file → RTSP stream at rtsp://rtsp-streamer:8554/stream
         │
         │  RTSP (H.264)
         ▼
[DL Streamer Container]
  GStreamer pipeline:
    rtspsrc → decodebin → videoconvert
    → gvadetect (person-detection-0202 / OpenVINO)
    → gvametaconvert
    → count aggregator (rolling window, every 5s)
         │
         │  POST /api/v1/queue/update {"count": 7, "timestamp": "..."}
         ▼
[kiosk-core — Queue Manager Library]
  QueueManagerService:
    count=7 → tier="high"   (threshold from queue_thresholds.yaml)
         │
    MenuAdapter:
    tier="high" → suppress slow-prep items
                → boost express combos, drinks, sides
         │
         │  adjusted weights applied to next product listing / upsell call
         ▼
[Ordering Agent via MCP]
  get_queue_status() → {"tier": "high", "count": 7}
  LLM adds context: "The queue is busy — here are our quickest options"
         │
         ▼
[UI]
  GET /api/v1/queue/status → live queue badge (green/yellow/red/critical)
```

**Key design decisions for queue management:**

| Decision | Choice | Rationale |
|---|---|---|
| DL Streamer as separate container | ✅ Separate Docker container | Keeps GStreamer/GLib runtime out of Python process; crash isolation; smaller kiosk-core image |
| Queue Manager as Python library | ✅ `kiosk_core/queue_manager/` module | Pure business logic (count → tier → menu weights); no GStreamer dependency; easily unit-tested |
| Push model (DL Streamer → kiosk-core) | HTTP POST every N seconds | Simple, stateless; kiosk-core does not need to poll the CV pipeline |
| Queue thresholds in YAML | `queue_thresholds.yaml` | Tunable without code changes; follows existing config pattern |

---

## 5b. Data Flow — Biometric Identity & Loyalty Injection

```
[Customer approaches kiosk]
         │
         │  Two parallel input paths:
         │
         ├─── Path A: UI-driven (browser camera + mic)
         │     REACT captures camera frame (JPEG) + mic audio (WAV)
         │     → base64 encode
         │     → kiosk-core SESSION
         │          → POST http://identity:8030/api/v1/identity/verify
         │               { image_b64: "...", audio_b64: "..." }
         │
         └─── Path B: DL Streamer real-time RTSP
               [DL Streamer — Identity container]
               Video pipeline: rtspsrc → gvadetect(face) → gvaclassify(face-reid)
               Audio pipeline: rtspsrc → gvaclassify(ECAPA-TDNN)
               → POST http://identity:8030/api/v1/identity/verify


[Identity Service — Verify]
   ┌── Face inference: base64/appsink frame
   │     face-detection-retail-0005 (OpenVINO GPU)
   │     → landmark crop + alignment
   │     → face-reidentification-retail-0095 (OpenVINO GPU)
   │     → 256-d L2-normalised vector
   │     → FAISS IndexFlatIP search on face_index.bin
   │     → (face_faiss_id, Sim_face)
   │
   ├── Voice inference: PCM WAV buffer
   │     resample to 16 kHz → log-mel filterbank
   │     → ECAPA-TDNN (OpenVINO CPU)
   │     → 192-d L2-normalised vector
   │     → FAISS IndexFlatIP search on voice_index.bin
   │     → (voice_faiss_id, Sim_voice)
   │
   └── Fusion Engine
         Both present:  Score = 0.6×Sim_face + 0.4×Sim_voice  ≥ 0.78 ?
         Face only:     Sim_face ≥ 0.80 ?
         Voice only:    Sim_voice ≥ 0.75 ?
              │
              │  MATCH → SELECT * FROM loyalty_profiles WHERE face_faiss_id = ?
              │
              └─→ Profile: { user_id, name, favorites, restrictions }

[kiosk-core SESSION]
   Receives profile → injects into LLM agent system prompt:
   "Customer: Alice. Favorites: Margherita Pizza, Cappuccino.
    Restrictions: nut allergy. Greet by name and avoid nut-based items."
         │
         ▼
[RAG Service — Ordering Agent]
   Personalised session: greets customer by name, filters menu,
   surfaces relevant items, applies loyalty promotions.
```



All services run as Docker containers on a single Intel edge node:

| Service | Port | Image | Notes |
|---|---|---|---|
| kiosk-ui | 7860 | intel/kiosk-ui (nginx + React SPA) | Serves UI + reverse-proxy |
| kiosk-core | 8012 | intel/kiosk-core (Python 3.11 / FastAPI) | Includes Queue Manager library |
| audio-analyzer | 8010 | intel/audio-analyzer (OpenVINO Whisper) | |
| text-to-speech | 8011 | intel/text-to-speech (OpenVINO TTS) | |
| rag-service | 8020 | intel/rag-service (OpenVINO LLM + ChromaDB) | |
| ovms-llm | 8000/9001 | openvino/model_server (Qwen3-4B, agent LLM) | |
| metrics-collector | 9000 | intel/metrics-collector | |
| identity-service | 8030 | intel/identity-service (OpenVINO + FAISS) | Face+Voice biometric auth; loyalty profiles |
| rtsp-streamer | 8554 | intel/rtsp-streamer (ffmpeg) | Mimics camera; swap for real RTSP camera |
| dlstreamer-queue | — | intel/dlstreamer-queue-counter (GStreamer + OpenVINO) | Person count → kiosk-core |
| dlstreamer-identity | — | intel/dlstreamer-identity (GStreamer + OpenVINO GPU) | Face/voice crops → identity-service |

**Networking:** all containers share a single Docker bridge network
(`smart-kiosk-assistant_default`). The UI's nginx reverse-proxies `/api`,
`/rag`, `/tts`, `/asr`, `/identity`, `/metrics-svc` to the respective backends.

**Volumes:**
- `generated_audio` — TTS WAV files shared between TTS service and kiosk-core
- `kiosk_db` — SQLite database file (`/app/data/kiosk.db`)
- `identity_db` — SQLite + FAISS indices (`/app/data/identity.db`, `face_index.bin`, `voice_index.bin`)
- `identity_models` — face-detection-retail-0005, face-reid-retail-0095, ECAPA-TDNN OpenVINO IR
- `ovms_models` — Qwen3-4B OpenVINO IR model artefacts
- `queue_models` — person-detection OpenVINO IR model (DL Streamer queue counter)
- Per-service model/cache volumes for ASR, TTS, and RAG

---

## 7. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM inference | OpenVINO (in-process) for RAG; OVMS for agent | RAG path optimised for latency; agent path needs tool-calling |
| Vector store | ChromaDB (`langchain-chroma`) persisted on disk | LangChain-native, supports cosine similarity, persistent across restarts without a separate server |
| Transactional store | SQLite + aiosqlite (WAL mode) | Zero-dependency, single-file, sufficient for edge throughput |
| Tool integration | MCP (fastmcp/SSE) | Standard protocol; agent and REST API share one service implementation |
| Speaker filtering | Pyannote diarization in ASR | Prevents bystander speech from corrupting the ordering context |
| UI streaming | AudioWorklet → 5-second WAV chunks | Balances latency vs. transcription accuracy on Whisper |
| Frontend | React + Vite + nginx (multi-stage Docker) | Replaces Gradio; SPA with full routing, live order panel, no Python dep |
| Queue CV pipeline | DL Streamer in dedicated container | GStreamer/GLib runtime must not share process with Python asyncio; crash isolation; image size |
| Queue business logic | Python module in kiosk-core | Pure policy code (count → tier → weights); no native deps; unit-testable without CV infrastructure |
| Queue integration model | Push (DL Streamer → kiosk-core REST) | Stateless; kiosk-core does not poll; pipeline can restart independently |
| Queue thresholds | YAML config (`queue_thresholds.yaml`) | Operator-tunable without code changes; consistent with products/upsell config pattern |
| Biometric vector store | FAISS IndexFlatIP (not ChromaDB) | Sub-millisecond exact cosine search over small identity corpus; no server process; binary file persisted on volume |
| Identity DB isolation | Separate `identity.db` (not `kiosk.db`) | Biometric data must not co-reside with ordering data; separate service boundary; independent backup/purge policy |
| Biometric fusion | Weighted score (0.6 face + 0.4 voice) | Face is higher confidence in controlled lighting; voice adds liveness assurance; configurable weights |
| Identity bootstrap | YAML seed on `BOOTSTRAP_ON_START` | Deterministic pre-registration for demos/testing; idempotent across restarts; no duplicate FAISS entries |
| RTSP identity pipeline | DL Streamer optional path | Enables passive/continuous recognition without UI interaction; UI base64 path always available as fallback |
