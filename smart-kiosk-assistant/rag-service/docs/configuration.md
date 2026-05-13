# Configuration

## Config File

Primary settings live in `config.yaml`.

### `server`

| Key | Default | Description |
|---|---|---|
| `host` | `0.0.0.0` | Bind address |
| `port` | `8020` | Service port |

### `api`

| Key | Default | Description |
|---|---|---|
| `cors_allow_origins` | `http://127.0.0.1`, `http://localhost` | Allowed browser origins |
| `openai_model_name` | `smart-kiosk-rag` | Model name returned from OpenAI-compatible APIs |

### `models.llm`

| Key | Default | Description |
|---|---|---|
| `hf_id` | `OpenVINO/Qwen2.5-3B-Instruct-int8-ov` | LLM repo or HF model ID |
| `device` | `GPU` | OpenVINO device for generation |
| `weight_format` | `int8` | Export precision for non-OpenVINO source models |
| `models_base_path` | `./models/llm` | Local model cache root |
| `hf_token` | `null` | Optional token for gated models |
| `model_path` | `null` | Explicit local model override |
| `max_new_tokens` | `220` | Default generation cap |
| `temperature` | `0.0` | Default generation temperature |
| `semantic_chunking_max_new_tokens` | `900` | Cap for LLM chunking passes |

### `models.embedding`

| Key | Default | Description |
|---|---|---|
| `provider` | `sentence_transformers` | `sentence_transformers` or `openvino` |
| `hf_id` | `BAAI/bge-base-en-v1.5` | Embedding model ID |
| `device` | `CPU` | Intended embedding runtime device |
| `models_base_path` | `./models/embeddings` | Local embedding cache root |
| `weight_format` | `fp16` | Export precision when `provider=openvino` |
| `normalize_embeddings` | `true` | Use normalized vectors for retrieval |

### `storage`

| Key | Default | Description |
|---|---|---|
| `persist_directory` | `./storage/vector_db` | Chroma persistence directory |
| `collection_name` | `smart-kiosk-assistant` | Chroma collection name |

### `retrieval`

| Key | Default | Description |
|---|---|---|
| `top_k` | `5` | Retrieved chunks used for prompting |
| `fetch_k` | `10` | Candidate documents fetched from vector store |
| `max_context_chars` | `12000` | Max prompt budget allocated to retrieved context |
| `score_threshold` | `null` | Optional numeric cutoff for Chroma scores |

### `chunking`

| Key | Default | Description |
|---|---|---|
| `strategy` | `semantic_llm` | `semantic_llm`, `semantic_embedding`, or `recursive` |
| `max_chunk_chars` | `1200` | Target upper bound per chunk |
| `min_chunk_chars` | `180` | Small chunk merge threshold |
| `overlap_chars` | `120` | Character overlap added between adjacent chunks |
| `semantic_similarity_threshold` | `0.72` | Boundary threshold for embedding chunking |
| `llm_passage_chars` | `6000` | Max passage size per LLM chunking pass |

### `answering`

| Key | Default | Description |
|---|---|---|
| `system_prompt` | retail assistant prompt | Base instruction for answers |
| `fallback_to_general_knowledge` | `true` | Allow non-store fallback when retrieved context is weak |
| `include_source_markers` | `false` | Add source labels directly into prompt blocks |

## Environment Overrides

Use double underscores to target nested keys.

Examples:

```bash
SMART_KIOSK_RAG__MODELS__LLM__DEVICE=CPU
SMART_KIOSK_RAG__RETRIEVAL__TOP_K=8
SMART_KIOSK_RAG__CHUNKING__STRATEGY=semantic_embedding
```

Use `SMART_KIOSK_RAG_CONFIG_OVERRIDE_PATHS` for comma-separated YAML override files.
