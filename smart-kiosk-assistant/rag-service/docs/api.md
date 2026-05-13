# API

## Health

### `GET /health`

Returns service liveness.

## Context Ingestion

### `POST /api/v1/context`

Request body:

```json
{
  "text": "Alfonso mangoes are in Produce near the tropical fruit display.",
  "source": "store-manual",
  "metadata": {
    "department": "produce"
  }
}
```

Response:

```json
{
  "chunks_added": 1,
  "source": "store-manual"
}
```

### `POST /api/v1/context/file`

Upload a UTF-8 `.txt` or `.md` file using multipart form-data with the `file` field.

### `GET /api/v1/context/stats`

Returns collection metadata and the current stored document count.

### `DELETE /api/v1/context`

Clears the active Chroma collection.

## Kiosk Query API

### `POST /api/v1/query`

Request body:

```json
{
  "transcription": "Where can I find Alfonso mangoes?",
  "context_text": "Customer is standing near produce.",
  "top_k": 5,
  "include_sources": true
}
```

Response type: `text/event-stream`

Events:

- `data: {"token":"..."}` for streamed answer fragments
- optional final `sources` event when `include_sources=true`
- `data: [DONE]` sentinel

## OpenAI-Compatible Chat

### `POST /v1/chat/completions`

Non-stream example:

```json
{
  "model": "smart-kiosk-rag",
  "messages": [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "Where is the bakery?"}
  ],
  "stream": false
}
```

Streaming example uses the same payload with `"stream": true` and returns standard SSE `data:` events ending with `[DONE]`.
