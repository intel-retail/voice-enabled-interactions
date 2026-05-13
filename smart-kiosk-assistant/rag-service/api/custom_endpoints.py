from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from dto.query_dto import ContextRequest, IngestResponse, QueryRequest
from pipeline import get_shared_pipeline


router = APIRouter()

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/v1/context", response_model=IngestResponse)
def ingest_context(request: ContextRequest) -> IngestResponse:
    pipeline = get_shared_pipeline()
    added = pipeline.ingest_text(request.text, source=request.source, metadata=request.metadata)
    return IngestResponse(chunks_added=added, source=request.source)


@router.post("/api/v1/context/file", response_model=IngestResponse)
async def ingest_context_file(file: UploadFile = File(...)) -> IngestResponse:
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md"}:
        raise HTTPException(status_code=415, detail="Only .txt and .md files are supported")

    raw = await file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="Context file must be UTF-8 encoded") from exc

    pipeline = get_shared_pipeline()
    added = pipeline.ingest_text(text, source=filename)
    return IngestResponse(chunks_added=added, source=filename)


@router.get("/api/v1/context/stats")
def context_stats():
    return JSONResponse(content=get_shared_pipeline().get_stats(), status_code=200)


@router.delete("/api/v1/context")
def clear_context():
    get_shared_pipeline().clear_context()
    return JSONResponse(content={"status": "cleared"}, status_code=200)


@router.post("/api/v1/query")
def query_context(request: QueryRequest) -> StreamingResponse:
    pipeline = get_shared_pipeline()
    prompt, sources = pipeline.plan_answer(
        request.transcription,
        context_text=request.context_text,
        top_k=request.top_k,
    )

    def _sse_generator():
        answer_tokens: list[str] = []
        try:
            for token in pipeline.stream_from_prompt(prompt):
                answer_tokens.append(token)
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            if request.include_sources and sources:
                payload = {
                    "event": "sources",
                    "sources": pipeline.source_payloads(sources),
                    "answer": "".join(answer_tokens).strip(),
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
