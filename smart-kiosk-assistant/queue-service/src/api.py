"""FastAPI application for the queue-service HTTP API.

Exposes:
  GET /health                   — liveness probe
  GET /api/v1/queue/count       — latest queue count + status (JSON)
  GET /stream                   — MJPEG live video stream

The uvicorn server is started in a daemon thread from ``main.py`` before the
GLib main loop starts so both coexist in the same process.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

import frame_buffer
import queue_state

logger = logging.getLogger(__name__)

app = FastAPI(title="queue-service", version="1.0.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/queue/count")
async def get_queue_count() -> JSONResponse:
    """Return the latest queue count with LOW / MEDIUM / HIGH status label."""
    return JSONResponse(content=queue_state.get())


@app.get("/stream")
async def mjpeg_stream() -> StreamingResponse:
    """Deliver a multipart/x-mixed-replace MJPEG stream.

    The browser renders this with a plain ``<img src="/stream">``.  Each frame
    is sent as soon as it is available; when no new frame has arrived the
    server waits up to 100 ms before re-sending the same frame so the
    connection stays alive without spinning.
    """

    async def generator():
        last: bytes | None = None
        while True:
            frame = frame_buffer.get_jpeg()
            if frame is None:
                # No frame yet — wait a little and retry
                await asyncio.sleep(0.1)
                continue
            if frame is not last:
                last = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            else:
                # Same frame — keep connection alive without flooding
                await asyncio.sleep(0.033)

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
