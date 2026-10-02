# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Dockerfile — kiosk-core / kiosk-ui
#
# Default CMD starts the kiosk-core FastAPI service on port 8012.
# The kiosk-ui React app (kiosk-ui/) is served separately via its own container.
FROM python:3.12-slim

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libportaudio2 is required by the sounddevice wheel at import time even
# when no mic device is used (Gradio file-upload mode). git is required by
# pip to install the vlm_metrics_logger dependency straight from
# performance-tools' git repo (see requirements.txt).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 curl git \
    && groupadd --gid ${APP_GID} appuser \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --home-dir /app appuser \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

# openwakeword==0.6.0 unconditionally declares a tflite-runtime dependency
# that has no PyPI wheels for Python 3.12+. kiosk-core only uses
# openwakeword's ONNX inference path by default
# (KIOSK_CORE_WAKEWORD_INFERENCE_FRAMEWORK=onnx), so tflite-runtime is never
# actually needed. Its real runtime dependencies are pinned in
# requirements.txt; openwakeword itself is installed here afterwards with
# --no-deps so pip does not try to resolve tflite-runtime. See
# requirements.txt for the full rationale.
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --no-deps openwakeword==0.6.0

COPY . .

RUN mkdir -p /app/generated_audio /app/data /app/conversations \
    && chown -R ${APP_UID}:${APP_GID} /app

USER appuser

EXPOSE 8012

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8012", "--log-level", "info"]
