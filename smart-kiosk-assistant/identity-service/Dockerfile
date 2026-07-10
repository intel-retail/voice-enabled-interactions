# identity-service — multimodal biometric authentication (Face ID + Voiceprint).
#
# Base image bundles Intel OpenVINO + DL Streamer + GStreamer, matching the
# gvadetect/gvaclassify pipelines and providing the OpenVINO runtime used for
# face/voice embedding extraction (Phase 4+).
FROM intel/dlstreamer:2026.1.0-ubuntu24

# The dlstreamer image runs as a non-root user; switch to root to install deps.
USER root

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

# Python deps. OpenVINO/GStreamer/DL Streamer come from the base image, so we
# only install the service-level packages here. Ubuntu 24.04 enforces PEP 668,
# hence --break-system-packages for the system interpreter.
COPY identity-service/requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Application code.
COPY identity-service/ ${APP_HOME}/

# Writable data dir for the shared SQLite DB + FAISS indices (mounted volumes
# override these at runtime).
RUN mkdir -p /app/data/identity && chown -R 1000:1000 /app

EXPOSE 8013

# Match the kiosk image convention (baked-in app user 1000:1000).
USER 1000:1000

HEALTHCHECK --interval=15s --timeout=10s --start-period=120s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8013/health', timeout=5).read()" || exit 1

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8013"]
