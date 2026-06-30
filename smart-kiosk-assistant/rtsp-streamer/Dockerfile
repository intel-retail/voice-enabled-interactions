# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Dockerfile — rtsp-streamer
#
# Publishes every *.mp4 under $MEDIA_DIR (/media) as an RTSP stream via
# MediaMTX + ffmpeg. Driven entirely by start.sh.
#
#   Stream path = <filename-without-extension>
#   e.g. /media/queue.mp4  ->  rtsp://<host>:8554/queue
FROM debian:stable-slim

ARG APP_UID=1000
ARG APP_GID=1000

# Pin the MediaMTX release. Override at build time with:
#   --build-arg MEDIAMTX_VERSION=v1.11.3 --build-arg MEDIAMTX_ARCH=linux_arm64v8
ARG MEDIAMTX_VERSION=v1.11.3
ARG MEDIAMTX_ARCH=linux_amd64

ENV DEBIAN_FRONTEND=noninteractive \
    MEDIA_DIR=/media \
    RTSP_PORT=8554 \
    MEDIAMTX_BIN=/opt/rtsp-streamer/mediamtx

# Runtime deps:
#   ffmpeg          -> stream publishing / black-frame generation
#   netcat-openbsd  -> `nc -z` readiness + healthcheck probe
#   ca-certificates -> HTTPS download of the MediaMTX release
#   curl, tar       -> fetch + unpack the MediaMTX binary (build only)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        netcat-openbsd \
        ca-certificates \
        curl \
        tar \
    && rm -rf /var/lib/apt/lists/*

# Download and install MediaMTX to the path start.sh expects.
RUN mkdir -p /opt/rtsp-streamer \
    && curl -fsSL \
        "https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/mediamtx_${MEDIAMTX_VERSION}_${MEDIAMTX_ARCH}.tar.gz" \
        -o /tmp/mediamtx.tar.gz \
    && tar -xzf /tmp/mediamtx.tar.gz -C /opt/rtsp-streamer mediamtx \
    && rm -f /tmp/mediamtx.tar.gz \
    && chmod +x "${MEDIAMTX_BIN}"

# Install the entrypoint script and the MediaMTX configuration.
# The catch-all mediamtx.yml is required so MediaMTX accepts the dynamic
# publish paths that start.sh derives from video filenames; without it
# MediaMTX loads an empty configuration and rejects every publish with
# "path '<name>' is not configured".
COPY start.sh /opt/rtsp-streamer/start.sh
COPY mediamtx.yml /opt/rtsp-streamer/mediamtx.yml
RUN chmod +x /opt/rtsp-streamer/start.sh

# WORKDIR = MediaMTX binary directory so the server auto-discovers
# ./mediamtx.yml when start.sh launches it with no config argument. This
# preserves start.sh's existing `"$MEDIAMTX_BIN"` invocation unchanged.
WORKDIR /opt/rtsp-streamer

# Run as a non-root user (matches repo convention: UID/GID 1000).
RUN groupadd --gid ${APP_GID} appuser \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --home-dir /home/appuser appuser \
    && mkdir -p "${MEDIA_DIR}" \
    && chown -R ${APP_UID}:${APP_GID} /opt/rtsp-streamer "${MEDIA_DIR}"

USER ${APP_UID}:${APP_GID}

EXPOSE 8554

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
    CMD nc -z 127.0.0.1 8554 || exit 1

ENTRYPOINT ["/opt/rtsp-streamer/start.sh"]