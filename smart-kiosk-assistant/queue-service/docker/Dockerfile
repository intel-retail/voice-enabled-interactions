# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# DLStreamer 2026.2 is REQUIRED, not merely preferred.
#
# The YOLO26 IR loads and runs correctly under the OpenVINO 2026.2 runtime
# bundled here, but on 2026.1 gvainference aborts at caps negotiation with
# "post-processing is NULL" / "Mapper is null" and the pipeline never
# prerolls. Verified with the identical yolo26n_static.xml in both images:
#
#   intel/dlstreamer:2026.1.0-ubuntu24      (OpenVINO 2026.1) -> fails
#   intel/dlstreamer:2026.2.0-ubuntu24-rc2  (OpenVINO 2026.2) -> works
#   intel/dlstreamer:2026.2.0-ubuntu24-rc3  (OpenVINO 2026.2) -> works
#
# person-detection-retail-0013 loads fine on both, so a green pipeline with
# the old detector says nothing about this. Do not downgrade this tag while
# conf/queue-config.yaml has `postproc: decoder`.
FROM intel/dlstreamer:2026.2.0-ubuntu24-rc3

ARG APP_UID=1000
ARG APP_GID=1000

USER root

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app/queue-service

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python3 -m venv --system-site-packages "$VIRTUAL_ENV" \
    && "$VIRTUAL_ENV/bin/pip" install --upgrade pip \
    && "$VIRTUAL_ENV/bin/pip" install -r requirements.txt

COPY . .

RUN mkdir -p /app/queue-service/models \
    && chown -R ${APP_UID}:${APP_GID} /app/queue-service

USER ${APP_UID}:${APP_GID}

EXPOSE 8090

ENTRYPOINT ["python3", "src/main.py"]
