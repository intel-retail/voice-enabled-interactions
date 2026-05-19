FROM openvino/ubuntu22_runtime:2026.1.0

USER root

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/rag-service/.cache/huggingface

WORKDIR /app/rag-service

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r ragservice \
    && useradd -r -g ragservice -d /app/rag-service ragservice

COPY requirements.txt ./requirements.txt

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/rag-service/models \
    /app/rag-service/storage \
    /app/rag-service/.cache/huggingface \
    /app/rag-service/.cache/matplotlib \
    && chown -R ragservice:ragservice /app/rag-service

USER ragservice

EXPOSE 8020

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8020/health', timeout=5).read()"

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8020"]
