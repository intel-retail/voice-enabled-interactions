# Run In Docker

Use this path when you want the service to run in an Intel OpenVINO runtime container.

## Start

From the `rag-service/` directory:

```bash
docker compose up -d --build
```

The built image will be tagged `rag-service:${RELEASE_TAG}` (e.g. `rag-service:v2026.1.0-rc1`). `RELEASE_TAG` is read from `.env` (defaults to `latest` if unset); the committed `.env` pins the current release tag.

This publishes:

- API: `http://127.0.0.1:8020`

## Verify

```bash
curl --noproxy '*' http://127.0.0.1:8020/health
```

## Notes

- The compose file passes `/dev/dri` and adds the `video` group so OpenVINO GPU can be used from the container.
- [config.yaml](../config.yaml) is the single source of truth; the same file drives standalone and container runs.
- Model cache and vector storage are persisted through bind mounts in `models/`, `storage/`, and `.cache/huggingface/`.
