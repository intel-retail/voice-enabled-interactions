# Run In Docker

Use this path when you want the service to run in an Intel OpenVINO runtime container.

## Start

From the `rag-service/` directory:

```bash
docker compose up -d --build
```

This publishes:

- API: `http://127.0.0.1:8020`

## Verify

```bash
curl --noproxy '*' http://127.0.0.1:8020/health
```

## Notes

- The compose file passes `/dev/dri` and adds the `video` group so OpenVINO GPU can be used from the container.
- `config.container.yaml` is mounted read-only and layered on top of `config.yaml`.
- Model cache and vector storage are persisted through bind mounts in `models/`, `storage/`, and `.cache/huggingface/`.
