"""Configuration loader for queue-service.

Loads ``conf/queue-config.yaml`` and applies ``QUEUE_SERVICE__SECTION__KEY``
environment overrides, exposing the result as a nested ``SimpleNamespace``.
Mirrors the loader used by the existing microservices (see
``edge-ai-libraries/microservices/audio-analyzer/utils/config_loader.py``).
"""
import json
import os
from types import SimpleNamespace

import yaml

# queue-service/ (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PREFIX = "QUEUE_SERVICE__"
CONFIG_PATH_ENV = "QUEUE_SERVICE_CONFIG_PATH"
DEFAULT_CONFIG_PATH = "conf/queue-config.yaml"


def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _load_yaml_file(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def _parse_env_value(raw_value, existing_value):
    if isinstance(existing_value, bool):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}

    if isinstance(existing_value, int) and not isinstance(existing_value, bool):
        return int(raw_value)

    if isinstance(existing_value, float):
        return float(raw_value)

    if isinstance(existing_value, list):
        stripped = raw_value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            return json.loads(stripped)
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    return raw_value


def _set_nested_value(data, path, value):
    current = data
    for segment in path[:-1]:
        next_value = current.get(segment)
        if not isinstance(next_value, dict):
            next_value = {}
            current[segment] = next_value
        current = next_value
    current[path[-1]] = value


def _apply_env_overrides(data):
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue

        path = [seg.lower() for seg in env_key[len(ENV_PREFIX):].split("__") if seg]
        if not path:
            continue

        existing_value = data
        for segment in path:
            if isinstance(existing_value, dict) and segment in existing_value:
                existing_value = existing_value[segment]
            else:
                existing_value = None
                break

        _set_nested_value(data, path, _parse_env_value(raw_value, existing_value))

    return data


def _dict_to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in value.items()})
    return value


def load_config(path=DEFAULT_CONFIG_PATH):
    path = _resolve_path(os.environ.get(CONFIG_PATH_ENV, path))
    data = _load_yaml_file(path)
    data = _apply_env_overrides(data)
    return _dict_to_namespace(data)


# Load once and expose.
config = load_config()
