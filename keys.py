import logging
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)
coolest_key = 0


def _load_secret_file(filename):
    """Load provider keys from a secrets file without failing at import time."""
    path = Path(filename)
    if not path.exists():
        logger.warning(
            "Secrets file '%s' not found. Keys from this provider are unavailable.",
            filename,
        )
        return {}

    try:
        with path.open("r", encoding="utf-8") as secret_file:
            secret = yaml.safe_load(secret_file) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.error("Failed to load '%s': %s", filename, exc)
        return {}

    api_keys = secret.get("api_keys", {})
    if not api_keys:
        logger.warning(
            "No 'api_keys' field found in '%s'. Keys from this provider are unavailable.",
            filename,
        )
    return api_keys


def _ordered_key_values(api_keys):
    """Normalize YAML api_keys to a stable list ordered by key index."""
    if isinstance(api_keys, list):
        return api_keys
    if not isinstance(api_keys, dict):
        return []

    def sort_key(item):
        key, _ = item
        try:
            return int(key)
        except (TypeError, ValueError):
            return str(key)

    return [value for _, value in sorted(api_keys.items(), key=sort_key)]


api_keys_openai = _load_secret_file("./secrets.yml")
api_keys_claude = _load_secret_file("./secrets_claude.yml")
api_keys_llama = _load_secret_file("./secrets_llama.yml")
api_keys_deepseek = _load_secret_file("./secrets_deepseek.yml")
api_keys_qwen = _load_secret_file("./secrets_qwen.yml")


def get_key(model):
    """Return a key such that all keys for a provider are used equally."""
    global coolest_key

    model_to_keys = {
        "openai": api_keys_openai,
        "claude": api_keys_claude,
        "deepseek_api": api_keys_deepseek,
        "deepseek": "ollama",
        "qwen": api_keys_qwen,
        "llama_api": api_keys_llama,
    }
    model_to_secret_file = {
        "openai": "secrets.yml",
        "claude": "secrets_claude.yml",
        "deepseek_api": "secrets_deepseek.yml",
        "qwen": "secrets_qwen.yml",
        "llama_api": "secrets_llama.yml",
    }

    if model not in model_to_keys:
        raise ValueError(f"Unknown model: {model}")

    if model == "deepseek":
        return model_to_keys[model]

    keys = _ordered_key_values(model_to_keys[model])
    if not keys:
        secret_file = model_to_secret_file.get(model, "the matching secrets file")
        raise RuntimeError(
            f"No API keys configured for model '{model}'. "
            f"Please add an 'api_keys' section to '{secret_file}'."
        )

    key = keys[coolest_key % len(keys)]
    coolest_key = (coolest_key + 1) % len(keys)
    return key


def get_base_url(model):
    base_urls = {
        "openai": "https://api.openai.com/v1/",
        "deepseek": "http://localhost:11434/v1/",
        "deepseek_api": "https://api.deepseek.com/v1",
        "claude": "https://api.anthropic.com/v1/complete",
        "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "llama_api": "https://api.llmapi.com/",
    }

    if model in base_urls:
        return base_urls[model]
    raise ValueError(f"Unknown model: {model}")
