# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared artifact helpers for optimizer backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_SECRET_VALUE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "password",
        "passwd",
        "secret",
        "token",
        "authorization",
        "access_token",
        "refresh_token",
        "client_secret",
        "nvidia_api_key",
    }
)


def sanitize_config_for_artifact(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy with secret-bearing fields redacted for persistent YAML/JSON."""

    def _redact(value: Any, *, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {str(k): _redact(v, key=str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [_redact(v, key=key) for v in value]
        if key is not None and isinstance(value, str) and value and not value.startswith("${"):
            if _is_secret_value_key(key):
                return "${REDACTED}"
        return value

    return _redact(config)


def _is_secret_value_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    # Reference fields hold env/secret *names*, not credentials.
    if lowered.endswith("_env") or lowered in {"api_key_secret", "api_key_env"}:
        return False
    if lowered in _SECRET_VALUE_KEYS:
        return True
    return any(lowered.endswith(f"_{suffix}") for suffix in ("api_key", "password", "token", "secret"))


__all__ = ["sanitize_config_for_artifact"]
