# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Redaction of credential-looking values before free-form runner settings are recorded as provenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_SECRET_KEY_MARKERS = ("api_key", "apikey", "token", "secret", "password", "passwd", "credential")
_REDACTED = "<redacted>"


def redact_credentials(settings: Mapping[str, Any], _prefix: str = "") -> dict[str, Any]:
    """Redact credential-looking values from free-form settings before they are recorded as provenance.

    Runner configs carry escape hatches forwarded verbatim to the harness (Gym's ``hydra_params``,
    Harbor's ``agent_kwargs``), so nothing stops a caller passing ``{"model": {"api_key": "sk-..."}}``.
    ``RunnerInfo.config`` is persisted into the run bundle, so a value that looks like a credential
    must not be written there.

    The *key* is always kept — knowing that a run set ``model.api_key`` is useful provenance; knowing
    the value is a leak. Matching is on the full dotted path, so a marker anywhere in it redacts, and
    nesting cannot hide a credential behind an innocuous leaf name.

    Lists are walked too, since a mapping inside one — ``{"models": [{"api_key": "sk-..."}]}`` —
    reaches the harness just as a nested mapping does. The index contributes no path segment: what
    marks a value as a credential is the key it sits under, not where in a list it happens to fall.
    """
    redacted: dict[str, Any] = {}
    for key, value in settings.items():
        path = f"{_prefix}{key}"
        if isinstance(value, Mapping):
            redacted[key] = redact_credentials(value, f"{path}.")
        elif any(marker in path.casefold() for marker in _SECRET_KEY_MARKERS):
            redacted[key] = _REDACTED
        elif isinstance(value, (list, tuple)):
            redacted[key] = [_redact_list_item(item, path) for item in value]
        else:
            redacted[key] = value
    return redacted


def _redact_list_item(item: Any, path: str) -> Any:
    """Redact inside one element of a list-valued setting. See :func:`redact_credentials`."""
    if isinstance(item, Mapping):
        return redact_credentials(item, f"{path}.")
    if isinstance(item, (list, tuple)):
        return [_redact_list_item(nested, path) for nested in item]
    return item
