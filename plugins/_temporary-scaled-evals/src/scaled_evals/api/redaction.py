# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secret redaction helpers for logs and uploaded text artifacts."""

from __future__ import annotations

import json
import re
from typing import Any

_SECRET_NAMES = (
    r"policy_api_key|openai_api_key|anthropic_api_key|ngc_inference_api_key|"
    r"daytona_api_key|api[_-]?key|access[_-]?token|sandbox_oc_token|bearer_token|"
    r"authorization|database_url|secret|password"
)
_SECRET_KEY_RE = re.compile(rf"(?i)(?:{_SECRET_NAMES})")
_ENV_REFERENCE_RE = re.compile(r"(?:\$[A-Z_][A-Z0-9_]*|\$\{[A-Z_][A-Z0-9_]*\})")
_ASSIGNMENT_RE = re.compile(
    rf"(?i)(\b(?:{_SECRET_NAMES})\b\s*[:=]\s*)"
    r"(\$\{[A-Z_][A-Z0-9_]*\}|[^,\s'\"}]+)"
)
_QUOTED_SECRET_FIELD_RE = re.compile(
    rf'(?i)("(?:{_SECRET_NAMES})"\s*:\s*")'
    r'((?:\\.|[^"\\\r\n])*)("?)'
)
_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|nvapi-[A-Za-z0-9._-]{8,}|"
    r"sha256~[A-Za-z0-9._~-]{8,}|"
    r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\b"
)
_DATABASE_URL_RE = re.compile(r"(?i)\b((?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?)://[^\s:/@]+:)([^\s@]+)(@)")


def redact_secret_text(text: str) -> str:
    """Mask common API-key shapes without needing the original secret value."""
    text = _ASSIGNMENT_RE.sub(_redact_assignment, text)
    text = _TOKEN_RE.sub("<redacted>", text)
    return _DATABASE_URL_RE.sub(r"\1<redacted>\3", text)


def redact_json_text(text: str, *, lines: bool = False) -> str:
    """Redact decoded values so regex matches cannot consume JSON escape syntax."""
    if lines:
        return "".join(redact_json_text(line) for line in text.splitlines(keepends=True))
    try:
        try:
            value = json.loads(text)
        except ValueError:
            # Preserve protection for malformed or partially written original artifacts.
            text = _QUOTED_SECRET_FIELD_RE.sub(r"\1<redacted>\3", text)
            return redact_secret_text(text)
        redacted = _redact_json_value(value)
        if redacted == value:
            return text
        suffix = "\r\n" if text.endswith("\r\n") else "\n" if text.endswith("\n") else ""
        # A trailing JSONL record separator does not make a record multiline.
        indent = 2 if "\n" in text.strip() or "\r" in text.strip() else None
        return json.dumps(redacted, ensure_ascii=False, indent=indent) + suffix
    except RecursionError as exc:
        # Text fallback would miss structured secret fields in deeply nested JSON.
        raise ValueError("JSON artifact nesting exceeds redaction capacity") from exc


def _redact_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        collision_suffix = 1
        for key, item in value.items():
            redacted_key = redact_secret_text(key)
            candidate = redacted_key
            if redacted_key != key:
                # Reserve original names as well as names already emitted.
                while candidate in value or candidate in result:
                    collision_suffix += 1
                    candidate = f"{redacted_key}#{collision_suffix}"
            result[candidate] = (
                "<redacted>"
                if _SECRET_KEY_RE.fullmatch(key) and item is not None and not _is_environment_reference(item)
                else _redact_json_value(item)
            )
        return result
    if isinstance(value, list):
        return [_redact_json_value(item) for item in value]
    return redact_secret_text(value) if isinstance(value, str) else value


def _redact_assignment(match: re.Match[str]) -> str:
    value = match.group(2)
    if _is_environment_reference(value):
        return match.group(0)
    return f"{match.group(1)}<redacted>"


def _is_environment_reference(value: Any) -> bool:
    return isinstance(value, str) and _ENV_REFERENCE_RE.fullmatch(value) is not None
