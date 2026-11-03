# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The token usage and served model an inference API's response body or stream chunk reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import JsonValue


class CallApi(StrEnum):
    """The inference API a call used, from its path."""

    CHAT_COMPLETIONS = "chat.completions"
    RESPONSES = "responses"
    MESSAGES = "messages"
    OTHER = "other"

    @classmethod
    def from_path(cls, path: str) -> CallApi:
        path = path.strip("/").removeprefix("v1/")
        if path == "chat/completions":
            return cls.CHAT_COMPLETIONS
        if path == "responses":
            return cls.RESPONSES
        if path == "messages":
            return cls.MESSAGES
        return cls.OTHER


@dataclass
class TokenUsage:
    """Token counts for one call, by class: every input token is exactly one of uncached, cache read, or cache write."""

    uncached_input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    output_tokens: int | None = None

    def merge(self, other: TokenUsage) -> None:
        """Take every count *other* reports."""
        for name in ("uncached_input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens"):
            value = getattr(other, name)
            if value is not None:
                setattr(self, name, value)

    @property
    def input_tokens(self) -> int | None:
        if self.uncached_input_tokens is None:
            return None
        return self.uncached_input_tokens + (self.cached_input_tokens or 0) + (self.cache_write_input_tokens or 0)

    def attributes(self) -> dict[str, JsonValue]:
        counts = {
            "gen_ai.usage.input_tokens": self.input_tokens,
            "nemo.usage.uncached_input_tokens": self.uncached_input_tokens,
            "gen_ai.usage.cached_tokens": self.cached_input_tokens,
            "llm.token_count.prompt_details.cache_write": self.cache_write_input_tokens,
            "gen_ai.usage.output_tokens": self.output_tokens,
        }
        return {key: value for key, value in counts.items() if value is not None}


def _count(mapping: Any, *path: str) -> int | None:
    for key in path:
        if not isinstance(mapping, Mapping):
            return None
        mapping = mapping.get(key)
    if isinstance(mapping, int) and not isinstance(mapping, bool):
        return mapping
    return None


def _openai_usage(total_input: int | None, cached: int | None, output: int | None) -> TokenUsage:
    uncached = None if total_input is None else total_input - (cached or 0)
    return TokenUsage(uncached_input_tokens=uncached, cached_input_tokens=cached, output_tokens=output)


def usage_from_payload(api: CallApi, payload: Mapping[str, Any]) -> TokenUsage | None:
    """The usage a response body or stream chunk reports, or ``None`` if it reports none."""
    if api is CallApi.MESSAGES:
        usage = payload.get("usage")
        if payload.get("type") == "message_start" and isinstance(payload.get("message"), Mapping):
            usage = payload["message"].get("usage")
        if not isinstance(usage, Mapping):
            return None
        return TokenUsage(
            uncached_input_tokens=_count(usage, "input_tokens"),
            cached_input_tokens=_count(usage, "cache_read_input_tokens"),
            cache_write_input_tokens=_count(usage, "cache_creation_input_tokens"),
            output_tokens=_count(usage, "output_tokens"),
        )
    usage = payload.get("usage")
    if api is CallApi.RESPONSES:
        if isinstance(payload.get("response"), Mapping):
            usage = payload["response"].get("usage")
        if not isinstance(usage, Mapping):
            return None
        return _openai_usage(
            _count(usage, "input_tokens"),
            _count(usage, "input_tokens_details", "cached_tokens"),
            _count(usage, "output_tokens"),
        )
    if not isinstance(usage, Mapping):
        return None
    return _openai_usage(
        _count(usage, "prompt_tokens"),
        _count(usage, "prompt_tokens_details", "cached_tokens"),
        _count(usage, "completion_tokens"),
    )


def model_from_payload(payload: Mapping[str, Any]) -> str | None:
    """The model a response body or stream chunk says served the call, or ``None`` if it names none."""
    for envelope in ("response", "message"):
        if isinstance(payload.get(envelope), Mapping):
            payload = payload[envelope]
            break
    model = payload.get("model")
    return model if isinstance(model, str) and model else None
