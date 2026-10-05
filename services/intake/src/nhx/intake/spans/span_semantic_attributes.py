# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed semantic span attributes normalized from source conventions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Self

from nhx.intake.spans.span_attribute_bags import SpanAttributeBags
from nhx.intake.spans.span_attribute_catalog import ATTRIBUTE_SPECS, to_semantic_value
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class NormalizedSpanAttributes:
    semantic: SpanSemanticAttributes
    source_attributes: dict[str, Any]
    consumed_keys: set[str]


class SpanSemanticAttributes(BaseModel):
    model: str | None = None
    provider: str | None = None
    prompt_id: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    agent_version: str | None = None
    tool_name: str | None = None
    project: str | None = None
    evaluation_name: str | None = None
    test_case_name: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    prompt_cache_write_tokens: int | None = Field(default=None, ge=0)
    prompt_audio_tokens: int | None = Field(default=None, ge=0)
    completion_reasoning_tokens: int | None = Field(default=None, ge=0)
    completion_audio_tokens: int | None = Field(default=None, ge=0)
    cost_total_usd: Decimal | None = None
    cost_input_usd: Decimal | None = None
    cost_output_usd: Decimal | None = None

    @classmethod
    def from_source_attributes(cls, attributes: Mapping[str, Any]) -> tuple[Self, set[str]]:
        # consumed_keys tracks every alias the catalog claimed so unhandled-attribute fallback skips them.
        values: dict[str, Any] = {}
        consumed_keys: set[str] = set()
        for spec in ATTRIBUTE_SPECS:
            present_source_keys = [key for key in spec.source_keys if key in attributes]
            consumed_keys.update(present_source_keys)
            for key in present_source_keys:
                value = to_semantic_value(attributes[key], spec)
                if value is not None:
                    values[spec.field.value] = value
                    break
        return cls(**values), consumed_keys

    @classmethod
    def from_source_attribute_layers(
        cls,
        *,
        resource_attributes: Mapping[str, Any],
        span_attributes: Mapping[str, Any],
    ) -> NormalizedSpanAttributes:
        # OTLP span attributes override resource attributes when both layers publish the same key.
        attributes = {**resource_attributes, **span_attributes}
        semantic_attributes, consumed_keys = cls.from_source_attributes(attributes)
        return NormalizedSpanAttributes(
            semantic=semantic_attributes,
            source_attributes=attributes,
            consumed_keys=consumed_keys,
        )

    @classmethod
    def from_bags(cls, bags: SpanAttributeBags) -> Self:
        values: dict[str, Any] = {}
        for spec in ATTRIBUTE_SPECS:
            value = bags.get_field(spec.field)
            if value is not None:
                values[spec.field.value] = value
        return cls(**values)

    def to_bags(self) -> SpanAttributeBags:
        bags = SpanAttributeBags()
        for spec in ATTRIBUTE_SPECS:
            bags.put_spec(spec, getattr(self, spec.field.value))
        return bags
