# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable helpers for metrics that declare resolvable references."""

from __future__ import annotations

from typing import Any, get_args

from nhx_evals_sdk.resolver_protocols import ModelResolver
from nhx_evals_sdk.values.models import ModelRef
from pydantic import BaseModel


def _annotation_contains_model_ref(annotation: Any) -> bool:
    """Return whether a field annotation allows ``ModelRef``."""
    if annotation is ModelRef:
        return True
    return any(_annotation_contains_model_ref(arg) for arg in get_args(annotation))


def model_ref_fields(metric: BaseModel) -> tuple[str, ...]:
    """Return fields annotated as allowing ``ModelRef`` values."""
    return tuple(
        field_name
        for field_name, field_info in type(metric).model_fields.items()
        if _annotation_contains_model_ref(field_info.annotation)
    )


def collect_model_refs(metric: BaseModel, fields: tuple[str, ...] | None = None) -> dict[str, ModelRef]:
    """Return model-reference-bearing fields that currently hold ``ModelRef`` values."""
    refs: dict[str, ModelRef] = {}
    for field_name in fields if fields is not None else model_ref_fields(metric):
        value = getattr(metric, field_name, None)
        if isinstance(value, ModelRef):
            refs[field_name] = value
    return refs


async def resolve_model_refs(
    metric: BaseModel,
    model_resolver: ModelResolver,
    fields: tuple[str, ...] | None = None,
) -> None:
    """Resolve model-reference-bearing fields in place."""
    for field_name, model_ref in collect_model_refs(metric, fields).items():
        setattr(metric, field_name, await model_resolver.resolve_model(model_ref))
