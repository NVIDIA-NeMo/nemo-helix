# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from nemo_helix_plugin.models.types import ModelMetadata

_MERGED_FIELDS = tuple(name for name in ModelMetadata.model_fields if name not in ("sources", "refreshed_at"))


def _is_empty(value: object) -> bool:
    return value is None or value == []


def merge_metadata(results: list[tuple[str, ModelMetadata]], refreshed_at: datetime) -> ModelMetadata | None:
    """Take each field from the first source, in priority order, that has a value for it."""
    merged: dict[str, object] = {}
    sources: list[str] = []
    for source_name, metadata in results:
        contributed = False
        for field_name in _MERGED_FIELDS:
            value = getattr(metadata, field_name)
            if field_name not in merged and not _is_empty(value):
                merged[field_name] = value
                contributed = True
        if contributed:
            sources.append(source_name)
    if not sources:
        return None
    return ModelMetadata.model_validate({**merged, "sources": sources, "refreshed_at": refreshed_at})
