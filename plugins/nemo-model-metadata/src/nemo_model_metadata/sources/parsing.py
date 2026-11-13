# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_helix_plugin.models.types import ModelModality

MAX_DISPLAY_NAME_LENGTH = 255
MAX_DESCRIPTION_LENGTH = 4000

_KNOWN_MODALITIES = {modality.value for modality in ModelModality}
_MODALITY_ALIASES = {"file": ModelModality.PDF}


def bounded_text(value: object, max_length: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:max_length]


def as_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def modalities(values: object) -> list[ModelModality]:
    if not isinstance(values, list):
        return []
    result: list[ModelModality] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.lower()
        modality = _MODALITY_ALIASES.get(normalized)
        if modality is None and normalized in _KNOWN_MODALITIES:
            modality = ModelModality(normalized)
        if modality is not None and modality not in result:
            result.append(modality)
    return result
