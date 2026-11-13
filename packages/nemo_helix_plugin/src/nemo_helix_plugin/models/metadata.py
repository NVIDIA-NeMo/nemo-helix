# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extension point for model metadata sources.

For each ModelProvider the Models service asks the one source that handles it: a plugin source registered
under ``nemo.models.metadata_sources``, or the built-in source that reads the provider's ``GET /v1/models``::

    [project.entry-points."nemo.models.metadata_sources"]
    ngc = "nemo_ngc_plugin.metadata:NgcMetadataSource"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import httpx
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.models.types import ModelMetadata, ModelModality, ModelProvider

METADATA_SOURCES_GROUP = "nemo.models.metadata_sources"


def _max_length(field_name: str) -> int:
    for constraint in ModelMetadata.model_fields[field_name].metadata:
        max_length = getattr(constraint, "max_length", None)
        if isinstance(max_length, int):
            return max_length
    raise ValueError(f"ModelMetadata.{field_name} has no max_length")


MAX_DISPLAY_NAME_LENGTH = _max_length("display_name")
MAX_DESCRIPTION_LENGTH = _max_length("description")

_KNOWN_MODALITIES = {modality.value for modality in ModelModality}
_MODALITY_ALIASES = {"file": ModelModality.PDF}


class SourceUnavailableError(Exception):
    """The source could not answer, so a missing result says nothing about the model."""


@dataclass(frozen=True)
class MetadataTarget:
    """A model as its provider knows it."""

    upstream_model_id: str
    provider: ModelProvider


@dataclass(frozen=True)
class MetadataSourceContext:
    """Clients a source may use. ``client`` is the Models service's platform client, for typed service clients."""

    http: httpx.AsyncClient
    client: AsyncNemoClient


class NemoModelMetadataSource(ABC):
    """Reads metadata for the models of the providers it handles."""

    name: ClassVar[str]

    @abstractmethod
    def handles(self, provider: ModelProvider) -> bool:
        """Whether this source is the one to ask about models served by ``provider``."""

    def begin_cycle(self) -> None:
        """Drop per-cycle caches before a reconcile pass."""

    @abstractmethod
    async def fetch(self, target: MetadataTarget, ctx: MetadataSourceContext) -> ModelMetadata | None:
        """Return what this source knows about the model, or ``None`` when it knows nothing.

        Raise :class:`SourceUnavailableError` when the source could not be asked, so the Models service
        keeps the current metadata and retries on the next cycle.
        """


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
