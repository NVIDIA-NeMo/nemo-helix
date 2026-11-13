# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass

from nemo_helix_plugin.models.types import ModelMetadata, ModelProvider


@dataclass(frozen=True)
class MetadataTarget:
    """A model as its upstream provider knows it."""

    upstream_model_id: str
    provider: ModelProvider | None


class MetadataSource(ABC):
    name: str

    def begin_cycle(self) -> None:
        """Drop per-cycle caches before a reconcile pass."""

    @abstractmethod
    async def fetch(self, target: MetadataTarget) -> ModelMetadata | None:
        """Return what this source knows about the model, or ``None`` when it knows nothing."""
