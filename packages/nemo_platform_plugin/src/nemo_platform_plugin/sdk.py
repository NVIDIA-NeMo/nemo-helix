# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin SDK resource container."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from nemo_platform import AsyncNeMoPlatform, NeMoPlatform

SyncResourceT = TypeVar("SyncResourceT")
AsyncResourceT = TypeVar("AsyncResourceT")


@dataclass(frozen=True, slots=True)
class NemoPluginSDKResources(Generic[SyncResourceT, AsyncResourceT]):
    """Container for plugin SDK resources exposed on legacy platform SDK owners.

    Typed clients should expose resources through explicit typed APIs instead
    of consuming this dynamic legacy ``nemo.sdk`` entry-point surface.
    """

    sync_resource: Callable[[NeMoPlatform], SyncResourceT] | None = None
    async_resource: Callable[[AsyncNeMoPlatform], AsyncResourceT] | None = None

    def __post_init__(self) -> None:
        if self.sync_resource is None and self.async_resource is None:
            raise ValueError("At least one of sync_resource or async_resource must be provided")


__all__ = [
    "NemoPluginSDKResources",
]
