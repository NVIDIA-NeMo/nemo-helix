# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin SDK resource container."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient

SyncHelixT = TypeVar("SyncHelixT", bound=NemoClient)
AsyncHelixT = TypeVar("AsyncHelixT", bound=AsyncNemoClient)
SyncResourceT = TypeVar("SyncResourceT")
AsyncResourceT = TypeVar("AsyncResourceT")


@dataclass(frozen=True, slots=True)
class NemoPluginSDKResources(Generic[SyncHelixT, SyncResourceT, AsyncHelixT, AsyncResourceT]):
    """Factories that build a plugin's SDK resource from a typed platform client.

    ``sync_resource`` receives the owning :class:`NemoClient`; ``async_resource``
    receives the owning :class:`AsyncNemoClient`. ``nemo.sdk`` entry points expose
    these factories so ``client.<plugin>`` resolves the plugin's resource namespace,
    except where ``<plugin>`` is already a typed service-client property of the
    client (``agents``, ``auditor``, ``evaluator``, ``data_designer``,
    ``agent_hardener``). Build those resources explicitly, e.g.
    ``AuditorPluginResource(client)``.
    """

    sync_resource: Callable[[SyncHelixT], SyncResourceT] | None = None
    async_resource: Callable[[AsyncHelixT], AsyncResourceT] | None = None

    def __post_init__(self) -> None:
        if self.sync_resource is None and self.async_resource is None:
            raise ValueError("At least one of sync_resource or async_resource must be provided")


__all__ = ["NemoPluginSDKResources"]
