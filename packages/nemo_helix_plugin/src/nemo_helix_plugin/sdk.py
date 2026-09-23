# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin SDK resource container."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from nemo_helix_plugin.client.adapter import AsyncHelixClient, SyncHelixClient

SyncHelixT = TypeVar("SyncHelixT", bound=SyncHelixClient)
AsyncHelixT = TypeVar("AsyncHelixT", bound=AsyncHelixClient)
SyncResourceT = TypeVar("SyncResourceT")
AsyncResourceT = TypeVar("AsyncResourceT")


@dataclass(frozen=True, slots=True)
class NemoPluginSDKResources(Generic[SyncHelixT, SyncResourceT, AsyncHelixT, AsyncResourceT]):
    """Container for plugin SDK resources exposed on legacy platform SDK owners.

    ``sync_resource`` receives the owning ``NeMoHelix``; ``async_resource``
    receives the owning ``AsyncNeMoHelix``. The parameters are typed as generics
    bound to the sync/async platform protocols rather than the SDK classes so this module,
    which plugin discovery imports, does not depend on the generated SDK.
    Typed clients should expose resources through explicit typed APIs instead
    of consuming this dynamic legacy ``nemo.sdk`` entry-point surface.
    """

    sync_resource: Callable[[SyncHelixT], SyncResourceT] | None = None
    async_resource: Callable[[AsyncHelixT], AsyncResourceT] | None = None

    def __post_init__(self) -> None:
        if self.sync_resource is None and self.async_resource is None:
            raise ValueError("At least one of sync_resource or async_resource must be provided")


__all__ = [
    "AsyncNeMoHelix",  # noqa: F822  (resolved lazily by module __getattr__)
    "NeMoHelix",  # noqa: F822  (resolved lazily by module __getattr__)
    "NemoPluginSDKResources",
]


def __getattr__(name: str) -> Any:
    # Plugins import the generated SDK classes from here. Resolve them on first
    # use so plugin discovery, which imports this module for the resource
    # container, does not require the generated SDK to be installed.
    if name in ("AsyncNeMoHelix", "NeMoHelix"):
        import nemo_helix

        return getattr(nemo_helix, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
