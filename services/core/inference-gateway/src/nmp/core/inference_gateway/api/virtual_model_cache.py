# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory cache for VirtualModel entities.

VirtualModels are fetched from the IGW's own VirtualModel API via the platform SDK
(``sdk.inference.virtual_models.list(workspace="-")``), following the same pattern used by
:mod:`model_cache` for ``ModelProvider``\\ s.

The cache is refreshed by :func:`refresh_virtual_model_cache`, which is called from
inside :func:`~nmp.core.inference_gateway.api.model_cache.refresh_model_cache` so both
caches stay in sync on the same background interval with no additional configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from nemo_platform import APIConnectionError, APIStatusError, AsyncNeMoPlatform
from nemo_platform.types.inference.virtual_model import VirtualModel
from nmp.common.entities.global_workspace import workspace_lookup_order
from nmp.core.inference_gateway.api.middleware_registry import (
    MiddlewareConfigRef,
    PrefetchResult,
    collect_config_refs,
    vm_uses_any_changed_config,
)

if TYPE_CHECKING:
    from nmp.core.inference_gateway.api.middleware_registry import MiddlewareRegistry

logger = logging.getLogger(__name__)


def sync_config_ref_versions(
    known_versions: dict[MiddlewareConfigRef, datetime | None],
    current_refs: set[MiddlewareConfigRef],
    prefetch: PrefetchResult,
) -> set[MiddlewareConfigRef]:
    """Update *known_versions* from a :class:`PrefetchResult`; return the set of changed refs.

    A ref is considered changed if:

    - It is in :attr:`PrefetchResult.fetched` with a different ``updated_at`` than the
      cached value, or
    - It is in :attr:`PrefetchResult.missing` **and** still present in *known_versions*
      (i.e. this is the first cycle that observed the deletion). The plugin signalled a
      definitive deletion via
      :class:`~nemo_platform_plugin.inference_middleware.MiddlewareConfigNotFoundError`, so any
      referencing VirtualModel must re-resolve (and the registry will mark it broken).
      On subsequent cycles the ref is already absent from *known_versions*, so it is
      **not** re-flagged — the dependent VMs are already broken and re-resolving them
      every poll cycle would only produce redundant warnings.

    Refs in :attr:`PrefetchResult.transient` leave their *known_versions* entry untouched
    and are **not** added to the returned set: this is the "don't flap on transient
    failures" invariant — a brief SDK / network glitch must not invalidate a previously
    healthy resolution.

    Refs no longer present in *current_refs* (no VM references them anymore) are pruned
    from *known_versions* so the map does not grow unboundedly. Refs in
    :attr:`PrefetchResult.missing` are also pruned so a recreated config with a fresh
    ``updated_at`` re-enters the cache as "newly seen" and triggers a re-resolve.
    """
    for stale_ref in set[MiddlewareConfigRef](known_versions) - current_refs:
        del known_versions[stale_ref]

    changed: set[MiddlewareConfigRef] = set[MiddlewareConfigRef]()

    for mref in prefetch.missing:
        if mref not in known_versions:
            continue
        known_versions.pop(mref)
        changed.add(mref)
        logger.info(
            "MiddlewareConfigRef %r / %r / %r was deleted upstream. "
            "Requests to VirtualModels that still reference it will fail until the referenced config is recreated, or removed from the VirtualModel.",
            mref.plugin_name,
            mref.config_type,
            mref.config_id,
        )

    for mref, config_object in prefetch.fetched.items():
        new_ts: datetime | None = getattr(config_object, "updated_at", None)
        if mref not in known_versions or new_ts != known_versions[mref]:
            changed.add(mref)
        known_versions[mref] = new_ts

    return changed


class VirtualModelCacheRefreshError(Exception):
    """Raised when the VirtualModel cache cannot be refreshed."""


@dataclass
class VirtualModelCache:
    """In-memory map of ``(workspace, name) → VirtualModel``."""

    virtual_model_map: dict[tuple[str, str], VirtualModel] = field(default_factory=dict)
    #: Last known ``updated_at`` per external middleware config ref, used to detect config entity updates.
    config_ref_versions: dict[MiddlewareConfigRef, datetime | None] = field(default_factory=dict)

    def get(self, workspace: str, name: str) -> VirtualModel | None:
        """Return the VirtualModel for ``workspace/name``, falling back to the global workspace (local wins)."""
        for candidate in workspace_lookup_order(workspace):
            virtual_model = self.virtual_model_map.get((candidate, name))
            if virtual_model is not None:
                return virtual_model
        return None

    def rebuild(self, virtual_models: list[VirtualModel]) -> None:
        """Replace the entire cache with *virtual_models*.

        Atomically swaps the map so in-flight lookups always see a consistent snapshot.
        """
        self.virtual_model_map = {(vm.workspace, vm.name): vm for vm in virtual_models}


async def refresh_virtual_model_cache(
    cache: VirtualModelCache,
    sdk: AsyncNeMoPlatform,
    registry: MiddlewareRegistry | None = None,
) -> None:
    """Fetch all VirtualModels from the IGW's own API, rebuild *cache*, and notify *registry*.

    Calls ``sdk.inference.virtual_models.list(workspace="-")`` (cross-workspace) and iterates
    all pages.  On any error a :class:`VirtualModelCacheRefreshError` is raised;
    callers are responsible for logging and deciding whether to retry.

    If *registry* is provided, after rebuilding the cache:

    - De-duplicates and fetches all external ``config_id`` responses via
      :meth:`MiddlewareRegistry.fetch_middleware_config_responses`\\ .
    - Updates :attr:`VirtualModelCache.middleware_config_ref_updated_at` and determines
      which config refs are new, removed, or updated; VMs that reference a changed ref
      re-resolve (even when the VirtualModel document is unchanged).
    - Pre-resolves middleware for VMs that are new, have a newer VM ``updated_at``, or
      touch a changed middleware ref.
    - Calls ``registry.notify_upserted`` for that same set of VMs and ``registry.notify_destroyed`` / ``evict`` for removed VMs.

    Notification errors are logged and do not fail the refresh.

    Args:
        cache: The cache instance to rebuild.
        sdk: Platform SDK authenticated as the ``inference-gateway`` service principal.
        registry: Optional :class:`~nmp.core.inference_gateway.api.middleware_registry.MiddlewareRegistry`
            to notify of VirtualModel lifecycle events.
    """
    try:
        virtual_models: list[VirtualModel] = []
        paginator = sdk.inference.virtual_models.list(workspace="-", page_size=200)
        async for vm in paginator:
            # Skip VMs with missing workspace/name — they cannot be keyed in the map
            if vm.workspace and vm.name:
                virtual_models.append(vm)
    except (APIConnectionError, APIStatusError) as exc:
        raise VirtualModelCacheRefreshError(f"Error refreshing VirtualModel cache from API: {exc}") from exc
    except Exception as exc:
        raise VirtualModelCacheRefreshError(f"Unexpected error refreshing VirtualModel cache: {exc}") from exc

    # Compute diff before rebuilding so we can notify registry of changes
    old_map = dict(cache.virtual_model_map)
    # vm.name/workspace are guaranteed non-None here (filtered above)
    new_map: dict[tuple[str, str], VirtualModel] = {(vm.workspace, vm.name): vm for vm in virtual_models}

    cache.rebuild(virtual_models)
    logger.debug("VirtualModel cache refreshed: %d entries", len(virtual_models))

    if registry is None:
        return

    current_refs = collect_config_refs(virtual_models)
    prefetch = await registry.prefetch_configs(current_refs)
    updated_config_refs = sync_config_ref_versions(cache.config_ref_versions, current_refs, prefetch)

    old_keys = set(old_map.keys())
    new_keys = set(new_map.keys())

    added = new_keys - old_keys
    removed = old_keys - new_keys
    changed_vms = {k for k in (old_keys & new_keys) if old_map[k].updated_at != new_map[k].updated_at}
    config_changed = {k for k in new_keys if vm_uses_any_changed_config(new_map[k], updated_config_refs)}
    upsert_keys = added | changed_vms | config_changed

    for key in upsert_keys:
        vm = new_map[key]
        try:
            await registry.resolve_configs_for_virtual_model(vm, prefetch=prefetch)
        except Exception:
            logger.warning("Error resolving configs for VirtualModel %s/%s", *key, exc_info=True)
            registry.evict(key)
            registry.broken_vms.add(key)
        try:
            await registry.notify_upserted(vm)
        except Exception:
            logger.warning("Error notifying upserted for VirtualModel %s/%s", *key, exc_info=True)

    # Evict and notify for removed VMs. broken_vms is also cleared here because
    # `evict` intentionally leaves it alone (so resolution-failure state survives
    # one diff cycle without the VM being deleted) — once the VM itself is gone
    # there is no future resolve to clear it for us.
    for key in removed:
        registry.evict(key)
        registry.broken_vms.discard(key)
        vm = old_map[key]
        try:
            await registry.notify_destroyed(vm)
        except Exception:
            logger.warning("Error notifying destroyed for VirtualModel %s/%s", *key, exc_info=True)
