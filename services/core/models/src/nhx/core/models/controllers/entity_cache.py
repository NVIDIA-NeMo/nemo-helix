# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched reads and writes of Model Entities for a single reconciliation phase.

Reconciling providers and deployments requires looking at the Model Entity behind
every served model, and often needs to link or unlink a provider on it. Doing that
one entity at a time costs two round trips per served model per pass, which grows
with the product of providers and the size of their model catalogues.

:class:`ModelEntityCache` reads the entities once per phase and accumulates the
intended changes, so each entity is written at most once and only when something
actually differs. Because the desired state for an entity is assembled in memory
before it is written, concurrent contributions from several providers cannot
overwrite one another.
"""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Callable

from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.types import CreateModelEntityRequest, ModelEntity, UpdateModelEntityRequest

logger = getLogger(__name__)

# Entity lists are read whole, so prefer few large pages over many small ones.
_PAGE_SIZE = 1000


@dataclass
class _PendingEntity:
    """Changes staged for one Model Entity within a phase."""

    workspace: str
    name: str
    create_kwargs: dict | None = None
    link_providers: list[str] = field(default_factory=list)
    unlink_providers: list[str] = field(default_factory=list)
    field_updates: dict = field(default_factory=dict)
    # Number of times a flush has tried to apply this entry. Distinguishes a
    # change that was never flushed from one that was flushed and failed.
    attempts: int = 0


class UnflushedMutationsError(RuntimeError):
    """Raised when the cache is refreshed while changes are still staged.

    Re-reading would discard the staged changes and reload the very state they
    were derived from, so the caller must flush first.
    """


class ModelEntityCache:
    """A per-phase snapshot of Model Entities plus the changes staged against it.

    Usage is strictly ``refresh()`` -> read/stage -> ``flush()``. Reads see staged
    changes layered over the snapshot, so a caller that stages a change and then
    reads the same entity within a phase observes its own write.
    """

    def __init__(self, models_sdk: AsyncNeMoHelix, emit_heartbeat: Callable[[], None]) -> None:
        """Initialize the cache.

        Args:
            models_sdk: SDK client for Models API interactions
            emit_heartbeat: Called as each entity is read or written. Reading and
                writing are both proportional to the number of entities, so they
                have to report progress or a large batch looks like a stall.
        """
        self._models_sdk = models_sdk
        self._emit_heartbeat = emit_heartbeat
        self._entities: dict[tuple[str, str], ModelEntity] = {}
        self._pending: dict[tuple[str, str], _PendingEntity] = {}
        self._loaded = False

    async def refresh(self) -> None:
        """Re-read every Model Entity across all workspaces.

        Changes that a flush already tried and failed to apply are kept and
        retried on a later flush. Because staged changes are expressed as
        differences rather than absolute state, replaying them against a newer
        snapshot stays correct.

        Raises:
            UnflushedMutationsError: If changes are staged that no flush has tried
                to apply yet, which would lose them silently.
        """
        never_attempted = [key for key, staged in self._pending.items() if staged.attempts == 0]
        if never_attempted:
            raise UnflushedMutationsError(
                f"{len(never_attempted)} staged Model Entity change(s) must be flushed before refreshing the cache"
            )

        entities: dict[tuple[str, str], ModelEntity] = {}
        models = client_from_platform(self._models_sdk, AsyncModelsClient)
        async for entity in (await models.list_models(workspace="-", query_params={"page_size": _PAGE_SIZE})).items():
            entities[(entity.workspace, entity.name)] = entity
            self._emit_heartbeat()

        self._entities = entities
        self._loaded = True
        logger.debug("Model Entity cache loaded %d entities", len(entities))

    @property
    def loaded(self) -> bool:
        """Whether the cache holds a snapshot."""
        return self._loaded

    def get(self, workspace: str, name: str) -> ModelEntity | None:
        """Return an entity as it will exist once staged changes are flushed.

        Returns ``None`` when the entity neither exists nor is staged for creation.
        """
        key = (workspace, name)
        entity = self._entities.get(key)
        pending = self._pending.get(key)
        if pending is None:
            return entity

        if entity is None:
            # An entity staged for creation is reported as absent rather than
            # fabricated. Callers stage what they want and the staged changes are
            # merged, so a second caller reaching the same conclusion is harmless.
            return None

        providers = [p for p in (entity.model_providers or []) if p not in pending.unlink_providers]
        providers.extend(p for p in pending.link_providers if p not in providers)
        return entity.model_copy(update={"model_providers": providers, **pending.field_updates})

    def _pending_for(self, workspace: str, name: str) -> _PendingEntity:
        key = (workspace, name)
        if key not in self._pending:
            self._pending[key] = _PendingEntity(workspace=workspace, name=name)
        return self._pending[key]

    def stage_create(self, workspace: str, name: str, **create_kwargs) -> None:
        """Stage creation of an entity that does not exist yet.

        Repeated calls for the same entity keep the first set of attributes; the
        providers that asked for it are merged by :meth:`stage_provider_link`.
        """
        pending = self._pending_for(workspace, name)
        if pending.create_kwargs is None:
            pending.create_kwargs = dict(create_kwargs)

    def stage_provider_link(self, workspace: str, name: str, provider_id: str) -> None:
        """Stage a provider reference to be present on an entity."""
        pending = self._pending_for(workspace, name)
        if provider_id in pending.unlink_providers:
            pending.unlink_providers.remove(provider_id)
        if provider_id not in pending.link_providers:
            pending.link_providers.append(provider_id)

    def stage_provider_unlink(self, workspace: str, name: str, provider_id: str) -> None:
        """Stage a provider reference to be absent from an entity."""
        pending = self._pending_for(workspace, name)
        if provider_id in pending.link_providers:
            pending.link_providers.remove(provider_id)
        if provider_id not in pending.unlink_providers:
            pending.unlink_providers.append(provider_id)

    def stage_field_updates(self, workspace: str, name: str, **updates) -> None:
        """Stage attribute values to write.

        Whether a value ought to be written is the caller's decision, since only
        the caller knows what counts as already-set for a given attribute. Values
        of ``None`` are dropped so callers can pass optional values through
        directly.
        """
        pending = self._pending_for(workspace, name)
        for key, value in updates.items():
            if value is not None:
                pending.field_updates.setdefault(key, value)

    async def flush(self) -> None:
        """Apply staged changes, writing each entity at most once.

        Entities whose staged state already matches the snapshot are skipped, so a
        pass that changes nothing performs no writes.

        An entity that fails to write keeps its staged change so a later flush can
        retry it. Some changes cannot be recomputed by a later pass -- unlinking a
        provider that is about to be deleted is derived from that provider, which
        will be gone -- so dropping a failure would leave the entity permanently
        inconsistent.
        """
        for (workspace, name), staged in list(self._pending.items()):
            staged.attempts += 1
            existing = self._entities.get((workspace, name))
            try:
                if existing is None:
                    await self._create(workspace, name, staged)
                else:
                    await self._update(workspace, name, staged, existing)
            except Exception:
                # One entity must not stop the rest, and the change is kept for a
                # later attempt rather than discarded.
                logger.warning(
                    "Failed to apply Model Entity changes for %s/%s, will retry (attempt %d)",
                    workspace,
                    name,
                    staged.attempts,
                    exc_info=True,
                )
            else:
                del self._pending[(workspace, name)]
            finally:
                self._emit_heartbeat()

    async def _create(self, workspace: str, name: str, staged: _PendingEntity) -> None:
        if staged.create_kwargs is None:
            # Only a link/unlink was staged for an entity that does not exist.
            logger.debug("Skipping Model Entity changes for missing entity %s/%s", workspace, name)
            return

        create_kwargs = {k: v for k, v in staged.create_kwargs.items() if k not in ("name", "workspace")}
        create_kwargs.update(staged.field_updates)
        if staged.link_providers:
            create_kwargs["model_providers"] = list(staged.link_providers)

        try:
            models = client_from_platform(self._models_sdk, AsyncModelsClient)
            created = (
                await models.create_model(
                    workspace=workspace, body=CreateModelEntityRequest(name=name, **create_kwargs)
                )
            ).data()
        except ConflictError:
            # Created concurrently; adopt it and apply the staged changes instead.
            logger.debug("Model Entity %s/%s already exists, applying staged changes", workspace, name)
            try:
                existing = (await models.get_model(workspace=workspace, name=name)).data()
            except NotFoundError:
                return
            self._entities[(workspace, name)] = existing
            await self._update(workspace, name, staged, existing)
            return

        if created is not None:
            self._entities[(workspace, name)] = created
        logger.debug("Created Model Entity %s/%s", workspace, name)

    async def _update(self, workspace: str, name: str, staged: _PendingEntity, existing: ModelEntity) -> None:
        update_params: dict = {}

        current_providers = list(existing.model_providers or [])
        desired_providers = [p for p in current_providers if p not in staged.unlink_providers]
        desired_providers.extend(p for p in staged.link_providers if p not in desired_providers)
        if desired_providers != current_providers:
            update_params["model_providers"] = desired_providers

        update_params.update(staged.field_updates)

        if not update_params:
            logger.debug("Model Entity %s/%s already matches desired state", workspace, name)
            return

        models = client_from_platform(self._models_sdk, AsyncModelsClient)
        updated = (
            await models.update_model(workspace=workspace, name=name, body=UpdateModelEntityRequest(**update_params))
        ).data()
        if updated is not None:
            self._entities[(workspace, name)] = updated
        logger.debug("Updated Model Entity %s/%s: %s", workspace, name, sorted(update_params))
