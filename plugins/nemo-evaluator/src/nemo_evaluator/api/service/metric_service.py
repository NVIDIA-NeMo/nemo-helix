# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service layer for stored metric operations.

Combines the Entity Store (the queryable index) with the Files service (the
canonical serialized bundle). The service owns the bundle's storage lifecycle.

Stored metrics are immutable: they can be created, read, listed, and deleted,
but not updated in place. Each bundle upload lands in its own uniquely-named
fileset, which makes every write/rollback target exactly one metric's bytes. As
a result no failure mode leaves a metric pointing at the wrong or missing data —
the worst case is an orphaned fileset (a storage leak), never a corrupted metric:

- create: upload first, then create the entity. On a name conflict (or any
  index failure) we delete the fileset we just created — never another
  metric's. The entity is only created once its bundle exists.
- delete: delete the entity first, then its fileset.
"""

from __future__ import annotations

import hashlib
import json
import logging

from nemo_evaluator.api.schemas import (
    Metric,
    MetricInline,
    MetricRef,
)
from nemo_evaluator.entities import MetricBundleEntity
from nemo_evaluator.metric_storage import delete_bundle_by_ref, store_bundle
from nemo_evaluator.shared.metric_bundles.bundles import MetricBundle as RuntimeMetricBundle
from nemo_helix_plugin.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClientProtocol,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
)
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.filter_ops import FilterOperation
from nemo_helix_plugin.log_utils import sanitize_for_log
from nemo_helix_plugin.schema import Page, PaginationData

#: Reserved name prefix for content-addressed derived metrics (auto-stored from inline task metrics).
_DERIVED_METRIC_PREFIX = "derived."

#: The entity store caps entity names at 63 characters (``^[a-z]...{1,62}...$``). The derived metric's
#: name is ``derived.<digest>``; we truncate the (hex) content digest to fit. The retained prefix is
#: far longer than needed to keep content-addressed dedup collision-free (e.g. 55 hex chars ≈ 220 bits).
_MAX_ENTITY_NAME_LENGTH = 63
_DERIVED_DIGEST_LENGTH = _MAX_ENTITY_NAME_LENGTH - len(_DERIVED_METRIC_PREFIX)

logger = logging.getLogger(__name__)


def _and_exclude_derived(filter_operation: FilterOperation | None) -> FilterOperation:
    """Combine an optional filter with "derived is not true", so derived metrics stay hidden.

    The filter grammar has no ``$ne``, so this is ``NOT(data.derived == True)`` — which also matches
    metrics created before the ``derived`` field existed (the key is simply absent).
    """
    not_derived = LogicalOperation(
        operator=FilterOperator.NOT,
        operations=[ComparisonOperation(field="data.derived", operator=FilterOperator.EQ, value=True)],
    )
    if filter_operation is None:
        return not_derived
    return LogicalOperation(operator=FilterOperator.AND, operations=[filter_operation, not_derived])


def _entity_to_schema(entity: MetricBundleEntity) -> Metric:
    """Convert a stored metric entity to its API representation."""
    created_at = entity.created_at
    updated_at = entity.updated_at
    if created_at is None or updated_at is None:
        raise ValueError(f"Stored metric '{entity.workspace}/{entity.name}' is missing persistence timestamps")
    return Metric(
        id=entity.id,
        name=entity.name,
        workspace=entity.workspace,
        project=entity.project,
        metric_type=entity.metric_type,
        description=entity.description,
        labels=entity.labels,
        outputs=entity.outputs,
        secrets=entity.secrets,
        payload_kind=entity.payload_kind,
        payload_digest=entity.payload_digest,
        bundle_ref=entity.bundle_ref,
        derived=entity.derived,
        created_at=created_at,
        updated_at=updated_at,
    )


def _entity_from_bundle(
    *,
    name: str,
    workspace: str,
    bundle: RuntimeMetricBundle,
    bundle_ref: str,
    project: str | None,
    derived: bool = False,
) -> MetricBundleEntity:
    """Build a stored-metric entity from a bundle and its Files reference."""
    return MetricBundleEntity(
        name=name,
        workspace=workspace,
        project=project,
        metric_type=bundle.metric_type,
        description=bundle.metadata.description,
        labels=bundle.metadata.labels,
        outputs=bundle.outputs,
        secrets=bundle.secrets,
        payload_kind=bundle.payload.kind,
        payload_digest=bundle.payload.digest,
        bundle_ref=bundle_ref,
        derived=derived,
    )


class MetricService:
    """Service layer for stored metric CRUD."""

    def __init__(self, entity_client: NemoEntitiesClientProtocol[MetricBundleEntity], files_client: AsyncFilesClient):
        self.entity_client = entity_client
        self.files_client = files_client

    async def create_metric(
        self,
        name: str,
        metric: MetricInline,
        *,
        workspace: str,
        project: str | None = None,
    ) -> Metric:
        """Store a new metric (addressed by workspace/name): upload its bundle, then index it."""
        logger.debug(
            "Creating metric", extra={"workspace": sanitize_for_log(workspace), "metric_name": sanitize_for_log(name)}
        )

        # Cheap pre-check to avoid uploading a (potentially large) bundle we would
        # only discard. The entity create below remains the authoritative,
        # race-safe uniqueness guard.
        try:
            await self.entity_client.get(MetricBundleEntity, name=name, workspace=workspace)
            raise ValueError(f"Metric with name '{name}' already exists in workspace '{workspace}'")
        except NemoEntityNotFoundError:
            pass

        # Convert the wire DTO into the runtime bundle (JSON round-trip keeps the
        # base64 payload handling consistent), then store/index it.
        runtime_bundle = RuntimeMetricBundle.model_validate_json(metric.model_dump_json())
        bundle_ref = await store_bundle(self.files_client, workspace, name, runtime_bundle)
        entity = _entity_from_bundle(
            name=name,
            workspace=workspace,
            bundle=runtime_bundle,
            bundle_ref=bundle_ref,
            project=project,
        )

        try:
            created = await self.entity_client.create(entity)
        except NemoEntityConflictError as e:
            # We own this uniquely-named fileset and the entity was not created;
            # deleting it cannot affect the existing metric's data.
            await self._discard_bundle(bundle_ref)
            raise ValueError(f"Metric with name '{name}' already exists in workspace '{workspace}'") from e
        except Exception:
            await self._discard_bundle(bundle_ref)
            raise

        logger.info(
            "Metric created",
            extra={"workspace": sanitize_for_log(created.workspace), "metric_name": sanitize_for_log(created.name)},
        )
        return _entity_to_schema(created)

    async def store_derived_metric(self, metric: MetricInline, *, workspace: str) -> MetricRef:
        """Store an inline metric as a content-addressed *derived* metric and return a reference to it.

        Used when persisting a task that carries an inline metric: rather than embedding the bundle in
        the task entity, we store it like any metric (Files-backed) but mark it ``derived`` (hidden
        from the default metric listing) and name it by a digest of its *full* content, so identical
        inline metrics across tasks dedupe to one stored bundle.

        The digest covers the whole bundle (metric_type, metadata, outputs, secrets, payload) — not
        just ``payload.digest`` — so two metrics that share scoring code but differ in secrets or
        output contracts get distinct names and are never silently collapsed onto each other.
        """
        runtime_bundle = RuntimeMetricBundle.model_validate_json(metric.model_dump_json())
        canonical = json.dumps(runtime_bundle.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        content_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        name = f"{_DERIVED_METRIC_PREFIX}{content_digest[:_DERIVED_DIGEST_LENGTH]}"
        ref = MetricRef(f"{workspace}/{name}")

        # Content-addressed: if this exact bundle is already stored, reuse it (dedup).
        try:
            await self.entity_client.get(MetricBundleEntity, name=name, workspace=workspace)
            return ref
        except NemoEntityNotFoundError:
            pass

        bundle_ref = await store_bundle(self.files_client, workspace, name, runtime_bundle)
        entity = _entity_from_bundle(
            name=name, workspace=workspace, bundle=runtime_bundle, bundle_ref=bundle_ref, project=None, derived=True
        )
        try:
            await self.entity_client.create(entity)
        except NemoEntityConflictError:
            # Raced another writer to the same content-addressed name; theirs is byte-identical, so
            # drop the fileset we just uploaded and reuse the existing entry.
            await self._discard_bundle(bundle_ref)
        except Exception:
            await self._discard_bundle(bundle_ref)
            raise
        return ref

    async def get_metric(self, workspace: str, name: str) -> Metric | None:
        """Get a stored metric by workspace and name."""
        try:
            entity = await self.entity_client.get(MetricBundleEntity, workspace=workspace, name=name)
            return _entity_to_schema(entity)
        except NemoEntityNotFoundError:
            return None

    async def list_metrics(
        self,
        workspace: str,
        page: int = 1,
        page_size: int = 100,
        sort: str | None = None,
        filter_operation: FilterOperation | None = None,
        include_derived: bool = False,
    ) -> Page[Metric]:
        """List stored metrics with filtering and pagination.

        Derived (task-internal) metrics are excluded unless ``include_derived`` is set — they're
        addressable by reference but shouldn't clutter the curated metric listing.
        """
        if not include_derived:
            filter_operation = _and_exclude_derived(filter_operation)
        result = await self.entity_client.list(
            MetricBundleEntity,
            workspace=workspace,
            filter_operation=filter_operation,
            sort=sort,
            page=page,
            page_size=page_size,
        )
        metrics = [_entity_to_schema(entity) for entity in result.data]
        return Page(
            data=metrics,
            pagination=PaginationData(
                page=result.pagination.page,
                page_size=result.pagination.page_size,
                current_page_size=len(metrics),
                total_pages=result.pagination.total_pages,
                total_results=result.pagination.total_results,
            ),
            sort=sort,
            filter=None,
        )

    async def delete_metric(self, workspace: str, name: str) -> bool:
        """Delete a stored metric and its backing bundle. Returns False if not found."""
        try:
            entity = await self.entity_client.get(MetricBundleEntity, workspace=workspace, name=name)
        except NemoEntityNotFoundError:
            return False

        try:
            await self.entity_client.delete(
                MetricBundleEntity,
                entity.name,
                workspace=workspace,
                expected_db_version=entity.db_version,
            )
        except NemoEntityNotFoundError:
            # Lost a delete race: another request already removed it.
            return False
        await self._discard_bundle(entity.bundle_ref)
        logger.info(
            "Metric deleted", extra={"workspace": sanitize_for_log(workspace), "metric_name": sanitize_for_log(name)}
        )
        return True

    async def _discard_bundle(self, bundle_ref: str) -> None:
        """Delete an unreferenced bundle fileset.

        Safe by construction: every reference names a fileset created for a
        single metric version, so deleting it can only remove bytes nothing
        else points at. A failure here leaks that one fileset (logged) but never
        affects a live metric.
        """
        try:
            await delete_bundle_by_ref(self.files_client, bundle_ref)
        except Exception:
            logger.warning(
                "Failed to delete unreferenced metric bundle fileset; storage may be leaked",
                extra={"bundle_ref": sanitize_for_log(bundle_ref)},
                exc_info=True,
            )
