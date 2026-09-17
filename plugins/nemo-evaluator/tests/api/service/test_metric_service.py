# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import nemo_evaluator.shared.metric_bundles.inline  # noqa: F401
import pytest
from nemo_evaluator.api.schemas import MetricInline
from nemo_evaluator.api.service.metric_service import MetricService
from nemo_evaluator.entities import MetricBundleEntity
from nemo_evaluator.metric_storage import parse_bundle_ref
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_platform_plugin.entities import ListResponse, PaginationInfo
from nemo_platform_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest
from nemo_platform_plugin.filter_ops import FilterOperation

_LEGACY_REQUIRED_BUNDLE_JSON = (
    '{"bundle_kind":"metric-bundle","bundle_format_version":"v1","metric_type":"exact-match",'
    '"metadata":{"description":null,"labels":{}},"outputs":[{"name":"exact-match","description":null,'
    '"value_json_schema":{"description":"Continuous numeric metric value.","title":"ContinuousScore",'
    '"type":"number"}}],"secrets":{},"payload":{"metric":{"type":"exact-match","description":null,'
    '"labels":{},"supported_job_types":["online","offline"],"reference":"x","candidate":"y"},'
    '"digest":"cdcbd7ba2c83a314ce2f682a1ff5a86fb48dceb33452006e373894d82984686c","kind":"inline"}}'
)

# ---- in-memory fakes -------------------------------------------------------


class _FakeDownloadResponse:
    def __init__(self, content: bytes) -> None:
        self._content = content

    async def read(self) -> bytes:
        return self._content


class _FakeFiles(AsyncFilesClient):
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], dict[str, bytes]] = {}

    async def create_fileset(
        self,
        *,
        workspace: str | None = None,
        body: CreateFilesetRequest,
        exist_ok: bool = False,
    ) -> object:
        del exist_ok
        self._store.setdefault((workspace or "default", body.name), {})
        return object()

    async def delete_fileset(self, *, workspace: str | None = None, name: str) -> object:
        self._store.pop((workspace or "default", name), None)
        return object()

    async def upload_file(
        self,
        *,
        content: bytes,
        path: str,
        name: str,
        workspace: str | None = None,
    ) -> object:
        self._store.setdefault((workspace or "default", name), {})[path] = bytes(content)
        return object()

    async def download_file(
        self,
        *,
        path: str,
        name: str,
        workspace: str | None = None,
    ) -> _FakeDownloadResponse:
        return _FakeDownloadResponse(self._store[(workspace or "default", name)][path])


class _FakeEntityClient:
    def __init__(self) -> None:
        self.entities: dict[tuple[str, str], MetricBundleEntity] = {}
        self.delete_error: Exception | None = None
        self.list_filter_operations: list[FilterOperation | None] = []

    async def get(
        self, entity_type: type[MetricBundleEntity], *, workspace: str, name: str, parent: str | None = None
    ) -> MetricBundleEntity:
        key = (workspace, name)
        if key not in self.entities:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")
        return self.entities[key]

    async def create(self, entity: MetricBundleEntity) -> MetricBundleEntity:
        key = (entity.workspace, entity.name)
        if key in self.entities:
            raise NemoEntityConflictError(f"{key} exists")
        now = datetime.now(timezone.utc)
        entity._id = f"metric_bundle-{entity.name}"
        entity._created_at = now
        entity._updated_at = now
        self.entities[key] = entity
        return entity

    async def delete(
        self,
        entity_type: type[MetricBundleEntity],
        name: str,
        *,
        workspace: str,
        parent: str | None = None,
        expected_db_version: int | None = None,
    ) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.entities.pop((workspace, name), None)

    async def list(
        self,
        entity_type: type[MetricBundleEntity],
        *,
        workspace: str,
        filter_operation: FilterOperation | None = None,
        sort: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ListResponse[MetricBundleEntity]:
        self.list_filter_operations.append(filter_operation)
        items = [e for (ws, _), e in self.entities.items() if ws == workspace]
        return ListResponse(
            data=items,
            pagination=PaginationInfo(
                page=page,
                page_size=page_size,
                current_page_size=len(items),
                total_pages=1,
                total_results=len(items),
            ),
        )


@pytest.fixture
def fake_files() -> _FakeFiles:
    return _FakeFiles()


@pytest.fixture
def fake_entity_client() -> _FakeEntityClient:
    return _FakeEntityClient()


@pytest.fixture
def service(fake_files: _FakeFiles, fake_entity_client: _FakeEntityClient) -> MetricService:
    return MetricService(fake_entity_client, fake_files)


def _bundle(metric=None) -> MetricInline:
    """Build a runtime bundle and return it as the API wire DTO (what requests carry)."""
    metric = metric or ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    runtime_bundle = bundle_metric(metric, CloudpickleMetricBundlePackager())
    return MetricInline.model_validate_json(runtime_bundle.model_dump_json())


def _legacy_required_bundle() -> MetricInline:
    """Load a frozen all-required bundle emitted by origin/main."""
    return MetricInline.model_validate_json(_LEGACY_REQUIRED_BUNDLE_JSON)


def _fileset_of(service: MetricService, bundle_ref: str) -> tuple[str, str]:
    workspace, fileset, _ = parse_bundle_ref(bundle_ref)
    return (workspace, fileset)


# ---- tests -----------------------------------------------------------------


async def test_create_stores_bundle_and_indexes_entity(
    service: MetricService, fake_files, fake_entity_client: _FakeEntityClient
) -> None:
    bundle = _bundle()
    created = await service.create_metric("exact", bundle, workspace="default")
    entity = fake_entity_client.entities[("default", "exact")]

    assert created.name == "exact"
    assert created.metric_type == bundle.metric_type
    assert created.payload_kind == "cloudpickle"
    assert created.payload_digest == bundle.payload.digest
    assert created.bundle_ref.startswith("default/metric-bundle.")
    assert created.description == bundle.metadata.description
    assert created.labels == bundle.metadata.labels
    assert _fileset_of(service, created.bundle_ref) in fake_files._store
    assert "required" not in entity.model_dump(mode="json")["outputs"][0]
    assert "required" not in created.model_dump(mode="json")["outputs"][0]


async def test_create_preserves_optional_output_in_entity_and_response(
    service: MetricService, fake_entity_client: _FakeEntityClient
) -> None:
    bundle = _bundle()
    optional = bundle.model_copy(update={"outputs": [bundle.outputs[0].model_copy(update={"required": False})]})

    created = await service.create_metric("optional", optional, workspace="default")
    entity = fake_entity_client.entities[("default", "optional")]

    assert entity.model_dump(mode="json")["outputs"][0]["required"] is False
    assert created.model_dump(mode="json")["outputs"][0]["required"] is False


async def test_create_rejects_duplicate_without_clobbering_existing(service: MetricService, fake_files) -> None:
    first = await service.create_metric("exact", _bundle(), workspace="default")

    with pytest.raises(ValueError, match="already exists"):
        await service.create_metric("exact", _bundle(), workspace="default")

    assert _fileset_of(service, first.bundle_ref) in fake_files._store


async def test_get_returns_none_when_missing(service: MetricService) -> None:
    assert await service.get_metric("default", "nope") is None


async def test_delete_removes_entity_and_bundle(service: MetricService, fake_files) -> None:
    created = await service.create_metric("m", _bundle(), workspace="default")

    assert await service.delete_metric("default", "m") is True
    assert await service.get_metric("default", "m") is None
    assert _fileset_of(service, created.bundle_ref) not in fake_files._store


async def test_delete_returns_false_when_missing(service: MetricService) -> None:
    assert await service.delete_metric("default", "nope") is False


async def test_delete_handles_concurrent_delete_race(
    service: MetricService, fake_entity_client: _FakeEntityClient
) -> None:
    await service.create_metric("m", _bundle(), workspace="default")

    fake_entity_client.delete_error = NemoEntityNotFoundError("deleted concurrently")
    assert await service.delete_metric("default", "m") is False


async def test_list_returns_workspace_metrics(service: MetricService) -> None:
    await service.create_metric("a", _bundle(), workspace="default")
    await service.create_metric("b", _bundle(), workspace="default")

    page = await service.list_metrics("default")

    assert {m.name for m in page.data} == {"a", "b"}
    assert page.pagination is not None
    assert page.pagination.total_results == 2


# ---- derived metrics -------------------------------------------------------


async def test_store_derived_metric_names_by_digest_and_marks_derived(
    service: MetricService, fake_files: _FakeFiles, fake_entity_client: _FakeEntityClient
) -> None:
    from nemo_evaluator.api.service.metric_service import _MAX_ENTITY_NAME_LENGTH

    ref = await service.store_derived_metric(_bundle(), workspace="default")

    workspace, _, name = ref.root.partition("/")
    assert workspace == "default"
    assert name.startswith("derived.")
    assert len(name) <= _MAX_ENTITY_NAME_LENGTH
    entity = fake_entity_client.entities[("default", name)]
    assert entity.derived is True
    assert _fileset_of(service, entity.bundle_ref) in fake_files._store


async def test_store_derived_metric_distinguishes_full_contract(
    service: MetricService, fake_entity_client: _FakeEntityClient
) -> None:
    bundle = _bundle()
    variant = bundle.model_copy(update={"metadata": bundle.metadata.model_copy(update={"description": "different"})})
    assert bundle.payload.digest == variant.payload.digest

    first = await service.store_derived_metric(bundle, workspace="default")
    second = await service.store_derived_metric(variant, workspace="default")

    assert first.root != second.root
    assert len(fake_entity_client.entities) == 2


async def test_store_derived_metric_is_content_addressed_dedup(
    service: MetricService, fake_files: _FakeFiles, fake_entity_client: _FakeEntityClient
) -> None:
    bundle = _bundle()

    first = await service.store_derived_metric(bundle, workspace="default")
    second = await service.store_derived_metric(bundle, workspace="default")

    assert first.root == second.root
    assert len(fake_entity_client.entities) == 1
    assert len(fake_files._store) == 1


async def test_store_derived_metric_preserves_legacy_required_identity_and_bytes(
    service: MetricService, fake_files: _FakeFiles, fake_entity_client: _FakeEntityClient
) -> None:
    legacy = _legacy_required_bundle()
    explicit_required = legacy.model_copy(update={"outputs": [legacy.outputs[0].model_copy(update={"required": True})]})
    optional = legacy.model_copy(update={"outputs": [legacy.outputs[0].model_copy(update={"required": False})]})
    legacy_content = json.loads(_LEGACY_REQUIRED_BUNDLE_JSON)
    expected_digest = hashlib.sha256(
        json.dumps(legacy_content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    first = await service.store_derived_metric(legacy, workspace="default")
    second = await service.store_derived_metric(explicit_required, workspace="default")
    third = await service.store_derived_metric(optional, workspace="default")

    assert first.root == second.root == f"default/derived.{expected_digest[:55]}"
    assert third.root != first.root
    assert len(fake_entity_client.entities) == 2
    _, _, name = first.root.partition("/")
    entity = fake_entity_client.entities[("default", name)]
    workspace, fileset = _fileset_of(service, entity.bundle_ref)
    assert fake_files._store[(workspace, fileset)]["bundle.json"] == _LEGACY_REQUIRED_BUNDLE_JSON.encode("utf-8")


async def test_list_excludes_derived_by_default(service: MetricService, fake_entity_client: _FakeEntityClient) -> None:
    await service.list_metrics("default")
    await service.list_metrics("default", include_derived=True)

    assert fake_entity_client.list_filter_operations[0] is not None
    assert fake_entity_client.list_filter_operations[1] is None
