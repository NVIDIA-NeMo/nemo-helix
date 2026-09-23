# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP route-level tests for the /metrics CRUD endpoints.

Drives the real FastAPI router + MetricService through a TestClient, with the
entity store and Files service replaced by in-memory fakes. Covers route wiring,
the get_metric_service dependency, and status-code mapping (201/204/404/409).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_evaluator.api.dependencies import get_metric_service
from nemo_evaluator.api.schemas import MetricInline
from nemo_evaluator.api.service.metric_service import MetricService
from nemo_evaluator.api.v2 import metrics as metrics_routes
from nemo_evaluator.entities import MetricBundleEntity
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_helix_plugin.entities import ListResponse, PaginationInfo
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.filter_ops import FilterOperation

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
        self.entity_versions: dict[tuple[str, str], int] = {}
        self.bump_version_on_next_delete = False
        self.delete_expected_db_versions: list[int | None] = []

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
        entity._db_version = 1
        self.entities[key] = entity
        self.entity_versions[key] = entity.db_version
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
        key = (workspace, name)
        if key not in self.entities:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")
        if self.bump_version_on_next_delete:
            self.entity_versions[key] += 1
            self.entities[key]._db_version = self.entity_versions[key]
            self.bump_version_on_next_delete = False
        self.delete_expected_db_versions.append(expected_db_version)
        if expected_db_version is not None and self.entity_versions[key] != expected_db_version:
            raise NemoEntityConflictError(f"{workspace}/{name} changed")
        del self.entities[key]
        del self.entity_versions[key]

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


@dataclass(frozen=True)
class _MetricsRouteHarness:
    client: TestClient
    entity_client: _FakeEntityClient


@pytest.fixture
def metrics_route_harness() -> Iterator[_MetricsRouteHarness]:
    app = FastAPI()
    app.include_router(metrics_routes.router, prefix="/v2/workspaces/{workspace}")
    fake_files = _FakeFiles()
    entity_client = _FakeEntityClient()
    service = MetricService(entity_client, fake_files)
    app.dependency_overrides[get_metric_service] = lambda: service
    yield _MetricsRouteHarness(TestClient(app), entity_client)


@pytest.fixture
def client(metrics_route_harness: _MetricsRouteHarness) -> TestClient:
    return metrics_route_harness.client


def _create_body() -> dict:
    """The create request body is a bare MetricInline (name comes from the path)."""
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    runtime_bundle = bundle_metric(metric, CloudpickleMetricBundlePackager())
    return MetricInline.model_validate_json(runtime_bundle.model_dump_json()).model_dump(mode="json")


_BASE = "/v2/workspaces/default/metrics"


def test_create_then_get(client: TestClient) -> None:
    resp = client.post(f"{_BASE}/exact", json=_create_body())
    assert resp.status_code == 201
    assert resp.json()["name"] == "exact"

    got = client.get(f"{_BASE}/exact")
    assert got.status_code == 200
    assert got.json()["payload_kind"] == "cloudpickle"


def test_create_duplicate_returns_409(client: TestClient) -> None:
    assert client.post(f"{_BASE}/exact", json=_create_body()).status_code == 201
    assert client.post(f"{_BASE}/exact", json=_create_body()).status_code == 409


def test_create_invalid_inline_metric_returns_422(client: TestClient) -> None:
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}")
    body = MetricInline.model_validate(
        bundle_metric(metric, InlineMetricBundlePackager()).model_dump(mode="json")
    ).model_dump(mode="json")
    body["payload"]["metric"].pop("type")

    assert client.post(f"{_BASE}/invalid", json=body).status_code == 422


def test_get_missing_returns_404(client: TestClient) -> None:
    assert client.get(f"{_BASE}/nope").status_code == 404


def test_list_returns_created_metrics(client: TestClient) -> None:
    client.post(f"{_BASE}/a", json=_create_body())
    client.post(f"{_BASE}/b", json=_create_body())

    resp = client.get(_BASE)
    assert resp.status_code == 200
    body = resp.json()
    assert {m["name"] for m in body["data"]} == {"a", "b"}
    assert body["pagination"]["total_results"] == 2


def test_create_then_delete(client: TestClient) -> None:
    client.post(f"{_BASE}/exact", json=_create_body())

    deleted = client.delete(f"{_BASE}/exact")
    assert deleted.status_code == 204
    assert client.get(f"{_BASE}/exact").status_code == 404


def test_delete_stale_version_returns_409_and_keeps_metric(metrics_route_harness: _MetricsRouteHarness) -> None:
    client = metrics_route_harness.client
    entity_client = metrics_route_harness.entity_client
    assert client.post(f"{_BASE}/exact", json=_create_body()).status_code == 201

    entity_client.bump_version_on_next_delete = True
    deleted = client.delete(f"{_BASE}/exact")

    assert deleted.status_code == 409
    assert entity_client.delete_expected_db_versions == [1]
    assert ("default", "exact") in entity_client.entities
    assert client.get(f"{_BASE}/exact").status_code == 200


def test_delete_missing_returns_404(client: TestClient) -> None:
    assert client.delete(f"{_BASE}/nope").status_code == 404


def test_metric_filter_translates_custom_fields_to_data_namespace() -> None:
    # metric_type/description are custom (data.*) fields; base columns (name) pass through. Without
    # this translation the entity store can't resolve the field and 500s (matches the result filters).
    from nemo_evaluator.api.schemas import MetricFilter
    from nemo_helix_plugin.api.filter import ComparisonOperation, FilterOperator, LogicalOperation

    assert MetricFilter._get_entity_field_map() == {
        "metric_type": "data.metric_type",
        "description": "data.description",
        "derived": "data.derived",
    }
    op = LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            ComparisonOperation(field="metric_type", operator=FilterOperator.EQ, value="exact-match"),
            ComparisonOperation(field="name", operator=FilterOperator.EQ, value="m"),
        ],
    )
    assert MetricFilter.translate_operation(op).to_dict() == {
        "$and": [{"data.metric_type": {"$eq": "exact-match"}}, {"name": {"$eq": "m"}}]
    }
