# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP route-level tests for the eval-result read endpoints.

Drives the real FastAPI routers + ResultService through a TestClient with an in-memory entity store.
Covers route wiring, the get_result_service dependency, and status-code mapping (200/204/404), plus
that the two collections (agent-eval vs row-eval) stay separate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TypeVar

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_evaluator.api.dependencies import get_result_service
from nemo_evaluator.api.service.result_service import ResultService
from nemo_evaluator.api.v2 import results as results_routes
from nemo_evaluator.entities import AgentEvalResultEntity, EvaluateResultEntity
from nemo_evaluator_sdk.values.results import AggregatedMetricResult
from nemo_helix_plugin.entities import ListResponse, PaginationInfo
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError

_ResultEntityT = TypeVar("_ResultEntityT", AgentEvalResultEntity, EvaluateResultEntity)


class _FakeEntityClient:
    def __init__(self) -> None:
        self.entities: dict[tuple[str, str, str], AgentEvalResultEntity | EvaluateResultEntity] = {}

    def seed(self, entity: _ResultEntityT) -> _ResultEntityT:
        now = datetime.now(timezone.utc)
        entity._id = f"{entity.__entity_type__}-{entity.name}"
        entity._created_at = now
        entity._updated_at = now
        self.entities[(entity.__entity_type__, entity.workspace, entity.name)] = entity
        return entity

    async def get(
        self, entity_type: type[_ResultEntityT], *, workspace: str, name: str, parent: str | None = None
    ) -> _ResultEntityT:
        key = (entity_type.__entity_type__, workspace, name)
        if key not in self.entities:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")
        entity = self.entities[key]
        assert isinstance(entity, entity_type)
        return entity

    async def delete(
        self,
        entity_type: type[_ResultEntityT],
        name: str,
        *,
        workspace: str,
        parent: str | None = None,
        expected_db_version: int | None = None,
    ) -> None:
        # Mirror the real EntityClient: raise NemoEntityNotFoundError when absent.
        key = (entity_type.__entity_type__, workspace, name)
        if key not in self.entities:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")
        del self.entities[key]

    async def list(
        self,
        entity_type: type[_ResultEntityT],
        *,
        workspace: str,
        filter_operation: object | None = None,
        sort: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ListResponse[_ResultEntityT]:
        items: list[_ResultEntityT] = []
        for (etype, ws, _), entity in self.entities.items():
            if etype == entity_type.__entity_type__ and ws == workspace:
                assert isinstance(entity, entity_type)
                items.append(entity)
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


def _agent_entity(name: str) -> AgentEvalResultEntity:
    return AgentEvalResultEntity(
        name=name,
        workspace="default",
        job_id=name,
        target_kind="fabric",
        target_name="gpt-5.5",
        target_url=None,
        scores=AggregatedMetricResult(scores=[]),
        bundle_ref="fileset://default/agent-eval-results#b",
    )


def _eval_entity(name: str) -> EvaluateResultEntity:
    return EvaluateResultEntity(
        name=name,
        workspace="default",
        job_id=name,
        target_kind="model",
        target_name="m",
        target_url=None,
        scores=AggregatedMetricResult(scores=[]),
        bundle_ref="fileset://default/eval-results#b",
        dataset_ref="default/ds",
        metric_types=["exact_match"],
    )


@pytest.fixture
def fake() -> _FakeEntityClient:
    return _FakeEntityClient()


@pytest.fixture
def client(fake: _FakeEntityClient) -> TestClient:
    app = FastAPI()
    prefix = "/v2/workspaces/{workspace}"
    app.include_router(results_routes.agent_eval_results_router, prefix=prefix)
    app.include_router(results_routes.evaluate_results_router, prefix=prefix)
    service = ResultService(fake)
    app.dependency_overrides[get_result_service] = lambda: service
    return TestClient(app)


_AGENT = "/v2/workspaces/default/agent-eval-results"
_EVAL = "/v2/workspaces/default/eval-results"


def test_list_agent_eval_results(client: TestClient, fake: _FakeEntityClient) -> None:
    fake.seed(_agent_entity("job-1"))
    fake.seed(_agent_entity("job-2"))

    resp = client.get(_AGENT)
    assert resp.status_code == 200
    body = resp.json()
    assert {e["name"] for e in body["data"]} == {"job-1", "job-2"}
    assert body["pagination"]["total_results"] == 2


def test_get_eval_result_returns_typed_payload(client: TestClient, fake: _FakeEntityClient) -> None:
    fake.seed(_eval_entity("job-9"))

    resp = client.get(f"{_EVAL}/job-9")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job-9"
    assert body["dataset_ref"] == "default/ds"
    assert body["metric_types"] == ["exact_match"]


def test_get_missing_returns_404(client: TestClient) -> None:
    assert client.get(f"{_AGENT}/nope").status_code == 404
    assert client.get(f"{_EVAL}/nope").status_code == 404


def test_delete_then_get_404(client: TestClient, fake: _FakeEntityClient) -> None:
    fake.seed(_agent_entity("job-1"))

    assert client.delete(f"{_AGENT}/job-1").status_code == 204
    assert client.get(f"{_AGENT}/job-1").status_code == 404


def test_delete_missing_returns_404(client: TestClient) -> None:
    assert client.delete(f"{_EVAL}/nope").status_code == 404


def test_delete_agent_eval_result_conflict_returns_409() -> None:
    class _Service:
        async def delete_agent_eval_result(self, workspace: str, name: str) -> bool:
            raise NemoEntityConflictError("changed")

    app = FastAPI()
    app.include_router(results_routes.agent_eval_results_router, prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_result_service] = lambda: _Service()
    client = TestClient(app)

    assert client.delete(f"{_AGENT}/job-1").status_code == 409


def test_delete_eval_result_conflict_returns_409() -> None:
    class _Service:
        async def delete_eval_result(self, workspace: str, name: str) -> bool:
            raise NemoEntityConflictError("changed")

    app = FastAPI()
    app.include_router(results_routes.evaluate_results_router, prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_result_service] = lambda: _Service()
    client = TestClient(app)

    assert client.delete(f"{_EVAL}/job-1").status_code == 409


def test_filter_translates_custom_fields_to_data_namespace() -> None:
    # Custom (non-base) trait fields must be rewritten to data.* for the entity store; base columns
    # (workspace, created_at) pass through. The plain Filter does no translation and the store 500s.
    from nemo_evaluator.api.v2.results import EvaluateResultFilter
    from nemo_helix_plugin.api.filter import ComparisonOperation, FilterOperator, LogicalOperation

    assert EvaluateResultFilter._get_entity_field_map() == {
        "job_id": "data.job_id",
        "target_kind": "data.target_kind",
        "target_name": "data.target_name",
        "dataset_ref": "data.dataset_ref",
    }
    op = LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            ComparisonOperation(field="job_id", operator=FilterOperator.EQ, value="j1"),
            ComparisonOperation(field="workspace", operator=FilterOperator.EQ, value="default"),
        ],
    )
    assert EvaluateResultFilter.translate_operation(op).to_dict() == {
        "$and": [{"data.job_id": {"$eq": "j1"}}, {"workspace": {"$eq": "default"}}]
    }


def test_collections_do_not_collide(client: TestClient, fake: _FakeEntityClient) -> None:
    # Same name in both collections: each endpoint sees only its own type.
    fake.seed(_agent_entity("shared"))
    fake.seed(_eval_entity("shared"))

    assert client.get(f"{_AGENT}/shared").json()["target_kind"] == "fabric"
    assert client.get(f"{_EVAL}/shared").json()["dataset_ref"] == "default/ds"
