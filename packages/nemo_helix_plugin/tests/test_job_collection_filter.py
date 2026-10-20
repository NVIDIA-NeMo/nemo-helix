# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Historical collection membership, pagination, attempts, and mutation authorization."""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from nemo_helix_plugin.api.filter import parse_json_filter
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client, get_request_authorizer
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.in_memory_filter import InMemoryFilterRepository
from nemo_helix_plugin.jobs.api_factory import JobRouteOption, job_route_factory
from pydantic import BaseModel, ConfigDict
from starlette.testclient import TestClient


class Spec(BaseModel):
    model_config = ConfigDict(extra="allow")
    value: str = "value"


def collection_filter(key: str, sibling: str) -> LogicalOperation:
    return LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            ComparisonOperation(operator=FilterOperator.HAS_KEY, field="spec", value=key),
            LogicalOperation(
                operator=FilterOperator.NOT,
                operations=[
                    ComparisonOperation(operator=FilterOperator.HAS_KEY, field="spec", value=sibling),
                ],
            ),
        ],
    )


def job(name: str, spec: dict[str, Any], source: str = "shared") -> SimpleNamespace:
    return SimpleNamespace(
        id=f"id-{name}",
        attempt_id="attempt-1",
        name=name,
        description=None,
        project=None,
        source=source,
        workspace="default",
        created_at=None,
        updated_at=None,
        spec=spec,
        status=None,
        status_details=None,
        error_details=None,
        ownership=None,
        custom_fields=None,
        control=None,
    )


def response(data: Any) -> SimpleNamespace:
    return SimpleNamespace(data=lambda: data)


class Jobs:
    def __init__(self) -> None:
        # Arbitrary names deliberately do not identify type. False and empty values are still present.
        self.rows = {
            "old-war": job("old-war", {"value": "old", "war": False}),
            "old-synth": job("old-synth", {"value": "old", "synth": ""}),
            "ambiguous": job("ambiguous", {"war": False, "synth": "interactive"}),
            "unknown": job("unknown", {"value": "unknown"}),
            "foreign": job("foreign", {"war": False}, source="other"),
        }
        self.actions: list[str] = []
        self.queries: list[dict[str, Any]] = []
        self.forbid_reads = False
        self.ignore_filter = False

    def with_retry(self, policy: Any) -> "Jobs":
        return self

    async def get_job(self, *, name: str, workspace: str) -> SimpleNamespace:
        assert not self.forbid_reads, "write-only caller must not need jobs.read"
        return response(self.rows[name])

    async def list_jobs(self, *, workspace: str, query_params: dict[str, Any]) -> SimpleNamespace:
        self.queries.append(query_params)
        operation = parse_json_filter(query_params["filter"])
        rows = list(self.rows.values())
        if not self.ignore_filter:
            rows = [row for row in rows if operation.apply(InMemoryFilterRepository(row))]
        size, page = query_params["page_size"], query_params["page"]
        selected = rows[(page - 1) * size : page * size]
        return SimpleNamespace(
            page=lambda: SimpleNamespace(
                items=selected,
                metadata={
                    "page": page,
                    "page_size": size,
                    "current_page_size": len(selected),
                    "total_results": len(rows),
                    "total_pages": (len(rows) + size - 1) // size,
                },
            )
        )

    async def create_job(self, *, workspace: str, body: Any) -> SimpleNamespace:
        self.actions.append("create")
        return response(job(body.name or "new", body.spec, body.source))

    async def update_job(self, *, name: str, workspace: str, body: Any) -> SimpleNamespace:
        self.actions.append("update")
        row = self.rows[name]
        if body.spec is not None:
            row.spec = body.spec
        return response(row)

    def attempt(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            id="attempt-1",
            name="attempt-1",
            job=f"id-{name}",
            workspace="default",
            seq=1,
            spec=self.rows[name].spec,
            status="error",
            status_details={},
            error_details=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            is_current=True,
        )

    async def get_job_attempt(self, *, job: str, **kwargs: Any) -> SimpleNamespace:
        self.actions.append("attempt")
        return response(self.attempt(job))

    async def list_job_attempts(self, *, job: str, **kwargs: Any) -> SimpleNamespace:
        self.actions.append("attempts")
        return SimpleNamespace(
            page=lambda: SimpleNamespace(
                items=[self.attempt(job)],
                metadata={
                    "page": 1,
                    "page_size": 25,
                    "current_page_size": 1,
                    "total_results": 1,
                    "total_pages": 1,
                },
            )
        )

    async def delete_job(self, **kwargs: Any) -> None:
        self.actions.append("delete")

    async def cancel_job(self, *, name: str, **kwargs: Any) -> SimpleNamespace:
        self.actions.append("cancel")
        return response(self.rows[name])

    async def pause_job(self, *, name: str, **kwargs: Any) -> SimpleNamespace:
        self.actions.append("pause")
        return response(self.rows[name])

    async def resume_job(self, *, name: str, **kwargs: Any) -> SimpleNamespace:
        self.actions.append("resume")
        return response(self.rows[name])


def build_app(jobs: Jobs, *, authorize: Any = None) -> FastAPI:
    async def allow(method: str, path: str) -> None:
        return None

    def compiler(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"steps": [{"name": "step", "executor": {"provider": "cpu", "container": {"image": "test"}}}]}

    app = FastAPI()
    for key, sibling in [("war", "synth"), ("synth", "war")]:
        app.include_router(
            job_route_factory(
                service_name="shared",
                job_type=key.capitalize(),
                job_input=Spec,
                platform_job_config_compiler=compiler,
                job_collection_filter=collection_filter(key, sibling),
                route_options=[JobRouteOption.CORE, JobRouteOption.PAUSE_RESUME],
            ),
            prefix=f"/{key}/{{workspace}}",
        )
    app.dependency_overrides[get_nemo_client] = lambda: SimpleNamespace(jobs=jobs)
    app.dependency_overrides[get_entity_client] = lambda: SimpleNamespace()
    app.dependency_overrides[get_request_authorizer] = lambda: authorize or allow
    return app


@pytest.fixture(autouse=True)
def patch_client() -> Any:
    with patch(
        "nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", side_effect=lambda client: client.jobs
    ):
        yield


def test_historical_rows_stay_visible_in_correct_collection() -> None:
    jobs = Jobs()
    client = TestClient(build_app(jobs))
    for collection in ("war", "synth"):
        result = client.get(f"/{collection}/default/jobs", params={"page_size": 1})
        assert result.status_code == 200, result.text
        assert [row["name"] for row in result.json()["data"]] == [f"old-{collection}"]
        assert result.json()["pagination"]["total_results"] == 1
        assert client.get(f"/{collection}/default/jobs/old-{collection}").status_code == 200


@pytest.mark.parametrize(
    "user_filter",
    [
        {"$or": [{"name": "old-war"}, {"name": "old-synth"}, {"name": "foreign"}]},
        {"$not": {"name": "unknown"}},
    ],
)
def test_user_logical_filters_cannot_escape_collection(user_filter: dict[str, Any]) -> None:
    client = TestClient(build_app(Jobs()))
    result = client.get("/war/default/jobs", params={"filter": json.dumps(user_filter), "page_size": 1})
    assert result.status_code == 200, result.text
    assert [row["name"] for row in result.json()["data"]] == ["old-war"]
    assert result.json()["pagination"]["total_results"] == 1


@pytest.mark.parametrize("name", ["old-synth", "ambiguous", "unknown", "foreign"])
@pytest.mark.parametrize(
    "method,suffix",
    [
        ("GET", ""),
        ("GET", "/status"),
        ("GET", "/logs"),
        ("GET", "/results"),
        ("GET", "/results/output"),
        ("GET", "/results/output/download"),
        ("GET", "/attempts"),
        ("GET", "/attempts/current"),
        ("DELETE", ""),
        ("POST", "/cancel"),
        ("POST", "/pause"),
        ("POST", "/resume"),
        ("PATCH", ""),
    ],
)
def test_all_named_routes_reject_siblings_before_action(name: str, method: str, suffix: str) -> None:
    jobs = Jobs()

    @asynccontextmanager
    async def internal(**kwargs: Any) -> Any:
        yield SimpleNamespace(jobs=jobs)

    with patch("nemo_helix_plugin.jobs.api_factory.get_async_nemo_client", internal):
        result = TestClient(build_app(jobs)).request(
            method,
            f"/war/default/jobs/{name}{suffix}",
            json={"description": "change"} if method == "PATCH" else None,
        )
    assert result.status_code == 404, result.text
    assert jobs.actions == []


@pytest.mark.parametrize(
    "method,suffix,action",
    [
        ("DELETE", "", "delete"),
        ("POST", "/cancel", "cancel"),
        ("POST", "/pause", "pause"),
        ("POST", "/resume", "resume"),
        ("PATCH", "", "update"),
    ],
)
def test_write_only_roles_authorized_before_internal_lookup(method: str, suffix: str, action: str) -> None:
    caller, internal_jobs = Jobs(), Jobs()
    caller.forbid_reads = True
    calls: list[str] = []

    async def authorize(actual_method: str, path: str) -> None:
        assert actual_method == method
        assert path == f"/apis/jobs/v2/workspaces/default/jobs/old-war{suffix}"
        calls.append("authorize")

    @asynccontextmanager
    async def internal(**kwargs: Any) -> Any:
        assert calls == ["authorize"]
        assert kwargs == {"as_service": "shared", "internal": True, "workspace": "default"}
        calls.append("lookup")
        yield SimpleNamespace(jobs=internal_jobs)

    with patch("nemo_helix_plugin.jobs.api_factory.get_async_nemo_client", internal):
        result = TestClient(build_app(caller, authorize=authorize)).request(
            method,
            f"/war/default/jobs/old-war{suffix}",
            json={"description": "change"} if method == "PATCH" else None,
        )
    assert result.status_code in (200, 204), result.text
    assert caller.actions == [action]


def test_matching_historical_job_retains_attempt_access() -> None:
    jobs = Jobs()
    client = TestClient(build_app(jobs))
    listed = client.get("/war/default/jobs/old-war/attempts")
    assert listed.status_code == 200, listed.text
    assert listed.json()["data"][0]["id"] == "attempt-1"
    selected = client.get("/war/default/jobs/old-war/attempts/current")
    assert selected.status_code == 200, selected.text
    assert selected.json()["spec"]["war"] is False
    assert jobs.actions == ["attempts", "attempt"]


def test_update_checks_new_spec_membership_before_compile_or_write() -> None:
    jobs = Jobs()

    @asynccontextmanager
    async def internal(**kwargs: Any) -> Any:
        yield SimpleNamespace(jobs=jobs)

    with patch("nemo_helix_plugin.jobs.api_factory.get_async_nemo_client", internal):
        client = TestClient(build_app(jobs))
        invalid = client.patch("/war/default/jobs/old-war", json={"spec": {"synth": "interactive"}})
        assert invalid.status_code == 422
        assert jobs.actions == []
        valid = client.patch("/war/default/jobs/old-war", json={"spec": {"war": False, "value": "new"}})
        assert valid.status_code == 200, valid.text
        assert valid.json()["spec"]["value"] == "new"
        assert jobs.actions == ["update"]


def test_denied_write_does_not_lookup_inventory() -> None:
    async def deny(method: str, path: str) -> None:
        raise HTTPException(403, "denied")

    with patch("nemo_helix_plugin.jobs.api_factory.get_async_nemo_client") as internal:
        result = TestClient(build_app(Jobs(), authorize=deny)).post("/war/default/jobs/old-war/cancel")
    assert result.status_code == 403
    internal.assert_not_called()


def test_rejects_invalid_compiled_collection_before_submission() -> None:
    jobs = Jobs()
    client = TestClient(build_app(jobs))
    result = client.post("/war/default/jobs", json={"name": "new", "spec": {"synth": "interactive"}})
    assert result.status_code == 422
    assert jobs.actions == []
    result = client.post("/war/default/jobs", json={"name": "new", "spec": {"war": False}})
    assert result.status_code == 201, result.text


def test_core_filter_violation_fails_closed() -> None:
    jobs = Jobs()
    jobs.ignore_filter = True
    result = TestClient(build_app(jobs)).get("/war/default/jobs")
    assert result.status_code == 502
