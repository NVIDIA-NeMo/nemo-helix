# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection isolation for job schemas that intentionally share one source."""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client, get_request_authorizer
from nemo_helix_plugin.jobs.api_factory import JobRouteOption, job_route_factory
from pydantic import BaseModel, ConfigDict, RootModel
from starlette.testclient import TestClient

_DISCRIMINATOR_KEY = "_nemo_job_route"


class _Spec(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: str


def _compiler(*_args: object, **_kwargs: object) -> dict[str, object]:
    for arg in _args:
        if isinstance(arg, BaseModel):
            assert _DISCRIMINATOR_KEY not in arg.model_dump()
    return {
        "steps": [
            {
                "name": "step",
                "executor": {"provider": "cpu", "container": {"image": "test-image"}},
            }
        ]
    }


def _job(name: str, discriminator: str | None, *, source: str = "agent-hardener") -> SimpleNamespace:
    spec: dict[str, object] = {"value": name}
    if discriminator is not None:
        spec[_DISCRIMINATOR_KEY] = discriminator
    return SimpleNamespace(
        id=f"id-{name}",
        name=name,
        description=None,
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
    )


class _Response:
    def __init__(self, value: object) -> None:
        self._value = value

    def data(self) -> object:
        return self._value


class _PageResponse:
    def __init__(self, items: Sequence[object]) -> None:
        self._items = list(items)

    def page(self) -> SimpleNamespace:
        return SimpleNamespace(
            items=self._items,
            metadata={
                "page": 1,
                "page_size": 10,
                "current_page_size": len(self._items),
                "total_pages": 1,
                "total_results": len(self._items),
            },
        )


class _JobsClient:
    def __init__(
        self,
        jobs: dict[str, SimpleNamespace] | None = None,
        *,
        list_items: list[SimpleNamespace] | None = None,
    ) -> None:
        self.jobs = jobs or {}
        self.list_items = list_items or []
        self.created_body: object | None = None
        self.list_query: dict[str, object] | None = None
        self.get_calls: list[str] = []
        self.action_calls: list[str] = []

    def with_retry(self, _policy: object) -> _JobsClient:
        return self

    async def create_job(self, *, workspace: str, body: object) -> _Response:
        self.created_body = body
        job = _job(getattr(body, "name", None) or "generated", None, source=getattr(body, "source"))
        job.workspace = workspace
        job.spec = dict(getattr(body, "spec"))
        self.jobs[job.name] = job
        return _Response(job)

    async def list_jobs(self, **kwargs: object) -> _PageResponse:
        query_params = kwargs["query_params"]
        assert isinstance(query_params, dict)
        self.list_query = dict(query_params)
        return _PageResponse(self.list_items)

    async def get_job(self, *, name: str, workspace: str) -> _Response:
        del workspace
        self.get_calls.append(name)
        return _Response(self.jobs[name])

    async def _unexpected_action(self, action: str) -> None:
        self.action_calls.append(action)
        raise AssertionError(f"{action} must not run before collection ownership is verified")

    async def delete_job(self, **_kwargs: object) -> None:
        await self._unexpected_action("delete")

    async def cancel_job(self, **_kwargs: object) -> _Response:
        await self._unexpected_action("cancel")
        raise AssertionError("unreachable")

    async def pause_job(self, **_kwargs: object) -> _Response:
        await self._unexpected_action("pause")
        raise AssertionError("unreachable")

    async def resume_job(self, **_kwargs: object) -> _Response:
        await self._unexpected_action("resume")
        raise AssertionError("unreachable")

    async def get_job_status(self, **_kwargs: object) -> _Response:
        await self._unexpected_action("status")
        raise AssertionError("unreachable")

    async def list_job_logs(self, **_kwargs: object) -> _PageResponse:
        await self._unexpected_action("logs")
        raise AssertionError("unreachable")

    async def list_job_results(self, **_kwargs: object) -> _Response:
        await self._unexpected_action("list-results")
        raise AssertionError("unreachable")

    async def get_job_result(self, **_kwargs: object) -> _Response:
        await self._unexpected_action("get-result")
        raise AssertionError("unreachable")


class _InternalNemoClient:
    """Async context-manager façade returned by the internal client factory."""

    def __init__(self, jobs_client: _JobsClient) -> None:
        self.jobs_client = jobs_client
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> _InternalNemoClient:
        self.entered = True
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.exited = True


class _WriteOnlyJobsClient(_JobsClient):
    """Caller client that can mutate but must never be used for ownership reads."""

    async def get_job(self, *, name: str, workspace: str) -> _Response:
        del name, workspace
        raise AssertionError("the caller's jobs.read permission must not be used for mutation preflight")

    async def delete_job(self, **_kwargs: object) -> None:
        self.action_calls.append("delete")

    async def cancel_job(self, **kwargs: object) -> _Response:
        return self._mutation_response("cancel", cast(str, kwargs["name"]), cast(str, kwargs["workspace"]))

    async def pause_job(self, **kwargs: object) -> _Response:
        return self._mutation_response("pause", cast(str, kwargs["name"]), cast(str, kwargs["workspace"]))

    async def resume_job(self, **kwargs: object) -> _Response:
        return self._mutation_response("resume", cast(str, kwargs["name"]), cast(str, kwargs["workspace"]))

    def _mutation_response(self, action: str, name: str, workspace: str) -> _Response:
        self.action_calls.append(action)
        job = _job(name, "war-game")
        job.workspace = workspace
        return _Response(job)


def _build_app(jobs_client: _JobsClient, *, authorize: Any | None = None) -> FastAPI:
    async def allow(_method: str, _path: str) -> None:
        return None

    app = FastAPI()
    app.include_router(
        job_route_factory(
            service_name="agent-hardener",
            job_type="WarGame",
            job_input=_Spec,
            platform_job_config_compiler=_compiler,
            route_options=[JobRouteOption.CORE, JobRouteOption.PAUSE_RESUME],
            job_discriminator="war-game",
        ),
        prefix="/war/{workspace}",
    )
    app.include_router(
        job_route_factory(
            service_name="agent-hardener",
            job_type="SynthBenign",
            job_input=_Spec,
            platform_job_config_compiler=_compiler,
            route_options=[JobRouteOption.CORE, JobRouteOption.PAUSE_RESUME],
            job_discriminator="synth-benign",
        ),
        prefix="/synth/{workspace}",
    )
    app.dependency_overrides[get_nemo_client] = lambda: SimpleNamespace(jobs_client=jobs_client)
    app.dependency_overrides[get_entity_client] = lambda: SimpleNamespace()
    app.dependency_overrides[get_request_authorizer] = lambda: authorize or allow
    return app


@pytest.fixture(autouse=True)
def _patch_jobs_client() -> Any:
    with patch(
        "nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client",
        side_effect=lambda client: client.jobs_client,
    ):
        yield


def test_create_stamps_unspoofable_discriminator_and_hides_it_from_response() -> None:
    jobs = _JobsClient()
    client = TestClient(_build_app(jobs))

    response = client.post(
        "/war/default/jobs",
        json={"name": "new-war", "spec": {"value": "payload", _DISCRIMINATOR_KEY: "synth-benign"}},
    )

    assert response.status_code == 201, response.text
    assert response.json()["spec"] == {"value": "payload"}
    assert jobs.created_body is not None
    assert getattr(jobs.created_body, "spec") == {
        "value": "payload",
        _DISCRIMINATOR_KEY: "war-game",
    }


def test_list_is_server_scoped_by_source_and_discriminator() -> None:
    jobs = _JobsClient()
    client = TestClient(_build_app(jobs))

    response = client.get("/war/default/jobs")

    assert response.status_code == 200, response.text
    assert jobs.list_query is not None
    assert json.loads(str(jobs.list_query["filter"])) == {
        "$and": [
            {"source": {"$eq": "agent-hardener"}},
            {f"spec.{_DISCRIMINATOR_KEY}": {"$eq": "war-game"}},
        ]
    }


def test_list_fails_closed_if_core_returns_another_subtype() -> None:
    jobs = _JobsClient(list_items=[_job("synth-1", "synth-benign")])
    client = TestClient(_build_app(jobs))

    response = client.get("/war/default/jobs")

    assert response.status_code == 502, response.text
    assert "outside the WarGame collection" in response.json()["detail"]


def test_get_accepts_only_matching_collection_and_hides_marker() -> None:
    jobs = _JobsClient({"war-1": _job("war-1", "war-game")})
    client = TestClient(_build_app(jobs))

    response = client.get("/war/default/jobs/war-1")

    assert response.status_code == 200, response.text
    assert response.json()["spec"] == {"value": "war-1"}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/war/default/jobs/synth-1"),
        ("get", "/war/default/jobs/synth-1/status"),
        ("delete", "/war/default/jobs/synth-1"),
        ("post", "/war/default/jobs/synth-1/cancel"),
        ("get", "/war/default/jobs/synth-1/logs"),
        ("get", "/war/default/jobs/synth-1/results"),
        ("get", "/war/default/jobs/synth-1/results/report"),
        ("get", "/war/default/jobs/synth-1/results/report/download"),
        ("post", "/war/default/jobs/synth-1/pause"),
        ("post", "/war/default/jobs/synth-1/resume"),
    ],
)
def test_every_name_scoped_route_rejects_another_subtype_before_action(method: str, path: str) -> None:
    jobs = _JobsClient({"synth-1": _job("synth-1", "synth-benign")})
    client = TestClient(_build_app(jobs))
    internal_client = _InternalNemoClient(jobs)

    with patch(
        "nemo_helix_plugin.jobs.api_factory.get_async_nemo_client",
        return_value=internal_client,
    ):
        response = client.request(method, path)

    assert response.status_code == 404, response.text
    assert jobs.action_calls == []


@pytest.mark.parametrize(
    ("method", "path", "status_code", "action", "authorized_method", "authorized_path"),
    [
        ("delete", "/war/default/jobs/war-1", 204, "delete", "DELETE", "/apis/jobs/v2/workspaces/default/jobs/war-1"),
        (
            "post",
            "/war/default/jobs/war-1/cancel",
            200,
            "cancel",
            "POST",
            "/apis/jobs/v2/workspaces/default/jobs/war-1/cancel",
        ),
        (
            "post",
            "/war/default/jobs/war-1/pause",
            200,
            "pause",
            "POST",
            "/apis/jobs/v2/workspaces/default/jobs/war-1/pause",
        ),
        (
            "post",
            "/war/default/jobs/war-1/resume",
            200,
            "resume",
            "POST",
            "/apis/jobs/v2/workspaces/default/jobs/war-1/resume",
        ),
    ],
)
def test_mutation_ownership_lookup_uses_internal_service_client_not_caller_read(
    method: str,
    path: str,
    status_code: int,
    action: str,
    authorized_method: str,
    authorized_path: str,
) -> None:
    authorization_calls: list[tuple[str, str]] = []

    async def authorize(call_method: str, call_path: str) -> None:
        authorization_calls.append((call_method, call_path))

    caller_jobs = _WriteOnlyJobsClient()
    ownership_jobs = _JobsClient({"war-1": _job("war-1", "war-game")})
    internal_client = _InternalNemoClient(ownership_jobs)
    client = TestClient(_build_app(caller_jobs, authorize=authorize))

    with patch(
        "nemo_helix_plugin.jobs.api_factory.get_async_nemo_client",
        return_value=internal_client,
    ) as client_factory:
        response = client.request(method, path)

    assert response.status_code == status_code, response.text
    assert authorization_calls == [(authorized_method, authorized_path)]
    assert ownership_jobs.get_calls == ["war-1"]
    assert caller_jobs.action_calls == [action]
    assert internal_client.entered and internal_client.exited
    client_factory.assert_called_once_with(
        as_service="agent-hardener",
        internal=True,
        workspace="default",
    )


def test_mutation_authorization_denial_happens_before_internal_ownership_lookup() -> None:
    async def deny(_method: str, _path: str) -> None:
        raise HTTPException(status_code=403, detail="mutation denied")

    caller_jobs = _WriteOnlyJobsClient()
    client = TestClient(_build_app(caller_jobs, authorize=deny))

    with patch("nemo_helix_plugin.jobs.api_factory.get_async_nemo_client") as client_factory:
        response = client.post("/war/default/jobs/unknown/cancel")

    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "mutation denied"
    client_factory.assert_not_called()
    assert caller_jobs.action_calls == []


def test_unmarked_legacy_job_is_not_guessed_into_a_discriminated_collection() -> None:
    jobs = _JobsClient({"legacy": _job("legacy", None)})
    client = TestClient(_build_app(jobs))

    response = client.get("/war/default/jobs/legacy")

    assert response.status_code == 404, response.text


@pytest.mark.parametrize("value", ["", "   ", 3])
def test_factory_rejects_invalid_discriminator(value: object) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        job_route_factory(
            service_name="agent-hardener",
            job_type="WarGame",
            job_input=_Spec,
            platform_job_config_compiler=_compiler,
            job_discriminator=cast(Any, value),
        )


def test_factory_rejects_root_model_for_discriminated_collection() -> None:
    class _ListSpec(RootModel[list[str]]):
        pass

    with pytest.raises(ValueError, match="object-shaped BaseModel"):
        job_route_factory(
            service_name="agent-hardener",
            job_type="WarGame",
            job_input=_ListSpec,
            platform_job_config_compiler=_compiler,
            job_discriminator="war-game",
        )
