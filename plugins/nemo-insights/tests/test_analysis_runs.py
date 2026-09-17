# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Insights analysis-runs facade over the generic ``agents.execute`` job."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, cast

import httpx
import pytest
from fastapi import HTTPException
from nemo_agents_plugin.entities import AgentInline
from nemo_agents_plugin.jobs.execute import ExecuteAgentJobConfig
from nemo_insights_plugin.analysis_runs import (
    ANALYSIS_RUN_NAME_PREFIX,
    CreateAnalysisRunRequest,
    build_execute_agent_job_config,
    create_analysis_run,
    get_analysis_run,
    list_analysis_runs,
    mint_analysis_run_name,
)
from nemo_insights_plugin.entities import AnalysisRun
from nemo_insights_plugin.schema import AnalysisRunPage
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.agents.client import AsyncAgentsClient
from nemo_platform_plugin.agents.types import CreateExecuteJobRequest
from nemo_platform_plugin.client.errors import NemoHTTPError, NemoTransportError, raise_for_status
from nemo_platform_plugin.entities.base import ListResponse, PaginationInfo
from nemo_platform_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError
from nemo_platform_plugin.entity_naming import NAME_MAX_LENGTH, NAME_PATTERN
from nemo_platform_plugin.models.client import AsyncModelsClient
from pydantic import ValidationError

DEFAULT_MODEL = "default/big"
FAST_MODEL = "default/small"
RUN_NAME = "insights-run-0123456789abcdef0123456789abcdef"
DECLARED_SORT_DEFAULT = "-created_at"


def _request(**overrides: Any) -> CreateAnalysisRunRequest:
    """A valid request; the model pair is required so every call must carry it."""
    return CreateAnalysisRunRequest(
        agent=overrides.pop("agent", "demo-agent"),
        default_model=overrides.pop("default_model", DEFAULT_MODEL),
        fast_model=overrides.pop("fast_model", FAST_MODEL),
        **overrides,
    )


class _StubExecuteJobs:
    def __init__(
        self,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
        get_error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.gets: list[str] = []
        self._response = response or {"name": RUN_NAME, "status": "created"}
        self._error = error
        self._get_error = get_error

    async def create(self, *, spec: dict[str, Any], name: str | None = None, workspace: str) -> dict[str, Any]:
        self.calls.append({"spec": spec, "name": name, "workspace": workspace})
        if self._error is not None:
            raise self._error
        return self._response

    async def get(self, name: str, *, workspace: str) -> dict[str, Any]:
        self.gets.append(name)
        if self._get_error is not None:
            raise self._get_error
        return self._response


class _StubModels:
    """Model Entity lookups the create path makes before recording a run."""

    def __init__(self, error: Exception | None = None) -> None:
        self.retrieved: list[tuple[str, str]] = []
        self._error = error

    async def retrieve(self, name: str, *, workspace: str) -> object:
        self.retrieved.append((workspace, name))
        if self._error is not None:
            raise self._error
        return object()


class _TypedResponse:
    def __init__(self, body: object) -> None:
        self._body = body

    def data(self) -> object:
        return self._body


class _TypedAgentsClient:
    def __init__(self, jobs: _StubExecuteJobs) -> None:
        self._jobs = jobs

    async def create_execute_job(
        self, *, body: CreateExecuteJobRequest, workspace: str | None = None
    ) -> _TypedResponse:
        assert workspace is not None
        return _TypedResponse(await self._jobs.create(spec=body.spec, name=body.name, workspace=workspace))

    async def get_execute_job(self, *, name: str, workspace: str | None = None) -> _TypedResponse:
        assert workspace is not None
        return _TypedResponse(await self._jobs.get(name, workspace=workspace))


class _TypedModelsClient:
    def __init__(self, models: _StubModels) -> None:
        self._models = models

    async def get_model(self, *, name: str, workspace: str | None = None) -> _TypedResponse:
        assert workspace is not None
        return _TypedResponse(await self._models.retrieve(name, workspace=workspace))


class _StubSdk:
    """Minimal stand-in for the request-scoped ``AsyncNeMoPlatform``."""

    def __init__(self, jobs: _StubExecuteJobs, models: _StubModels | None = None) -> None:
        self.jobs = jobs
        self.agents = type("_Agents", (), {"jobs": type("_Jobs", (), {"execute": jobs})()})()
        self.models = models or _StubModels()


@pytest.fixture(autouse=True)
def _patch_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_client_from_platform(platform: _StubSdk, client_cls: type[object]) -> object:
        if client_cls is AsyncAgentsClient:
            return _TypedAgentsClient(platform.jobs)
        if client_cls is AsyncModelsClient:
            return _TypedModelsClient(platform.models)
        raise AssertionError(f"unexpected client type: {client_cls!r}")

    monkeypatch.setattr("nemo_insights_plugin.analysis_runs.client_from_platform", fake_client_from_platform)


class _StubEntities:
    """Records entity writes so the ordering against job creation is observable."""

    def __init__(self, existing: AnalysisRun | None = None, create_error: Exception | None = None) -> None:
        self.created: list[AnalysisRun] = []
        self.list_queries: list[dict[str, Any]] = []
        self._existing = existing
        self._create_error = create_error

    async def create(self, entity: AnalysisRun) -> AnalysisRun:
        if self._create_error is not None:
            raise self._create_error
        self.created.append(entity)
        return entity

    async def get(self, _type: type, *, name: str, workspace: str) -> AnalysisRun:
        if self._existing is None:
            raise NemoEntityNotFoundError(f"{workspace}/{name}")
        return self._existing

    async def list(self, _type: type, **query: Any) -> ListResponse[AnalysisRun]:
        self.list_queries.append(query)
        return ListResponse(
            data=[self._existing] if self._existing is not None else [],
            pagination=PaginationInfo(
                page=query["page"],
                page_size=query["page_size"],
                current_page_size=1 if self._existing is not None else 0,
                total_pages=1,
                total_results=1 if self._existing is not None else 0,
            ),
        )


def _sdk(jobs: _StubExecuteJobs, models: _StubModels | None = None) -> AsyncNeMoPlatform:
    """The route touches ``sdk.agents.jobs.execute`` and ``sdk.models``; cast past the concrete type."""
    return cast(AsyncNeMoPlatform, _StubSdk(jobs, models))


def _entities(stub: _StubEntities) -> NemoEntitiesClient:
    return cast(NemoEntitiesClient, stub)


def _raise(error: Exception) -> Any:
    """Replace a stub method with one that fails, for the error-path tests."""

    async def _fail(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    return _fail


def _nemo_http_error(status_code: int, body: Any) -> NemoHTTPError:
    request = httpx.Request("POST", "http://platform/apis/agents/v2/workspaces/default/jobs/execute")
    response = httpx.Response(status_code, json=body, request=request)
    try:
        raise_for_status(response)
    except NemoHTTPError as exc:
        return exc
    raise AssertionError("expected raise_for_status to fail")


def _transport_error(method: str, url: str) -> NemoTransportError:
    return NemoTransportError(httpx.ConnectError("boom", request=httpx.Request(method, url)))


def _run(**overrides: Any) -> AnalysisRun:
    return AnalysisRun(
        name=overrides.pop("name", RUN_NAME),
        workspace=overrides.pop("workspace", "default"),
        agent=overrides.pop("agent", "demo-agent"),
        **overrides,
    )


# ---------------------------------------------------------------------------
# Run naming — the link between a run and its job
# ---------------------------------------------------------------------------


def test_minted_run_name_is_a_valid_entity_name() -> None:
    """The name doubles as the job name, so it must satisfy the platform pattern."""
    import re

    name = mint_analysis_run_name()

    assert name.startswith(ANALYSIS_RUN_NAME_PREFIX)
    assert len(name) <= NAME_MAX_LENGTH
    assert re.match(NAME_PATTERN, name)


def test_minted_run_names_are_unique() -> None:
    """Uniqueness comes from the uuid, not the agent — an agent-derived name collided."""
    assert len({mint_analysis_run_name() for _ in range(100)}) == 100


# ---------------------------------------------------------------------------
# Job spec construction
# ---------------------------------------------------------------------------


def test_execute_job_config_validates_against_the_real_job_schema() -> None:
    spec = build_execute_agent_job_config(_request(), workspace="team-a", run_name=RUN_NAME)

    config = ExecuteAgentJobConfig.model_validate(spec)

    assert config.extension is not None
    assert config.extension.kind == "insights.analysis"


def test_the_extension_is_scoped_to_the_agent_and_workspace() -> None:
    """The extension needs no run identity: nothing it writes is stamped with one."""
    spec = build_execute_agent_job_config(_request(), workspace="team-a", run_name=RUN_NAME)

    config = ExecuteAgentJobConfig.model_validate(spec)

    assert config.extension is not None
    assert config.extension.config == {
        "agent": "demo-agent",
        "workspace": "team-a",
    }


def test_analyst_is_submitted_inline_with_the_requested_models() -> None:
    """There is no Analyst Agent entity; the request composes one per run."""
    spec = build_execute_agent_job_config(_request(), workspace="team-a", run_name=RUN_NAME)

    config = ExecuteAgentJobConfig.model_validate(spec)

    assert isinstance(config.agent, AgentInline)
    assert config.agent.config["models"]["default"]["model"] == DEFAULT_MODEL
    assert config.agent.config["models"]["fast"]["model"] == FAST_MODEL
    assert config.agent.config["harnesses"]["insights"]["settings"]["agent"] == "demo-agent"


def test_read_scope_reaches_the_inline_analyst_settings() -> None:
    request = _request(since=datetime(2026, 8, 1, tzinfo=timezone.utc), evaluation_id="eval-123")

    config = ExecuteAgentJobConfig.model_validate(
        build_execute_agent_job_config(request, workspace="default", run_name=RUN_NAME)
    )

    assert isinstance(config.agent, AgentInline)
    settings = config.agent.config["harnesses"]["insights"]["settings"]
    assert settings["since"] == "2026-08-01T00:00:00+00:00"
    assert settings["evaluation_id"] == "eval-123"


def test_ethos_is_inlined_into_the_analyst_harness_settings() -> None:
    """Parity with AnalyzeSpec.ethos: the Fabric adapter has no Files access to resolve a ref."""
    request = _request(ethos="# Ethos\n\nBe careful.")

    config = ExecuteAgentJobConfig.model_validate(
        build_execute_agent_job_config(request, workspace="default", run_name=RUN_NAME)
    )

    assert isinstance(config.agent, AgentInline)
    assert config.agent.config["harnesses"]["insights"]["settings"]["ethos"] == "# Ethos\n\nBe careful."


def test_ethos_is_omitted_when_unset() -> None:
    config = ExecuteAgentJobConfig.model_validate(
        build_execute_agent_job_config(_request(), workspace="default", run_name=RUN_NAME)
    )

    assert isinstance(config.agent, AgentInline)
    assert "ethos" not in config.agent.config["harnesses"]["insights"]["settings"]


@pytest.mark.parametrize("field", ["agent", "evaluation_id", "ethos", "default_model", "fast_model"])
@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_settings_are_rejected(field: str, blank: str) -> None:
    """These reach the adapter verbatim, which rejects a blank only once the job is running."""
    body = {"agent": "demo-agent", "default_model": DEFAULT_MODEL, "fast_model": FAST_MODEL, field: blank}

    with pytest.raises(ValidationError):
        CreateAnalysisRunRequest.model_validate(body)


def test_surrounding_whitespace_is_trimmed_before_it_reaches_the_inline_analyst() -> None:
    request = _request(agent="  demo-agent  ", evaluation_id="  eval-123\n", ethos="\n# Ethos\n")

    config = ExecuteAgentJobConfig.model_validate(
        build_execute_agent_job_config(request, workspace="team-a", run_name=RUN_NAME)
    )

    assert isinstance(config.agent, AgentInline)
    settings = config.agent.config["harnesses"]["insights"]["settings"]
    assert settings["agent"] == "demo-agent"
    assert settings["evaluation_id"] == "eval-123"
    assert settings["ethos"] == "# Ethos"
    assert config.extension is not None
    assert config.extension.config["agent"] == "demo-agent"


def test_execute_job_config_omits_timeout_when_unset() -> None:
    spec = build_execute_agent_job_config(_request(), workspace="default", run_name=RUN_NAME)

    assert "timeout_seconds" not in spec


def test_execute_job_config_carries_timeout_when_set() -> None:
    spec = build_execute_agent_job_config(_request(timeout_seconds=120.0), workspace="default", run_name=RUN_NAME)

    assert ExecuteAgentJobConfig.model_validate(spec).timeout_seconds == 120.0


def test_model_refs_are_required() -> None:
    """The pair lives only in the operator's CLI config, so the request must carry it."""
    with pytest.raises(ValidationError):
        CreateAnalysisRunRequest.model_validate({"agent": "demo-agent"})


# ---------------------------------------------------------------------------
# Create: record first, then submit under the same name
# ---------------------------------------------------------------------------


async def test_create_records_the_run_before_submitting_the_job() -> None:
    jobs = _StubExecuteJobs()
    entities = _StubEntities()

    response = await create_analysis_run("team-a", _request(), _sdk(jobs), _entities(entities))

    assert len(entities.created) == 1
    assert response.run.agent == "demo-agent"
    assert response.run.workspace == "team-a"
    assert response.job == {"name": RUN_NAME, "status": "created"}


async def test_the_job_takes_the_run_name_so_the_link_needs_no_write_back() -> None:
    jobs = _StubExecuteJobs()
    entities = _StubEntities()

    response = await create_analysis_run("default", _request(), _sdk(jobs), _entities(entities))

    assert jobs.calls[0]["name"] == response.run.name
    assert response.run.name.startswith(ANALYSIS_RUN_NAME_PREFIX)


async def test_the_run_captures_the_request_scope() -> None:
    entities = _StubEntities()
    request = _request(since=datetime(2026, 8, 1, tzinfo=timezone.utc), evaluation_id="eval-123")

    response = await create_analysis_run("default", request, _sdk(_StubExecuteJobs()), _entities(entities))

    assert response.run.since == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert response.run.evaluation_id == "eval-123"
    assert response.run.default_model == DEFAULT_MODEL
    assert response.run.fast_model == FAST_MODEL


async def test_a_bare_model_name_is_looked_up_in_the_run_workspace() -> None:
    models = _StubModels()

    await create_analysis_run(
        "team-a",
        _request(default_model="big", fast_model="small"),
        _sdk(_StubExecuteJobs(), models),
        _entities(_StubEntities()),
    )

    assert models.retrieved == [("team-a", "big"), ("team-a", "small")]


async def test_a_denied_model_lookup_keeps_the_status_the_store_returned() -> None:
    """A cross-workspace ref the caller cannot read is a 403, not a malformed request."""
    denied = _nemo_http_error(403, {"detail": "no access to workspace-b"})
    entities = _StubEntities()

    with pytest.raises(HTTPException) as excinfo:
        await create_analysis_run(
            "workspace-a",
            _request(default_model="workspace-b/foo"),
            _sdk(_StubExecuteJobs(), _StubModels(error=denied)),
            _entities(entities),
        )

    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "no access to workspace-b"
    assert entities.created == []


async def test_an_unreachable_models_service_is_not_reported_as_a_bad_request() -> None:
    """Nothing is recorded yet, so this is a 503 rather than a 422 blaming the caller."""
    unreachable = _transport_error("GET", "http://platform/models")
    entities = _StubEntities()

    with pytest.raises(HTTPException) as excinfo:
        await create_analysis_run(
            "default", _request(), _sdk(_StubExecuteJobs(), _StubModels(error=unreachable)), _entities(entities)
        )

    assert excinfo.value.status_code == 503
    assert entities.created == []


async def test_nothing_is_submitted_when_the_run_cannot_be_recorded() -> None:
    jobs = _StubExecuteJobs()
    entities = _StubEntities(create_error=RuntimeError("store down"))

    with pytest.raises(HTTPException) as excinfo:
        await create_analysis_run("default", _request(), _sdk(jobs), _entities(entities))

    assert excinfo.value.status_code == 500
    assert jobs.calls == []


async def test_a_failed_submission_leaves_the_run_record_in_place() -> None:
    """Deleting it could orphan a job that a timed-out create actually landed."""
    jobs = _StubExecuteJobs(error=_nemo_http_error(422, {"detail": "bad model ref"}))
    entities = _StubEntities()

    with pytest.raises(HTTPException) as excinfo:
        await create_analysis_run("default", _request(), _sdk(jobs), _entities(entities))

    assert excinfo.value.status_code == 422
    assert excinfo.value.detail == {"error": "bad model ref", "run": entities.created[0].name}
    assert len(entities.created) == 1


async def test_an_unreachable_jobs_service_leaves_a_findable_run_record() -> None:
    """Transport errors are siblings of HTTP status errors, so they need their own arm.

    This is the case a retry could plausibly fix, so the orphan has to be
    findable afterwards rather than vanishing into an unhandled 500.
    """
    unreachable = _transport_error("POST", "http://platform/jobs/execute")
    jobs = _StubExecuteJobs(error=unreachable)
    entities = _StubEntities()

    with pytest.raises(HTTPException) as excinfo:
        await create_analysis_run("default", _request(), _sdk(jobs), _entities(entities))

    assert excinfo.value.status_code == 503
    assert len(entities.created) == 1
    # The caller cannot recover a run it was never told the name of.
    assert isinstance(excinfo.value.detail, dict)
    assert excinfo.value.detail["run"] == entities.created[0].name


async def test_a_failed_submission_falls_back_to_the_raw_error_body() -> None:
    """A body with no ``detail`` key is surfaced whole rather than dropped."""
    jobs = _StubExecuteJobs(error=_nemo_http_error(500, {"message": "upstream exploded"}))
    entities = _StubEntities()

    with pytest.raises(HTTPException) as excinfo:
        await create_analysis_run("default", _request(), _sdk(jobs), _entities(entities))

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == {"error": {"message": "upstream exploded"}, "run": entities.created[0].name}


# ---------------------------------------------------------------------------
# Read: join the run with its job by derived name
# ---------------------------------------------------------------------------


async def test_get_joins_the_run_with_its_backing_job() -> None:
    jobs = _StubExecuteJobs()
    entities = _StubEntities(existing=_run())

    response = await get_analysis_run("default", RUN_NAME, _sdk(jobs), _entities(entities))

    assert jobs.gets == [RUN_NAME]
    assert response.job is not None
    assert response.run.name == RUN_NAME


async def test_a_run_whose_job_is_missing_reads_as_never_submitted() -> None:
    """This is the disambiguation: no job under the run's name means it never landed."""
    jobs = _StubExecuteJobs(get_error=_nemo_http_error(404, {"detail": "not found"}))
    entities = _StubEntities(existing=_run())

    response = await get_analysis_run("default", RUN_NAME, _sdk(jobs), _entities(entities))

    assert response.job is None
    assert response.run.name == RUN_NAME


async def test_a_non_404_job_lookup_failure_is_not_swallowed() -> None:
    jobs = _StubExecuteJobs(get_error=_nemo_http_error(503, {"detail": "jobs down"}))
    entities = _StubEntities(existing=_run())

    with pytest.raises(NemoHTTPError):
        await get_analysis_run("default", RUN_NAME, _sdk(jobs), _entities(entities))


async def test_get_returns_404_for_an_unknown_run() -> None:
    entities = _StubEntities()

    with pytest.raises(HTTPException) as excinfo:
        await get_analysis_run("default", RUN_NAME, _sdk(_StubExecuteJobs()), _entities(entities))

    assert excinfo.value.status_code == 404


# ---------------------------------------------------------------------------
# List: filter, sort and pagination assembly
# ---------------------------------------------------------------------------


async def _list(entities: _StubEntities, **overrides: Any) -> AnalysisRunPage:
    """Call the handler directly, supplying what FastAPI would otherwise resolve.

    A direct call bypasses dependency resolution, so the ``Query`` defaults
    arrive as ``Query`` objects rather than values; pass them explicitly.
    """
    return await list_analysis_runs(
        overrides.pop("workspace", "default"),
        page=overrides.pop("page", 1),
        page_size=overrides.pop("page_size", 20),
        sort=overrides.pop("sort", DECLARED_SORT_DEFAULT),
        agent=overrides.pop("agent", None),
        entity_client=_entities(entities),
        **overrides,
    )


def test_the_route_declares_the_expected_sort_default() -> None:
    """Direct handler calls cannot observe it, so pin the declaration itself."""
    assert inspect.signature(list_analysis_runs).parameters["sort"].default.default == "-created_at"


async def test_list_passes_the_page_and_sort_through_to_the_entity_query() -> None:
    entities = _StubEntities(existing=_run())

    page = await list_analysis_runs(
        "team-a", page=2, page_size=5, sort="created_at", agent=None, entity_client=_entities(entities)
    )

    query = entities.list_queries[0]
    assert (query["workspace"], query["page"], query["page_size"], query["sort"]) == ("team-a", 2, 5, "created_at")
    assert page.sort == "created_at"
    assert page.pagination is not None
    assert (page.pagination.page, page.pagination.total_results) == (2, 1)


async def test_list_builds_an_agent_filter_only_when_one_is_asked_for() -> None:
    """An empty filter must go down as None, not as an empty dict the store would apply."""
    with_agent = _StubEntities(existing=_run())
    without_agent = _StubEntities(existing=_run())

    filtered = await _list(with_agent, agent="demo-agent")
    unfiltered = await _list(without_agent, agent=None)

    assert with_agent.list_queries[0]["filter_obj"] == {"agent": "demo-agent"}
    assert filtered.filter == {"agent": "demo-agent"}
    assert without_agent.list_queries[0]["filter_obj"] is None
    assert unfiltered.filter is None


async def test_list_surfaces_a_store_failure_as_a_500() -> None:
    entities = _StubEntities()
    entities.list = _raise(RuntimeError("entity store down"))  # type: ignore[method-assign]

    with pytest.raises(HTTPException) as excinfo:
        await _list(entities)

    assert excinfo.value.status_code == 500
