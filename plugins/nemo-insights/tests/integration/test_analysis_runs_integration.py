# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analysis runs through the SDK against real Insights, Agents and Jobs services.

The unit tests stub every seam in this path: the route tests stub the entity
client and the execute-agent SDK, the SDK tests stub the HTTP client, and the
CLI tests stub the SDK. So nothing shows that the SDK's request shape actually
satisfies the route, that its response parsing handles real bodies, or that
submitting a run produces a real ``agents.execute`` job the read path can join.

These drive the SDK — the interface a user actually holds — against the real
services in-process. The job *body* is deliberately not run: Fabric executes the
Analyst in a subprocess against a live platform, and standing that up here would
mean faking the transport, model reconciliation and SDK wiring that the e2e
suite exists to prove. Inference belongs to e2e; this tier covers the platform
plumbing around it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from nemo_agents_plugin.service import AgentsService
from nemo_insights_plugin.service import InsightsService
from nhx.core.entities.service import EntitiesService
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.core.models.service import ModelsService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client, subprocess_job_executor_patch
from pydantic import ValidationError

pytestmark = pytest.mark.integration

WORKSPACE = "default"
ANALYSIS_RUNS_URL = f"/apis/insights/v2/workspaces/{WORKSPACE}/analysis-runs"
DEFAULT_MODEL = f"{WORKSPACE}/analyst-default"
FAST_MODEL = f"{WORKSPACE}/analyst-fast"
ETHOS = "# Ethos\n\nAnswer only from retrieved context."


class _TestInsightsService(NemoServiceAdapter):
    def __init__(self) -> None:
        super().__init__(InsightsService())


class _TestAgentsService(NemoServiceAdapter):
    def __init__(self) -> None:
        super().__init__(AgentsService())


@contextmanager
def _platform(*, workspaces: list[str] | None = None, workspace: str = WORKSPACE) -> Iterator[ClientContext]:
    """Insights in front of the Agents, Jobs and Models services it actually calls."""
    with (
        subprocess_job_executor_patch(),
        create_test_client(
            _TestInsightsService,
            _TestAgentsService,
            EntitiesService,
            JobsService,
            FilesService,
            ModelsService,
            client_type=ClientContext,
            workspaces=workspaces or [workspace],
            workspace=workspace,
        ) as ctx,
    ):
        # The create path resolves the model pair before recording a run, so
        # the referenced entities have to exist.
        if workspace == WORKSPACE:
            for ref in (DEFAULT_MODEL, FAST_MODEL):
                ctx.sdk.models.create(workspace=WORKSPACE, name=ref.split("/", 1)[1], backend_format="OPENAI_CHAT")
        yield ctx


def _create(ctx: ClientContext, **overrides: Any) -> Any:
    """Submit a run the way a user does, which submits a real job."""
    return ctx.sdk.insights.analysis_runs.create(
        workspace=WORKSPACE,
        agent=overrides.pop("agent", "demo-agent"),
        default_model=overrides.pop("default_model", DEFAULT_MODEL),
        fast_model=overrides.pop("fast_model", FAST_MODEL),
        **overrides,
    )


def _backing_job(ctx: ClientContext, run_name: str) -> dict[str, Any]:
    return ctx.sdk.agents.jobs.execute.get(run_name, workspace=WORKSPACE)


# ---------------------------------------------------------------------------
# Create: the run is recorded and a job is submitted under its name
# ---------------------------------------------------------------------------


def test_creating_a_run_submits_a_job_under_its_name() -> None:
    """The shared name is the only link between run and job, so prove it holds."""
    with _platform() as ctx:
        created = _create(ctx)

        assert created.job is not None
        assert created.job["name"] == created.run.name
        # The job is real: the Jobs service answers for it independently.
        assert _backing_job(ctx, created.run.name)["name"] == created.run.name


def test_the_submitted_job_carries_the_analyst_config_the_facade_built() -> None:
    """The request's scope has to survive into the job spec, or the run analyzes the wrong thing."""
    with _platform() as ctx:
        created = _create(ctx, agent="scoped-agent", ethos=ETHOS, evaluation_id="eval-123")

        spec = _backing_job(ctx, created.run.name)["spec"]

        assert spec["extension"]["kind"] == "insights.analysis"
        assert spec["extension"]["config"]["agent"] == "scoped-agent"
        settings = spec["agent"]["config"]["harnesses"]["insights"]["settings"]
        assert settings["agent"] == "scoped-agent"
        assert settings["ethos"] == ETHOS
        assert settings["evaluation_id"] == "eval-123"


def test_the_sdk_serializes_a_since_bound_the_route_accepts() -> None:
    """``since`` crosses the wire as JSON, so the SDK's encoding has to survive the round trip."""
    with _platform() as ctx:
        created = _create(ctx, since=datetime(2026, 8, 1, tzinfo=timezone.utc), timeout_seconds=30.0)

        assert created.run.since == datetime(2026, 8, 1, tzinfo=timezone.utc)
        spec = _backing_job(ctx, created.run.name)["spec"]
        assert spec["agent"]["config"]["harnesses"]["insights"]["settings"]["since"] == "2026-08-01T00:00:00+00:00"
        # The step config preserves the original request alongside the resolved agent.
        assert spec["request"]["timeout_seconds"] == 30.0


def test_a_bare_model_name_resolves_in_the_run_workspace_and_is_stored_qualified() -> None:
    """The Analyst only accepts qualified refs, so normalization has to happen before storage."""
    with _platform() as ctx:
        created = _create(ctx, default_model="analyst-default", fast_model="analyst-fast")

        assert created.run.default_model == DEFAULT_MODEL
        assert created.run.fast_model == FAST_MODEL
        models = _backing_job(ctx, created.run.name)["spec"]["agent"]["config"]["models"]
        assert models["default"]["model"] == DEFAULT_MODEL
        assert models["fast"]["model"] == FAST_MODEL


@pytest.mark.parametrize(
    "ref",
    [
        "does-not-exist",
        f"{WORKSPACE}/does-not-exist",
        "too/many/slashes",
    ],
)
def test_a_model_ref_that_does_not_resolve_is_rejected_before_anything_is_recorded(ref: str) -> None:
    """A bogus ref used to 201, persisting a run behind a job that could never start."""
    with _platform() as ctx:
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            _create(ctx, default_model=ref)

        assert excinfo.value.response.status_code == 422, excinfo.value.response.text
        assert ctx.sdk.insights.analysis_runs.list_runs(workspace=WORKSPACE).data == []


def test_a_qualified_ref_may_name_a_model_in_another_workspace() -> None:
    """A run is not confined to its own workspace's models, only to what the caller can read."""
    with _platform(workspaces=["workspace-a", "workspace-b"], workspace="workspace-a") as ctx:
        for name in ("foo", "bar"):
            ctx.sdk.models.create(workspace="workspace-b", name=name, backend_format="OPENAI_CHAT")

        created = ctx.sdk.insights.analysis_runs.create(
            workspace="workspace-a",
            agent="borrowing-agent",
            default_model="workspace-b/foo",
            fast_model="workspace-b/bar",
        )

        assert created.run.workspace == "workspace-a"
        assert created.run.default_model == "workspace-b/foo"
        job = ctx.sdk.agents.jobs.execute.get(created.run.name, workspace="workspace-a")
        assert job["spec"]["agent"]["config"]["models"]["default"]["model"] == "workspace-b/foo"


# ---------------------------------------------------------------------------
# Read: join the run with its job, and list from the store
# ---------------------------------------------------------------------------


def test_reading_a_run_joins_it_with_its_backing_job() -> None:
    with _platform() as ctx:
        created = _create(ctx, agent="joined-agent")

        read = ctx.sdk.insights.analysis_runs.get(workspace=WORKSPACE, name=created.run.name)

        assert read.run.agent == "joined-agent"
        assert read.job is not None, "a submitted run must read back with its job"
        assert read.job_status is not None


def test_runs_are_listed_from_the_store_and_filtered_by_agent() -> None:
    with _platform() as ctx:
        alpha = {_create(ctx, agent="alpha").run.name for _ in range(2)}
        beta = _create(ctx, agent="beta").run.name

        filtered = ctx.sdk.insights.analysis_runs.list_runs(workspace=WORKSPACE, agent="alpha")
        unfiltered = ctx.sdk.insights.analysis_runs.list_runs(workspace=WORKSPACE)

        assert {run.name for run in filtered.data} == alpha
        assert beta in {run.name for run in unfiltered.data}


def test_the_sort_the_sdk_sends_is_accepted_and_echoed() -> None:
    """The route's sort field is only reachable through the SDK's kwarg."""
    with _platform() as ctx:
        _create(ctx, agent="sorted-agent")

        page = ctx.sdk.insights.analysis_runs.list_runs(workspace=WORKSPACE, agent="sorted-agent", sort="created_at")

        assert page.sort == "created_at"


def test_pagination_reaches_the_store_and_is_reported_back() -> None:
    with _platform() as ctx:
        for _ in range(3):
            _create(ctx, agent="paged-agent")

        page = ctx.sdk.insights.analysis_runs.list_runs(workspace=WORKSPACE, agent="paged-agent", page=1, page_size=2)

        assert len(page.data) == 2
        assert page.pagination is not None
        assert page.pagination.page_size == 2
        assert page.pagination.total_results == 3


def test_getting_an_unknown_run_is_a_404() -> None:
    """Needs no job, Fabric or model, so it belongs here rather than in e2e."""
    with _platform() as ctx:
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            ctx.sdk.insights.analysis_runs.get(workspace=WORKSPACE, name="insights-run-does-not-exist")

        assert excinfo.value.response.status_code == 404


# ---------------------------------------------------------------------------
# Validation: the SDK and the wire reject blanks at different points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["agent", "evaluation_id", "ethos", "default_model", "fast_model"])
def test_the_sdk_rejects_a_blank_setting_without_a_round_trip(field: str) -> None:
    """``_build_create_body`` builds the request model, so the SDK fails before sending."""
    with _platform() as ctx:
        with pytest.raises(ValidationError):
            _create(ctx, **{field: "   "})


@pytest.mark.parametrize("field", ["agent", "evaluation_id", "ethos", "default_model", "fast_model"])
def test_the_route_rejects_a_blank_setting_on_the_wire(field: str) -> None:
    """A caller that is not the SDK still gets a 422 rather than a run that fails later."""
    with _platform() as ctx:
        response = ctx.test_client.post(
            ANALYSIS_RUNS_URL,
            json={
                "agent": "demo-agent",
                "default_model": DEFAULT_MODEL,
                "fast_model": FAST_MODEL,
                field: "   ",
            },
        )

        assert response.status_code == 422, response.text
