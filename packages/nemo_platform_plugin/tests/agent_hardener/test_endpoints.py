# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agent Hardener service endpoint definitions."""

from __future__ import annotations

import json
from collections.abc import Callable
from types import GenericAlias
from typing import get_args, get_origin

import pytest
from nemo_platform_plugin.agent_hardener import endpoints
from nemo_platform_plugin.agent_hardener.types import (
    AgentHardenerManifest,
    AgentHardenerRun,
    ApplyMitigationRequest,
    ApplyMitigationResponse,
    ComposeDefenseRequest,
    ComposeDefenseResponse,
    EventIn,
    EventsResponse,
    InspectAgentRequest,
    InspectAgentResponse,
    InspectProjectRequest,
    InspectProjectResponse,
    JsonMap,
    ManifestInit,
    ManifestUpdate,
    ModelConfigDefaults,
    SynthBenignJob,
    SynthBenignJobRequest,
    SynthBenignSpec,
    ValidateModelRequest,
    ValidateModelResponse,
    WarGameJob,
    WarGameJobRequest,
    WarGameSpec,
)
from nemo_platform_plugin.client.types import BinaryContent, CursorPagination, Paginated, PreparedRequest
from nemo_platform_plugin.jobs.schemas import (
    PlatformJobListResultResponse,
    PlatformJobLog,
    PlatformJobResultResponse,
    PlatformJobStatusResponse,
)
from pydantic import BaseModel

ResponseType = type[BaseModel] | type[BinaryContent] | GenericAlias | None


def _manifest_body() -> ManifestInit:
    return ManifestInit(name="manifest-1", agent="default/agent")


def _war_game_body() -> WarGameJobRequest:
    return WarGameJobRequest(name="job-1", spec=WarGameSpec(manifest_id="manifest-1"))


def _synth_body() -> SynthBenignJobRequest:
    return SynthBenignJobRequest(name="synth-1", spec=SynthBenignSpec(manifest_id="manifest-1"))


@pytest.mark.parametrize(
    ("build", "method", "path_template", "path_params", "response_type"),
    [
        (lambda: endpoints.healthz(), "GET", "/apis/agent-hardener/v1/healthz", {}, JsonMap),
        (
            lambda: endpoints.create_war_game_job(workspace="default", body=_war_game_body()),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/jobs",
            {"workspace": "default"},
            WarGameJob,
        ),
        (
            lambda: endpoints.get_war_game_job(workspace="default", name="job-1"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/jobs/{name}",
            {"workspace": "default", "name": "job-1"},
            WarGameJob,
        ),
        (
            lambda: endpoints.delete_war_game_job(workspace="default", name="job-1"),
            "DELETE",
            "/apis/agent-hardener/v2/workspaces/{workspace}/jobs/{name}",
            {"workspace": "default", "name": "job-1"},
            None,
        ),
        (
            lambda: endpoints.cancel_war_game_job(workspace="default", name="job-1"),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/jobs/{name}/cancel",
            {"workspace": "default", "name": "job-1"},
            WarGameJob,
        ),
        (
            lambda: endpoints.get_war_game_job_status(workspace="default", name="job-1"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/jobs/{name}/status",
            {"workspace": "default", "name": "job-1"},
            PlatformJobStatusResponse,
        ),
        (
            lambda: endpoints.list_war_game_job_results(workspace="default", name="job-1"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/jobs/{name}/results",
            {"workspace": "default", "name": "job-1"},
            PlatformJobListResultResponse,
        ),
        (
            lambda: endpoints.get_war_game_job_result(workspace="default", job="job-1", name="out"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/jobs/{job}/results/{name}",
            {"workspace": "default", "job": "job-1", "name": "out"},
            PlatformJobResultResponse,
        ),
        (
            lambda: endpoints.download_war_game_job_result(workspace="default", job="job-1", name="out"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/jobs/{job}/results/{name}/download",
            {"workspace": "default", "job": "job-1", "name": "out"},
            BinaryContent,
        ),
        (
            lambda: endpoints.get_manifest(workspace="default", name="manifest-1"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/manifests/{name}",
            {"workspace": "default", "name": "manifest-1"},
            AgentHardenerManifest,
        ),
        (
            lambda: endpoints.create_manifest(workspace="default", body=_manifest_body()),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/manifests",
            {"workspace": "default"},
            AgentHardenerManifest,
        ),
        (
            lambda: endpoints.inspect_project(
                workspace="default", body=InspectProjectRequest(project_fileset="fileset-1")
            ),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/manifests/inspect-project",
            {"workspace": "default"},
            InspectProjectResponse,
        ),
        (
            lambda: endpoints.inspect_agent(workspace="default", body=InspectAgentRequest(agent="default/agent")),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/manifests/inspect-agent",
            {"workspace": "default"},
            InspectAgentResponse,
        ),
        (
            lambda: endpoints.update_manifest(workspace="default", name="manifest-1", body=ManifestUpdate(port=9000)),
            "PATCH",
            "/apis/agent-hardener/v2/workspaces/{workspace}/manifests/{name}",
            {"workspace": "default", "name": "manifest-1"},
            AgentHardenerManifest,
        ),
        (
            lambda: endpoints.delete_manifest(workspace="default", name="manifest-1"),
            "DELETE",
            "/apis/agent-hardener/v2/workspaces/{workspace}/manifests/{name}",
            {"workspace": "default", "name": "manifest-1"},
            None,
        ),
        (
            lambda: endpoints.refresh_manifest(workspace="default", name="manifest-1"),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/manifests/{name}/refresh",
            {"workspace": "default", "name": "manifest-1"},
            AgentHardenerManifest,
        ),
        (
            lambda: endpoints.get_model_config_defaults(workspace="default"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/model-config-defaults",
            {"workspace": "default"},
            ModelConfigDefaults,
        ),
        (
            lambda: endpoints.validate_model(
                workspace="default",
                body=ValidateModelRequest(model="model-1", base_url="https://api.example.test/v1"),
            ),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/model-config/validate",
            {"workspace": "default"},
            ValidateModelResponse,
        ),
        (
            lambda: endpoints.get_run(workspace="default", name="run-1"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/runs/{name}",
            {"workspace": "default", "name": "run-1"},
            AgentHardenerRun,
        ),
        (
            lambda: endpoints.delete_run(workspace="default", name="run-1"),
            "DELETE",
            "/apis/agent-hardener/v2/workspaces/{workspace}/runs/{name}",
            {"workspace": "default", "name": "run-1"},
            None,
        ),
        (
            lambda: endpoints.apply_mitigation(
                workspace="default", name="run-1", body=ApplyMitigationRequest(guardrails_toml="[[guardrail]]\n")
            ),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/runs/{name}/apply-mitigation",
            {"workspace": "default", "name": "run-1"},
            ApplyMitigationResponse,
        ),
        (
            lambda: endpoints.compose_defense(
                workspace="default", name="run-1", body=ComposeDefenseRequest(mitigations={"defenses": []})
            ),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/runs/{name}/compose-defense",
            {"workspace": "default", "name": "run-1"},
            ComposeDefenseResponse,
        ),
        (
            lambda: endpoints.ingest_event(workspace="default", name="run-1", body=EventIn(event="started")),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/runs/{name}/events",
            {"workspace": "default", "name": "run-1"},
            None,
        ),
        (
            lambda: endpoints.get_events(workspace="default", name="run-1"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/runs/{name}/events",
            {"workspace": "default", "name": "run-1"},
            EventsResponse,
        ),
        (
            lambda: endpoints.create_synth_benign_job(workspace="default", body=_synth_body()),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/synth-benign/jobs",
            {"workspace": "default"},
            SynthBenignJob,
        ),
        (
            lambda: endpoints.get_synth_benign_job(workspace="default", name="synth-1"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/synth-benign/jobs/{name}",
            {"workspace": "default", "name": "synth-1"},
            SynthBenignJob,
        ),
        (
            lambda: endpoints.delete_synth_benign_job(workspace="default", name="synth-1"),
            "DELETE",
            "/apis/agent-hardener/v2/workspaces/{workspace}/synth-benign/jobs/{name}",
            {"workspace": "default", "name": "synth-1"},
            None,
        ),
        (
            lambda: endpoints.cancel_synth_benign_job(workspace="default", name="synth-1"),
            "POST",
            "/apis/agent-hardener/v2/workspaces/{workspace}/synth-benign/jobs/{name}/cancel",
            {"workspace": "default", "name": "synth-1"},
            SynthBenignJob,
        ),
        (
            lambda: endpoints.get_synth_benign_job_status(workspace="default", name="synth-1"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/synth-benign/jobs/{name}/status",
            {"workspace": "default", "name": "synth-1"},
            PlatformJobStatusResponse,
        ),
        (
            lambda: endpoints.list_synth_benign_job_results(workspace="default", name="synth-1"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/synth-benign/jobs/{name}/results",
            {"workspace": "default", "name": "synth-1"},
            PlatformJobListResultResponse,
        ),
        (
            lambda: endpoints.get_synth_benign_job_result(workspace="default", job="synth-1", name="suite"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/synth-benign/jobs/{job}/results/{name}",
            {"workspace": "default", "job": "synth-1", "name": "suite"},
            PlatformJobResultResponse,
        ),
        (
            lambda: endpoints.download_synth_benign_job_result(workspace="default", job="synth-1", name="suite"),
            "GET",
            "/apis/agent-hardener/v2/workspaces/{workspace}/synth-benign/jobs/{job}/results/{name}/download",
            {"workspace": "default", "job": "synth-1", "name": "suite"},
            BinaryContent,
        ),
    ],
)
def test_endpoint_shapes(
    build: Callable[[], PreparedRequest],
    method: str,
    path_template: str,
    path_params: dict[str, str],
    response_type: ResponseType,
) -> None:
    prepared = build()

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == method
    assert prepared.path_template == path_template
    assert prepared.path_params == path_params
    assert prepared.response_type == response_type


def test_create_manifest_uses_name_for_conflict_resolution() -> None:
    body = _manifest_body()
    prepared = endpoints.create_manifest(workspace="default", body=body, exist_ok=True)

    assert prepared.on_conflict_get is not None
    assert prepared.on_conflict_get.path_template.endswith("/manifests/{name}")
    assert prepared.on_conflict_get.path_params == {"workspace": "default", "name": "manifest-1"}


def test_create_manifest_body_serializes_name_and_api_key_secret() -> None:
    body = ManifestInit(name="manifest-1", agent="default/agent")
    prepared = endpoints.create_manifest(workspace="default", body=body)

    assert isinstance(prepared.content, bytes)
    assert json.loads(prepared.content) == {"name": "manifest-1", "agent": "default/agent"}

    validation = endpoints.validate_model(
        workspace="default",
        body=ValidateModelRequest(
            model="model-1",
            base_url="https://api.example.test/v1",
            api_key_secret="secret-1",
        ),
    )
    assert isinstance(validation.content, bytes)
    assert json.loads(validation.content)["api_key_secret"] == "secret-1"


def test_list_endpoint_query_params() -> None:
    manifests = endpoints.list_manifests(
        workspace="default",
        query_params={"page": 2, "page_size": 10, "sort": "-created_at", "filter[agent]": "default/agent"},
    )
    runs = endpoints.list_runs(
        workspace="default",
        query_params={"page": 3, "page_size": 5, "filter[manifest_id]": "manifest-1"},
    )
    jobs = endpoints.list_war_game_jobs(workspace="default", query_params={"filter": '{"status":"active"}'})
    synth = endpoints.list_synth_benign_jobs(workspace="default", query_params={"sort": "updated_at"})
    events = endpoints.get_events(workspace="default", name="run-1", query_params={"after": 42})

    assert manifests.query_params == {
        "page": 2,
        "page_size": 10,
        "sort": "-created_at",
        "filter[agent]": "default/agent",
    }
    assert runs.query_params == {"page": 3, "page_size": 5, "filter[manifest_id]": "manifest-1"}
    assert jobs.query_params == {"filter": '{"status":"active"}'}
    assert synth.query_params == {"sort": "updated_at"}
    assert events.query_params == {"after": 42}


def test_list_response_markers() -> None:
    for prepared, expected_item in (
        (endpoints.list_war_game_jobs(workspace="default"), WarGameJob),
        (endpoints.list_manifests(workspace="default"), AgentHardenerManifest),
        (endpoints.list_runs(workspace="default"), AgentHardenerRun),
        (endpoints.list_synth_benign_jobs(workspace="default"), SynthBenignJob),
    ):
        assert get_origin(prepared.response_type) is Paginated
        assert get_args(prepared.response_type)[0] is expected_item


def test_log_response_markers_use_cursor_pagination() -> None:
    for prepared in (
        endpoints.list_war_game_job_logs(workspace="default", name="job-1"),
        endpoints.list_synth_benign_job_logs(workspace="default", name="synth-1"),
    ):
        assert get_origin(prepared.response_type) is Paginated
        assert get_args(prepared.response_type) == (PlatformJobLog, CursorPagination)
