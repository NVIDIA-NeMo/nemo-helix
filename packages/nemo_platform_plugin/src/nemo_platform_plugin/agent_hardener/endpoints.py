# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Agent Hardener service."""

from __future__ import annotations

from abc import abstractmethod

from nemo_platform_plugin.agent_hardener.types import (
    AgentHardenerManifest,
    AgentHardenerRun,
    ApplyMitigationRequest,
    ApplyMitigationResponse,
    ComposeDefenseRequest,
    ComposeDefenseResponse,
    EventIn,
    EventsResponse,
    GetEventsQueryParams,
    InspectAgentRequest,
    InspectAgentResponse,
    InspectProjectRequest,
    InspectProjectResponse,
    JobLogsQueryParams,
    JsonMap,
    ListManifestsQueryParams,
    ListRunsQueryParams,
    ListSynthBenignJobsQueryParams,
    ListWarGameJobsQueryParams,
    ManifestInit,
    ManifestUpdate,
    ModelConfigDefaults,
    SynthBenignJob,
    SynthBenignJobRequest,
    ValidateModelRequest,
    ValidateModelResponse,
    WarGameJob,
    WarGameJobRequest,
)
from nemo_platform_plugin.client.endpoint import delete, get, patch, post
from nemo_platform_plugin.client.types import BinaryContent, CursorPagination, Paginated, PreparedRequest
from nemo_platform_plugin.jobs.schemas import (
    PlatformJobListResultResponse,
    PlatformJobLog,
    PlatformJobResultResponse,
    PlatformJobStatusResponse,
)

_HEALTH = "/apis/agent-hardener/v1/healthz"
_ROOT = "/apis/agent-hardener/v2/workspaces/{workspace}"
_JOBS = f"{_ROOT}/jobs"
_MANIFESTS = f"{_ROOT}/manifests"
_RUNS = f"{_ROOT}/runs"
_SYNTH_JOBS = f"{_ROOT}/synth-benign/jobs"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@get(_HEALTH)
@abstractmethod
def healthz() -> JsonMap: ...


# ---------------------------------------------------------------------------
# War-game jobs
# ---------------------------------------------------------------------------


@post(_JOBS)
@abstractmethod
def create_war_game_job(*, workspace: str | None = None, body: WarGameJobRequest) -> WarGameJob: ...


@get(_JOBS)
@abstractmethod
def list_war_game_jobs(
    *, workspace: str | None = None, query_params: ListWarGameJobsQueryParams | None = None
) -> Paginated[WarGameJob]: ...


@get(f"{_JOBS}/{{name}}")
@abstractmethod
def get_war_game_job(*, workspace: str | None = None, name: str) -> WarGameJob: ...


@delete(f"{_JOBS}/{{name}}")
@abstractmethod
def delete_war_game_job(*, workspace: str | None = None, name: str) -> None: ...


@post(f"{_JOBS}/{{name}}/cancel")
@abstractmethod
def cancel_war_game_job(*, workspace: str | None = None, name: str) -> WarGameJob: ...


@get(f"{_JOBS}/{{name}}/status")
@abstractmethod
def get_war_game_job_status(*, workspace: str | None = None, name: str) -> PlatformJobStatusResponse: ...


@get(f"{_JOBS}/{{name}}/logs")
@abstractmethod
def list_war_game_job_logs(
    *, workspace: str | None = None, name: str, query_params: JobLogsQueryParams | None = None
) -> Paginated[PlatformJobLog, CursorPagination]: ...


@get(f"{_JOBS}/{{name}}/results")
@abstractmethod
def list_war_game_job_results(*, workspace: str | None = None, name: str) -> PlatformJobListResultResponse: ...


@get(f"{_JOBS}/{{job}}/results/{{name}}")
@abstractmethod
def get_war_game_job_result(*, workspace: str | None = None, job: str, name: str) -> PlatformJobResultResponse: ...


@get(f"{_JOBS}/{{job}}/results/{{name}}/download")
@abstractmethod
def download_war_game_job_result(*, workspace: str | None = None, job: str, name: str) -> BinaryContent: ...


# ---------------------------------------------------------------------------
# Manifest CRUD + inspection
# ---------------------------------------------------------------------------


@get(f"{_MANIFESTS}/{{name}}")
@abstractmethod
def get_manifest(*, workspace: str | None = None, name: str) -> AgentHardenerManifest: ...


@get(_MANIFESTS)
@abstractmethod
def list_manifests(
    *, workspace: str | None = None, query_params: ListManifestsQueryParams | None = None
) -> Paginated[AgentHardenerManifest]: ...


def _get_manifest_on_conflict(body: ManifestInit, workspace: str | None) -> PreparedRequest[AgentHardenerManifest]:
    return get_manifest(name=body.name, workspace=workspace)


@post(_MANIFESTS, get_on_conflict=_get_manifest_on_conflict)
@abstractmethod
def create_manifest(
    *, workspace: str | None = None, body: ManifestInit, exist_ok: bool = False
) -> AgentHardenerManifest: ...


@post(f"{_MANIFESTS}/inspect-project")
@abstractmethod
def inspect_project(*, workspace: str | None = None, body: InspectProjectRequest) -> InspectProjectResponse: ...


@post(f"{_MANIFESTS}/inspect-agent")
@abstractmethod
def inspect_agent(*, workspace: str | None = None, body: InspectAgentRequest) -> InspectAgentResponse: ...


@patch(f"{_MANIFESTS}/{{name}}")
@abstractmethod
def update_manifest(*, workspace: str | None = None, name: str, body: ManifestUpdate) -> AgentHardenerManifest: ...


@post(f"{_MANIFESTS}/{{name}}/refresh")
@abstractmethod
def refresh_manifest(*, workspace: str | None = None, name: str) -> AgentHardenerManifest: ...


@delete(f"{_MANIFESTS}/{{name}}")
@abstractmethod
def delete_manifest(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Model config validation
# ---------------------------------------------------------------------------


@get(f"{_ROOT}/model-config-defaults")
@abstractmethod
def get_model_config_defaults(*, workspace: str | None = None) -> ModelConfigDefaults: ...


@post(f"{_ROOT}/model-config/validate")
@abstractmethod
def validate_model(*, workspace: str | None = None, body: ValidateModelRequest) -> ValidateModelResponse: ...


# ---------------------------------------------------------------------------
# Run records + actions + events
# ---------------------------------------------------------------------------


@get(f"{_RUNS}/{{name}}")
@abstractmethod
def get_run(*, workspace: str | None = None, name: str) -> AgentHardenerRun: ...


@get(_RUNS)
@abstractmethod
def list_runs(
    *, workspace: str | None = None, query_params: ListRunsQueryParams | None = None
) -> Paginated[AgentHardenerRun]: ...


@delete(f"{_RUNS}/{{name}}")
@abstractmethod
def delete_run(*, workspace: str | None = None, name: str) -> None: ...


@post(f"{_RUNS}/{{name}}/apply-mitigation")
@abstractmethod
def apply_mitigation(
    *, workspace: str | None = None, name: str, body: ApplyMitigationRequest
) -> ApplyMitigationResponse: ...


@post(f"{_RUNS}/{{name}}/compose-defense")
@abstractmethod
def compose_defense(
    *, workspace: str | None = None, name: str, body: ComposeDefenseRequest
) -> ComposeDefenseResponse: ...


@post(f"{_RUNS}/{{name}}/events")
@abstractmethod
def ingest_event(*, workspace: str | None = None, name: str, body: EventIn) -> None: ...


@get(f"{_RUNS}/{{name}}/events")
@abstractmethod
def get_events(
    *, workspace: str | None = None, name: str, query_params: GetEventsQueryParams | None = None
) -> EventsResponse: ...


# ---------------------------------------------------------------------------
# Synth-benign jobs
# ---------------------------------------------------------------------------


@post(_SYNTH_JOBS)
@abstractmethod
def create_synth_benign_job(*, workspace: str | None = None, body: SynthBenignJobRequest) -> SynthBenignJob: ...


@get(_SYNTH_JOBS)
@abstractmethod
def list_synth_benign_jobs(
    *, workspace: str | None = None, query_params: ListSynthBenignJobsQueryParams | None = None
) -> Paginated[SynthBenignJob]: ...


@get(f"{_SYNTH_JOBS}/{{name}}")
@abstractmethod
def get_synth_benign_job(*, workspace: str | None = None, name: str) -> SynthBenignJob: ...


@delete(f"{_SYNTH_JOBS}/{{name}}")
@abstractmethod
def delete_synth_benign_job(*, workspace: str | None = None, name: str) -> None: ...


@post(f"{_SYNTH_JOBS}/{{name}}/cancel")
@abstractmethod
def cancel_synth_benign_job(*, workspace: str | None = None, name: str) -> SynthBenignJob: ...


@get(f"{_SYNTH_JOBS}/{{name}}/status")
@abstractmethod
def get_synth_benign_job_status(*, workspace: str | None = None, name: str) -> PlatformJobStatusResponse: ...


@get(f"{_SYNTH_JOBS}/{{name}}/logs")
@abstractmethod
def list_synth_benign_job_logs(
    *, workspace: str | None = None, name: str, query_params: JobLogsQueryParams | None = None
) -> Paginated[PlatformJobLog, CursorPagination]: ...


@get(f"{_SYNTH_JOBS}/{{name}}/results")
@abstractmethod
def list_synth_benign_job_results(*, workspace: str | None = None, name: str) -> PlatformJobListResultResponse: ...


@get(f"{_SYNTH_JOBS}/{{job}}/results/{{name}}")
@abstractmethod
def get_synth_benign_job_result(*, workspace: str | None = None, job: str, name: str) -> PlatformJobResultResponse: ...


@get(f"{_SYNTH_JOBS}/{{job}}/results/{{name}}/download")
@abstractmethod
def download_synth_benign_job_result(*, workspace: str | None = None, job: str, name: str) -> BinaryContent: ...
