# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP contract for the UserSim plugin's routes and the job collections mounted by ``add_job_routes``."""

from abc import abstractmethod
from datetime import datetime
from typing import Literal

from nemo_helix_plugin.client.endpoint import delete, get, post
from nemo_helix_plugin.client.types import BinaryContent, Paginated
from nemo_helix_plugin.jobs.schemas import HelixJobStatus, HelixJobStatusResponse
from nemo_user_sim_plugin.model_check import ModelStatus
from nemo_user_sim_plugin.probes import Probe
from pydantic import BaseModel, Field, JsonValue

JobCollection = Literal["simulate", "export"]
_JOBS = "/apis/user-sim/v2/workspaces/{workspace}/jobs/{collection}"


class JobRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    spec: dict[str, JsonValue] = Field(default_factory=dict)


class Job(BaseModel):
    name: str
    workspace: str | None = None
    description: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    spec: dict[str, JsonValue] = Field(default_factory=dict)
    status: HelixJobStatus | None = None
    status_details: dict[str, JsonValue] | None = None
    error_details: dict[str, JsonValue] | None = None


@get("/apis/user-sim/v1/healthz")
@abstractmethod
def health() -> dict[str, JsonValue]: ...


@get("/apis/user-sim/v1/probes")
@abstractmethod
def probes() -> list[Probe]: ...


class SuggestionRequest(BaseModel):
    agent: str


class Suggestion(BaseModel):
    name: str
    reason: str


class ProbeSuggestions(BaseModel):
    agent: str
    probes: list[Suggestion]
    read: list[str] = Field(description="The agent files the suggestion was inferred from.")


@get("/apis/user-sim/v2/workspaces/{workspace}/model-check")
@abstractmethod
def check_models(*, workspace: str | None = None) -> list[ModelStatus]: ...


@post("/apis/user-sim/v2/workspaces/{workspace}/probe-suggestions")
@abstractmethod
def suggest_probes(*, workspace: str | None = None, body: SuggestionRequest) -> ProbeSuggestions: ...


@post(_JOBS)
@abstractmethod
def create_job(*, workspace: str | None = None, collection: JobCollection, body: JobRequest) -> Job: ...


@get(_JOBS)
@abstractmethod
def list_jobs(*, workspace: str | None = None, collection: JobCollection) -> Paginated[Job]: ...


@get(_JOBS + "/{name}")
@abstractmethod
def get_job(*, workspace: str | None = None, collection: JobCollection, name: str) -> Job: ...


@get(_JOBS + "/{name}/status")
@abstractmethod
def get_job_status(*, workspace: str | None = None, collection: JobCollection, name: str) -> HelixJobStatusResponse: ...


@get(_JOBS + "/{job}/results/{name}/download")
@abstractmethod
def download_result(
    *, workspace: str | None = None, collection: JobCollection, job: str, name: str
) -> BinaryContent: ...


@delete(_JOBS + "/{name}")
@abstractmethod
def delete_job(*, workspace: str | None = None, collection: JobCollection, name: str) -> None: ...


@post(_JOBS + "/{name}/cancel")
@abstractmethod
def cancel_job(*, workspace: str | None = None, collection: JobCollection, name: str) -> Job: ...
