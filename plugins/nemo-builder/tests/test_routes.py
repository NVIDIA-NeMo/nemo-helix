# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's routes, over HTTP, with the entity store faked."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_builder_plugin import completion, remote, service
from nemo_builder_plugin.backend import BackendRejectedError
from nemo_builder_plugin.completion import (
    Caller,
    JobNotRecorded,
    RecordedStep,
    current_caller,
    current_delegation,
    step_refusal,
    submit_refusal,
    verified_delegation,
)
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, ExternalImage, ImageRuntimeConfig, Provenance
from nemo_builder_plugin.remote import RemoteBackend
from nemo_builder_plugin.schema import JOB_NAME_PREFIX, BuildSet, BuildSpec
from nemo_builder_plugin.service import BuilderService, get_allow_unverified, get_job_of, get_settle, get_token_exchange
from nemo_builder_plugin.settle import settle
from nemo_builder_plugin.submit import (
    BackendChanged,
    BuildConflict,
    CreatedJob,
    ExistingJob,
    MissingSecrets,
    SubmitResult,
    request_digest,
    submit_build_set,
)
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import WorkloadDelegationEntity, WorkloadDelegationScope
from nemo_helix_plugin.auth.workload_identity import (
    WorkloadIdentityConfigError,
    build_kubernetes_pod_uid_workload_delegation,
)
from nemo_helix_plugin.authz import CallerKind, get_path_rules
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.config import Configuration
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entities.types import Entity, EntityCreateInput, EntityUpdate
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError, get_entity_client
from nemo_helix_plugin.jobs.execution_profiles import KubernetesJobExecutionProfile
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse
from pydantic import BaseModel, ValidationError, field_validator
from starlette.requests import Request

#: A setting of the deployment as a route asks for it, or the error that reading it ends in.
Setting = bool | Exception | Callable[[], bool]


def _asked(setting: Setting) -> Callable[[], bool]:
    def read() -> bool:
        if isinstance(setting, Exception):
            raise setting
        return setting() if callable(setting) else setting

    return read


def _app(
    entities: AsyncMock,
    caller: Caller = Caller(auth=False),
    job_status: AsyncMock | None = None,
    builder: AsyncMock | None = None,
    delegation: str | None = None,
    job_of: AsyncMock | None = None,
    exchange: Setting = False,
    unverified: Setting = False,
) -> FastAPI:
    """``entities`` is the caller's entity store; ``builder`` the builder's, which settling writes through.
    ``delegation`` is the workload delegation the caller's token names, and ``job_of`` the platform's record of
    the job step each one was made for. ``exchange`` is whether the platform mints its workloads' tokens
    (workload token exchange), and ``unverified`` whether the operator accepted completions nothing verifies
    (`builder.remote.allow_unverified_completion`)."""
    app = FastAPI()
    for spec in BuilderService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    # As the platform registers them on the service's app.
    for kind, handler in BuilderService().get_exception_handlers().items():
        app.add_exception_handler(kind, handler)
    app.dependency_overrides[get_entity_client] = lambda: entities
    app.dependency_overrides[current_caller] = lambda: caller
    # Every build job still running, unless a test says otherwise.
    jobs = job_status or AsyncMock(return_value=HelixJobStatus.ACTIVE)
    rows = builder or AsyncMock(update=AsyncMock(side_effect=lambda updated: updated))

    async def settle_as_builder(image: ContainerImage) -> ContainerImage:
        return (await settle(rows, [image], jobs))[0]

    app.dependency_overrides[get_settle] = lambda: settle_as_builder
    app.dependency_overrides[current_delegation] = lambda: delegation
    if job_of is not None:
        app.dependency_overrides[get_job_of] = lambda: job_of
    app.dependency_overrides[get_token_exchange] = lambda: _asked(exchange)
    app.dependency_overrides[get_allow_unverified] = lambda: _asked(unverified)
    return app


def _empty_page() -> MagicMock:
    page = MagicMock()
    page.data = []
    return page


class TestListingOneJobsImages:
    def test_a_job_and_a_status_filter_on_the_stored_fields(self) -> None:
        entities = AsyncMock()
        entities.list.return_value = _empty_page()
        response = TestClient(_app(entities)).get(
            "/v2/workspaces/team-a/container-images", params={"job": "demo-1", "status": "pending"}
        )
        assert response.status_code == 200
        assert entities.list.await_args.kwargs["filter_obj"] == {
            "provenance.job": "team-a/demo-1",
            "status": "pending",
        }

    def test_a_build_set_and_a_revision_filter_on_the_stored_provenance(self) -> None:
        entities = AsyncMock()
        entities.list.return_value = _empty_page()
        response = TestClient(_app(entities)).get(
            "/v2/workspaces/team-a/container-images", params={"build_set": "demo", "revision": "1"}
        )
        assert response.status_code == 200
        assert entities.list.await_args.kwargs["filter_obj"] == {
            "provenance.build_set": "demo",
            "provenance.revision": 1,
        }

    @pytest.mark.parametrize("revision", ["0", "one"])
    def test_a_revision_that_no_set_can_have_is_refused(self, revision: str) -> None:
        entities = AsyncMock()
        response = TestClient(_app(entities)).get(
            "/v2/workspaces/team-a/container-images", params={"revision": revision}
        )
        assert response.status_code == 422
        entities.list.assert_not_awaited()

    def test_the_job_is_always_in_the_path_workspace(self) -> None:
        entities = AsyncMock()
        response = TestClient(_app(entities)).get(
            "/v2/workspaces/team-a/container-images", params={"job": "team-b/demo-1"}
        )
        assert response.status_code == 422
        entities.list.assert_not_awaited()

    def test_no_filter_lists_the_workspace(self) -> None:
        entities = AsyncMock()
        entities.list.return_value = _empty_page()
        TestClient(_app(entities)).get("/v2/workspaces/team-a/container-images")
        assert entities.list.await_args.kwargs["filter_obj"] is None

    def test_an_unknown_status_is_refused(self) -> None:
        entities = AsyncMock()
        response = TestClient(_app(entities)).get("/v2/workspaces/team-a/container-images", params={"status": "done"})
        assert response.status_code == 422
        entities.list.assert_not_awaited()

    def test_the_deprecated_queued_status_still_filters(self) -> None:
        """0.10's builder wrote it, and a 0.10 client may still ask for it."""
        entities = AsyncMock()
        page = MagicMock()
        page.data = [_row().model_copy(update={"status": "queued"})]
        entities.list.return_value = page
        response = TestClient(_app(entities)).get("/v2/workspaces/team-a/container-images", params={"status": "queued"})
        assert response.status_code == 200
        assert entities.list.await_args.kwargs["filter_obj"] == {"status": "queued"}
        assert [image["status"] for image in response.json()] == ["queued"]

    def test_listing_returns_the_stored_status_and_asks_jobs_nothing(self) -> None:
        """Settling as it lists would move rows out of a `status` filter between pages: builder-settle does it."""
        entities = _entities(_row())
        page = MagicMock()
        page.data = [_row()]
        entities.list.return_value = page
        job_status = AsyncMock(return_value=HelixJobStatus.ERROR)
        response = TestClient(_app(entities, job_status=job_status)).get(
            "/v2/workspaces/ws-a/container-images", params={"status": "pending"}
        )
        assert response.status_code == 200
        assert [image["status"] for image in response.json()] == ["pending"]
        job_status.assert_not_awaited()
        entities.update.assert_not_awaited()


DIGEST = "sha256:" + "d" * 64
IMAGE = "/v2/workspaces/ws-a/container-images/build-1-0"
STEP = Caller(auth=True, actor="job-step", submitter="alice")
#: The push step without workload token exchange: the builder's own service principal, acting for the submitter.
SERVICE_STEP = Caller(auth=True, actor="service:builder", submitter="alice")
#: Acting for the image's submitter, and not its build job: without workload token exchange, any request whose
#: headers name a service.
NAMED_SERVICE = Caller(auth=True, actor="service:evals", submitter="alice")
#: Under workload token exchange a step acts for the submitter as the ServiceAccount its pod runs as, and its
#: token names the delegation the platform recorded for its pod: the image's own remote-build step's, and that
#: of a step of another job of the same submitter's.
EXCHANGED_STEP = Caller(auth=True, actor="system:serviceaccount:builds:build-remote", submitter="alice")
#: A token the platform signs for a service acting for the submitter, under workload token exchange: one
#: platform service calls another with such a token, and Jobs writes one for a step it runs outside Kubernetes.
#: It names no workload delegation.
SUBPROCESS_STEP = Caller(auth=True, actor="service:jobs", submitter="alice")
OWN, OTHER_JOBS = "delegation-own", "delegation-other"
#: The job a remote build's submit created, and its first attempt, as Jobs answered the create and the row
#: records them.
JOB_UID, ATTEMPT = "platform-job-1", "attempt-1"
#: What the platform has on record for the pod of that attempt's step: the job by its name, and the attempt.
OWN_STEP = RecordedStep(workspace="ws-a", job="build-1", attempt_id=ATTEMPT)
ANOTHER_JOBS_STEP = RecordedStep(workspace="ws-a", job="eval-7", attempt_id="attempt-7")


def _job_of(steps: Mapping[str, RecordedStep | None] | None = None) -> AsyncMock:
    """The job step each delegation was made for, as the platform has it on record."""
    known = {OWN: OWN_STEP, OTHER_JOBS: ANOTHER_JOBS_STEP} if steps is None else steps
    return AsyncMock(side_effect=lambda delegation: known.get(delegation))


def _row(
    status: Literal["pending", "ready", "failed"] = "pending",
    digest: str | None = None,
    backend: str = "execution",
    recorded: bool = True,
) -> ContainerImage:
    """A row as its submit left it. One a build service builds records the job the submit created, unless
    ``recorded`` is False: the submit has created the job and not yet written it."""
    records = backend != "execution" and recorded
    row = ContainerImage(
        name="build-1-0",
        workspace="ws-a",
        registry="reg.example.com",
        repository="ws-a/app",
        provenance=Provenance(
            build_set="build",
            revision=1,
            job="ws-a/build-1",
            job_uid=JOB_UID if records else None,
            job_attempt_id=ATTEMPT if records else None,
            request_digest="sha256:" + "a" * 64,
            backend=backend,
        ),
    )
    row.status, row.digest = status, digest
    # What the entity store records for whoever created the row: the submit's effective principal.
    row._created_by = "alice"
    return row


def _entities(row: ContainerImage) -> AsyncMock:
    entities = AsyncMock()
    entities.get.return_value = row
    entities.update.side_effect = lambda updated: updated
    return entities


class TestComplete:
    def _post(
        self,
        row: ContainerImage,
        *,
        caller: Caller = Caller(auth=False),
        body: dict | None = None,
        delegation: str | None = None,
        job_of: AsyncMock | None = None,
        exchange: Setting = False,
        unverified: Setting = False,
    ):
        entities = _entities(row)
        body = {"digest": DIGEST} if body is None else body
        app = _app(entities, caller, delegation=delegation, job_of=job_of, exchange=exchange, unverified=unverified)
        response = TestClient(app).post(f"{IMAGE}/complete", json=body)
        return response, entities

    def test_the_pushed_digest_makes_the_image_ready(self) -> None:
        response, entities = self._post(_row())
        assert response.status_code == 200
        assert (response.json()["status"], response.json()["digest"]) == ("ready", DIGEST)
        assert response.json()["image_ref"] == f"reg.example.com/ws-a/app@{DIGEST}"
        entities.update.assert_awaited_once()

    def test_how_the_image_starts_is_recorded_with_it(self) -> None:
        runtime_config = {
            "entrypoint": ["/bin/bash", "-l", "-c"],
            "cmd": ["/app/run.sh"],
            "working_dir": "/app",
            "user": "agent",
        }
        response, _ = self._post(_row(), body={"digest": DIGEST, "runtime_config": runtime_config})
        assert response.status_code == 200
        assert response.json()["runtime_config"] == runtime_config

    @pytest.mark.parametrize(
        ("caller", "how"),
        [
            (Caller(auth=False), {}),
            (EXCHANGED_STEP, {"delegation": OWN, "exchange": True}),
            (SERVICE_STEP, {"unverified": True}),
        ],
        ids=["auth-off", "its-own-step-under-token-exchange", "a-step-the-operator-accepts-unverified"],
    )
    def test_a_remote_build_steps_handle_is_recorded_with_it(self, caller: Caller, how: dict[str, Any]) -> None:
        external = {"image_id": "img-1", "build_id": "build-1", "attributes": {"builder": "v1"}}
        row = _row(backend="remote")
        response, entities = self._post(
            row, caller=caller, body={"digest": DIGEST, "external": external}, job_of=_job_of(), **how
        )
        assert response.status_code == 200
        assert response.json()["external"] == external
        assert response.json()["runtime_config"] is None
        entities.update.assert_awaited_once()

    @pytest.mark.parametrize("row", [_row(), _row("ready", DIGEST), _row("failed")], ids=["pending", "ready", "failed"])
    def test_a_handle_for_an_image_this_cluster_builds_is_a_409(self, row: ContainerImage) -> None:
        """Whatever the row's status: the request doesn't fit the image, before how it settled is looked at."""
        body = {"digest": DIGEST, "external": {"image_id": "img-1"}}
        response, entities = self._post(row, body=body)
        assert response.status_code == 409
        assert "built in this cluster" in response.json()["detail"]
        entities.update.assert_not_awaited()

    def test_a_misspelt_handle_is_refused_not_dropped(self) -> None:
        body = {"digest": DIGEST, "external": {"imageid": "img-1"}}
        response, entities = self._post(_row(backend="remote"), body=body)
        assert response.status_code == 422
        entities.update.assert_not_awaited()

    def test_an_image_this_cluster_built_has_no_handle(self) -> None:
        response, _ = self._post(_row())
        assert response.json()["external"] is None
        assert response.json()["provenance"]["backend"] == "execution"

    def test_an_image_already_ready_at_that_digest_is_returned_as_it_is(self) -> None:
        response, entities = self._post(_row("ready", DIGEST))
        assert response.status_code == 200
        entities.update.assert_not_awaited()

    @pytest.mark.parametrize("row", [_row("ready", "sha256:" + "e" * 64), _row("failed")])
    def test_an_image_that_settled_otherwise_is_a_409(self, row: ContainerImage) -> None:
        response, entities = self._post(row)
        assert response.status_code == 409
        entities.update.assert_not_awaited()

    def test_a_repeat_that_starts_a_remote_image_another_way_is_a_409(self) -> None:
        """Answered 200, the caller would take the runtime configuration the row has for the one it sent."""
        row = _row("ready", DIGEST, backend="remote")
        row.external = ExternalImage(build_id="build-1")
        row.runtime_config = ImageRuntimeConfig(entrypoint=["/altered"])
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}, "runtime_config": {"entrypoint": ["/published"]}}
        response, entities = self._post(row, body=body)
        assert response.status_code == 409
        assert "another runtime configuration" in response.json()["detail"]
        entities.update.assert_not_awaited()
        # The completion the row was made ready by, repeated, is still returned.
        response, entities = self._post(row, body=body | {"runtime_config": {"entrypoint": ["/altered"]}})
        assert (response.status_code, response.json()["runtime_config"]["entrypoint"]) == (200, ["/altered"])
        entities.update.assert_not_awaited()

    def test_a_repeat_for_an_image_built_here_before_how_it_starts_was_recorded_returns_it(self) -> None:
        """A push step's completion, repeated across an upgrade, carries what the row was written without."""
        body = {"digest": DIGEST, "runtime_config": {"entrypoint": ["/bin/app"], "user": "agent"}}
        response, entities = self._post(_row("ready", DIGEST), body=body)
        assert (response.status_code, response.json()["runtime_config"]) == (200, None)
        entities.update.assert_not_awaited()

    def test_a_repeat_that_starts_an_image_built_here_another_way_is_a_409(self) -> None:
        row = _row("ready", DIGEST)
        row.runtime_config = ImageRuntimeConfig(entrypoint=["/altered"], user="")
        body = {"digest": DIGEST, "runtime_config": {"entrypoint": ["/bin/app"], "user": ""}}
        response, entities = self._post(row, body=body)
        assert response.status_code == 409
        assert "another runtime configuration" in response.json()["detail"]
        entities.update.assert_not_awaited()

    @pytest.mark.parametrize("caller", [STEP, SERVICE_STEP], ids=["with-token-exchange", "without"])
    def test_a_job_step_acting_for_the_images_submitter_may_complete_it(self, caller: Caller) -> None:
        assert self._post(_row(), caller=caller)[0].status_code == 200

    @pytest.mark.parametrize(
        "caller",
        [
            Caller(auth=True, actor="alice"),
            Caller(auth=True, actor="job-step", submitter="bob"),
            Caller(auth=True, actor="service:builder"),
            Caller(auth=True, actor="service:builder", submitter="bob"),
        ],
        ids=["a-users-own-token", "a-step-acting-for-someone-else", "a-service-alone", "a-service-for-someone-else"],
    )
    def test_any_other_caller_is_a_403_and_nothing_is_written(self, caller: Caller) -> None:
        response, entities = self._post(_row(), caller=caller)
        assert response.status_code == 403
        entities.update.assert_not_awaited()

    @pytest.mark.parametrize(
        "caller",
        [SERVICE_STEP, NAMED_SERVICE, Caller(auth=True, actor="mallory", submitter="alice")],
        ids=["the-images-own-step", "a-request-naming-another-service", "a-request-naming-anyone"],
    )
    def test_without_token_exchange_no_request_completes_a_remote_build(self, caller: Caller) -> None:
        """With platform auth on and no workload token exchange, who is calling is whatever a request's headers
        say. The row's own step and anything else acting for its submitter look alike, and no secret the
        builder could hand the step is out of the other's reach: none of them is let make the row ready."""
        job_of = _job_of()
        chosen = {"digest": "sha256:" + "f" * 64, "external": {"build_id": "made-up"}}
        response, entities = self._post(_row(backend="remote"), caller=caller, body=chosen, job_of=job_of)
        assert response.status_code == 403
        # The reason says which case it was, and what the operator can set.
        assert "platform auth is on without workload token exchange" in response.json()["detail"]
        assert "builder.remote.allow_unverified_completion is not set" in response.json()["detail"]
        entities.update.assert_not_awaited()
        job_of.assert_not_awaited()

    def test_what_an_operator_accepts_by_allowing_unverified_completions(self) -> None:
        """Not wanted, and what the setting says it accepts: without workload token exchange nothing the
        platform hands the route says which job is calling, so with it on, a remote row's own step and
        anything else acting for its submitter are let through alike, on whom they say they act for."""
        entities = _entities(_row(backend="remote"))
        chosen = "sha256:" + "f" * 64
        starts = {"entrypoint": ["/chosen"], "cmd": None, "working_dir": None, "user": None}
        first = TestClient(_app(entities, NAMED_SERVICE, unverified=True)).post(
            f"{IMAGE}/complete",
            json={"digest": chosen, "external": {"build_id": "made-up"}, "runtime_config": starts},
        )
        assert (first.status_code, first.json()["status"]) == (200, "ready")
        # What a consumer then pins and runs: the digest, the build and the entrypoint the other workload chose.
        assert first.json()["image_ref"] == f"reg.example.com/ws-a/app@{chosen}"
        assert first.json()["external"]["build_id"] == "made-up"
        assert first.json()["runtime_config"] == starts
        # And the row's own step, with the digest its build published, is the one refused.
        own = TestClient(_app(entities, SERVICE_STEP, unverified=True)).post(
            f"{IMAGE}/complete", json={"digest": DIGEST, "external": {"build_id": "build-1"}}
        )
        assert own.status_code == 409
        entities.update.assert_awaited_once()
        # Whom the caller acts for is still asked: the setting accepts that nothing shows which job it is.
        someone_elses = Caller(auth=True, actor="service:builder", submitter="bob")
        refused, _ = self._post(_row(backend="remote"), caller=someone_elses, unverified=True)
        assert (refused.status_code, "submitted by someone else" in refused.json()["detail"]) == (403, True)

    def test_another_jobs_step_is_refused_and_the_images_own_makes_the_first_completion(self) -> None:
        """Under workload token exchange each step's token names the platform's record of its pod, which says
        which job the pod is a step of: another job of the submitter's is told from the image's own."""
        entities = _entities(_row(backend="remote"))
        job_of = _job_of()
        other = TestClient(_app(entities, EXCHANGED_STEP, delegation=OTHER_JOBS, job_of=job_of, exchange=True)).post(
            f"{IMAGE}/complete", json={"digest": "sha256:" + "f" * 64, "external": {"build_id": "made-up"}}
        )
        assert other.status_code == 403
        assert "a step of another job" in other.json()["detail"]
        entities.update.assert_not_awaited()
        # The row is still pending, and its own step completes it with what its build published.
        own = TestClient(_app(entities, EXCHANGED_STEP, delegation=OWN, job_of=job_of, exchange=True)).post(
            f"{IMAGE}/complete", json={"digest": DIGEST, "external": {"build_id": "build-1"}}
        )
        assert (own.status_code, own.json()["status"], own.json()["digest"]) == (200, "ready", DIGEST)
        assert [call.args for call in job_of.await_args_list] == [(OTHER_JOBS,), (OWN,)]
        entities.update.assert_awaited_once()

    def test_a_repeat_from_another_jobs_step_is_refused_too(self) -> None:
        """It is not told what the image's own step wrote by sending the same."""
        row = _row("ready", DIGEST, backend="remote")
        row.external = ExternalImage(build_id="build-1")
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        how: dict[str, Any] = {"caller": EXCHANGED_STEP, "body": body, "job_of": _job_of(), "exchange": True}
        response, _ = self._post(row, delegation=OTHER_JOBS, **how)
        assert response.status_code == 403
        response, _ = self._post(row, delegation=OWN, **how)
        assert response.status_code == 200

    @pytest.mark.parametrize("steps", [{}, {OWN: None}], ids=["no-longer-on-record", "not-a-job-steps"])
    def test_a_step_the_platform_has_no_job_on_record_for_is_refused(
        self, steps: dict[str, RecordedStep | None]
    ) -> None:
        """A token outlives the record it was minted from by a little: its job has ended, and the record with it."""
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        response, entities = self._post(
            _row(backend="remote"),
            caller=EXCHANGED_STEP,
            body=body,
            delegation=OWN,
            job_of=_job_of(steps),
            exchange=True,
        )
        assert response.status_code == 403
        assert "not on record as a job step" in response.json()["detail"]
        entities.update.assert_not_awaited()

    def test_a_record_that_cant_be_read_admits_no_one(self, caplog: pytest.LogCaptureFixture) -> None:
        """Answered as an outage, which the step waits out: the image is built, and only this call is left."""
        job_of = AsyncMock(side_effect=RuntimeError("entity store down"))
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        response, entities = self._post(
            _row(backend="remote"), caller=EXCHANGED_STEP, body=body, delegation=OWN, job_of=job_of, exchange=True
        )
        assert response.status_code == 503
        assert "entity store down" not in response.text
        assert "could not tell which job is completing ws-a/build-1-0" in caplog.text
        entities.update.assert_not_awaited()

    @pytest.mark.parametrize(
        ("how", "unread"),
        [
            ({"exchange": WorkloadIdentityConfigError("no auth section: hunter2")}, "hunter2"),
            ({"unverified": BackendRejectedError("the builder's configuration can't be read: hunter2")}, "hunter2"),
            ({"exchange": WorkloadIdentityConfigError("no auth section: hunter2"), "unverified": True}, "hunter2"),
        ],
        ids=[
            "how-the-platform-authenticates-a-workload",
            "what-the-operator-accepted",
            "how-it-authenticates-a-workload-though-the-operator-accepts-unverified",
        ],
    )
    def test_a_deployment_whose_settings_cant_be_read_admits_no_one(
        self, how: dict[str, Any], unread: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Answered as a record that couldn't be read is: an outage, which the step waits out. What the
        operator accepted where nothing is verified accepts nothing where that can't be told: the platform
        may be verifying its workloads. So a submit there is refused first, whatever the setting."""
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        response, entities = self._post(
            _row(backend="remote"), caller=EXCHANGED_STEP, body=body, delegation=OWN, job_of=_job_of(), **how
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "Couldn't check which job the caller is a step of; try again."
        assert unread not in response.text
        assert "could not tell which job is completing ws-a/build-1-0" in caplog.text
        entities.update.assert_not_awaited()

    def test_with_auth_off_no_caller_is_asked_anything(self) -> None:
        """As the README says of that mode: anyone may complete an image, a remote build's included, whether
        or not its row records its job yet. How the platform would authenticate a caller is not read."""
        job_of, never = _job_of(), MagicMock(side_effect=AssertionError("not asked with auth off"))
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        for row in (_row(backend="remote"), _row(backend="remote", recorded=False)):
            response, entities = self._post(row, body=body, job_of=job_of, exchange=never, unverified=never)
            assert (response.status_code, response.json()["status"]) == (200, "ready")
            entities.update.assert_awaited_once()
        job_of.assert_not_awaited()
        never.assert_not_called()

    @pytest.mark.parametrize("unverified", [False, True], ids=["by-default", "though-the-operator-accepts-unverified"])
    @pytest.mark.parametrize(
        "caller",
        [NAMED_SERVICE, SUBPROCESS_STEP, SERVICE_STEP],
        ids=["a-platform-service-for-the-submitter", "a-step-jobs-runs-outside-kubernetes", "the-builders-own-name"],
    )
    def test_under_token_exchange_a_caller_whose_token_names_no_workload_is_refused(
        self, caller: Caller, unverified: bool
    ) -> None:
        """A platform service acting for the submitter calls with such a token, and Jobs writes one for each
        step it runs outside Kubernetes. Neither is the build's step, which always has a record there. What
        the operator accepted where nothing can be verified accepts nothing here."""
        job_of = _job_of()
        chosen = {"digest": "sha256:" + "f" * 64, "external": {"build_id": "made-up"}}
        response, entities = self._post(
            _row(backend="remote"), caller=caller, body=chosen, job_of=job_of, exchange=True, unverified=unverified
        )
        assert response.status_code == 403
        assert "the caller's token names no workload" in response.json()["detail"]
        entities.update.assert_not_awaited()
        job_of.assert_not_awaited()

    def test_no_record_is_read_for_a_caller_refused_already(self) -> None:
        job_of = _job_of()
        someone_elses = Caller(auth=True, actor="system:serviceaccount:builds:build-remote", submitter="bob")
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        response, _ = self._post(
            _row(backend="remote"), caller=someone_elses, body=body, delegation=OWN, job_of=job_of, exchange=True
        )
        assert response.status_code == 403
        job_of.assert_not_awaited()

    @pytest.mark.parametrize(
        ("step", "reason"),
        [
            (RecordedStep("ws-a", "build-1", "attempt-2"), "another run than the one its submit created"),
            (RecordedStep("ws-a", "build-1", "attempt-2", job_uid=JOB_UID), "another run than the one its submit"),
            (RecordedStep("ws-a", "build-1", "attempt-2", job_uid="platform-job-2"), "created under its build job's"),
            (RecordedStep("ws-a", "build-1", ATTEMPT, job_uid="platform-job-2"), "created under its build job's"),
            (RecordedStep("ws-b", "build-1", ATTEMPT), "a step of another job"),
        ],
        ids=[
            "a-job-put-under-the-name-once-the-builds-was-deleted",
            "a-rerun-of-the-builds-own-job",
            "a-job-under-the-name-where-the-record-names-the-jobs-own-id",
            "another-job-though-the-attempt-reads-the-same",
            "a-job-of-the-same-name-in-another-workspace",
        ],
    )
    def test_a_step_of_any_run_but_the_one_the_submit_created_is_refused(self, step: RecordedStep, reason: str) -> None:
        """The platform's record names a job as the row does, by workspace and name, and a job created once
        the build's is deleted can take its name. So can a rerun run a specification someone changed since.
        The row records the attempt its submit created, and the record must be a step of that one."""
        body = {"digest": "sha256:" + "f" * 64, "external": {"build_id": "made-up"}}
        response, entities = self._post(
            _row(backend="remote"),
            caller=EXCHANGED_STEP,
            body=body,
            delegation=OTHER_JOBS,
            job_of=_job_of({OTHER_JOBS: step}),
            exchange=True,
        )
        assert response.status_code == 403
        assert reason in response.json()["detail"]
        entities.update.assert_not_awaited()

    def test_a_step_whose_record_names_the_jobs_own_id_and_its_attempt_completes_it(self) -> None:
        """Where a later platform records the job's own ID for a step's pod, both are held to the row's."""
        step = RecordedStep("ws-a", "build-1", ATTEMPT, job_uid=JOB_UID)
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        response, entities = self._post(
            _row(backend="remote"),
            caller=EXCHANGED_STEP,
            body=body,
            delegation=OWN,
            job_of=_job_of({OWN: step}),
            exchange=True,
        )
        assert (response.status_code, response.json()["status"]) == (200, "ready")
        entities.update.assert_awaited_once()

    def test_a_row_that_does_not_record_its_job_yet_is_a_503(self, caplog: pytest.LogCaptureFixture) -> None:
        """The submit writes the rows, creates the job, then writes on the rows which job it created. A step
        that calls in between is not refused: answered as an outage, which it waits out."""
        row = _row(backend="remote", recorded=False)
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        how: dict[str, Any] = {"caller": EXCHANGED_STEP, "body": body, "job_of": _job_of(), "exchange": True}
        response, entities = self._post(row, delegation=OWN, **how)
        assert response.status_code == 503
        assert response.json()["detail"] == "build-1-0 does not record its build job yet; try again."
        assert "completing ws-a/build-1-0 put off: its row records no build job yet" in caplog.text
        entities.update.assert_not_awaited()
        # A caller that is no step of a job of that name is told so at once: no row will come to admit it.
        response, entities = self._post(row, delegation=OTHER_JOBS, **how)
        assert (response.status_code, "a step of another job" in response.json()["detail"]) == (403, True)
        response, entities = self._post(row, **how)
        assert (response.status_code, "names no workload" in response.json()["detail"]) == (403, True)
        # Once the submit has written it, the step's next try completes the image.
        response, entities = self._post(_row(backend="remote"), delegation=OWN, **how)
        assert (response.status_code, response.json()["status"]) == (200, "ready")

    @pytest.mark.parametrize(
        ("caller", "how", "reason"),
        [
            (SERVICE_STEP, {}, "platform auth is on without workload token exchange"),
            (NAMED_SERVICE, {"exchange": True}, "the caller's token names no workload"),
            (EXCHANGED_STEP, {"exchange": True, "delegation": OWN, "steps": {}}, "not on record as a job step"),
            (EXCHANGED_STEP, {"exchange": True, "delegation": OTHER_JOBS}, "a step of another job"),
            (
                EXCHANGED_STEP,
                {"exchange": True, "delegation": OWN, "steps": {OWN: RecordedStep("ws-a", "build-1", "attempt-2")}},
                "another run than the one its submit created",
            ),
        ],
        ids=["nothing-verifies", "no-workload", "not-on-record", "another-job", "another-run"],
    )
    def test_each_refusal_says_which_it_was_and_quotes_nothing_the_caller_presented(
        self, caller: Caller, how: dict[str, Any], reason: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each is a 403 that names the image and the case. None repeats the delegation the caller's token
        named, nor the job the row records: what was presented, and what would pass, stay out of the answer
        and the log."""
        how = dict(how)
        steps = how.pop("steps", None)
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        response, entities = self._post(_row(backend="remote"), caller=caller, body=body, job_of=_job_of(steps), **how)
        assert response.status_code == 403
        assert reason in response.json()["detail"] and "build-1-0" in response.json()["detail"]
        assert f"completing ws-a/build-1-0 refused: {response.json()['detail']}" in caplog.text
        for unsaid in (OWN, OTHER_JOBS, JOB_UID, ATTEMPT, "attempt-2"):
            assert unsaid not in response.text and unsaid not in caplog.text
        entities.update.assert_not_awaited()

    @pytest.mark.parametrize(
        ("how", "logged"),
        [
            ({}, "completing ws-a/build-1-0forged line refused: build-1-0 is built by a build service"),
            ({"exchange": True, "delegation": OWN, "recorded": False}, "completing ws-a/build-1-0forged line put off"),
            (
                {"exchange": True, "delegation": OWN, "job_of": AsyncMock(side_effect=RuntimeError("down"))},
                "could not tell which job is completing ws-a/build-1-0forged line",
            ),
        ],
        ids=["refused", "put-off", "could-not-tell"],
    )
    def test_what_the_request_named_is_logged_on_one_line(
        self, how: dict[str, Any], logged: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A name in the path can carry a line break. Each line the route logs of it is one line."""
        how = {"job_of": _job_of()} | how
        entities = _entities(_row(backend="remote", recorded=how.pop("recorded", True)))
        app = _app(entities, EXCHANGED_STEP, **how)
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        response = TestClient(app).post(f"{IMAGE}%0Aforged line/complete", json=body)
        assert response.status_code in (403, 503)
        (line,) = [record.getMessage() for record in caplog.records if record.name == service.logger.name]
        assert line.startswith(logged) and "\n" not in line

    @pytest.mark.parametrize(
        ("caller", "how"),
        [
            (EXCHANGED_STEP, {"delegation": OTHER_JOBS, "exchange": True}),
            (NAMED_SERVICE, {"exchange": True}),
            (SERVICE_STEP, {}),
            (NAMED_SERVICE, {"unverified": True}),
        ],
        ids=["another-jobs-step", "a-service-token-under-exchange", "headers-by-default", "headers-accepted"],
    )
    def test_an_image_this_cluster_builds_is_completed_by_whoever_acts_for_its_submitter(
        self, caller: Caller, how: dict[str, Any]
    ) -> None:
        """Not wanted, and stated in the README's Security section: only a row a build service built is held
        to its own job. The push step's completion is as it was before the builder asked, however the
        platform authenticates a caller and whatever the operator set for remote builds: none of it is read."""
        job_of, never = _job_of(), MagicMock(side_effect=AssertionError("not asked of a row built here"))
        entities = _entities(_row())
        app = _app(entities, caller, delegation=how.get("delegation"), job_of=job_of, exchange=never, unverified=never)
        response = TestClient(app).post(f"{IMAGE}/complete", json={"digest": DIGEST})
        assert (response.status_code, response.json()["status"]) == (200, "ready")
        job_of.assert_not_awaited()
        never.assert_not_called()
        entities.update.assert_awaited_once()
        # Nor does a row built here come to record more of its job than its name.
        stored = entities.update.await_args.args[0]
        assert (stored.provenance.job_uid, stored.provenance.job_attempt_id) == (None, None)

    def test_what_a_build_services_client_reported_of_how_the_image_starts_is_recorded(self) -> None:
        """A remote-build step sends it when its service's client reported one, as the push step sends what it
        read: a consumer that must start the image's own entrypoint has nowhere else to learn it."""
        runtime_config = {"entrypoint": ["/entrypoint.sh"], "cmd": ["serve"], "working_dir": "/srv", "user": ""}
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}, "runtime_config": runtime_config}
        response, entities = self._post(
            _row(backend="remote"), caller=EXCHANGED_STEP, body=body, delegation=OWN, job_of=_job_of(), exchange=True
        )
        assert response.status_code == 200
        assert response.json()["runtime_config"] == runtime_config
        assert response.json()["external"]["build_id"] == "build-1"
        entities.update.assert_awaited_once()

    def test_an_image_that_does_not_exist_is_a_404(self) -> None:
        entities = AsyncMock()
        entities.get.side_effect = NemoEntityNotFoundError("no")
        response = TestClient(_app(entities)).post(f"{IMAGE}/complete", json={"digest": DIGEST})
        assert response.status_code == 404

    @pytest.mark.parametrize(
        "body",
        [
            {"digest": "sha256:abc"},
            {"digest": DIGEST, "signature": "x"},
            {},
            {"digest": DIGEST, "runtime_config": {"entrypoint": "/start"}},
        ],
    )
    def test_anything_but_a_digest_and_how_the_image_starts_is_refused(self, body: dict) -> None:
        response, entities = self._post(_row(), body=body)
        assert response.status_code == 422
        entities.update.assert_not_awaited()


#: The claims of the token the platform's auth service mints for a job step that has proved which pod it is.
MINTED = {
    "sub": "alice",
    "act": {"sub": "system:serviceaccount:builds:build-remote"},
    "nhx_service_attributes": {"auth": {"delegation_id": "workload-delegation-1"}},
}


def _authenticated(
    kind: str | None, claims: object, *, actor: str | None = "system:serviceaccount:builds:build-remote"
) -> SimpleNamespace:
    """The auth client the platform's middleware leaves for a route, with the token it verified for the request."""
    acting = SimpleNamespace(subject=actor, groups=[]) if actor else None
    token = SimpleNamespace(token_kind=kind, claims=SimpleNamespace(actor=acting, raw_claims=claims))
    return SimpleNamespace(resolved_bearer_token=token)


def _delegation(
    job: str = "build-1",
    *,
    kind: str = "job",
    now: datetime | None = None,
    steps: bool = True,
    attempt: str = ATTEMPT,
    record_id: str = "workload-delegation-1",
) -> WorkloadDelegationEntity:
    """The record Jobs' Kubernetes backend makes for a step's pod, built with what it builds one from: the
    job's name, for the workload and for ``job_id``, and the ID of the attempt the step belongs to."""
    record = build_kubernetes_pod_uid_workload_delegation(
        scope=WorkloadDelegationScope(workload_workspace="ws-a", workload_kind=kind, workload_instance_id=job),
        workload_audience="platform",
        workload_generation=f"{attempt}/step-1/pod-of-{record_id}",
        namespace="builds",
        service_account_name="build-remote",
        pod_uid=f"pod-of-{record_id}",
        job_id=job if steps else None,
        attempt_id=attempt if steps else None,
        step_id="step-1" if steps else None,
        auth_context=AuthContext(principal_id="alice"),
        ttl_seconds_active=3600,
        now=now,
    )
    record._id = record_id
    return record


async def _refusal(
    row: ContainerImage,
    caller: Caller,
    delegation: str | None,
    job_of: AsyncMock,
    *,
    exchange: Setting = True,
    unverified: Setting = False,
) -> str | None:
    """``step_refusal``, in a deployment that authenticates a caller as ``exchange`` and ``unverified`` say."""
    return await step_refusal(
        row, caller, delegation, job_of, token_exchange=_asked(exchange), allow_unverified=_asked(unverified)
    )


class TestWhichJobIsCalling:
    """Under workload token exchange, a step's token names the platform's record of its pod."""

    def test_a_workload_tokens_delegation_is_read_from_its_verified_claims(self) -> None:
        assert verified_delegation(_authenticated("workload_access_token", MINTED)) == "workload-delegation-1"

    @pytest.mark.parametrize(
        ("kind", "actor"),
        [
            ("access_key", "service:evals"),
            ("oidc_access_token", "service:evals"),
            ("workload_subject_token", "service:evals"),
            (None, "service:evals"),
            ("workload_access_token", None),
        ],
        ids=["an-access-key", "a-users-token", "a-pods-own-token", "no-kind", "a-workload-acting-for-no-one"],
    )
    def test_the_same_claim_in_any_other_token_is_not_the_platforms_word(
        self, kind: str | None, actor: str | None
    ) -> None:
        """An access key carries attributes a platform service chose for it. Only the token the auth service
        mints for a workload it has verified, acting for someone, names a delegation."""
        assert verified_delegation(_authenticated(kind, MINTED, actor=actor)) is None

    @pytest.mark.parametrize(
        "client",
        [
            SimpleNamespace(),
            SimpleNamespace(resolved_bearer_token=None),
            _authenticated("workload_access_token", {"sub": "alice", "act": {"sub": "service:evals"}}),
            _authenticated("workload_access_token", {"nhx_service_attributes": {"auth": {"delegation_id": ""}}}),
            _authenticated("workload_access_token", {"nhx_service_attributes": {"auth": {"delegation_id": 7}}}),
            _authenticated("workload_access_token", {"nhx_service_attributes": {"auth": "workload-delegation-1"}}),
            _authenticated("workload_access_token", {"nhx_service_attributes": ["workload-delegation-1"]}),
            _authenticated("workload_access_token", None),
        ],
        ids=[
            "headers",
            "no-token",
            "a-services-own-token",
            "an-empty-id",
            "an-id-that-is-no-text",
            "no-mapping-for-auth",
            "no-mapping-of-attributes",
            "no-claims",
        ],
    )
    def test_a_request_whose_token_names_none_has_none(self, client: SimpleNamespace) -> None:
        assert verified_delegation(client) is None

    def test_the_claim_read_is_the_one_the_platform_mints_for_a_job_step(self) -> None:
        """Against the platform's own auth, where it is installed: the builder reads a verified token by the
        names the platform gives its parts, and would ask nothing if one of them changed."""
        exchange = pytest.importorskip("nhx.core.auth.api.v2.workload_token_exchange")
        token_claims = pytest.importorskip("nhx.common.auth.token_claims")
        token_resolver = pytest.importorskip("nhx.common.auth.token_resolver")
        auth_client = pytest.importorskip("nhx.common.auth.client")
        record = _delegation()
        minted = exchange._build_delegated_claims(
            record, exchange.VerifiedSubjectToken(subject=record.workload_subject, groups=[])
        )
        claims = token_claims.TokenClaims(
            subject=minted["sub"],
            email=None,
            groups=[],
            scopes=[],
            raw_claims=minted,
            actor=token_claims.ActorClaims(subject=minted["act"]["sub"], groups=[]),
        )
        token = token_resolver.ResolvedBearerToken(claims=claims, token_kind="workload_access_token")
        # Where a route finds the token: on the auth client the middleware leaves for it.
        assert "resolved_bearer_token" in auth_client.AuthClient.model_fields
        assert verified_delegation(SimpleNamespace(resolved_bearer_token=token)) == record.id

    def test_the_route_reads_the_delegation_the_requests_token_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Through the route, with the caller and its token read as the platform leaves them for it."""
        monkeypatch.setattr(completion, "platform_auth_enabled", lambda: True)
        monkeypatch.setattr(
            completion,
            "current_auth_context",
            lambda: AuthContext(
                principal_id="system:serviceaccount:builds:build-remote", principal_on_behalf_of="alice"
            ),
        )
        monkeypatch.setattr(
            completion, "current_auth_client", lambda _request: _authenticated("workload_access_token", MINTED)
        )
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        for step, answer in ((ANOTHER_JOBS_STEP, 403), (OWN_STEP, 200)):
            job_of = _job_of({"workload-delegation-1": step})
            app = _app(_entities(_row(backend="remote")), job_of=job_of, exchange=True)
            del app.dependency_overrides[current_caller], app.dependency_overrides[current_delegation]
            assert TestClient(app).post(f"{IMAGE}/complete", json=body).status_code == answer
            job_of.assert_awaited_once_with("workload-delegation-1")

    def test_with_auth_off_no_token_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anyone may complete an image then, so which workload one of them is changes nothing."""
        monkeypatch.setattr(completion, "platform_auth_enabled", lambda: False)
        monkeypatch.setattr(
            completion, "current_auth_client", lambda _request: _authenticated("workload_access_token", MINTED)
        )
        job_of = _job_of({"workload-delegation-1": ANOTHER_JOBS_STEP})
        app = _app(_entities(_row(backend="remote")), job_of=job_of, exchange=True)
        del app.dependency_overrides[current_caller], app.dependency_overrides[current_delegation]
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        assert TestClient(app).post(f"{IMAGE}/complete", json=body).status_code == 200
        job_of.assert_not_awaited()

    async def test_only_a_step_of_the_attempt_the_images_submit_created_passes(self) -> None:
        row = _row(backend="remote")
        assert await _refusal(row, EXCHANGED_STEP, OWN, _job_of()) is None
        assert "a step of another job" in (await _refusal(row, EXCHANGED_STEP, OTHER_JOBS, _job_of()) or "")
        assert "not on record as a job step" in (await _refusal(row, EXCHANGED_STEP, OWN, _job_of({})) or "")
        rerun = _job_of({OWN: RecordedStep("ws-a", "build-1", "attempt-2")})
        assert "another run than the one its submit created" in (await _refusal(row, EXCHANGED_STEP, OWN, rerun) or "")

    async def test_a_caller_that_presents_nothing_to_check_is_refused_not_passed(self) -> None:
        """Under workload token exchange the check of which job is calling is what ties a completion to the
        row's build. It is failed, not skipped, for a token that names no workload."""
        job_of = _job_of()
        for caller in (NAMED_SERVICE, SUBPROCESS_STEP, EXCHANGED_STEP):
            refusal = await _refusal(_row(backend="remote"), caller, None, job_of)
            assert "the caller's token names no workload" in (refusal or "")
        job_of.assert_not_awaited()

    async def test_without_token_exchange_no_caller_can_be_shown_to_be_the_step(self) -> None:
        """Nothing is read of the caller, which has nothing verified to present: the operator's setting decides."""
        job_of = _job_of()
        row = _row(backend="remote")
        refusal = await _refusal(row, SERVICE_STEP, None, job_of, exchange=False)
        assert "builder.remote.allow_unverified_completion is not set" in (refusal or "")
        assert await _refusal(row, SERVICE_STEP, None, job_of, exchange=False, unverified=True) is None
        # A row that records no job yet is completed there too: nothing would be compared with what it records.
        unrecorded = _row(backend="remote", recorded=False)
        assert await _refusal(unrecorded, SERVICE_STEP, None, job_of, exchange=False, unverified=True) is None
        job_of.assert_not_awaited()

    async def test_a_row_that_records_no_attempt_yet_can_admit_no_step_and_refuse_none(self) -> None:
        row = _row(backend="remote", recorded=False)
        with pytest.raises(JobNotRecorded, match="build-1-0 does not record its build job yet"):
            await _refusal(row, EXCHANGED_STEP, OWN, _job_of())

    async def test_nothing_is_asked_where_no_caller_is_checked_or_the_image_is_built_here(self) -> None:
        job_of, never = _job_of(), MagicMock(side_effect=AssertionError("not asked"))
        for row in (_row(backend="remote"), _row(backend="remote", recorded=False), _row()):
            assert await _refusal(row, Caller(auth=False), None, job_of, exchange=never, unverified=never) is None
        for caller, delegation in ((EXCHANGED_STEP, OTHER_JOBS), (NAMED_SERVICE, None), (SERVICE_STEP, None)):
            assert await _refusal(_row(), caller, delegation, job_of, exchange=never, unverified=never) is None
        job_of.assert_not_awaited()
        never.assert_not_called()

    def test_what_the_operator_accepted_is_asked_only_where_nothing_can_be_verified(self) -> None:
        """With token exchange on the setting is not read: a builder whose settings can't be read still
        completes a build whose step it can verify."""
        unreadable = BackendRejectedError("the builder's configuration can't be read")
        response = TestClient(
            _app(
                _entities(_row(backend="remote")),
                EXCHANGED_STEP,
                delegation=OWN,
                job_of=_job_of(),
                exchange=True,
                unverified=unreadable,
            )
        ).post(f"{IMAGE}/complete", json={"digest": DIGEST, "external": {"build_id": "build-1"}})
        assert response.status_code == 200

    @pytest.mark.parametrize(
        ("setting", "answer"),
        [(False, 403), (True, 200), (BackendRejectedError("the builder's configuration can't be read"), 503)],
        ids=["not-set", "set", "unreadable"],
    )
    def test_the_route_reads_what_the_operator_accepted_from_the_builders_configuration(
        self, setting: bool | Exception, answer: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def load() -> BuilderConfig:
            if isinstance(setting, Exception):
                raise setting
            return BuilderConfig.model_validate({"remote": {"allow_unverified_completion": setting}})

        monkeypatch.setattr(service, "_load_config", load)
        app = _app(_entities(_row(backend="remote")), SERVICE_STEP)
        del app.dependency_overrides[get_allow_unverified]
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        assert TestClient(app).post(f"{IMAGE}/complete", json=body).status_code == answer

    @pytest.mark.parametrize(
        ("exchange", "answer"),
        [(lambda: True, 200), (lambda: False, 403), (_asked(WorkloadIdentityConfigError("unreadable")), 503)],
        ids=["on", "off", "unreadable"],
    )
    def test_the_route_asks_the_platform_whether_it_mints_its_workloads_tokens(
        self, exchange: Callable[[], bool], answer: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(service, "is_workload_identity_token_exchange_enabled", exchange)
        app = _app(_entities(_row(backend="remote")), EXCHANGED_STEP, delegation=OWN, job_of=_job_of())
        del app.dependency_overrides[get_token_exchange]
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        assert TestClient(app).post(f"{IMAGE}/complete", json=body).status_code == answer

    def test_a_delegations_step_is_of_the_job_its_images_row_names_and_of_one_attempt_of_it(self) -> None:
        """Jobs records the workspace, the job's name and the attempt for a step's pod. The row names its build
        job by the first two, which another job can come to hold, and records the attempt its submit created."""
        row = _row(backend="remote")
        step = service._step_on_record(_delegation())
        assert step == OWN_STEP == RecordedStep(workspace="ws-a", job="build-1", attempt_id=ATTEMPT, job_uid=None)
        assert f"{step.workspace}/{step.job}" == row.provenance.job == "ws-a/build-1"
        assert step.attempt_id == row.provenance.job_attempt_id
        assert service._step_on_record(_delegation("eval-7", attempt="attempt-7")) == ANOTHER_JOBS_STEP
        # Another job under the build job's name, or a rerun of it: the same job to its name, and another attempt.
        later = service._step_on_record(_delegation(attempt="attempt-2", record_id="workload-delegation-2"))
        assert later == RecordedStep(workspace="ws-a", job="build-1", attempt_id="attempt-2")

    def test_the_jobs_own_id_is_read_where_a_record_carries_it(self) -> None:
        """The platform records none for a step's pod yet. A record that does is read with it."""
        record = _delegation()
        assert "job_uid" not in type(record).model_fields
        fields = {name: getattr(record, name) for name in ("workload_workspace", "workload_kind", "job_id")}
        carrying = SimpleNamespace(**fields, attempt_id=record.attempt_id, job_uid=JOB_UID, is_active=lambda: True)
        assert service._step_on_record(cast(WorkloadDelegationEntity, carrying)) == RecordedStep(
            workspace="ws-a", job="build-1", attempt_id=ATTEMPT, job_uid=JOB_UID
        )

    def test_what_jobs_records_for_a_steps_pod_is_the_jobs_name_and_the_steps_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Against the platform's Jobs controller, where it is installed: the record its Kubernetes backend
        registers for a step's pod names the job by its name, though the step carries the job's own ID, and
        the attempt by its ID. So a row is held to the attempt: the name alone is another job's to take."""
        delegations = pytest.importorskip("nhx.core.jobs.controllers.backends.kubernetes.workload_delegations")
        labels = pytest.importorskip("nhx.core.jobs.app.constants")
        kubernetes = pytest.importorskip("kubernetes.client")
        job_types = pytest.importorskip("nemo_helix_plugin.jobs.types")
        monkeypatch.setattr(delegations, "is_workload_identity_token_exchange_enabled", lambda: True)
        step = job_types.HelixJobStepWithContext(
            id="step-1",
            job="build-1",
            job_id=JOB_UID,
            attempt_id=ATTEMPT,
            fileset="job-fileset-build-1",
            workspace="ws-a",
            name="remote-build",
            auth_context=job_types.AuthContext(principal_id="alice"),
        )
        pods = MagicMock()
        pods.list_namespaced_pod.return_value = kubernetes.V1PodList(
            items=[kubernetes.V1Pod(metadata=kubernetes.V1ObjectMeta(uid="pod-1", name="remote-build-pod"))]
        )
        registered: list[WorkloadDelegationEntity] = []
        manager = delegations.KubernetesPodBoundWorkloadDelegationManager(
            core_v1=pods,
            namespace="builds",
            ttl_seconds_active=lambda _step: 3600,
            workload_audience=lambda: "platform",
            register_workload_delegation=registered.append,
            revoke_workload_delegation=MagicMock(),
        )
        target = delegations.KubernetesPodBoundWorkloadDelegationTarget(
            namespace="builds",
            name="job-of-the-step",
            labels={
                labels.JOB_TYPE_LABEL: labels.JOB_TYPE_JOB,
                labels.JOB_WORKSPACE_ID_LABEL: "ws-a",
                labels.JOB_ID_LABEL: step.job,
                labels.JOB_ATTEMPT_ID_LABEL: step.attempt_id,
                labels.JOB_STEP_NAME_LABEL: step.name,
                labels.JOB_STEP_ID_LABEL: step.id,
                labels.JOB_MANAGED_BY_LABEL: labels.JOB_MANAGED_BY_JOBS_CONTROLLER,
            },
            service_account_name="build-remote",
        )
        assert manager.ensure_for_target(step, target) is True
        (record,) = registered
        assert JOB_UID not in record.model_dump_json()
        assert service._step_on_record(record) == OWN_STEP

    @pytest.mark.parametrize(
        "record",
        [
            None,
            _delegation(now=datetime(2020, 1, 1, tzinfo=UTC)),
            _delegation(kind="deployment", steps=False),
            _delegation(kind="deployment"),
            _delegation().model_copy(update={"attempt_id": None}),
            _delegation().model_copy(update={"job_id": None}),
        ],
        ids=[
            "no-longer-on-record",
            "expired",
            "a-deployments",
            "a-deployment-that-names-a-job",
            "a-job-steps-that-names-no-attempt",
            "a-job-steps-that-names-no-job",
        ],
    )
    def test_a_record_that_is_not_a_standing_job_steps_names_no_step(
        self, record: WorkloadDelegationEntity | None
    ) -> None:
        assert service._step_on_record(record) is None

    async def test_the_record_is_read_by_its_id_as_the_builder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not as the caller: the record is the platform's own, kept where no submitter reads."""
        record = _delegation()
        stored = {
            "entity_type": "workload_delegation",
            "id": record.id,
            "workspace": "system",
            "name": record.name,
            "data": record.model_dump(mode="json", exclude_computed_fields=True),
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "db_version": 1,
        }

        def entities(request: httpx.Request) -> httpx.Response:
            if request.url.path == f"/apis/entities/v2/entities/{record.id}":
                return httpx.Response(200, json=stored)
            return httpx.Response(404, json={"detail": "Entity not found"})

        client = AsyncNemoClient(
            base_url="http://platform.test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(entities))
        )
        asked: list[dict[str, object]] = []

        @asynccontextmanager
        async def as_builder(**kwargs: object) -> AsyncGenerator[AsyncNemoClient]:
            asked.append(kwargs)
            yield client

        monkeypatch.setattr(service, "get_async_nemo_client", as_builder)
        assert await service._job_of_as_builder(record.id) == OWN_STEP
        assert asked == [{"as_service": "builder", "internal": True}]
        assert await service._job_of_as_builder("workload-delegation-2") is None


class TestReadingOneImage:
    def _get(self, caller: Caller):
        return TestClient(_app(_entities(_row()), caller)).get(IMAGE)

    @pytest.mark.parametrize(
        "caller",
        [
            Caller(auth=False),
            Caller(auth=True, actor="carol"),
            STEP,
            SERVICE_STEP,
            Caller(auth=True, actor="service:evals", submitter="bob"),
        ],
        ids=[
            "auth-off",
            "a-person",
            "a-step-with-token-exchange",
            "the-push-step-without",
            "a-service-for-another-person",
        ],
    )
    def test_whom_the_policy_admits_reads_it(self, caller: Caller) -> None:
        """A service acting for someone other than the submitter reads it too, so a build can be reused."""
        response = self._get(caller)
        assert response.status_code == 200
        assert response.json()["name"] == "build-1-0"

    @pytest.mark.parametrize(
        "caller",
        [Caller(auth=True, actor="service:evals"), Caller(auth=True, actor="service:evals", submitter="service:other")],
        ids=["acting-for-no-one", "acting-for-a-service"],
    )
    def test_a_service_not_acting_for_a_person_is_a_403(self, caller: Caller) -> None:
        """The policy admits any service on its own permissions, so the handler holds it to acting for a person."""
        response = self._get(caller)
        assert response.status_code == 403
        assert "acting for a person" in response.json()["detail"]

    def test_an_image_that_does_not_exist_is_a_404(self) -> None:
        entities = AsyncMock()
        entities.get.side_effect = NemoEntityNotFoundError("no")
        assert TestClient(_app(entities, SERVICE_STEP)).get(IMAGE).status_code == 404

    def test_a_pending_image_whose_job_ended_fails_as_it_is_read(self) -> None:
        entities = _entities(_row())
        builder = _entities(_row())
        job_status = AsyncMock(return_value=HelixJobStatus.COMPLETED)
        response = TestClient(_app(entities, job_status=job_status, builder=builder)).get(IMAGE)
        assert response.status_code == 200
        assert response.json()["status"] == "failed"
        assert "ended (completed) without publishing it" in response.json()["status_detail"]
        # As the builder: a reader who may not write the image must not be refused, nor be its writer.
        builder.update.assert_awaited_once()
        entities.update.assert_not_awaited()

    def test_a_pending_image_whose_job_runs_is_read_as_it_is(self) -> None:
        entities = _entities(_row())
        response = TestClient(_app(entities)).get(IMAGE)
        assert response.json()["status"] == "pending"
        entities.update.assert_not_awaited()

    def test_a_refused_reader_settles_nothing(self) -> None:
        entities = _entities(_row())
        job_status = AsyncMock(return_value=HelixJobStatus.ERROR)
        builder = _entities(_row())
        app = _app(entities, Caller(auth=True, actor="service:evals"), job_status=job_status, builder=builder)
        assert TestClient(app).get(IMAGE).status_code == 403
        job_status.assert_not_awaited()
        builder.update.assert_not_awaited()

    @pytest.mark.parametrize("fails", ["constructing", "entering"])
    def test_a_builder_client_that_cant_be_had_returns_the_image_as_stored(
        self, fails: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Settling is best effort: the next read, or builder-settle, settles it."""

        @asynccontextmanager
        async def unenterable(**_kwargs: object) -> AsyncGenerator[AsyncNemoClient]:
            raise RuntimeError("no service token")
            yield

        def unconstructible(**_kwargs: object) -> object:
            raise RuntimeError("no service token")

        monkeypatch.setattr(
            service, "get_async_nemo_client", unconstructible if fails == "constructing" else unenterable
        )
        app = _app(_entities(_row()))
        del app.dependency_overrides[get_settle]
        response = TestClient(app).get(IMAGE)
        assert response.status_code == 200
        assert response.json()["status"] == "pending"
        assert "could not settle image ws-a/" in caplog.text


class TestSubmitting:
    BUILDS = "/v2/workspaces/ws-a/builds"
    BODY = {"name": "demo", "revision": 1, "build_specs": [{"source": {"fileset": "fs-a"}}]}

    @pytest.mark.parametrize(
        "caller",
        [Caller(auth=True, actor="service:evals"), Caller(auth=True, actor="service:evals", submitter="service:other")],
        ids=["acting-for-no-one", "acting-for-a-service"],
    )
    def test_a_service_not_acting_for_a_person_is_a_403_and_nothing_is_written(self, caller: Caller) -> None:
        entities, client = AsyncMock(), MagicMock()
        app = _app(entities, caller)
        app.dependency_overrides[get_nemo_client] = lambda: client
        response = TestClient(app).post(self.BUILDS, json=self.BODY)
        assert response.status_code == 403
        assert "acting for a person" in response.json()["detail"]
        assert entities.mock_calls == client.mock_calls == []

    @pytest.mark.parametrize(
        "caller",
        [
            Caller(auth=False),
            Caller(auth=True, actor="alice"),
            Caller(auth=True, actor="service:evals", submitter="alice"),
        ],
        ids=["auth-off", "a-person", "a-service-acting-for-a-person"],
    )
    def test_anyone_else_may(self, caller: Caller) -> None:
        assert submit_refusal(caller) is None

    def _post(self, monkeypatch: pytest.MonkeyPatch, config: BuilderConfig | None = None):
        """Submit to a deployment configured as ``config``, whose Jobs has no execution profiles."""

        def jobs(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/execution-profiles"), "nothing else is asked before a refusal"
            return httpx.Response(200, json=[])

        if config is not None:
            monkeypatch.setattr(service, "_load_config", lambda: config)
        client = AsyncNemoClient(
            base_url="http://platform.test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(jobs))
        )
        entities = AsyncMock()
        app = _app(entities)
        app.dependency_overrides[get_nemo_client] = lambda: client
        response = TestClient(app).post(self.BUILDS, json=self.BODY)
        entities.create.assert_not_awaited()
        return response

    @pytest.mark.parametrize(
        ("config", "named"),
        [
            ({"backend": "remote"}, "builder.remote.registry and builder.remote.repository"),
            ({"backend": "remot"}, "'remot', which is no backend"),
            ({}, "builder.registry"),
        ],
        ids=["remote-unconfigured", "no-such-backend", "in-cluster-unconfigured"],
    )
    def test_a_backends_refusal_is_a_409_that_says_it_is_one(
        self, config: dict, named: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller must tell it from a revision another request holds: this one the same request meets again."""
        response = self._post(monkeypatch, BuilderConfig.model_validate(config))
        assert response.status_code == 409
        assert response.json()["code"] == "backend_rejected"
        # `detail` is the sentence it always was, for a caller that only reads that.
        assert named in response.json()["detail"]

    def test_a_misspelt_setting_is_named_in_the_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def get(cls: type[BuilderConfig]) -> BuilderConfig:
            return BuilderConfig.model_validate({"remote": {"repositry": "team/images"}})

        monkeypatch.setattr(BuilderConfig, "get", classmethod(get))
        with pytest.raises(ValidationError):
            BuilderConfig.get()
        response = self._post(monkeypatch)
        assert response.status_code == 409
        assert response.json()["code"] == "backend_rejected"
        assert "configuration is invalid (remote.repositry: " in response.json()["detail"]

    def test_a_refused_value_is_named_but_not_repeated(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A validator's own sentence quotes what it refused, and an operator may have pasted a credential there."""

        def get(cls: type[BuilderConfig]) -> BuilderConfig:
            return BuilderConfig.model_validate({"remote": {"registry": "bot:t0ken@registry.example.com/team"}})

        monkeypatch.setattr(BuilderConfig, "get", classmethod(get))
        response = self._post(monkeypatch)
        assert response.status_code == 409
        assert response.json()["detail"] == (
            "the builder's configuration is invalid (remote.registry: invalid value); the API log says more"
        )
        # The operator's log has the rest, but not the value either: here it holds a password.
        assert "remote.registry must be a host" in caplog.text
        assert "t0ken" not in caplog.text

    @pytest.mark.parametrize(
        ("refused", "message", "logged"),
        [
            ("bot:t0ken@reg", "Value error, bad host 'bot:t0ken@reg'", "Value error, bad host <the value>"),
            (
                ["ghcr.io", "bot:t0ken@reg"],
                "Value error, not a host: bot:t0ken@reg",
                "Value error, not a host: <the value>",
            ),
            (
                {"registry": "/bot:t0ken@reg/"},
                "Value error, bad path bot:t0ken@reg",
                "Value error, bad path <the value>",
            ),
            (
                "/bot:pa\\ss@reg/team",
                "Value error, invalid repository path: " + repr("bot:pa\\ss@reg/team"),
                "Value error, invalid repository path: <the value>",
            ),
            # A value too short to be anything but a word of the sentence is left where it is.
            ("to", "Value error, set it to a host", "Value error, set it to a host"),
        ],
        ids=["a-string", "in-a-list", "trimmed-in-a-mapping", "trimmed-then-escaped", "too-short-to-take-out"],
    )
    def test_a_validator_that_quotes_what_it_refused_has_it_taken_out_of_the_log(
        self, refused: object, message: str, logged: str
    ) -> None:
        """The builder's own validators don't repeat a value. This is for one that does all the same."""
        error: Any = {"type": "value_error", "loc": ("registry",), "msg": message, "input": refused}
        assert service._without_input(error) == logged

    def test_pydantics_own_sentence_is_logged_as_it_is(self) -> None:
        """It doesn't quote the value, and the value may be in it by chance: here it is the limit itself."""
        error: Any = {
            "type": "less_than",
            "loc": ("remote", "build_timeout_seconds"),
            "msg": "Input should be less than 3600",
            "input": "3600",
        }
        assert service._without_input(error) == "Input should be less than 3600"

    def test_a_registry_to_copy_from_that_is_refused_is_not_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def get(cls: type[BuilderConfig]) -> BuilderConfig:
            return BuilderConfig.model_validate({"import_registries": ["ghcr.io", "bot:t0ken@registry.example.com/x"]})

        monkeypatch.setattr(BuilderConfig, "get", classmethod(get))
        response = self._post(monkeypatch)
        assert (response.status_code, response.json()["code"]) == (409, "backend_rejected")
        # The whole sentence: none of it is an entry of the list, which the log takes out.
        assert "import_registries must be registry hosts, with no scheme, login or path" in caplog.text
        assert "t0ken" not in caplog.text and "t0ken" not in response.text

    def test_a_check_on_the_section_as_a_whole_is_named_for_the_section(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Such a check has no one setting to name. An answer of `(: invalid value)` would name nothing."""

        def get(cls: type[BuilderConfig]) -> BuilderConfig:
            return BuilderConfig.model_validate({"registry_username_secret": "user", "registry_password_secret": None})

        monkeypatch.setattr(BuilderConfig, "get", classmethod(get))
        response = self._post(monkeypatch)
        assert response.status_code == 409
        assert response.json()["detail"] == (
            "the builder's configuration is invalid (builder: invalid value); the API log says more"
        )
        # And the log does: the check is given the whole section, whose keys are the settings its sentence names.
        assert "registry_username_secret and registry_password_secret are set together, or both null" in caplog.text

    def test_a_validator_that_repeats_what_it_refused_does_not_put_it_in_the_log(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """None of the builder's own does. One added later, or one of a library's, may: through the route."""

        class Careless(BaseModel):
            registry: str

            @field_validator("registry")
            @classmethod
            def _quotes_it(cls, value: str) -> str:
                raise ValueError(f"{value!r} is not a registry host")

        def get(cls: type[BuilderConfig]) -> BuilderConfig:
            Careless.model_validate({"registry": "bot:t0ken@registry.example.com"})
            raise AssertionError("unreachable")

        monkeypatch.setattr(BuilderConfig, "get", classmethod(get))
        response = self._post(monkeypatch)
        assert response.json()["detail"] == (
            "the builder's configuration is invalid (registry: invalid value); the API log says more"
        )
        assert "registry: Value error, <the value> is not a registry host" in caplog.text
        assert "t0ken" not in caplog.text

    def test_a_config_that_cant_be_read_at_all_is_a_refusal_too(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Such as a `builder:` section with every line under it commented out."""

        def get(cls: type[BuilderConfig]) -> BuilderConfig:
            raise ValueError("Settings for service 'builder' must be a mapping, got 'hunter2'.")

        monkeypatch.setattr(BuilderConfig, "get", classmethod(get))
        response = self._post(monkeypatch)
        assert response.status_code == 409
        assert response.json()["code"] == "backend_rejected"
        assert response.json()["detail"] == "the builder's configuration can't be read; the API log says why"
        assert "must be a mapping" in caplog.text

    @pytest.mark.parametrize(
        ("failure", "code"),
        [
            (MissingSecrets("the build job needs platform secrets"), "missing_secrets"),
            (BackendChanged("already submitted under the 'execution' backend"), "backend_changed"),
            (BuildConflict("already submitted with a different request"), None),
        ],
        ids=["secrets", "another-backends-row", "another-requests-revision"],
    )
    def test_each_other_409_says_which_it_is(
        self, failure: Exception, code: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def submit(*_args: object, **_kwargs: object) -> SubmitResult:
            raise failure

        monkeypatch.setattr(service, "load_backend", lambda *_args: None)
        monkeypatch.setattr(service, "submit_build_set", submit)
        response = self._post(monkeypatch, BuilderConfig.model_validate({}))
        assert response.status_code == 409
        assert response.json()["detail"] == str(failure)
        # A revision another request holds is the one a caller gets past by reading or resubmitting.
        assert response.json().get("code") == code


class TestWhoTheRoutesAdmit:
    """Submitting and the push step's two routes admit a service; listing, people only."""

    @staticmethod
    def _callers() -> dict[tuple[str, str], set[CallerKind]]:
        router = BuilderService().get_routers()[0].router
        return {
            (method, route.path): {kind for rule in get_path_rules(route.endpoint) for kind in rule.callers}
            for route in router.routes
            if isinstance(route, APIRoute)
            for method in route.methods or ()
        }

    def test_submitting_and_the_push_steps_routes_admit_a_service(self) -> None:
        callers = self._callers()
        both = {CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL}
        assert callers[("POST", "/builds")] == both
        assert callers[("GET", "/container-images/{name}")] == both
        assert callers[("POST", "/container-images/{name}/complete")] == both

    def test_listing_admits_people_only(self) -> None:
        """A service could see every row at once; one row at a time, it must act for a person."""
        assert self._callers()[("GET", "/container-images")] == {CallerKind.PRINCIPAL}


class TestAResubmitFindsItsJob:
    """When the build set's job already exists, the route reads that job from Jobs: the ID Jobs gave it, and the
    ``custom_fields`` its creator did."""

    def test_the_route_reads_the_named_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fields = {"images": ["build-1-0"], "request_digest": "sha256:" + "a" * 64}
        requests: list[httpx.Request] = []

        def jobs(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path.endswith("/execution-profiles"):
                return httpx.Response(200, json=[])
            return httpx.Response(
                200,
                json={
                    "id": "job-1",
                    "attempt_id": "attempt-1",
                    "name": "builder-build-1",
                    "workspace": "ws-a",
                    "source": "builder",
                    "platform_spec": {
                        "steps": [{"name": "fetch", "executor": {"provider": "cpu", "container": {"image": "busybox"}}}]
                    },
                    "fileset": "job-fileset-builder-build-1",
                    "status": "active",
                    "custom_fields": fields,
                },
            )

        # The submit sequence has its own tests; here it only asks for the job a resubmit found.
        async def submit(*_args, get_job, **_kwargs) -> SubmitResult:
            assert await get_job("builder-build-1") == ExistingJob(uid="job-1", custom_fields=fields)
            return SubmitResult(job="builder-build-1", images=[])

        monkeypatch.setattr(service, "load_backend", lambda *_args: None)
        monkeypatch.setattr(service, "submit_build_set", submit)
        client = AsyncNemoClient(
            base_url="http://platform.test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(jobs))
        )
        app = _app(AsyncMock())
        app.dependency_overrides[get_nemo_client] = lambda: client
        body = {"name": "build", "revision": 1, "build_specs": [{"source": {"fileset": "ws-a/ctx"}}]}

        response = TestClient(app).post("/v2/workspaces/ws-a/builds", json=body)

        assert response.status_code == 201, response.text
        assert ("GET", "/apis/jobs/v2/workspaces/ws-a/jobs/builder-build-1") in [
            (r.method, r.url.path) for r in requests
        ]


# ---------------------------------------------------------------------------
# The routes over the platform's own entity client, with a store in memory where its typed client would be
# ---------------------------------------------------------------------------

DEMO, DEMO_JOB = "demo-1", f"{JOB_NAME_PREFIX}demo-1"
DEMO_COMPLETE = f"/v2/workspaces/ws-a/container-images/{DEMO}/complete"
#: The request, as a caller posts it: the set whose one image is ``DEMO`` and whose job is ``DEMO_JOB``.
DEMO_REQUEST: dict[str, Any] = {"name": "demo", "revision": 1, "build_specs": [{"source": {"fileset": "fs-a"}}]}
#: What Alice's remote-build step reports of the image her build published.
ALICES: dict[str, Any] = {
    "digest": "sha256:" + "a" * 64,
    "external": {"build_id": "alices-build", "image_id": "alices-image"},
    "runtime_config": {"entrypoint": ["/alices/entrypoint"]},
}
REMOTE_PROFILES = [
    KubernetesJobExecutionProfile.model_validate({"profile": "build-remote", "config": {"namespace": "builds"}})
]

Elsewhere = Callable[[], Awaitable[None]]


def _remote_config() -> BuilderConfig:
    built_by = {"registry": "registry.example.com", "repository": "team/images", "image": "steps.example.com/remote:1"}
    return BuilderConfig.model_validate({"backend": "remote", "default_platform": "linux/amd64", "remote": built_by})


def _answer(status: int) -> httpx.Response:
    return httpx.Response(status, json={"detail": "refused"}, request=httpx.Request("GET", "http://platform.test"))


class _Answer:
    """What a call of the typed client returns: the entity is its ``data()``."""

    def __init__(self, entity: Entity) -> None:
        self._entity = entity

    def data(self) -> Entity:
        return self._entity.model_copy(deep=True)


class Store:
    """The entity store, in memory, where the platform's ``EntityClient`` has its typed client.

    A row is found by its name. It is updated by its ID, on condition that it is still at the version the
    update names, and is then at the next. A row created under a name whose row was deleted is another row: a
    new ID, at version 1, created by whoever the request acts for.
    """

    def __init__(self) -> None:
        self._rows: dict[str, Entity] = {}
        self._made = 0
        #: Whom the store takes a create to be for: the request's effective principal.
        self.acting_for = ""
        #: What the completion did, in order: ``read <id>``, and ``write <id>: <how it ended>``.
        self.trace: list[str] = []
        #: What happens elsewhere just before the completion's nth read, or its nth write, is answered.
        self.before_read: dict[int, Elsewhere] = {}
        self.before_write: dict[int, Elsewhere] = {}
        self._elsewhere = False
        #: What the routes and the submit are handed: the platform's client, over this store.
        self.client = EntityClient(cast(AsyncEntitiesClient, self))

    @asynccontextmanager
    async def elsewhere(self) -> AsyncGenerator[None]:
        """What is done between these is not the completion's: it is not traced, and nothing happens meanwhile."""
        was, self._elsewhere = self._elsewhere, True
        try:
            yield
        finally:
            self._elsewhere = was

    def _named(self, workspace: str, entity_type: str, name: str) -> Entity | None:
        found = [
            row
            for row in self._rows.values()
            if (row.workspace, row.entity_type, row.name) == (workspace, entity_type, name)
        ]
        assert len(found) <= 1, "a name holds one row"
        return found[0] if found else None

    async def _meanwhile(self, pending: dict[int, Elsewhere], kind: str) -> None:
        if self._elsewhere:
            return
        happens = pending.pop(sum(entry.startswith(kind) for entry in self.trace) + 1, None)
        if happens is not None:
            async with self.elsewhere():
                await happens()

    def _did(self, entry: str) -> None:
        if not self._elsewhere:
            self.trace.append(entry)

    async def create_entity(self, *, workspace: str, entity_type: str, body: EntityCreateInput) -> _Answer:
        assert body.name is not None
        if self._named(workspace, entity_type, body.name) is not None:
            raise ConflictError(_answer(409))
        self._made += 1
        now = datetime.now(UTC)
        row = Entity(
            entity_type=entity_type,
            id=f"container-image-{self._made}",
            workspace=workspace,
            name=body.name,
            data=body.data,
            created_at=now,
            created_by=self.acting_for,
            updated_at=now,
            updated_by=self.acting_for,
            db_version=1,
        )
        self._rows[row.id] = row
        return _Answer(row)

    async def get_entity_by_name(
        self, *, entity_type: str, name: str, workspace: str, query_params: object = None
    ) -> _Answer:
        await self._meanwhile(self.before_read, "read")
        row = self._named(workspace, entity_type, name)
        if row is None:
            raise NotFoundError(_answer(404))
        self._did(f"read {row.id}")
        return _Answer(row)

    async def update_entity_by_id(
        self, *, entity_type: str, entity_id: str, workspace: str, body: EntityUpdate, query_params: object = None
    ) -> _Answer:
        await self._meanwhile(self.before_write, "write")
        row = self._rows.get(entity_id)
        if row is None or (row.workspace, row.entity_type) != (workspace, entity_type):
            self._did(f"write {entity_id}: no such row")
            raise NotFoundError(_answer(404))
        if body.expected_db_version is not None and body.expected_db_version != row.db_version:
            self._did(f"write {entity_id}: changed since it was read")
            raise ConflictError(_answer(409))
        # Who created a row is the store's to say, and no update changes it.
        row = row.model_copy(update={"data": body.data, "db_version": row.db_version + 1})
        self._rows[entity_id] = row
        self._did(f"write {entity_id}: written")
        return _Answer(row)

    def delete(self, row: ContainerImage) -> None:
        """As whatever may delete entities deletes one, through the platform's generic route: by its ID."""
        del self._rows[row.id]

    def touch(self, row: ContainerImage, **provenance: str) -> None:
        """Any other write of the row: it is at the next version, and its provenance says ``provenance`` too."""
        stored = self._rows[row.id]
        data = {**stored.data, "provenance": {**stored.data["provenance"], **provenance}}
        self._rows[row.id] = stored.model_copy(update={"data": data, "db_version": stored.db_version + 1})

    async def now(self) -> ContainerImage:
        """The row now stored under the image's name, read as the platform's client reads it."""
        async with self.elsewhere():
            return await self.client.get(ContainerImage, name=DEMO, workspace="ws-a")


class Jobs:
    """Jobs, as far as a submit meets it: a name holds one job at a time, and every create makes a new job, with
    an ID and a first attempt of its own, which it answers with."""

    def __init__(self) -> None:
        self.jobs: dict[str, CreatedJob] = {}
        self._made = 0

    async def create_job(self, request: CreateHelixJobRequest) -> CreatedJob:
        assert request.name is not None
        if request.name in self.jobs:
            raise ConflictError(_answer(409))
        self._made += 1
        self.jobs[request.name] = CreatedJob(uid=f"platform-job-{self._made}", attempt_id=f"attempt-{self._made}")
        return self.jobs[request.name]

    async def get_job(self, name: str) -> ExistingJob:
        return ExistingJob(uid=self.jobs[name].uid, custom_fields={})


async def _submit_remote(store: Store, jobs: Jobs, *, by: str, fileset: str = "fs-a") -> ContainerImage:
    """The builder's own submit of ``demo`` revision 1 for ``by``, from ``fileset``, under the remote backend:
    the one row it writes, which records the job it created."""

    async def secret_readable(workspace: str, name: str) -> bool:
        return True

    async def resolve_upstream(reference: object, platform: str) -> str:
        raise AssertionError("nothing here copies an image")

    store.acting_for = by
    async with store.elsewhere():
        submitted = await submit_build_set(
            BuildSet(name="demo", revision=1, build_specs=[BuildSpec.model_validate({"source": {"fileset": fileset}})]),
            backend=RemoteBackend(_remote_config(), REMOTE_PROFILES),
            workspace="ws-a",
            entity_client=store.client,
            create_job=jobs.create_job,
            get_job=jobs.get_job,
            secret_readable=secret_readable,
            resolve_upstream=resolve_upstream,
            default_platform="linux/amd64",
        )
    (row,) = submitted.images
    assert (row.name, row.status, row.created_by, row.db_version) == (DEMO, "pending", by, 2)
    created = jobs.jobs[DEMO_JOB]
    assert (row.provenance.job, row.provenance.backend) == (f"ws-a/{DEMO_JOB}", "remote")
    assert (row.provenance.job_uid, row.provenance.job_attempt_id) == (created.uid, created.attempt_id)
    return row


def _bob_submits_anew(store: Store, jobs: Jobs, row: ContainerImage, *, fileset: str = "fs-a") -> Elsewhere:
    """``row`` and its build job are deleted, as a privileged cleanup could, and Bob submits ``demo`` revision
    1 himself: from the same fileset the very request that wrote ``row``, from another a request of his own."""

    async def happens() -> None:
        store.delete(row)
        del jobs.jobs[DEMO_JOB]
        await _submit_remote(store, jobs, by="bob", fileset=fileset)

    return happens


def _steps_record(row: ContainerImage) -> RecordedStep:
    """What the platform has on record for the pod of the remote-build step of the job ``row`` records."""
    assert row.provenance.job_attempt_id is not None
    return RecordedStep(workspace="ws-a", job=DEMO_JOB, attempt_id=row.provenance.job_attempt_id)


def _complete_demo(store: Store, caller: Caller, **how: Any):
    """Alice's step's completion, through the route, as ``caller``."""
    app = _app(cast(AsyncMock, store.client), caller, **how)
    return TestClient(app).post(DEMO_COMPLETE, json=ALICES)


def _says_what_alices_step_sent(row: ContainerImage) -> bool:
    return (
        (row.status, row.digest) == ("ready", ALICES["digest"])
        and row.external is not None
        and (row.external.build_id, row.external.image_id) == ("alices-build", "alices-image")
        and row.runtime_config is not None
        and row.runtime_config.entrypoint == ["/alices/entrypoint"]
    )


class TestTheRowThatIsCompleted:
    """The route reads the row by its name and decides who may complete it. The write is of that row: by its
    ID, at the version it was decided of. A row deleted and submitted anew meanwhile is another row under the
    same name, with another submitter, and is not the one made ``ready``."""

    async def test_the_row_the_route_decided_of_is_written_and_the_name_is_read_once(self) -> None:
        store, jobs = Store(), Jobs()
        alices = await _submit_remote(store, jobs, by="alice")
        job_of = AsyncMock(return_value=_steps_record(alices))
        response = _complete_demo(store, EXCHANGED_STEP, delegation=OWN, job_of=job_of, exchange=True)
        assert (response.status_code, response.json()["id"]) == (200, alices.id)
        assert _says_what_alices_step_sent(await store.now())
        assert store.trace == [f"read {alices.id}", f"write {alices.id}: written"]

    async def test_a_row_replaced_after_the_route_read_it_is_not_written(self) -> None:
        """The route decides that Alice's step may complete Alice's row. Before the write, the row and its job
        are deleted, and Bob submits a request of his own under the same set name and revision. The store has
        no row under the ID that was decided of, so the request is a 404, and Bob's row waits for his build."""
        store, jobs = Store(), Jobs()
        alices = await _submit_remote(store, jobs, by="alice")
        store.before_write[1] = _bob_submits_anew(store, jobs, alices, fileset="fs-b")

        response = _complete_demo(store, SERVICE_STEP, unverified=True)

        bobs = await store.now()
        assert response.status_code == 404
        assert bobs.id != alices.id and bobs.provenance.request_digest != alices.provenance.request_digest
        assert (bobs.created_by, bobs.status, bobs.digest, bobs.db_version) == ("bob", "pending", None, 2)
        assert store.trace == [f"read {alices.id}", f"write {alices.id}: no such row"]

    async def test_a_row_replaced_while_the_route_asks_which_job_is_calling_is_not_written(self) -> None:
        """Under workload token exchange the route waits on the platform for its record of the caller, after
        it has read the row. Bob's submit of the very request Alice made leaves a row that names the same job
        and the same request: only the store's ID for it, whom it says created it, and the job it records
        tell the two apart. Both checks were made of Alice's row, and Bob's is not written."""
        store, jobs = Store(), Jobs()
        alices = await _submit_remote(store, jobs, by="alice")
        replaced = _bob_submits_anew(store, jobs, alices)

        async def the_platforms_record(delegation: str) -> RecordedStep:
            async with store.elsewhere():
                await replaced()
            return _steps_record(alices)

        job_of = AsyncMock(side_effect=the_platforms_record)
        response = _complete_demo(store, EXCHANGED_STEP, delegation=OWN, job_of=job_of, exchange=True)

        bobs = await store.now()
        assert response.status_code == 404
        assert (bobs.provenance.job, bobs.provenance.request_digest) == (
            alices.provenance.job,
            alices.provenance.request_digest,
        )
        assert (bobs.id != alices.id, bobs.created_by, bobs.status) == (True, "bob", "pending")
        job_of.assert_awaited_once_with(OWN)
        assert store.trace == [f"read {alices.id}", f"write {alices.id}: no such row"]

    async def test_a_row_replaced_before_a_retry_reads_the_name_again_is_a_409(self) -> None:
        """When the store refuses the write because the row changed, the name is read again. By then the row
        is gone and Bob's is under its name: nothing was decided of that one, and the completion ends there."""
        store, jobs = Store(), Jobs()
        alices = await _submit_remote(store, jobs, by="alice")

        async def another_write_gets_in_first() -> None:
            store.touch(alices)

        store.before_write[1] = another_write_gets_in_first
        store.before_read[2] = _bob_submits_anew(store, jobs, alices)
        job_of = AsyncMock(return_value=_steps_record(alices))

        response = _complete_demo(store, EXCHANGED_STEP, delegation=OWN, job_of=job_of, exchange=True)

        bobs = await store.now()
        assert response.status_code == 409
        assert response.json()["detail"] == "demo-1 was replaced while it was being completed"
        assert (bobs.id != alices.id, bobs.created_by, bobs.status) == (True, "bob", "pending")
        assert store.trace == [
            f"read {alices.id}",
            f"write {alices.id}: changed since it was read",
            f"read {bobs.id}",
        ]
        job_of.assert_awaited_once_with(OWN)

    async def test_a_row_that_only_changed_is_asked_about_again_and_then_written(self) -> None:
        store, jobs = Store(), Jobs()
        alices = await _submit_remote(store, jobs, by="alice")

        async def another_write_gets_in_first() -> None:
            store.touch(alices)

        store.before_write[1] = another_write_gets_in_first
        job_of = AsyncMock(return_value=_steps_record(alices))

        response = _complete_demo(store, EXCHANGED_STEP, delegation=OWN, job_of=job_of, exchange=True)

        assert (response.status_code, response.json()["id"]) == (200, alices.id)
        assert _says_what_alices_step_sent(await store.now())
        assert store.trace == [
            f"read {alices.id}",
            f"write {alices.id}: changed since it was read",
            f"read {alices.id}",
            f"write {alices.id}: written",
        ]
        # Who may complete it was decided of the row as it was first read, and again of the row as it is.
        assert job_of.await_count == 2

    async def test_a_row_that_changed_so_that_its_caller_may_no_longer_complete_it_is_a_403(self) -> None:
        """The checks are the route's own, asked again: here the build's job was created anew meanwhile, and
        the row records the new one, which the caller is no step of."""
        store, jobs = Store(), Jobs()
        alices = await _submit_remote(store, jobs, by="alice")

        async def the_job_is_created_anew() -> None:
            store.touch(alices, job_uid="platform-job-9", job_attempt_id="attempt-9")

        store.before_write[1] = the_job_is_created_anew
        job_of = AsyncMock(return_value=_steps_record(alices))

        response = _complete_demo(store, EXCHANGED_STEP, delegation=OWN, job_of=job_of, exchange=True)

        assert response.status_code == 403
        assert "another run than the one its submit created" in response.json()["detail"]
        assert (await store.now()).status == "pending"
        assert store.trace == [
            f"read {alices.id}",
            f"write {alices.id}: changed since it was read",
            f"read {alices.id}",
        ]
        assert job_of.await_count == 2


def _a_job(
    job_id: str, attempt_id: str, *, source: str = "builder.build", custom_fields: dict[str, Any] | None = None
) -> HelixJobResponse:
    """A job as Jobs answers with it, to a create or a read."""
    spec = HelixJobSpec.model_validate(
        {"steps": [{"name": "a-step", "executor": {"provider": "cpu", "container": {"image": "busybox"}}}]}
    )
    return HelixJobResponse(
        id=job_id,
        attempt_id=attempt_id,
        name=DEMO_JOB,
        workspace="ws-a",
        platform_spec=spec,
        fileset=f"job-fileset-{DEMO_JOB}",
        status=HelixJobStatus.CREATED,
        source=source,
        custom_fields=custom_fields,
    )


class TestARemoteBuildsJob:
    """Through the submit route, under the remote backend, with Jobs answering on the wire as it does."""

    @staticmethod
    def _post(
        monkeypatch: pytest.MonkeyPatch, store: Store, *, create: httpx.Response, found: HelixJobResponse | None = None
    ):
        """The submit's answer, and what Jobs was asked: each request's method and path, and how it was answered."""
        jobs = "/apis/jobs/v2/workspaces/ws-a/jobs"
        asked: list[tuple[str, str, int]] = []

        def platform(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/apis/jobs/v2/execution-profiles":
                answer = httpx.Response(200, json=[profile.model_dump(mode="json") for profile in REMOTE_PROFILES])
            elif (request.method, request.url.path) == ("POST", jobs):
                answer = create
            elif (request.method, request.url.path) == ("GET", f"{jobs}/{DEMO_JOB}") and found is not None:
                answer = httpx.Response(200, json=found.model_dump(mode="json"))
            else:
                answer = httpx.Response(404, json={"detail": "not found"})
            asked.append((request.method, request.url.path, answer.status_code))
            return answer

        monkeypatch.setattr(service, "_load_config", _remote_config)
        client = AsyncNemoClient(
            base_url="http://platform.test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(platform))
        )
        store.acting_for = "alice"
        app = _app(cast(AsyncMock, store.client), Caller(auth=True, actor="alice"))
        app.dependency_overrides[get_nemo_client] = lambda: client
        return TestClient(app).post("/v2/workspaces/ws-a/builds", json=DEMO_REQUEST), asked

    async def test_each_row_records_what_jobs_answered_the_create_with(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The job's own ID and its first attempt's, which Jobs gives no other job and no rerun."""
        store = Store()
        created = _a_job("platform-job-CREATED", "attempt-CREATED")
        response, asked = self._post(
            monkeypatch, store, create=httpx.Response(201, json=created.model_dump(mode="json"))
        )
        assert response.status_code == 201, response.text
        (image,) = response.json()["images"]
        assert (image["status"], image["provenance"]["job"]) == ("pending", f"ws-a/{DEMO_JOB}")
        assert (image["provenance"]["job_uid"], image["provenance"]["job_attempt_id"]) == (
            "platform-job-CREATED",
            "attempt-CREATED",
        )
        stored = await store.now()
        assert (stored.provenance.job_uid, stored.provenance.job_attempt_id, stored.db_version) == (
            "platform-job-CREATED",
            "attempt-CREATED",
            2,
        )
        assert [(method, status) for method, _, status in asked] == [("GET", 200), ("POST", 201)]

    async def test_a_job_made_beforehand_under_the_builds_name_is_not_taken_for_the_builds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Jobs answers the create 409, and the read with a job that carries the request's digest in metadata
        its creator wrote. Its ``source`` and its spec are someone's own. The submit is answered 409, with
        what to do next, and the row is left waiting and records nothing of the job: nothing tells this job
        from one that another submit of the request created a moment ago and has yet to record."""
        digest = request_digest(BuildSet.model_validate(DEMO_REQUEST).with_default_platform("linux/amd64"))
        beforehand = _a_job(
            "platform-job-MADE-BEFOREHAND",
            "attempt-MADE-BEFOREHAND",
            source="my-own-tool",
            custom_fields={"request_digest": digest},
        )
        store = Store()
        response, asked = self._post(
            monkeypatch,
            store,
            create=httpx.Response(409, json={"detail": f"Job with name '{DEMO_JOB}' already exists."}),
            found=beforehand,
        )
        assert response.status_code == 409, response.text
        assert response.json() == {
            "detail": (
                f"a job named {DEMO_JOB!r} already exists, and this request's images do not record it as their "
                "build job. Another submit of the same request may be recording it now: submit the request again "
                "in a moment. If the answer is still this, submit the next revision"
            )
        }
        stored = await store.now()
        assert (stored.status, stored.status_detail, stored.db_version) == ("pending", None, 1)
        # Nothing of the job that was found is recorded: not its ID, nor the attempt it is at.
        assert (stored.provenance.job_uid, stored.provenance.job_attempt_id) == (None, None)
        assert "MADE-BEFOREHAND" not in response.text
        assert [(method, status) for method, _, status in asked] == [("GET", 200), ("POST", 409), ("GET", 200)]

    async def test_a_row_that_records_an_earlier_job_is_left_waiting_when_another_is_found_under_the_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row records the job its submit created, which is deleted since. Jobs answers the create 409,
        and the read with another job under the build's name. The submit is answered 409, with what to do
        next, and the row is left as it was, still recording its own job: nothing tells the job found from
        one that another submit of the request created a moment ago and has yet to record in its place."""
        store = Store()
        submitted = await _submit_remote(store, Jobs(), by="alice")
        own = (submitted.provenance.job_uid, submitted.provenance.job_attempt_id)
        response, asked = self._post(
            monkeypatch,
            store,
            create=httpx.Response(409, json={"detail": f"Job with name '{DEMO_JOB}' already exists."}),
            found=_a_job("platform-job-MADE-SINCE", "attempt-MADE-SINCE"),
        )
        assert response.status_code == 409, response.text
        assert response.json() == {
            "detail": (
                f"a job named {DEMO_JOB!r} already exists, and this request's images do not record it as their "
                "build job. Another submit of the same request may be recording it now: submit the request again "
                "in a moment. If the answer is still this, submit the next revision"
            )
        }
        stored = await store.now()
        # Not failed, and not written: at the version its own submit left it at.
        assert (stored.status, stored.status_detail, stored.db_version) == ("pending", None, submitted.db_version)
        assert (stored.provenance.job_uid, stored.provenance.job_attempt_id) == own
        assert "MADE-SINCE" not in response.text
        assert [(method, status) for method, _, status in asked] == [("GET", 200), ("POST", 409), ("GET", 200)]

    async def test_a_platform_that_cannot_show_who_completes_an_image_refuses_the_submit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With platform auth on and no workload token exchange a remote build's completion would be refused,
        so no build is made: nothing is written, and Jobs is asked for no job."""
        monkeypatch.setattr(remote, "platform_auth_enabled", lambda: True)
        monkeypatch.setattr(remote, "is_workload_identity_token_exchange_enabled", lambda: False)
        store = Store()
        response, asked = self._post(monkeypatch, store, create=httpx.Response(500))
        assert (response.status_code, response.json()["code"]) == (409, "backend_rejected")
        assert "builder.remote.allow_unverified_completion" in response.json()["detail"]
        assert store.trace == [] and [method for method, _, _ in asked] == ["GET"]
        with pytest.raises(NemoEntityNotFoundError):
            await store.now()


# ---------------------------------------------------------------------------
# The completion route behind the platform's own auth, where it is installed
# ---------------------------------------------------------------------------

#: Where the platform serves the route: the service's name, its router's prefix, then the route's own path.
SERVED = "/apis/builder/v2/workspaces/ws-a/container-images/build-1-0/complete"
AUDIENCE = "platform"
#: What the row's own build published, as its remote-build step reports it.
BUILT: dict[str, Any] = {"digest": DIGEST, "external": {"build_id": "build-1", "image_id": "image-1"}}
#: What a caller that ran no build sends instead: a digest, a build and a way to start the image, of its choosing.
CHOSEN: dict[str, Any] = {
    "digest": "sha256:" + "f" * 64,
    "external": {"image_id": "image-of-its-choosing", "build_id": "no-such-build"},
    "runtime_config": {"entrypoint": ["/bin/sh", "-c"], "cmd": ["its-own-program"]},
}


def _platform() -> SimpleNamespace:
    """The platform's own auth, where it is installed. A plugin's code may not import it; these tests put it in
    front of the builder's routes, as a deployment does."""
    return SimpleNamespace(
        jwt=pytest.importorskip("jwt"),
        config=pytest.importorskip("nhx.common.config.base"),
        middleware=pytest.importorskip("nhx.common.auth.middleware"),
        tokens=pytest.importorskip("nhx.common.auth.workload_tokens"),
        exchange=pytest.importorskip("nhx.core.auth.api.v2.workload_token_exchange"),
        bundle=pytest.importorskip("nhx.core.auth.app.bundle"),
        engine=pytest.importorskip("nhx.core.auth.app.embedded_pdp.engine"),
        wasm=pytest.importorskip("nhx.core.auth.app.embedded_pdp.policy_wasm"),
    )


async def _policy(nhx: SimpleNamespace) -> Any:
    """The platform's authorization policy, as compiled, over what a deployment loads into it: the roles, and
    every installed service's routes with the rules their handlers carry, the builder's among them. To that,
    the workspace's members: the row's submitter and another who may build, and one who may only read."""
    try:
        compiled = nhx.wasm.ensure_embedded_policy_wasm(auto_build=False)
    except nhx.wasm.PolicyWasmError:
        pytest.skip("the platform's authorization policy is not compiled here (`make build-policy`)")
    data = await nhx.bundle.build_authorization_data(None)
    if "/apis/builder/v2/workspaces/{workspace}/container-images/{name}/complete" not in data["authz"]["endpoints"]:
        pytest.skip("the builder is not among the services installed here, so the policy has no rule for its routes")
    members = {"alice": "Editor", "carol": "Editor", "bob": "Viewer"}
    data["authz"]["principals"] = {member: {"workspaces": {"ws-a": [role]}} for member, role in members.items()}
    policy = nhx.engine.OPAPolicy(str(compiled))
    policy.set_data(data)
    return policy


def _auth_config(nhx: SimpleNamespace, tmp_path: Path, *, token_exchange: bool) -> Any:
    """Platform auth on. With ``token_exchange``, as a deployment that mints its workloads' tokens has it;
    without, as one that takes a request's identity from its headers."""
    if not token_exchange:
        return nhx.config.AuthConfig(enabled=True, policy_decision_point_base_url="https://platform.test")
    key_file = tmp_path / "token-signing.pem"
    key_file.write_bytes(
        rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return nhx.config.AuthConfig(
        enabled=True,
        policy_decision_point_base_url="https://platform.test",
        oidc=nhx.config.OIDCConfig(
            enabled=True,
            audience=AUDIENCE,
            issuer="https://idp.example.com",
            public_client=nhx.config.OIDCPublicClientConfig(client_id="cli"),
            workload=nhx.config.OIDCWorkloadConfig(client_id="workload"),
        ),
        token_signing=nhx.config.TokenSigningConfig(
            issuer="https://platform.test/apis/auth", key_id="builder-tests", private_key_file=str(key_file)
        ),
    )


@dataclass
class Deployment:
    """The builder's completion route behind the platform's own auth middleware, with platform auth on, over
    one pending row a build service builds."""

    nhx: SimpleNamespace
    config: Any
    client: httpx.AsyncClient
    entities: AsyncMock
    #: The pods of three steps, as Jobs records them under token exchange: the row's own remote-build step, a
    #: step of another job the submitter runs, and a step of a later run under the build job's name.
    own: WorkloadDelegationEntity
    another_jobs: WorkloadDelegationEntity
    a_later_runs: WorkloadDelegationEntity
    job_of: AsyncMock
    #: What the route was handed for each request that reached it: who is calling, and the workload delegation
    #: its token names. Both as the builder's own functions read them from what the middleware left.
    handed: list[tuple[Caller, str | None]]

    async def complete(self, body: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        return await self.client.post(SERVED, json=body, headers=headers)

    async def service_token(self, name: str, on_behalf_of: str | None = None) -> str:
        """The token a service calls the platform with under token exchange, acting for ``on_behalf_of``: what
        a client made with ``get_async_nemo_client(as_service=name, on_behalf_of=...)`` sends."""
        provider = self.nhx.tokens.ServiceWorkloadAccessTokenProvider(self.config, name, on_behalf_of)
        return await provider.get_access_token_async()

    async def step_token(self, record: WorkloadDelegationEntity) -> str:
        """The token the platform's auth service mints for a job step that has proved which pod it is: the
        claims its exchange builds from the pod's record, signed with the key and the header it signs with."""
        exchange, tokens = self.nhx.exchange, self.nhx.tokens
        proved = exchange.VerifiedSubjectToken(subject=record.workload_subject, groups=[])
        key = await tokens.workload_signing_key_async(self.config)
        now = int(time.time())
        claims = {
            "iss": tokens.workload_token_issuer(self.config),
            **exchange._build_delegated_claims(record, proved),
            "aud": AUDIENCE,
            "iat": now,
            "nbf": now,
            "exp": now + 300,
        }
        return self.nhx.jwt.encode(claims, key.private_key, algorithm="RS256", headers={"kid": key.kid})


@asynccontextmanager
async def _deployment(tmp_path: Path, *, token_exchange: bool, unverified: bool = False) -> AsyncGenerator[Deployment]:
    nhx = _platform()
    policy = await _policy(nhx)
    config = _auth_config(nhx, tmp_path, token_exchange=token_exchange)
    own = _delegation(record_id="delegation-of-the-builds-step")
    another_jobs = _delegation("eval-7", attempt="attempt-7", record_id="delegation-of-another-job")
    a_later_runs = _delegation(attempt="attempt-2", record_id="delegation-of-a-later-run")
    known = {record.id: record for record in (own, another_jobs, a_later_runs)}
    job_of = AsyncMock(side_effect=lambda delegation: service._step_on_record(known.get(delegation)))
    entities = _entities(_row(backend="remote"))
    handed: list[tuple[Caller, str | None]] = []

    def decide(request: httpx.Request) -> httpx.Response:
        """The policy decision point: the compiled policy, answering what the middleware asks it."""
        answer = policy.evaluate(json.loads(request.content)["input"], nhx.engine.ENTRYPOINT_MAP["allow"])
        return httpx.Response(200, json={"result": answer[0]["result"] if answer else {"allowed": False}})

    def recording(request: Request) -> str | None:
        """``current_delegation`` itself, with what it and ``current_caller`` made of the request kept."""
        handed.append((current_caller(), current_delegation(request)))
        return handed[-1][1]

    app = FastAPI()
    builder = BuilderService()
    for spec in builder.get_routers():
        app.include_router(spec.router, prefix=f"/apis/{builder.name}{spec.prefix}")
    app.dependency_overrides[get_entity_client] = lambda: entities
    app.dependency_overrides[get_job_of] = lambda: job_of
    app.dependency_overrides[current_delegation] = recording
    # Whether the platform mints its workloads' tokens is left for the route to ask the platform itself.
    app.dependency_overrides[get_allow_unverified] = lambda: _asked(unverified)
    Configuration.set_override(config)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(decide)) as decision_point:
            app.add_middleware(
                nhx.middleware.AuthorizationMiddleware, service_name="builder", http_client=decision_point
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="https://platform.test") as client:
                yield Deployment(nhx, config, client, entities, own, another_jobs, a_later_runs, job_of, handed)
    finally:
        Configuration.clear_override(type(config))


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _naming(actor: str, on_behalf_of: str) -> dict[str, str]:
    """The identity headers the platform takes on trust without token exchange. A step sends these, built from
    its job's creator; nothing stops any other request inside the cluster from carrying the same."""
    return {"X-NHX-Principal-Id": actor, "X-NHX-Principal-On-Behalf-Of": on_behalf_of}


class TestBehindThePlatformsOwnAuth:
    """Its middleware, tokens it signs and resolves itself, and its compiled policy deciding who is let
    through, in front of the route: who then completes an image a build service built."""

    @pytest.mark.parametrize("unverified", [False, True], ids=["by-default", "though-the-operator-accepts-unverified"])
    async def test_under_token_exchange_only_a_step_of_the_builds_own_run_completes_it(
        self, tmp_path: Path, unverified: bool
    ) -> None:
        """A token the platform signed for a service acting for the submitter passes the policy, and acts for
        the submitter as a step's does: one platform service calls another with such a token, and Jobs writes
        one, as ``service:jobs``, for each step it runs as a subprocess. It names no record of a pod, and is
        refused. So is a step of another job, and one of another run under the build job's name. The row's
        own step, with what its build published, then makes the one write."""
        async with _deployment(tmp_path, token_exchange=True, unverified=unverified) as deployed:
            for name in ("evals", "jobs"):
                impostor = await deployed.complete(CHOSEN, _bearer(await deployed.service_token(name, "alice")))
                assert impostor.status_code == 403
                assert "the caller's token names no workload" in impostor.json()["detail"]
                assert deployed.handed[-1] == (Caller(auth=True, actor=f"service:{name}", submitter="alice"), None)
            deployed.job_of.assert_not_awaited()

            for record, reason in (
                (deployed.another_jobs, "a step of another job"),
                (deployed.a_later_runs, "another run than the one its submit created"),
            ):
                refused = await deployed.complete(CHOSEN, _bearer(await deployed.step_token(record)))
                assert refused.status_code == 403
                assert reason in refused.json()["detail"]
                assert deployed.handed[-1][1] == record.id
            deployed.entities.update.assert_not_awaited()

            own = deployed.own
            completed = await deployed.complete(BUILT, _bearer(await deployed.step_token(own)))
            assert (completed.status_code, completed.json()["status"], completed.json()["digest"]) == (
                200,
                "ready",
                DIGEST,
            )
            assert deployed.handed[-1] == (Caller(auth=True, actor=own.workload_subject, submitter="alice"), own.id)
            deployed.entities.update.assert_awaited_once()

    async def test_with_identity_from_headers_no_request_completes_it(self, tmp_path: Path) -> None:
        """Without token exchange the platform takes who is calling from a request's headers. The step's own
        request and any other that names the submitter are alike, and both are refused: nothing is written."""
        async with _deployment(tmp_path, token_exchange=False) as deployed:
            for actor in ("service:builder", "service:evals", "mallory"):
                refused = await deployed.complete(CHOSEN, _naming(actor, "alice"))
                assert refused.status_code == 403
                assert "platform auth is on without workload token exchange" in refused.json()["detail"]
                assert deployed.handed[-1] == (Caller(auth=True, actor=actor, submitter="alice"), None)
            deployed.entities.update.assert_not_awaited()
            deployed.job_of.assert_not_awaited()

    async def test_with_identity_from_headers_the_operator_can_accept_what_nothing_verifies(
        self, tmp_path: Path
    ) -> None:
        """Whom a request says it acts for is then all that is asked, as before the builder asked more."""
        async with _deployment(tmp_path, token_exchange=False, unverified=True) as deployed:
            for_someone_else = await deployed.complete(BUILT, _naming("service:builder", "carol"))
            assert for_someone_else.status_code == 403
            assert "submitted by someone else" in for_someone_else.json()["detail"]
            completed = await deployed.complete(BUILT, _naming("service:builder", "alice"))
            assert (completed.status_code, completed.json()["status"]) == (200, "ready")
            deployed.entities.update.assert_awaited_once()
            deployed.job_of.assert_not_awaited()
