# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's routes, over HTTP, with the entity store faked."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_builder_plugin import completion, service
from nemo_builder_plugin.completion import (
    Caller,
    current_caller,
    current_delegation,
    step_refusal,
    submit_refusal,
    verified_delegation,
)
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, ExternalImage, ImageRuntimeConfig, Provenance
from nemo_builder_plugin.service import BuilderService, get_job_of, get_settle
from nemo_builder_plugin.settle import settle
from nemo_builder_plugin.submit import BackendChanged, BuildConflict, MissingSecrets, SubmitResult
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import WorkloadDelegationEntity, WorkloadDelegationScope
from nemo_helix_plugin.auth.workload_identity import build_kubernetes_pod_uid_workload_delegation
from nemo_helix_plugin.authz import CallerKind, get_path_rules
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError, get_entity_client
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from pydantic import BaseModel, ValidationError, field_validator


def _app(
    entities: AsyncMock,
    caller: Caller = Caller(auth=False),
    job_status: AsyncMock | None = None,
    builder: AsyncMock | None = None,
    delegation: str | None = None,
    job_of: AsyncMock | None = None,
) -> FastAPI:
    """``entities`` is the caller's entity store; ``builder`` the builder's, which settling writes through.
    ``delegation`` is the workload delegation the caller's token names, and ``job_of`` the platform's record of
    which job each one was made for a step of."""
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
OWN, OTHER_JOBS = "delegation-own", "delegation-other"


def _job_of(jobs: Mapping[str, str | None] | None = None) -> AsyncMock:
    """The job each delegation was made for a step of, as the platform has it on record."""
    known = {OWN: "ws-a/build-1", OTHER_JOBS: "ws-a/eval-7"} if jobs is None else jobs
    return AsyncMock(side_effect=lambda delegation: known.get(delegation))


def _row(
    status: Literal["pending", "ready", "failed"] = "pending", digest: str | None = None, backend: str = "execution"
) -> ContainerImage:
    row = ContainerImage(
        name="build-1-0",
        workspace="ws-a",
        registry="reg.example.com",
        repository="ws-a/app",
        provenance=Provenance(
            build_set="build",
            revision=1,
            job="ws-a/build-1",
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
    ):
        entities = _entities(row)
        body = {"digest": DIGEST} if body is None else body
        app = _app(entities, caller, delegation=delegation, job_of=job_of)
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

    @pytest.mark.parametrize("caller", [Caller(auth=False), STEP, SERVICE_STEP], ids=["auth-off", "step", "service"])
    def test_a_remote_build_steps_handle_is_recorded_with_it(self, caller: Caller) -> None:
        external = {"image_id": "img-1", "build_id": "build-1", "attributes": {"builder": "v1"}}
        row = _row(backend="remote")
        response, entities = self._post(row, caller=caller, body={"digest": DIGEST, "external": external})
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

    def test_a_request_naming_a_service_for_the_submitter_makes_the_first_completion(self) -> None:
        """Not wanted, and stated in the README's Security section: without workload token exchange nothing the
        platform hands the route says which job is calling, so a remote row's own step and anything else acting
        for its submitter look alike."""
        entities = _entities(_row(backend="remote"))
        chosen = "sha256:" + "f" * 64
        starts = {"entrypoint": ["/chosen"], "cmd": None, "working_dir": None, "user": None}
        first = TestClient(_app(entities, NAMED_SERVICE)).post(
            f"{IMAGE}/complete",
            json={"digest": chosen, "external": {"build_id": "made-up"}, "runtime_config": starts},
        )
        assert (first.status_code, first.json()["status"]) == (200, "ready")
        # What a consumer then pins and runs: the digest, the build and the entrypoint the other workload chose.
        assert first.json()["image_ref"] == f"reg.example.com/ws-a/app@{chosen}"
        assert first.json()["external"]["build_id"] == "made-up"
        assert first.json()["runtime_config"] == starts
        # And the row's own step, with the digest its build published, is the one refused.
        own = TestClient(_app(entities, SERVICE_STEP)).post(
            f"{IMAGE}/complete", json={"digest": DIGEST, "external": {"build_id": "build-1"}}
        )
        assert own.status_code == 409
        entities.update.assert_awaited_once()

    def test_another_jobs_step_is_refused_and_the_images_own_makes_the_first_completion(self) -> None:
        """Under workload token exchange each step's token names the platform's record of its pod, which says
        which job the pod is a step of: another job of the submitter's is told from the image's own."""
        entities = _entities(_row(backend="remote"))
        job_of = _job_of()
        other = TestClient(_app(entities, EXCHANGED_STEP, delegation=OTHER_JOBS, job_of=job_of)).post(
            f"{IMAGE}/complete", json={"digest": "sha256:" + "f" * 64, "external": {"build_id": "made-up"}}
        )
        assert other.status_code == 403
        assert "a step of another job" in other.json()["detail"]
        entities.update.assert_not_awaited()
        # The row is still pending, and its own step completes it with what its build published.
        own = TestClient(_app(entities, EXCHANGED_STEP, delegation=OWN, job_of=job_of)).post(
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
        response, _ = self._post(row, caller=EXCHANGED_STEP, body=body, delegation=OTHER_JOBS, job_of=_job_of())
        assert response.status_code == 403
        response, _ = self._post(row, caller=EXCHANGED_STEP, body=body, delegation=OWN, job_of=_job_of())
        assert response.status_code == 200

    @pytest.mark.parametrize("jobs", [{}, {OWN: None}], ids=["no-longer-on-record", "not-a-job-steps"])
    def test_a_step_the_platform_has_no_job_on_record_for_is_refused(self, jobs: dict[str, str | None]) -> None:
        """A token outlives the record it was minted from by a little: its job has ended, and the record with it."""
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        response, entities = self._post(
            _row(backend="remote"), caller=EXCHANGED_STEP, body=body, delegation=OWN, job_of=_job_of(jobs)
        )
        assert response.status_code == 403
        assert "not on record as a job step" in response.json()["detail"]
        entities.update.assert_not_awaited()

    def test_a_record_that_cant_be_read_admits_no_one(self, caplog: pytest.LogCaptureFixture) -> None:
        """Answered as an outage, which the step waits out: the image is built, and only this call is left."""
        job_of = AsyncMock(side_effect=RuntimeError("entity store down"))
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        response, entities = self._post(
            _row(backend="remote"), caller=EXCHANGED_STEP, body=body, delegation=OWN, job_of=job_of
        )
        assert response.status_code == 503
        assert "entity store down" not in response.text
        assert "could not tell which job is completing ws-a/build-1-0" in caplog.text
        entities.update.assert_not_awaited()

    @pytest.mark.parametrize(
        "caller",
        [Caller(auth=False), SERVICE_STEP, NAMED_SERVICE],
        ids=["auth-off", "the-step-without-exchange", "a-request-naming-a-service"],
    )
    def test_a_caller_whose_token_names_no_workload_is_asked_nothing_more(self, caller: Caller) -> None:
        job_of = _job_of()
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        response, _ = self._post(_row(backend="remote"), caller=caller, body=body, job_of=job_of)
        assert response.status_code == 200
        job_of.assert_not_awaited()

    def test_no_record_is_read_for_a_caller_refused_already(self) -> None:
        job_of = _job_of()
        someone_elses = Caller(auth=True, actor="system:serviceaccount:builds:build-remote", submitter="bob")
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        response, _ = self._post(_row(backend="remote"), caller=someone_elses, body=body, delegation=OWN, job_of=job_of)
        assert response.status_code == 403
        job_of.assert_not_awaited()

    def test_a_step_of_a_job_under_the_build_jobs_name_passes_as_its_step(self) -> None:
        """Not wanted, and stated in the README's Security section: the platform's record names a job as the
        row does, by workspace and name, and a job created once the build's is deleted can take its name."""
        job_of = _job_of({OTHER_JOBS: "ws-a/build-1"})
        body = {"digest": "sha256:" + "f" * 64, "external": {"build_id": "made-up"}}
        response, entities = self._post(
            _row(backend="remote"), caller=EXCHANGED_STEP, body=body, delegation=OTHER_JOBS, job_of=job_of
        )
        assert (response.status_code, response.json()["external"]["build_id"]) == (200, "made-up")
        entities.update.assert_awaited_once()

    def test_an_image_this_cluster_builds_is_completed_by_whoever_acts_for_its_submitter(self) -> None:
        """Not wanted, and stated in the README's Security section: only a row a build service built is held
        to its own job. The push step's completion is as it was before the builder asked."""
        job_of = _job_of()
        response, entities = self._post(_row(), caller=EXCHANGED_STEP, delegation=OTHER_JOBS, job_of=job_of)
        assert (response.status_code, response.json()["status"]) == (200, "ready")
        job_of.assert_not_awaited()
        entities.update.assert_awaited_once()

    def test_what_a_build_services_client_reported_of_how_the_image_starts_is_recorded(self) -> None:
        """A remote-build step sends it when its service's client reported one, as the push step sends what it
        read: a consumer that must start the image's own entrypoint has nowhere else to learn it."""
        runtime_config = {"entrypoint": ["/entrypoint.sh"], "cmd": ["serve"], "working_dir": "/srv", "user": ""}
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}, "runtime_config": runtime_config}
        response, entities = self._post(_row(backend="remote"), caller=STEP, body=body)
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
    job: str = "build-1", *, kind: str = "job", now: datetime | None = None, steps: bool = True
) -> WorkloadDelegationEntity:
    """The record Jobs' Kubernetes backend makes for a step's pod, built with what it builds one from."""
    record = build_kubernetes_pod_uid_workload_delegation(
        scope=WorkloadDelegationScope(workload_workspace="ws-a", workload_kind=kind, workload_instance_id=job),
        workload_audience="platform",
        workload_generation="attempt-1/step-1/pod-1",
        namespace="builds",
        service_account_name="build-remote",
        pod_uid="pod-1",
        job_id=job if steps else None,
        attempt_id="attempt-1" if steps else None,
        step_id="step-1" if steps else None,
        auth_context=AuthContext(principal_id="alice"),
        ttl_seconds_active=3600,
        now=now,
    )
    record._id = "workload-delegation-1"
    return record


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
        for job, answer in (("ws-a/eval-7", 403), ("ws-a/build-1", 200)):
            job_of = _job_of({"workload-delegation-1": job})
            app = _app(_entities(_row(backend="remote")), job_of=job_of)
            del app.dependency_overrides[current_caller], app.dependency_overrides[current_delegation]
            assert TestClient(app).post(f"{IMAGE}/complete", json=body).status_code == answer
            job_of.assert_awaited_once_with("workload-delegation-1")

    def test_with_auth_off_no_token_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anyone may complete an image then, so which workload one of them is changes nothing."""
        monkeypatch.setattr(completion, "platform_auth_enabled", lambda: False)
        monkeypatch.setattr(
            completion, "current_auth_client", lambda _request: _authenticated("workload_access_token", MINTED)
        )
        job_of = _job_of({"workload-delegation-1": "ws-a/eval-7"})
        app = _app(_entities(_row(backend="remote")), job_of=job_of)
        del app.dependency_overrides[current_caller], app.dependency_overrides[current_delegation]
        body = {"digest": DIGEST, "external": {"build_id": "build-1"}}
        assert TestClient(app).post(f"{IMAGE}/complete", json=body).status_code == 200
        job_of.assert_not_awaited()

    async def test_only_a_step_of_the_images_own_job_passes(self) -> None:
        row = _row(backend="remote")
        assert await step_refusal(row, OWN, _job_of()) is None
        assert "a step of another job" in (await step_refusal(row, OTHER_JOBS, _job_of()) or "")
        assert "not on record as a job step" in (await step_refusal(row, OWN, _job_of({})) or "")

    async def test_nothing_is_asked_where_no_token_names_a_workload_or_the_image_is_built_here(self) -> None:
        job_of = _job_of()
        assert await step_refusal(_row(backend="remote"), None, job_of) is None
        assert await step_refusal(_row(), OTHER_JOBS, job_of) is None
        job_of.assert_not_awaited()

    def test_a_delegations_job_is_the_one_its_images_row_names(self) -> None:
        """Jobs records the workspace and the job's name for a step's pod, and the row its build job by both."""
        assert service._job_named(_delegation()) == _row().provenance.job == "ws-a/build-1"
        assert service._job_named(_delegation("eval-7")) == "ws-a/eval-7"

    @pytest.mark.parametrize(
        "record",
        [
            None,
            _delegation(now=datetime(2020, 1, 1, tzinfo=UTC)),
            _delegation(kind="deployment", steps=False),
            _delegation(kind="deployment"),
        ],
        ids=["no-longer-on-record", "expired", "a-deployments", "a-deployment-that-names-a-job"],
    )
    def test_a_record_that_is_not_a_standing_job_steps_names_no_job(
        self, record: WorkloadDelegationEntity | None
    ) -> None:
        assert service._job_named(record) is None

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
        assert await service._job_of_as_builder(record.id) == "ws-a/build-1"
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
    """When the build set's job already exists, the route reads that job's ``custom_fields`` from Jobs."""

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
        async def submit(*_args, get_job_fields, **_kwargs) -> SubmitResult:
            assert await get_job_fields("builder-build-1") == fields
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
