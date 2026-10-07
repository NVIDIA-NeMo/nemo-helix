# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's routes, over HTTP, with the entity store faked."""

from __future__ import annotations

from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_builder_plugin.completion import Caller, current_caller
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.service import BuilderService
from nemo_helix_plugin.authz import CallerKind, get_path_rules
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError, get_entity_client


def _app(entities: AsyncMock, caller: Caller = Caller(auth=False)) -> FastAPI:
    app = FastAPI()
    for spec in BuilderService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_entity_client] = lambda: entities
    app.dependency_overrides[current_caller] = lambda: caller
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


DIGEST = "sha256:" + "d" * 64
IMAGE = "/v2/workspaces/ws-a/container-images/build-1-0"
STEP = Caller(auth=True, actor="job-step", submitter="alice")
#: The push step without workload token exchange: the builder's own service principal, acting for the submitter.
SERVICE_STEP = Caller(auth=True, actor="service:builder", submitter="alice")


def _row(status: Literal["pending", "ready", "failed"] = "pending", digest: str | None = None) -> ContainerImage:
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
    def _post(self, row: ContainerImage, *, caller: Caller = Caller(auth=False), body: dict[str, str] | None = None):
        entities = _entities(row)
        body = {"digest": DIGEST} if body is None else body
        response = TestClient(_app(entities, caller)).post(f"{IMAGE}/complete", json=body)
        return response, entities

    def test_the_pushed_digest_makes_the_image_ready(self) -> None:
        response, entities = self._post(_row())
        assert response.status_code == 200
        assert (response.json()["status"], response.json()["digest"]) == ("ready", DIGEST)
        entities.update.assert_awaited_once()

    def test_an_image_already_ready_at_that_digest_is_returned_as_it_is(self) -> None:
        response, entities = self._post(_row("ready", DIGEST))
        assert response.status_code == 200
        entities.update.assert_not_awaited()

    @pytest.mark.parametrize("row", [_row("ready", "sha256:" + "e" * 64), _row("failed")])
    def test_an_image_that_settled_otherwise_is_a_409(self, row: ContainerImage) -> None:
        response, entities = self._post(row)
        assert response.status_code == 409
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

    def test_an_image_that_does_not_exist_is_a_404(self) -> None:
        entities = AsyncMock()
        entities.get.side_effect = NemoEntityNotFoundError("no")
        response = TestClient(_app(entities)).post(f"{IMAGE}/complete", json={"digest": DIGEST})
        assert response.status_code == 404

    @pytest.mark.parametrize("body", [{"digest": "sha256:abc"}, {"digest": DIGEST, "signature": "x"}, {}])
    def test_anything_but_a_digest_is_refused(self, body: dict[str, str]) -> None:
        response, entities = self._post(_row(), body=body)
        assert response.status_code == 422
        entities.update.assert_not_awaited()


class TestReadingOneImage:
    def _get(self, caller: Caller):
        return TestClient(_app(_entities(_row()), caller)).get(IMAGE)

    @pytest.mark.parametrize(
        "caller",
        [Caller(auth=False), Caller(auth=True, actor="carol"), STEP, SERVICE_STEP],
        ids=["auth-off", "a-person", "a-step-with-token-exchange", "the-push-step-without"],
    )
    def test_whom_the_policy_admits_reads_it(self, caller: Caller) -> None:
        response = self._get(caller)
        assert response.status_code == 200
        assert response.json()["name"] == "build-1-0"

    @pytest.mark.parametrize(
        "caller",
        [Caller(auth=True, actor="service:builder"), Caller(auth=True, actor="service:builder", submitter="bob")],
        ids=["acting-for-no-one", "acting-for-someone-else"],
    )
    def test_a_service_not_acting_for_the_submitter_is_a_403(self, caller: Caller) -> None:
        """The policy admits any service on its own permissions, so the handler holds it to the submitter."""
        assert self._get(caller).status_code == 403

    def test_an_image_that_does_not_exist_is_a_404(self) -> None:
        entities = AsyncMock()
        entities.get.side_effect = NemoEntityNotFoundError("no")
        assert TestClient(_app(entities, SERVICE_STEP)).get(IMAGE).status_code == 404


class TestWhoTheRoutesAdmit:
    """The push step's two routes admit a service; the others, people only."""

    @staticmethod
    def _callers() -> dict[tuple[str, str], set[CallerKind]]:
        router = BuilderService().get_routers()[0].router
        return {
            (method, route.path): {kind for rule in get_path_rules(route.endpoint) for kind in rule.callers}
            for route in router.routes
            if isinstance(route, APIRoute)
            for method in route.methods or ()
        }

    def test_the_push_steps_routes_admit_a_service(self) -> None:
        callers = self._callers()
        both = {CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL}
        assert callers[("GET", "/container-images/{name}")] == both
        assert callers[("POST", "/container-images/{name}/complete")] == both

    def test_submitting_and_listing_admit_people_only(self) -> None:
        callers = self._callers()
        assert callers[("POST", "/builds")] == {CallerKind.PRINCIPAL}
        assert callers[("GET", "/container-images")] == {CallerKind.PRINCIPAL}
