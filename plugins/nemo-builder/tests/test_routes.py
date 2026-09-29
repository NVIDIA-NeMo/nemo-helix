# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's routes, over HTTP, with the entity store faked."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_builder_plugin.completion import SIMPLE_SIGNING_TYPE
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.service import BuilderService
from nemo_builder_plugin.signing import signature_annotations
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError, get_entity_client


def _app(entities: AsyncMock) -> FastAPI:
    app = FastAPI()
    for spec in BuilderService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_entity_client] = lambda: entities
    return app


def _empty_page() -> MagicMock:
    page = MagicMock()
    page.data = []
    return page


class TestListingOneJobsImages:
    """How the credential broker finds what a job may publish, without reading the workspace."""

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
        """A job name is only unique within a workspace, so another workspace's cannot be named."""
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


# --- The signature route ----------------------------------------------------------------------

KEY = ec.generate_private_key(ec.SECP256R1())
PUBLIC_KEY = KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
CONFIG = BuilderConfig(
    registry="reg.example.com", credential_broker="http://broker:8080", signing_public_key=PUBLIC_KEY.decode()
)


def _row() -> ContainerImage:
    return ContainerImage(
        name="build-1-0",
        workspace="ws-a",
        registry="reg.example.com",
        repository="ws-a/app",
        provenance=Provenance(
            build_set="build",
            revision=1,
            job="ws-a/build-1",
            system_tag="ws-a--build-1-0",
            request_digest="sha256:" + "a" * 64,
        ),
    )


def _signed(row: ContainerImage, *, key: ec.EllipticCurvePrivateKey = KEY) -> dict[str, str]:
    payload = json.dumps(
        {
            "critical": {
                "identity": {"docker-reference": f"{row.registry}/{row.repository}"},
                "image": {"docker-manifest-digest": "sha256:" + "d" * 64},
                "type": SIMPLE_SIGNING_TYPE,
            },
            "optional": signature_annotations(row),
        }
    ).encode()
    signature = key.sign(payload, ec.ECDSA(hashes.SHA256()))
    return {"payload": base64.b64encode(payload).decode(), "signature": base64.b64encode(signature).decode()}


class TestDeliveringSignatures:
    """A row goes ready only on a signature the route verified; the caller's say-so is not enough."""

    @pytest.fixture(autouse=True)
    def _config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(BuilderConfig, "get", classmethod(lambda cls: CONFIG))

    def _entities(self, row: ContainerImage) -> AsyncMock:
        entities = AsyncMock()
        entities.get.return_value = row
        entities.update.side_effect = lambda updated: updated
        return entities

    def _post(self, entities: AsyncMock, body: dict[str, str]):  # returns TestClient's own response type
        return TestClient(_app(entities)).post("/v2/workspaces/ws-a/container-images/build-1-0/signature", json=body)

    def test_a_verified_signature_makes_the_row_ready(self) -> None:
        entities = self._entities(_row())
        response = self._post(entities, _signed(_row()))
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert response.json()["signature"]["verified_against"].startswith("key:sha256:")
        entities.update.assert_awaited_once()

    def test_a_signature_by_another_key_is_a_422_and_writes_nothing(self) -> None:
        entities = self._entities(_row())
        response = self._post(entities, _signed(_row(), key=ec.generate_private_key(ec.SECP256R1())))
        assert response.status_code == 422
        entities.update.assert_not_awaited()

    def test_a_row_that_does_not_exist_is_a_404(self) -> None:
        entities = AsyncMock()
        entities.get.side_effect = NemoEntityNotFoundError("no")
        assert self._post(entities, _signed(_row())).status_code == 404

    def test_a_failed_row_is_a_409(self) -> None:
        row = _row()
        row.status = "failed"
        assert self._post(self._entities(row), _signed(_row())).status_code == 409

    def test_anything_but_a_payload_and_a_signature_is_refused(self) -> None:
        body = _signed(_row()) | {"digest": "sha256:" + "e" * 64}
        assert self._post(self._entities(_row()), body).status_code == 422
