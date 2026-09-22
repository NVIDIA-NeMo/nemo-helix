# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Writes through one workspace never act on an entity shared from the global workspace.

A shareable model or provider in ``default`` is *readable* by name from any workspace.
A write addressed to another workspace must still only ever find entities that live
there: resolving the shared one instead would let an Editor of that workspace modify
or delete an entity in ``default`` it may only be allowed to read, and would stop it
from creating its own entity under the shared name.
"""

import uuid

from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import ConflictError
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest
from nmp.testing import ClientContext

GLOBAL = "default"


def _team_workspace(test_clients: ClientContext) -> str:
    name = f"team-{uuid.uuid4().hex[:8]}"
    try:
        client_from_platform(test_clients.sdk, WorkspacesClient).create_workspace(
            body=CreateWorkspaceRequest(name=name, description=name)
        ).data()
    except ConflictError:
        pass
    return name


def _models(workspace: str, name: str = "") -> str:
    base = f"/apis/models/v2/workspaces/{workspace}/models"
    return f"{base}/{name}" if name else base


def _providers(workspace: str, name: str = "") -> str:
    base = f"/apis/models/v2/workspaces/{workspace}/providers"
    return f"{base}/{name}" if name else base


def test_shared_model_still_resolves_from_another_workspace(test_clients: ClientContext):
    client = test_clients.test_client
    team = _team_workspace(test_clients)
    name = f"shared-{uuid.uuid4().hex[:8]}"
    assert client.post(_models(GLOBAL), json={"name": name}).status_code == 201
    try:
        response = client.get(_models(team, name))
        assert response.status_code == 200
        assert response.json()["workspace"] == GLOBAL
    finally:
        client.delete(_models(GLOBAL, name))


def test_update_through_another_workspace_does_not_modify_the_shared_model(test_clients: ClientContext):
    client = test_clients.test_client
    team = _team_workspace(test_clients)
    name = f"shared-{uuid.uuid4().hex[:8]}"
    assert client.post(_models(GLOBAL), json={"name": name, "description": "original"}).status_code == 201
    try:
        response = client.patch(_models(team, name), json={"description": "changed through team"})

        assert response.status_code == 404, response.text
        assert client.get(_models(GLOBAL, name)).json()["description"] == "original"
    finally:
        client.delete(_models(GLOBAL, name))


def test_a_workspace_can_create_its_own_model_under_a_shared_name(test_clients: ClientContext):
    client = test_clients.test_client
    team = _team_workspace(test_clients)
    name = f"shared-{uuid.uuid4().hex[:8]}"
    assert client.post(_models(GLOBAL), json={"name": name, "description": "global"}).status_code == 201
    try:
        response = client.post(_models(team), json={"name": name, "description": "local"})

        assert response.status_code == 201, response.text
        assert response.json()["workspace"] == team
        # The local model now shadows the shared one for reads through the team workspace.
        assert client.get(_models(team, name)).json()["description"] == "local"
        assert client.get(_models(GLOBAL, name)).json()["description"] == "global"
    finally:
        client.delete(_models(team, name))
        client.delete(_models(GLOBAL, name))


def test_a_workspace_can_create_its_own_provider_under_a_shared_name(test_clients: ClientContext):
    client = test_clients.test_client
    team = _team_workspace(test_clients)
    name = f"shared-{uuid.uuid4().hex[:8]}"
    shared = {"name": name, "host_url": "https://global.example.com"}
    assert client.post(_providers(GLOBAL), json=shared).status_code == 201
    try:
        response = client.post(_providers(team), json={"name": name, "host_url": "https://team.example.com"})

        assert response.status_code == 201, response.text
        assert response.json()["workspace"] == team
    finally:
        client.delete(_providers(team, name))
        client.delete(_providers(GLOBAL, name))


def test_upsert_through_another_workspace_creates_a_local_provider(test_clients: ClientContext):
    client = test_clients.test_client
    team = _team_workspace(test_clients)
    name = f"shared-{uuid.uuid4().hex[:8]}"
    assert (
        client.post(_providers(GLOBAL), json={"name": name, "host_url": "https://global.example.com"}).status_code
        == 201
    )
    try:
        response = client.put(_providers(team, name), json={"host_url": "https://team.example.com"})

        assert response.status_code in (200, 201), response.text
        assert response.json()["workspace"] == team
        assert client.get(_providers(GLOBAL, name)).json()["host_url"] == "https://global.example.com"
    finally:
        client.delete(_providers(team, name))
        client.delete(_providers(GLOBAL, name))


def test_delete_through_another_workspace_leaves_the_shared_provider(test_clients: ClientContext):
    client = test_clients.test_client
    team = _team_workspace(test_clients)
    name = f"shared-{uuid.uuid4().hex[:8]}"
    assert (
        client.post(_providers(GLOBAL), json={"name": name, "host_url": "https://global.example.com"}).status_code
        == 201
    )
    try:
        response = client.delete(_providers(team, name))

        assert response.status_code == 404, response.text
        assert client.get(_providers(GLOBAL, name)).status_code == 200
    finally:
        client.delete(_providers(GLOBAL, name))
