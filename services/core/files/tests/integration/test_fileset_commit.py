# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit commit markers through the SDK and real Entity version conflicts."""

from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetOutput, UpdateFilesetRequest
from nhx.common.entities.client import EntityClient
from nhx.core.files.api.v2.filesets import endpoints
from nhx.core.files.app.revision import commit_fileset_revision
from nhx.core.files.entities import Fileset

BASE = "/apis/files/v2/workspaces/default/filesets"


def test_commit_rotates_only_commit_marker(files_client: FilesClient, fileset: FilesetOutput):
    assert fileset.commit_revision and UUID(fileset.commit_revision).version == 4
    assert fileset.commit_revision != fileset.revision
    previous = fileset.commit_revision
    for _ in range(2):
        committed = files_client.commit_fileset(workspace="default", name=fileset.name).data()
        assert committed.revision == fileset.revision
        assert committed.commit_revision and committed.commit_revision != previous
        observed = files_client.get_fileset_by_id(workspace="default", fileset_id=fileset.id).data()
        assert observed.commit_revision == committed.commit_revision
        listed = next(fs for fs in files_client.list_filesets(workspace="default").items() if fs.id == fileset.id)
        assert listed.commit_revision == committed.commit_revision
        previous = committed.commit_revision

    for content in (b"first", b"second"):
        uploaded = files_client.upload_file(workspace="default", name=fileset.name, path="file", content=content).data()
        observed = files_client.get_fileset(workspace="default", name=fileset.name).data()
        assert observed.revision == uploaded.fileset_revision
        assert observed.commit_revision == previous
    updated = files_client.update_fileset(
        workspace="default", name=fileset.name, body=UpdateFilesetRequest(description="changed")
    ).data()
    assert updated.commit_revision == previous
    files_client.delete_file(workspace="default", name=fileset.name, path="file")
    observed = files_client.get_fileset(workspace="default", name=fileset.name).data()
    assert observed.commit_revision == previous
    committed = files_client.commit_fileset(workspace="default", name=fileset.name).data()
    assert committed.revision == observed.revision
    assert committed.commit_revision != previous


@pytest.mark.parametrize("incoming_commit", [False, True])
def test_conflicts_preserve_the_other_token(
    files_client: FilesClient,
    fileset: FilesetOutput,
    monkeypatch: pytest.MonkeyPatch,
    incoming_commit: bool,
):
    update = EntityClient.update
    competing_token = str(uuid4())
    updates = 0

    async def race(store: EntityClient, proposed: Fileset, *, original_name: str | None = None):
        nonlocal updates
        updates += 1
        if updates == 1:
            other = await store.get_by_id(Fileset, proposed.id)
            if incoming_commit:
                other.revision = competing_token
            else:
                other.commit_revision = competing_token
            other.description = "concurrent metadata"
            await update(store, other)
        return await update(store, proposed, original_name=original_name)

    with monkeypatch.context() as patch:
        patch.setattr(EntityClient, "update", race)
        if incoming_commit:
            committed = files_client.commit_fileset(workspace="default", name=fileset.name).data()
            receipt = committed.commit_revision
        else:
            uploaded = files_client.upload_file(
                workspace="default", name=fileset.name, path="file", content=b"hello"
            ).data()
            receipt = uploaded.fileset_revision
    assert updates == 2
    observed = files_client.get_fileset(workspace="default", name=fileset.name).data()
    assert observed.description == "concurrent metadata"
    if incoming_commit:
        assert observed.revision == competing_token
        assert observed.commit_revision == receipt
    else:
        assert observed.commit_revision == competing_token
        assert observed.revision == receipt


def test_delayed_commit_returns_its_own_marker(
    files_client: FilesClient, fileset: FilesetOutput, monkeypatch: pytest.MonkeyPatch
):
    first_token = ""

    async def commit_then_supersede(original: Fileset, store: EntityClient):
        nonlocal first_token
        first = await commit_fileset_revision(original, store)
        first_token = first.commit_revision
        await commit_fileset_revision(first, store)
        return first

    monkeypatch.setattr(endpoints, "commit_fileset_revision", commit_then_supersede)
    result = files_client.commit_fileset(workspace="default", name=fileset.name).data()
    assert result.commit_revision == first_token
    assert result.revision == fileset.revision
    observed = files_client.get_fileset(workspace="default", name=fileset.name).data()
    assert observed.commit_revision != first_token
    assert observed.revision == fileset.revision


def test_commit_never_retargets_recreated_name(files_client: FilesClient, monkeypatch: pytest.MonkeyPatch):
    fileset = files_client.create_fileset(
        workspace="default", body=CreateFilesetRequest(name="recreated-commit")
    ).data()
    update = EntityClient.update
    replacement_token = str(uuid4())

    async def replace_before_update(store: EntityClient, proposed: Fileset, *, original_name: str | None = None):
        await store.delete_by_id(Fileset, proposed.id)
        await store.create(
            Fileset(
                name=proposed.name,
                workspace=proposed.workspace,
                purpose=proposed.purpose,
                storage=proposed.storage,
                commit_revision=replacement_token,
            )
        )
        return await update(store, proposed, original_name=original_name)

    with monkeypatch.context() as patch:
        patch.setattr(EntityClient, "update", replace_before_update)
        with pytest.raises(NotFoundError):
            files_client.commit_fileset(workspace="default", name=fileset.name)
    replacement = files_client.get_fileset(workspace="default", name=fileset.name).data()
    assert replacement.id != fileset.id
    assert replacement.commit_revision == replacement_token
    files_client.delete_fileset_by_id(workspace="default", fileset_id=replacement.id)


@pytest.mark.parametrize("read_only", [False, True])
def test_legacy_commit_has_no_storage_effects(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch, read_only: bool
):
    storage = (
        {"type": "github", "owner": "org", "repo": "repo", "revision": "main"}
        if read_only
        else {"type": "local", "path": "/unused"}
    )
    created = test_client.post(
        "/apis/entities/v2/workspaces/default/entities/fileset",
        json={
            "name": "legacy-commit",
            "data": {"purpose": "generic", "storage": storage},
        },
    )
    assert created.status_code == 201, created.text
    url = f"{BASE}/legacy-commit"
    initial = test_client.get(url).json()
    assert initial["commit_revision"] == f"legacy-commit:{initial['id']}"
    assert test_client.get(url).json()["commit_revision"] == initial["commit_revision"]

    def no_storage(*args, **kwargs):
        raise AssertionError("commit must not construct a storage backend")

    monkeypatch.setattr(endpoints, "storage_impl_factory", no_storage)
    response = test_client.post(f"{url}/commit")
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == initial["revision"]
    assert response.json()["commit_revision"] != initial["commit_revision"]
    assert test_client.get(url).json()["commit_revision"] == response.json()["commit_revision"]


def test_commit_requires_update_permission_without_create_permission():
    from unittest.mock import patch

    from nemo_helix_plugin.client.client import NemoClient
    from nemo_helix_plugin.client.errors import PermissionDeniedError
    from nemo_helix_plugin.workspaces.client import WorkspacesClient
    from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
    from nhx.core.auth.app.bundle import build_authorization_data
    from nhx.core.files.service import FilesService
    from nhx.testing import TEST_ADMIN_EMAIL, as_user, create_test_client, grant_workspace_role

    async def with_commit_role(entities_client: EntityClient | None = None):
        data = await build_authorization_data(entities_client)
        data["authz"]["roles"]["FilesetCommitter"] = {"includes": ["Viewer"], "permissions": ["filesets.update"]}
        return data

    with (
        patch("nhx.core.auth.app.bundle.build_authorization_data", side_effect=with_commit_role),
        patch("nhx.core.auth.app.embedded_pdp.data.build_authorization_data", side_effect=with_commit_role),
        create_test_client(FilesService, auth_enabled=True, client_type=NemoClient) as client,
    ):
        admin = as_user(client, TEST_ADMIN_EMAIL)
        workspace = "commit-auth"
        WorkspacesClient.from_client(admin).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        admin_files = FilesClient.from_client(admin)
        created = admin_files.create_fileset(workspace=workspace, body=CreateFilesetRequest(name="files")).data()
        grant_workspace_role(admin, workspace=workspace, principal="committer@example.com", roles=["FilesetCommitter"])
        files = FilesClient.from_client(as_user(client, "committer@example.com"))
        with pytest.raises(PermissionDeniedError):
            files.create_fileset(workspace=workspace, body=CreateFilesetRequest(name="forbidden-create"))
        committed = files.commit_fileset(workspace=workspace, name=created.name).data()
        assert committed.revision == created.revision
        assert committed.commit_revision != created.commit_revision
