# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Revision receipts through Files HTTP handlers and the real Entity store."""

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import FilesetOutput, UpdateFilesetRequest
from nhx.common.entities.client import EntityClient
from nhx.core.files.api.v2.filesets import endpoints
from nhx.core.files.app.backends.local import LocalStorageImpl
from nhx.core.files.app.revision import advance_fileset_revision
from nhx.core.files.entities import DEFAULT_REVISION, Fileset
from nhx.testing import ClientContext

BASE = "/apis/files/v2/workspaces/default/filesets"


def test_revision_round_trip(files_client: FilesClient, fileset: FilesetOutput, test_client: TestClient):
    assert fileset.revision is not None
    assert UUID(fileset.revision).version == 4
    current = fileset.revision
    url = f"{BASE}/{fileset.name}"
    assert files_client.get_fileset_by_id(workspace="default", fileset_id=fileset.id).data().revision == current
    assert (
        next(fs for fs in files_client.list_filesets(workspace="default").items() if fs.id == fileset.id).revision
        == current
    )
    assert test_client.patch(url, json={}).json()["revision"] == current

    for content_type in ("raw", "multipart", "identical"):
        if content_type == "multipart":
            result = test_client.put(f"{url}/-/file.txt", files={"file": ("file.txt", b"hello")})
            assert result.status_code == 200, result.text
            uploaded = result.json()["fileset_revision"]
        else:
            uploaded = (
                files_client.upload_file(workspace="default", name=fileset.name, path="file.txt", content=b"hello")
                .data()
                .fileset_revision
            )
        assert uploaded is not None and uploaded != current
        assert files_client.get_fileset(workspace="default", name=fileset.name).data().revision == uploaded
        current = uploaded
        assert test_client.get(f"{url}/-/file.txt").content == b"hello"
        assert test_client.head(f"{url}/-/file.txt").status_code == 200
        files_client.list_files(workspace="default", name=fileset.name).data()
        assert files_client.get_fileset(workspace="default", name=fileset.name).data().revision == current

    updated = files_client.update_fileset(
        workspace="default", name=fileset.name, body=UpdateFilesetRequest(description="changed")
    ).data()
    assert updated.revision != current
    deleted = files_client.delete_file(workspace="default", name=fileset.name, path="file.txt").data()
    assert deleted.fileset_revision is not None and deleted.fileset_revision != updated.revision
    assert test_client.delete(f"{url}/-/file.txt").status_code == 404
    assert files_client.get_fileset(workspace="default", name=fileset.name).data().revision == deleted.fileset_revision


def test_unset_revisions_are_stable_and_first_upload_persists_revision(client_context: ClientContext, tmp_path: Path):
    portal = client_context.test_client.portal
    assert portal is not None
    store = client_context.entity_client
    created = client_context.test_client.post(
        "/apis/entities/v2/workspaces/default/entities/fileset",
        json={
            "name": "legacy-files",
            "data": {
                "purpose": "generic",
                "storage": {"type": "local", "path": str(tmp_path)},
            },
        },
    )
    assert created.status_code == 201, created.text

    async def get_legacy() -> Fileset:
        return await store.get(Fileset, "legacy-files", workspace="default")

    legacy = portal.call(get_legacy)
    files = FilesClient.from_client(client_context.client)
    for _ in range(2):
        assert files.get_fileset(workspace="default", name=legacy.name).data().revision == DEFAULT_REVISION
    stored = portal.call(store.get_by_id, Fileset, legacy.id)
    assert stored.db_version == legacy.db_version
    assert stored.revision == DEFAULT_REVISION
    result = files.upload_file(workspace="default", name=legacy.name, path="file.txt", content=b"hello").data()
    stored = portal.call(store.get_by_id, Fileset, legacy.id)
    assert result.fileset_revision == stored.revision
    assert UUID(stored.revision).version == 4
    assert stored.commit_revision == DEFAULT_REVISION
    assert files.get_fileset(workspace="default", name=legacy.name).data().commit_revision == DEFAULT_REVISION
    committed = files.commit_fileset(workspace="default", name=legacy.name).data()
    assert committed.revision == stored.revision
    assert committed.commit_revision and UUID(committed.commit_revision).version == 4


def test_real_version_conflict_preserves_metadata_without_repeating_upload(
    files_client: FilesClient,
    fileset: FilesetOutput,
    monkeypatch: pytest.MonkeyPatch,
):
    update = EntityClient.update
    upload = LocalStorageImpl.upload
    updates = 0
    uploads = 0

    async def concurrent_update(store: EntityClient, proposed: Fileset, *, original_name: str | None = None):
        nonlocal updates
        updates += 1
        if updates == 1:
            competing = await store.get_by_id(Fileset, proposed.id)
            competing.description = "concurrent description"
            competing.custom_fields = {"other": "kept"}
            await update(store, competing)
        return await update(store, proposed, original_name=original_name)

    async def counted_upload(storage, path, data, content_length=None):
        nonlocal uploads
        uploads += 1
        return await upload(storage, path, data, content_length=content_length)

    with monkeypatch.context() as patch:
        patch.setattr(EntityClient, "update", concurrent_update)
        patch.setattr(LocalStorageImpl, "upload", counted_upload)
        receipt = files_client.upload_file(
            workspace="default", name=fileset.name, path="file.txt", content=b"hello"
        ).data()
    assert uploads == 1
    assert updates == 2
    fetched = files_client.get_fileset(workspace="default", name=fileset.name).data()
    assert fetched.description == "concurrent description"
    assert fetched.custom_fields == {"other": "kept"}
    assert fetched.revision == receipt.fileset_revision


def test_response_keeps_its_revision_when_later_mutation_commits(
    files_client: FilesClient,
    fileset: FilesetOutput,
    monkeypatch: pytest.MonkeyPatch,
):
    first_revision = ""

    async def update_then_supersede(original: Fileset, store: EntityClient):
        nonlocal first_revision
        first = await advance_fileset_revision(original, store)
        first_revision = first.revision
        await advance_fileset_revision(first, store)
        return first

    monkeypatch.setattr(endpoints, "advance_fileset_revision", update_then_supersede)
    result = files_client.upload_file(workspace="default", name=fileset.name, path="file.txt", content=b"hello").data()
    assert result.fileset_revision == first_revision
    assert files_client.get_fileset(workspace="default", name=fileset.name).data().revision != first_revision


def test_revision_failure_is_an_error_after_storage_success(
    test_client: TestClient,
    fileset: FilesetOutput,
    monkeypatch: pytest.MonkeyPatch,
):
    async def unavailable(original: Fileset, store: EntityClient):
        raise HTTPException(503, "revision persistence unavailable")

    monkeypatch.setattr(endpoints, "advance_fileset_revision", unavailable)
    url = f"{BASE}/{fileset.name}"
    response = test_client.put(f"{url}/-/file.txt", content=b"written")
    assert response.status_code == 503
    assert test_client.get(f"{url}/-/file.txt").content == b"written"
    assert test_client.get(url).json()["revision"] == fileset.revision


def test_late_receipt_does_not_update_replacement(client_context: ClientContext, monkeypatch: pytest.MonkeyPatch):
    files = FilesClient.from_client(client_context.client)
    from nemo_helix_plugin.files.types import CreateFilesetRequest

    original = files.create_fileset(workspace="default", body=CreateFilesetRequest(name="reused-name")).data()
    update = EntityClient.update
    replacement_revision = str(uuid4())

    async def replace_before_update(store: EntityClient, proposed: Fileset, *, original_name: str | None = None):
        await store.delete_by_id(Fileset, proposed.id)
        await store.create(
            Fileset(
                name=proposed.name,
                workspace=proposed.workspace,
                storage=proposed.storage,
                purpose=proposed.purpose,
                revision=replacement_revision,
            )
        )
        return await update(store, proposed, original_name=original_name)

    with monkeypatch.context() as patch:
        patch.setattr(EntityClient, "update", replace_before_update)
        response = client_context.test_client.put(f"{BASE}/{original.name}/-/file.txt", content=b"hello")
    assert response.status_code == 404, response.text
    replacement = files.get_fileset(workspace="default", name=original.name).data()
    assert replacement.id != original.id
    assert replacement.revision == replacement_revision


def test_metadata_conflict_reapplies_only_the_patch(
    files_client: FilesClient,
    fileset: FilesetOutput,
    monkeypatch: pytest.MonkeyPatch,
):
    update = EntityClient.update
    updates = 0

    async def concurrent_update(store: EntityClient, proposed: Fileset, *, original_name: str | None = None):
        nonlocal updates
        updates += 1
        if updates == 1:
            competing = await store.get_by_id(Fileset, proposed.id)
            competing.custom_fields = {"concurrent": "kept"}
            await update(store, competing)
        return await update(store, proposed, original_name=original_name)

    with monkeypatch.context() as patch:
        patch.setattr(EntityClient, "update", concurrent_update)
        result = files_client.update_fileset(
            workspace="default", name=fileset.name, body=UpdateFilesetRequest(description="requested")
        ).data()
    assert updates == 2
    assert result.description == "requested"
    assert result.custom_fields == {"concurrent": "kept"}
    assert result.revision != fileset.revision


def test_revision_bookkeeping_preserves_authorization():
    from nemo_helix_plugin.client.client import NemoClient
    from nemo_helix_plugin.client.errors import PermissionDeniedError
    from nemo_helix_plugin.files.types import CreateFilesetRequest
    from nemo_helix_plugin.workspaces.client import WorkspacesClient
    from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
    from nhx.core.files.service import FilesService
    from nhx.core.secrets.service import SecretsService
    from nhx.testing import TEST_ADMIN_EMAIL, as_user, create_test_client, grant_workspace_role

    with create_test_client(FilesService, SecretsService, auth_enabled=True, client_type=NemoClient) as client:
        admin = as_user(client, TEST_ADMIN_EMAIL)
        workspace = "revision-auth"
        WorkspacesClient.from_client(admin).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        grant_workspace_role(admin, workspace=workspace, principal="writer@example.com", roles=["Editor"])
        grant_workspace_role(admin, workspace=workspace, principal="reader@example.com", roles=["Viewer"])
        writer = FilesClient.from_client(as_user(client, "writer@example.com"))
        reader = FilesClient.from_client(as_user(client, "reader@example.com"))
        created = writer.create_fileset(workspace=workspace, body=CreateFilesetRequest(name="authorized-files")).data()
        committed = writer.commit_fileset(workspace=workspace, name=created.name).data()
        assert committed.revision == created.revision
        assert committed.commit_revision != created.commit_revision
        receipt = writer.upload_file(workspace=workspace, name=created.name, path="file.txt", content=b"hello").data()
        assert receipt.fileset_revision != created.revision
        assert reader.get_fileset(workspace=workspace, name=created.name).data().revision == receipt.fileset_revision
        with pytest.raises(PermissionDeniedError):
            reader.commit_fileset(workspace=workspace, name=created.name)
        assert (
            reader.get_fileset(workspace=workspace, name=created.name).data().commit_revision
            == committed.commit_revision
        )
        with pytest.raises(PermissionDeniedError):
            reader.upload_file(workspace=workspace, name=created.name, path="file.txt", content=b"denied")
        assert writer.get_fileset(workspace=workspace, name=created.name).data().revision == receipt.fileset_revision


def test_whole_delete_returns_last_revision_and_recreation_is_distinct(files_client: FilesClient):
    from nemo_helix_plugin.client.errors import NotFoundError
    from nemo_helix_plugin.files.types import CreateFilesetRequest

    original = files_client.create_fileset(workspace="default", body=CreateFilesetRequest(name="deleted-files")).data()
    deleted = files_client.delete_fileset_by_id(workspace="default", fileset_id=original.id).data()
    assert deleted.revision == original.revision
    with pytest.raises(NotFoundError):
        files_client.get_fileset(workspace="default", name=original.name)
    replacement = files_client.create_fileset(workspace="default", body=CreateFilesetRequest(name=original.name)).data()
    assert replacement.id != original.id
    assert replacement.revision != original.revision
