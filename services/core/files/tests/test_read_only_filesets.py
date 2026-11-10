# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only filesets reject mutations without touching storage or the body."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from nhx.common.api.common import SecretRef
from nhx.common.files.storage_config import (
    GithubStorageConfig,
    GitStorageConfig,
    HuggingfaceStorageConfig,
    LocalStorageConfig,
    NGCStorageConfig,
    S3StorageConfig,
)
from nhx.core.files.api.v2.filesets import endpoints
from nhx.core.files.app.backends.base import FileInfo


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(endpoints.router)
    for dependency in (endpoints.get_entity_client, endpoints.get_nemo_client, endpoints.get_auth_client):
        app.dependency_overrides[dependency] = lambda: MagicMock()
    return app


@pytest.mark.parametrize(
    "storage",
    [
        GitStorageConfig(
            url="ssh://git@example.com/org/repo.git",
            ssh_key_secret=SecretRef(root="git-key"),
            known_hosts="example.com ssh-ed25519 test",
        ),
        GithubStorageConfig(owner="org", repo="repo"),
        HuggingfaceStorageConfig(repo_id="org/repo"),
        NGCStorageConfig(org="org", team="team", target="model", api_key_secret=SecretRef(root="ngc-key")),
    ],
    ids=["git", "github", "huggingface", "ngc"],
)
@pytest.mark.parametrize("method", ["PUT", "DELETE"])
@pytest.mark.parametrize("path", ["existing.txt", "missing.txt"])
@pytest.mark.parametrize("content_type", ["application/octet-stream", "multipart/form-data"])
async def test_read_only_mutations_do_not_read_body_or_access_storage(
    app, monkeypatch, storage, method, path, content_type
):
    get_fileset = AsyncMock(return_value=SimpleNamespace(storage=storage))
    resolve_secrets = AsyncMock(side_effect=AssertionError("must not resolve secrets"))
    factory = MagicMock(side_effect=AssertionError("must not access storage"))
    monkeypatch.setattr(endpoints, "get_fileset", get_fileset)
    monkeypatch.setattr(endpoints, "resolve_storage_secrets_for_user", resolve_secrets)
    monkeypatch.setattr(endpoints, "storage_impl_factory", factory)
    receive = AsyncMock(side_effect=AssertionError("must not read the upload body"))
    send = AsyncMock()

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": f"/v2/workspaces/default/filesets/readonly/-/{path}",
            "query_string": b"",
            # Deliberately malformed upload metadata must not mask the read-only error.
            "headers": [(b"content-type", content_type.encode()), (b"content-length", b"invalid")],
        },
        receive,
        send,
    )

    messages = [call.args[0] for call in send.await_args_list]
    assert messages[0]["status"] == 409
    assert messages[1]["body"] == (f'{{"detail":"Fileset storage type \'{storage.type}\' is read-only"}}'.encode())
    get_fileset.assert_awaited_once()
    receive.assert_not_awaited()
    resolve_secrets.assert_not_awaited()
    factory.assert_not_called()


@pytest.mark.parametrize("backend", ["local", "s3"])
async def test_writable_storage_still_uploads_and_deletes(app, monkeypatch, tmp_path, backend):
    config = (
        LocalStorageConfig(path=str(tmp_path))
        if backend == "local"
        else S3StorageConfig(bucket="test", use_sdk_auth=True)
    )
    fileset = SimpleNamespace(storage=config, name="writable", custom_fields={})
    monkeypatch.setattr(endpoints, "get_fileset", AsyncMock(return_value=fileset))
    monkeypatch.setattr(endpoints, "resolve_storage_secrets_for_user", AsyncMock(return_value={}))
    storage = endpoints.storage_impl_factory(config)
    if backend == "s3":
        # No network needed: check that the endpoint still invokes the writable backend.
        storage.upload = AsyncMock(return_value=FileInfo(path="file.txt", size=7))
        storage.delete = AsyncMock(return_value=FileInfo(path="file.txt", size=7))
    monkeypatch.setattr(endpoints, "storage_impl_factory", lambda *_: storage)
    url = "/v2/workspaces/default/filesets/writable/-/file.txt"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.put(url, content=b"content")
        assert response.status_code == 200, response.text
        if backend == "local":
            assert (tmp_path / "file.txt").read_bytes() == b"content"
        else:
            storage.upload.assert_awaited_once()
        response = await client.delete(url)
        assert response.status_code == 200, response.text
        if backend == "local":
            assert not (tmp_path / "file.txt").exists()
        else:
            storage.delete.assert_awaited_once_with("file.txt")
