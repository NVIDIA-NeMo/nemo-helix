# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only filesets reject mutations without touching storage or the body."""

import json
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
from nhx.core.files.api.v2.otlp import endpoints as otlp_endpoints
from nhx.core.files.app.backends.base import FileInfo


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(endpoints.router)
    app.include_router(otlp_endpoints.router)
    app.dependency_overrides[otlp_endpoints.dep_log_storage] = lambda: AsyncMock()
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
@pytest.mark.parametrize(
    "method,content_type",
    [
        ("PUT", "application/octet-stream"),
        ("PUT", "multipart/form-data"),
        ("DELETE", "application/octet-stream"),
        ("POST", "application/json"),
        ("POST", "application/x-protobuf"),
    ],
)
@pytest.mark.parametrize("path", ["existing.txt", "missing.txt"])
async def test_read_only_mutations_do_not_read_body_or_access_storage(
    app, monkeypatch, storage, method, path, content_type
):
    """All mutation routes reject read-only storage before body or backend work."""
    get_fileset = AsyncMock(return_value=SimpleNamespace(storage=storage, custom_fields={}))
    resolve_secrets = AsyncMock(side_effect=AssertionError("must not resolve secrets"))
    factory = MagicMock(side_effect=AssertionError("must not access storage"))
    for module in (endpoints, otlp_endpoints):
        monkeypatch.setattr(module, "get_fileset", get_fileset)
        monkeypatch.setattr(module, "resolve_storage_secrets_for_user", resolve_secrets)
        monkeypatch.setattr(module, "storage_impl_factory", factory)
    receive = AsyncMock(side_effect=AssertionError("must not read the upload body"))
    send = AsyncMock()

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": (
                "/v2/workspaces/default/filesets/readonly/otlp/v1/logs"
                if method == "POST"
                else f"/v2/workspaces/default/filesets/readonly/-/{path}"
            ),
            "query_string": b"",
            # Deliberately malformed upload metadata must not mask the read-only error.
            "headers": [(b"content-type", content_type.encode()), (b"content-length", b"invalid")],
        },
        receive,
        send,
    )

    messages = [call.args[0] for call in send.await_args_list]
    assert messages[0]["status"] == 409
    assert json.loads(messages[1]["body"]) == {"detail": f"Fileset storage type '{storage.type}' is read-only"}
    get_fileset.assert_awaited_once()
    receive.assert_not_awaited()
    resolve_secrets.assert_not_awaited()
    factory.assert_not_called()


@pytest.mark.parametrize("backend", ["local", "s3"])
async def test_writable_storage_still_uploads_and_deletes(app, monkeypatch, tmp_path, backend):
    """Writable backends keep successful mutations and local missing-file 404s."""
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
            response = await client.delete(url)
            assert response.status_code == 404, response.text
        else:
            storage.delete.assert_awaited_once_with("file.txt")


@pytest.mark.parametrize("backend", ["local", "huggingface"])
@pytest.mark.parametrize("principal", ["user@example.com", "service:jobs"])
async def test_immutable_upload_preserves_authorization_precedence(app, monkeypatch, tmp_path, backend, principal):
    """Users get the existing 403 before read-only checks; services retain write access."""
    config = (
        LocalStorageConfig(path=str(tmp_path)) if backend == "local" else HuggingfaceStorageConfig(repo_id="org/repo")
    )
    fileset = SimpleNamespace(storage=config, name="immutable", custom_fields={"service_source": "models"})
    monkeypatch.setattr(endpoints, "get_fileset", AsyncMock(return_value=fileset))
    resolve_secrets = AsyncMock(return_value={})
    monkeypatch.setattr(endpoints, "resolve_storage_secrets_for_user", resolve_secrets)
    app.dependency_overrides[endpoints.get_auth_client] = lambda: SimpleNamespace(
        principal=SimpleNamespace(id=principal)
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.put("/v2/workspaces/default/filesets/immutable/-/file.txt", content=b"content")
    expected_status = 403 if not principal.startswith("service:") else (409 if config.read_only else 200)
    assert response.status_code == expected_status, response.text
    if expected_status != 200:
        resolve_secrets.assert_not_awaited()
        assert not (tmp_path / "file.txt").exists()


@pytest.mark.parametrize(
    "method,path",
    [
        ("put", "/v2/workspaces/{workspace}/filesets/{name}/-/{path}"),
        ("delete", "/v2/workspaces/{workspace}/filesets/{name}/-/{path}"),
        ("post", "/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs"),
    ],
)
def test_read_only_conflict_has_typed_error_schema(app, method, path):
    """Each mutation route documents the detail field of its read-only error."""
    schema = app.openapi()["paths"][path][method]["responses"]["409"]["content"]["application/json"]["schema"]
    assert schema["type"] == "object"
    assert schema["properties"]["detail"]["type"] == "string"
