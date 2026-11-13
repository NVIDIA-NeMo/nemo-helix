# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the git repository lookup endpoint."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from nemo_helix_plugin.files.types import FindGitRepositoryFilesRequest, SshHostKey
from nhx.common.api.common import SecretRef
from nhx.common.auth import get_auth_client
from nhx.common.service.dependencies import get_nemo_client
from nhx.core.files.api.v2.git_repositories.endpoints import find_git_repository_files, router
from nhx.core.files.app.backends.base import FileInfo
from nhx.core.files.app.backends.git import (
    GitAccessError,
    GitConfigError,
    GitServerFault,
    GitStorageConfig,
    GitUnavailableError,
)
from nhx.core.files.config import FilesConfig
from pydantic import ValidationError

KEY = SshHostKey(key_type="ssh-ed25519", fingerprint="SHA256:abc", known_hosts_line="[h]:2222 ssh-ed25519 AAAA")
SCAN = "nhx.core.files.app.backends.git.scan_host_keys"
MODULE = "nhx.core.files.api.v2.git_repositories.endpoints"
STORAGE = GitStorageConfig(
    url="ssh://git@gitlab.example.com:2222/org/repo.git",
    ssh_key_secret=SecretRef("key"),
    known_hosts="[gitlab.example.com]:2222 ssh-ed25519 AAAA",
)
UNTRUSTED = STORAGE.model_copy(update={"known_hosts": ""})


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> FilesConfig:
    monkeypatch.delenv("NHX_FILES_ALLOWED_EXTERNAL_HOSTS", raising=False)
    return FilesConfig(allowed_external_hosts="ssh://gitlab.example.com:2222")


class _FakeGit:
    files: list[FileInfo] = []
    error: Exception | None = None

    def __init__(self, config: GitStorageConfig, secrets: dict[str, str]):
        self.config = config

    async def validate_storage(self) -> None:
        if self.error:
            raise self.error

    async def resolve_config(self) -> GitStorageConfig:
        return self.config.model_copy(update={"revision": "a" * 40, "original_revision": self.config.revision})

    async def list_files(self) -> list[FileInfo]:
        return self.files


async def _find(
    config: FilesConfig,
    storage: GitStorageConfig = STORAGE,
    files: list[FileInfo] | None = None,
    error: Exception | None = None,
):
    fake = type("Fake", (_FakeGit,), {"files": files or [], "error": error})
    with (
        patch(f"{MODULE}.resolve_storage_secrets_for_user", AsyncMock(return_value={"ssh_key": "k"})),
        patch("nhx.core.files.app.backends.git.GitStorageImpl", fake),
    ):
        return await find_git_repository_files(
            "default",
            FindGitRepositoryFilesRequest(storage=storage, file_name="agent.yaml"),
            client=AsyncMock(),
            auth_client=AsyncMock(),
            config=config,
        )


async def test_find_files_returns_the_resolved_commit_and_matching_paths(config):
    response = await _find(
        config,
        files=[
            FileInfo(path="agents/b/agent.yaml", size=1),
            FileInfo(path="README.md", size=1),
            FileInfo(path="agent.yaml", size=1),
            FileInfo(path="agents/a/agent.yaml", size=1),
            FileInfo(path="agents/c/not-agent.yaml", size=1),
        ],
    )
    assert response.revision == "a" * 40
    assert response.paths == ["agent.yaml", "agents/a/agent.yaml", "agents/b/agent.yaml"]
    assert response.host_keys == []


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (GitAccessError("key rejected"), 400),
        (GitConfigError("no such branch"), 400),
        (GitUnavailableError("unreachable"), 502),
    ],
)
async def test_find_files_reports_failures_like_create(config, error, status):
    with pytest.raises(HTTPException) as raised:
        await _find(config, error=error)
    assert raised.value.status_code == status


@pytest.mark.parametrize("file_name", ["", "agents/agent.yaml"])
def test_find_files_takes_a_bare_file_name(file_name):
    with pytest.raises(ValidationError):
        FindGitRepositoryFilesRequest(storage=STORAGE, file_name=file_name)


async def test_find_files_leaves_a_server_fault_to_the_server_error_handler(config):
    with pytest.raises(GitServerFault):
        await _find(config, error=GitServerFault("git is not installed in the files service"))


async def test_without_known_hosts_returns_the_host_keys_and_reads_nothing(config):
    with patch(SCAN, AsyncMock(return_value=[KEY])) as scan:
        response = await _find(config, storage=UNTRUSTED, error=AssertionError("the repository was read"))
    assert response.host_keys == [KEY]
    assert response.revision is None
    assert response.paths == []
    assert scan.await_args is not None
    assert scan.await_args.args[0].port == 2222


@pytest.mark.parametrize(
    "url",
    ["git@other.example.com:org/repo.git", "git@gitlab.example.com:org/repo.git"],
)
async def test_refuses_to_scan_a_host_not_allowlisted(config, url):
    with patch(SCAN, AsyncMock()) as scan, pytest.raises(HTTPException) as raised:
        await _find(config, storage=UNTRUSTED.model_copy(update={"url": url}))
    assert raised.value.status_code == 400
    scan.assert_not_awaited()


async def test_a_host_that_cannot_be_scanned_is_a_bad_gateway(config):
    with (
        patch(SCAN, AsyncMock(side_effect=GitUnavailableError("no keys"))),
        pytest.raises(HTTPException) as raised,
    ):
        await _find(config, storage=UNTRUSTED)
    assert raised.value.status_code == 502


async def test_scan_leaves_a_server_fault_to_the_server_error_handler(config):
    with (
        patch(SCAN, AsyncMock(side_effect=GitServerFault("ssh-keyscan is not installed in the files service"))),
        pytest.raises(GitServerFault),
    ):
        await _find(config, storage=UNTRUSTED)


@pytest.fixture
def http(config: FilesConfig) -> Iterator[TestClient]:
    app = FastAPI()
    app.state.service_configs = {FilesConfig: config}
    app.dependency_overrides[get_nemo_client] = lambda: AsyncMock()
    app.dependency_overrides[get_auth_client] = lambda: AsyncMock()
    app.include_router(router)
    with TestClient(app) as client:
        yield client


def _post_find(http: TestClient, known_hosts: str | None):
    storage = {"type": "git", "url": STORAGE.url, "ssh_key_secret": "key"}
    if known_hosts is not None:
        storage["known_hosts"] = known_hosts
    return http.post(
        "/v2/workspaces/default/git-repositories/find-files",
        json={"storage": storage, "file_name": "agent.yaml"},
    )


def test_find_files_without_known_hosts_is_routed_to_the_scan(http):
    with patch(SCAN, AsyncMock(return_value=[KEY])):
        response = _post_find(http, known_hosts=None)
    assert response.status_code == 200
    assert response.json()["host_keys"][0]["fingerprint"] == KEY.fingerprint


def test_find_files_is_routed_under_the_workspace(http):
    fake = type("Fake", (_FakeGit,), {"files": [FileInfo(path="a/agent.yaml", size=1)]})
    with (
        patch(f"{MODULE}.resolve_storage_secrets_for_user", AsyncMock(return_value={"ssh_key": "k"})),
        patch("nhx.core.files.app.backends.git.GitStorageImpl", fake),
    ):
        response = _post_find(http, known_hosts=STORAGE.known_hosts)
    assert response.status_code == 200
    assert response.json()["paths"] == ["a/agent.yaml"]
