# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the git repository lookup endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from nemo_helix_plugin.files.types import FindGitRepositoryFilesRequest, ScanSshHostKeysRequest, SshHostKey
from nhx.common.api.common import SecretRef
from nhx.core.files.api.v2.git_repositories.endpoints import find_git_repository_files, scan_ssh_host_keys
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


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> FilesConfig:
    # The environment outranks constructor arguments in service configs.
    monkeypatch.delenv("NHX_FILES_ALLOWED_EXTERNAL_HOSTS", raising=False)
    return FilesConfig(allowed_external_hosts="ssh://gitlab.example.com:2222")


SCAN = "nhx.core.files.api.v2.git_repositories.endpoints.scan_host_keys"


async def test_returns_keys_for_an_allowed_host(config):
    with patch(SCAN, AsyncMock(return_value=[KEY])) as scan:
        response = await scan_ssh_host_keys(
            "default", ScanSshHostKeysRequest(url="ssh://git@gitlab.example.com:2222/org/repo.git"), config
        )
    assert response.host == "gitlab.example.com:2222"
    assert response.keys == [KEY]
    assert scan.await_args is not None
    assert scan.await_args.args[0].port == 2222


@pytest.mark.parametrize(
    "url",
    [
        "git@other.example.com:org/repo.git",
        "git@gitlab.example.com:org/repo.git",
        "https://gitlab.example.com/org/repo",
    ],
)
async def test_refuses_to_scan_anything_not_allowlisted(url, config):
    with patch(SCAN, AsyncMock()) as scan, pytest.raises(HTTPException) as raised:
        await scan_ssh_host_keys("default", ScanSshHostKeysRequest(url=url), config)
    assert raised.value.status_code == 400
    scan.assert_not_awaited()


async def test_unreachable_host_is_a_bad_gateway(config):
    with (
        patch(SCAN, AsyncMock(side_effect=GitUnavailableError("no keys"))),
        pytest.raises(HTTPException) as raised,
    ):
        await scan_ssh_host_keys(
            "default", ScanSshHostKeysRequest(url="ssh://gitlab.example.com:2222/org/repo.git"), config
        )
    assert raised.value.status_code == 502


STORAGE = GitStorageConfig(
    url="ssh://git@gitlab.example.com:2222/org/repo.git",
    ssh_key_secret=SecretRef("key"),
    known_hosts="[gitlab.example.com]:2222 ssh-ed25519 AAAA",
)
MODULE = "nhx.core.files.api.v2.git_repositories.endpoints"


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


async def _find(files: list[FileInfo] | None = None, error: Exception | None = None):
    fake = type("Fake", (_FakeGit,), {"files": files or [], "error": error})
    with (
        patch(f"{MODULE}.resolve_storage_secrets_for_user", AsyncMock(return_value={"ssh_key": "k"})),
        patch(f"{MODULE}.GitStorageImpl", fake),
    ):
        return await find_git_repository_files(
            "default",
            FindGitRepositoryFilesRequest(storage=STORAGE, file_name="agent.yaml"),
            client=AsyncMock(),
            auth_client=AsyncMock(),
        )


async def test_find_files_returns_the_resolved_commit_and_matching_paths():
    response = await _find(
        [
            FileInfo(path="agents/b/agent.yaml", size=1),
            FileInfo(path="README.md", size=1),
            FileInfo(path="agent.yaml", size=1),
            FileInfo(path="agents/a/agent.yaml", size=1),
            FileInfo(path="agents/c/not-agent.yaml", size=1),
        ]
    )
    assert response.revision == "a" * 40
    assert response.paths == ["agent.yaml", "agents/a/agent.yaml", "agents/b/agent.yaml"]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (GitAccessError("key rejected"), 400),
        (GitConfigError("no such branch"), 400),
        (GitUnavailableError("unreachable"), 502),
    ],
)
async def test_find_files_reports_failures_like_create(error, status):
    with pytest.raises(HTTPException) as raised:
        await _find(error=error)
    assert raised.value.status_code == status


@pytest.mark.parametrize("file_name", ["", "agents/agent.yaml"])
def test_find_files_takes_a_bare_file_name(file_name):
    with pytest.raises(ValidationError):
        FindGitRepositoryFilesRequest(storage=STORAGE, file_name=file_name)


async def test_find_files_leaves_a_server_fault_to_the_server_error_handler():
    with pytest.raises(GitServerFault):
        await _find(error=GitServerFault("git is not installed in the files service"))


async def test_scan_leaves_a_server_fault_to_the_server_error_handler(config):
    with (
        patch(SCAN, AsyncMock(side_effect=GitServerFault("ssh-keyscan is not installed in the files service"))),
        pytest.raises(GitServerFault),
    ):
        await scan_ssh_host_keys(
            "default", ScanSshHostKeysRequest(url="ssh://gitlab.example.com:2222/org/repo.git"), config
        )
