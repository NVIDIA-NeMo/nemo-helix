# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient

from tests.auth.integration.jobs_auth_helpers import job_exists_in_pages, managed_admin_workspace


class _StubWorkspaces:
    """Recording fake matching the typed WorkspacesClient shape."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.deleted: list[str] = []

    def create_workspace(self, *, body, **kwargs) -> SimpleNamespace:
        self.created.append(body.name)
        return SimpleNamespace(data=lambda: None)

    def delete_workspace(self, *, name, **kwargs) -> SimpleNamespace:
        self.deleted.append(name)
        return SimpleNamespace(data=lambda: None)


def _stub_workspaces_client(monkeypatch: pytest.MonkeyPatch) -> _StubWorkspaces:
    stub = _StubWorkspaces()
    monkeypatch.setattr(WorkspacesClient, "from_client", classmethod(lambda cls, client: stub))
    return stub


class _StubJob:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubJobsResponse:
    def __init__(self, job_names: list[str]) -> None:
        self._job_names = job_names

    def items(self) -> Iterator[_StubJob]:
        for name in self._job_names:
            yield _StubJob(name)


def test_managed_admin_workspace_deletes_workspace_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _stub_workspaces_client(monkeypatch)

    with managed_admin_workspace(cast(NemoClient, object()), "workspace-a") as workspace_name:
        assert workspace_name == "workspace-a"

    assert stub.created == ["workspace-a"]
    assert stub.deleted == ["workspace-a"]


def test_managed_admin_workspace_deletes_workspace_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _stub_workspaces_client(monkeypatch)

    with pytest.raises(RuntimeError, match="boom"):
        with managed_admin_workspace(cast(NemoClient, object()), "workspace-b"):
            raise RuntimeError("boom")

    assert stub.created == ["workspace-b"]
    assert stub.deleted == ["workspace-b"]


def test_job_exists_in_pages_checks_later_pages() -> None:
    jobs_response = _StubJobsResponse(["other-job", "target-job"])

    assert job_exists_in_pages(jobs_response.items(), "target-job") is True
