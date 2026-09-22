# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for ``plugins/nemo-agents/tests/unit/``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.files.types import FilesetFileOutput, FilesetOutput
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults

PLATFORM_BASE_URL = "http://test"


def _reject_request(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected platform request: {request.method} {request.url}")


def _make_platform_client(
    handler: Callable[[httpx.Request], httpx.Response] = _reject_request,
    *,
    workspace: str | None = None,
) -> NemoClient:
    return NemoClient(
        base_url=PLATFORM_BASE_URL,
        workspace=workspace,
        http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url=PLATFORM_BASE_URL),
    )


PlatformClientFactory = Callable[..., NemoClient]


@pytest.fixture
def make_platform_client() -> PlatformClientFactory:
    """Build a :class:`NemoClient` over ``httpx.MockTransport`` standing in for the CLI's platform handle.

    Without a handler every request fails loudly, so a test that only expects
    fileset transfers (recorded by :class:`FakeFilesetTransfers`) catches any
    stray platform call.
    """
    return _make_platform_client


def _fileset_output(workspace: str, name: str) -> FilesetOutput:
    return FilesetOutput.model_validate(
        {
            "id": f"id-{name}",
            "name": name,
            "workspace": workspace,
            "description": "",
            "purpose": "generic",
            "storage": {"type": "local", "path": f"/data/{name}"},
            "metadata": {},
            "custom_fields": {},
            "project": "",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )


@dataclass
class FakeFilesetTransfers:
    """Stand-in for the ``filesets.transfer`` helpers the agents jobs drive.

    Records every call with the typed Files client it was handed plus its
    keyword arguments. ``on_download`` materializes a downloaded tree at
    ``local_path``; ``download_error`` makes downloads raise instead.
    """

    downloads: list[dict[str, Any]] = field(default_factory=list)
    uploads: list[dict[str, Any]] = field(default_factory=list)
    lists: list[dict[str, Any]] = field(default_factory=list)
    deletes: list[dict[str, Any]] = field(default_factory=list)
    on_download: Callable[[Path, str | None, str | None, str], None] | None = None
    download_error: Exception | None = None
    listed: list[FilesetFileOutput] = field(default_factory=list)

    def download(self, client: FilesClient, **kwargs: Any) -> None:
        assert isinstance(client, FilesClient), client
        self.downloads.append({"client": client, **kwargs})
        if self.download_error is not None:
            raise self.download_error
        if self.on_download is not None:
            self.on_download(
                Path(kwargs["local_path"]),
                kwargs.get("fileset"),
                kwargs.get("workspace"),
                kwargs.get("remote_path", ""),
            )

    def upload(self, client: FilesClient, **kwargs: Any) -> FilesetOutput:
        assert isinstance(client, FilesClient), client
        self.uploads.append({"client": client, **kwargs})
        return _fileset_output(kwargs.get("workspace") or "default", kwargs.get("fileset") or "fileset-auto")

    def list_files(self, client: FilesClient, **kwargs: Any) -> Any:
        from filesets.transfer import ListFilesResponse

        assert isinstance(client, FilesClient), client
        self.lists.append({"client": client, **kwargs})
        return ListFilesResponse(data=list(self.listed))

    async def async_list_files(self, client: AsyncFilesClient, **kwargs: Any) -> Any:
        from filesets.transfer import ListFilesResponse

        assert isinstance(client, AsyncFilesClient), client
        self.lists.append({"client": client, **kwargs})
        return ListFilesResponse(data=list(self.listed))

    def delete(self, client: FilesClient, **kwargs: Any) -> None:
        assert isinstance(client, FilesClient), client
        self.deletes.append({"client": client, **kwargs})


@pytest.fixture
def fake_transfers(monkeypatch: pytest.MonkeyPatch) -> FakeFilesetTransfers:
    """Route the agents jobs' fileset transfers through :class:`FakeFilesetTransfers`."""
    import nemo_agents_plugin.jobs.evaluate_agent as evaluate_agent
    import nemo_agents_plugin.jobs.fileset_io as fileset_io
    import nemo_agents_plugin.tasks.execute.workdir as workdir

    fake = FakeFilesetTransfers()
    monkeypatch.setattr(fileset_io, "download", fake.download)
    monkeypatch.setattr(fileset_io, "upload", fake.upload)
    monkeypatch.setattr(evaluate_agent, "upload", fake.upload)
    monkeypatch.setattr(workdir, "download", fake.download)
    monkeypatch.setattr(workdir, "async_list_files", fake.async_list_files)
    return fake


@pytest.fixture(autouse=True)
def _isolate_nmp_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    """Pin ``NMP_DATA_DIR`` to a per-test tempdir for every test in this tree.

    Belt-and-braces against accidental writes to ``~/.local/share/nemo``: any
    test that resolves :func:`nmp_user_data_dir` (directly or via
    ``AgentsConfig`` defaults) will land under a tempdir that pytest
    cleans up, never the developer's real homedir.  Tests that need to
    exercise the env-var resolution itself (XDG, explicit path) override
    this in their own fixtures via ``monkeypatch``.
    """
    isolated = tmp_path_factory.mktemp("nmp-data")
    monkeypatch.setenv("NMP_DATA_DIR", str(isolated))
    # Configuration is cached; reset so the override takes effect for tests
    # that read `AgentsConfig.get()`.
    from nemo_platform_plugin.config import Configuration

    Configuration.clear_cache()
    yield
    Configuration.clear_cache()


@pytest.fixture
def ctx(tmp_path: Path) -> JobContext:
    """:class:`JobContext` with platform-style storage rooted in ``tmp_path``.

    Mirrors the shape ``run_task`` builds via
    :func:`nemo_platform_plugin.tasks.dispatcher.build_ctx_from_env` and the
    scheduler builds via ``_build_local_context``: a per-job tempdir
    containing ``persistent/`` and ``ephemeral/`` subdirs plus a
    :class:`LocalJobResults` sink rooted at ``persistent/results/``.
    """
    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"
    persistent.mkdir(exist_ok=True)
    ephemeral.mkdir(exist_ok=True)
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
        results=LocalJobResults(root=persistent / "results"),
    )


@pytest.fixture(autouse=True)
def _released_contract_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """A source checkout reports a version the packaging guard rejects."""
    import nemo_agents_plugin.container.template as template

    monkeypatch.setattr(template, "get_contract_version", lambda: "1.0.0")
