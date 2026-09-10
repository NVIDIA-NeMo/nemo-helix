# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.types import FilesetOutput
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults


@pytest.fixture
def ctx(tmp_path: Path) -> JobContext:
    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"
    persistent.mkdir()
    ephemeral.mkdir()
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
        results=LocalJobResults(root=persistent / "results"),
    )


def _reject_request(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected platform request: {request.method} {request.url}")


@pytest.fixture
def make_platform_client() -> Callable[..., NemoClient]:
    """Build a :class:`NemoClient` over ``httpx.MockTransport`` standing in for the scheduler's ``sdk``.

    Without a handler every request fails loudly, so tests that only expect
    fileset transfers (recorded by ``fake_transfers``) catch stray platform calls.
    """

    def make(handler: Callable[[httpx.Request], httpx.Response] = _reject_request) -> NemoClient:
        return NemoClient(
            base_url="http://test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test"),
        )

    return make


@dataclass
class FakeFilesetTransfers:
    """Stand-in for the ``filesets.transfer`` helpers ``nemo_agents_plugin.jobs.fileset_io`` drives.

    Records download/upload calls with their typed Files client and keyword
    arguments; ``on_download`` materializes files at ``local_path``.
    """

    downloads: list[dict[str, Any]] = field(default_factory=list)
    uploads: list[dict[str, Any]] = field(default_factory=list)
    on_download: Callable[[Path, str | None, str | None], None] | None = None

    def download(self, client: FilesClient, **kwargs: Any) -> None:
        assert isinstance(client, FilesClient), client
        self.downloads.append({"client": client, **kwargs})
        if self.on_download is not None:
            self.on_download(Path(kwargs["local_path"]), kwargs.get("fileset"), kwargs.get("workspace"))

    def upload(self, client: FilesClient, **kwargs: Any) -> FilesetOutput:
        assert isinstance(client, FilesClient), client
        self.uploads.append({"client": client, **kwargs})
        name = kwargs.get("fileset") or "fileset-auto"
        return FilesetOutput.model_validate(
            {
                "id": f"id-{name}",
                "name": name,
                "workspace": kwargs.get("workspace") or "default",
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


@pytest.fixture
def fake_transfers(monkeypatch: pytest.MonkeyPatch) -> FakeFilesetTransfers:
    import nemo_agents_plugin.jobs.fileset_io as fileset_io

    fake = FakeFilesetTransfers()
    monkeypatch.setattr(fileset_io, "download", fake.download)
    monkeypatch.setattr(fileset_io, "upload", fake.upload)
    return fake
