# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-entrypoint test state; client factories remain real."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client_provider import DefaultNemoClientProvider, set_nemo_client_provider
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nhx.common.config import AuthConfig, Configuration, HelixConfig

Mode = Literal["headers", "token-file"]


@dataclass
class TaskCapture:
    ctx: JobContext
    config_path: Path
    token_path: Path
    clients: list[NemoClient] = field(default_factory=list)

    def context(self, client: NemoClient) -> JobContext:
        self.clients.append(client)
        return self.ctx


@contextmanager
def task_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: Mode) -> Generator[TaskCapture]:
    for variable in ("NHX_ACCESS_TOKEN", "NHX_ACCESS_TOKEN_FILE", "NHX_WORKLOAD_IDENTITY_TOKEN_FILE"):
        monkeypatch.delenv(variable, raising=False)
    # A service-created job must delegate the original user, including group/account grants.
    monkeypatch.setenv(
        "NHX_PRINCIPAL",
        json.dumps(
            {
                "id": "service:jobs",
                "on_behalf_of": "creator",
                "on_behalf_of_email": "creator@example.test",
                "on_behalf_of_groups": ["readers"],
                "on_behalf_of_account_id": "creator-account",
                "on_behalf_of_authz_aliases": ["creator", "creator@example.test"],
            }
        ),
    )
    monkeypatch.setenv("NHX_BASE_URL", "https://localhost:8080")
    config_path = tmp_path / "config.json"
    monkeypatch.setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", str(config_path))
    token_path = tmp_path / "access-token"
    if mode == "token-file":
        token_path.write_text("first-task-token")
        monkeypatch.setenv("NHX_ACCESS_TOKEN_FILE", str(token_path))
    set_nemo_client_provider(DefaultNemoClientProvider())
    Configuration.set_override(HelixConfig(base_url="https://localhost:8080", services=""))
    Configuration.set_override(AuthConfig(enabled=False))
    ctx = JobContext(
        workspace="allowed",
        storage=StoragePaths(ephemeral=tmp_path, persistent=tmp_path),
        results=LocalJobResults(root=tmp_path / "results"),
    )
    capture = TaskCapture(ctx, config_path, token_path)
    try:
        yield capture
    finally:
        for client in capture.clients:
            client.close()
        set_nemo_client_provider(None)
        Configuration.clear_override(AuthConfig)
        Configuration.clear_override(HelixConfig)


def assert_task_identity(capture: TaskCapture, mode: Mode, service: str) -> None:
    assert len(capture.clients) == 1
    client = capture.clients[0]
    headers = client.default_headers
    assert headers["X-NHX-Internal"] == "true"
    if mode == "headers":
        assert headers["X-NHX-Principal-Id"] == f"service:{service}"
        assert headers["X-NHX-Principal-On-Behalf-Of"] == "creator"
        assert headers["X-NHX-Principal-On-Behalf-Of-Email"] == "creator@example.test"
        assert headers["X-NHX-Principal-On-Behalf-Of-Groups"] == "readers"
        assert headers["X-NHX-Subject-Account-Id"] == "creator-account"
        assert headers["X-NHX-Subject-Aliases"] == "creator,creator@example.test"
    else:
        assert not any(name.lower().startswith("x-nhx-principal") for name in headers)
        assert client._auth is not None
        assert client._auth.get_access_token() == "first-task-token"
        capture.token_path.write_text("rotated-task-token")
        assert client._auth.get_access_token() == "rotated-task-token"
