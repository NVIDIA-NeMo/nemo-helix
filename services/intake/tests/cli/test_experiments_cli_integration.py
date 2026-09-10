# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``nemo experiments`` against the in-process Intake service.

Experiments live in the entity store, so the ASGI app runs without ClickHouse;
the ClickHouse-backed span routes are covered by the wire-level tests instead.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_plugin.client.client import NemoClient
from nmp.intake.api.v2.experiments.dependencies import get_evaluation_rollup_repository
from nmp.intake.cli import ExperimentsCLI
from nmp.intake.config import ClickHouseConfig, IntakeConfig
from nmp.intake.service import IntakeService
from nmp.testing import SDKTestClientAdapter, create_test_client
from typer.testing import CliRunner

app = ExperimentsCLI().get_cli()


@pytest.fixture(scope="module")
def asgi_client() -> Iterator[NemoClient]:
    """Typed client whose transport is the in-process Intake ASGI app."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IntakeService, "is_ready", AsyncMock(return_value=True))
        intake_config = IntakeConfig(clickhouse_config=ClickHouseConfig(url="http://127.0.0.1:1"))
        with create_test_client(
            IntakeService,
            client_type=TestClient,
            dependency_overrides={get_evaluation_rollup_repository: lambda: None},
            service_configs={IntakeService: intake_config},
        ) as test_client:
            yield NemoClient(
                base_url=str(test_client.base_url), workspace="default", http_client=SDKTestClientAdapter(test_client)
            )


@pytest.fixture
def run(asgi_client: NemoClient):
    def _run(args: list[str], **kwargs):
        state = CLIContext(overrides={"base_url": "http://test", "output_format": "json"}, _client=asgi_client)
        return CliRunner().invoke(app, args, obj=state, **kwargs)

    return _run


def _name() -> str:
    return f"exp-{uuid.uuid4().hex[:8]}"


def test_experiments_lifecycle(run) -> None:
    name = _name()

    result = run(["create", name, "--description", "demo", "--metadata", '{"team": "core"}'])
    assert result.exit_code == 0, result.output
    created = json.loads(result.stdout)
    assert created["name"] == name
    assert created["description"] == "demo"
    assert created["metadata"] == {"team": "core"}
    assert created["default_sort"] == "-created_at"

    result = run(["get", name])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == created["id"]

    result = run(["list", "--filter.name", name])
    assert result.exit_code == 0, result.output
    listed = json.loads(result.stdout)
    assert [item["name"] for item in listed["data"]] == [name]
    assert listed["pagination"]["total_results"] == 1

    result = run(["update", name, "--body-name", name, "--summary", "done", "--is-favorite"])
    assert result.exit_code == 0, result.output
    updated = json.loads(result.stdout)
    assert updated["summary"] == "done"
    assert updated["is_favorite"] is True
    assert updated["description"] == "demo"

    result = run(["create", name, "--exist-ok"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == created["id"]

    result = run(["create", name])
    assert result.exit_code == 3
    assert "Conflict" in result.stderr

    result = run(["delete", name])
    assert result.exit_code == 0, result.output
    assert "Deleted successfully" in result.stdout

    result = run(["get", name])
    assert result.exit_code == 3
    assert "Not found" in result.stderr


def test_experiments_update_rename_is_rejected(run) -> None:
    name = _name()
    assert run(["create", name]).exit_code == 0

    result = run(["update", name, "--body-name", f"{name}-renamed"])

    assert result.exit_code == 3
    assert "Conflict: (409)" in result.stderr


def test_experiments_list_all_pages(run) -> None:
    prefix = _name()
    names = sorted(f"{prefix}-{i}" for i in range(3))
    for name in names:
        assert run(["create", name, "--insight-id", prefix]).exit_code == 0

    result = run(["list", "--filter.insight-id", prefix, "--page-size", "1", "--sort", "name"])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["data"]) == 1
    assert "More pages" in result.stderr

    result = run(["list", "--filter.insight-id", prefix, "--page-size", "1", "--sort", "name", "--all-pages"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == names
    assert body["pagination"]["total_results"] == 3
    assert "More pages" not in result.stderr
