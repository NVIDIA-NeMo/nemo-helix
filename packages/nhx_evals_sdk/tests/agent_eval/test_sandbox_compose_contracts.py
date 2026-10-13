# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Docker Compose sandbox provider."""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path

import pytest
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers import compose
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers.compose import (
    ComposeCleanupError,
    ComposeCommandResult,
    ComposeServiceTopology,
    DockerComposeSandboxProvider,
)

from packages.nhx_evals_sdk.tests.agent_eval._compose_testkit import _TOPOLOGY, _provider


def test_public_compatibility_surface_is_stable() -> None:
    """Keep the Compose provider's supported public imports stable."""
    assert compose.__all__ == [
        "ComposeCleanupError",
        "ComposeCommandResult",
        "ComposeServiceTopology",
        "ComposeTeardownContext",
        "DockerComposeSandboxProvider",
        "ProgressCallback",
        "PullPolicy",
        "TeardownHook",
    ]

    for cls in (
        ComposeCleanupError,
        ComposeCommandResult,
        ComposeServiceTopology,
        compose.ComposeTeardownContext,
        DockerComposeSandboxProvider,
    ):
        assert getattr(importlib.import_module(cls.__module__), cls.__name__) is cls


def test_public_value_types_pickle_round_trip() -> None:
    """Keep supported public value types pickle-compatible."""
    cleanup_error = ComposeCleanupError("teardown failed")
    restored_cleanup_error = pickle.loads(pickle.dumps(cleanup_error))
    assert type(restored_cleanup_error) is ComposeCleanupError
    assert restored_cleanup_error.args == cleanup_error.args

    command_result = ComposeCommandResult(("docker", "compose", "ps"), 0, "running\\n", "")
    restored_command_result = pickle.loads(pickle.dumps(command_result))
    assert restored_command_result == command_result

    topology = ComposeServiceTopology("agent", frozenset({"agent", "redis"}), frozenset({"init"}))
    restored_topology = pickle.loads(pickle.dumps(topology))
    assert restored_topology == topology


def test_invalid_configuration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        DockerComposeSandboxProvider(compose_files=(), service_topology=_TOPOLOGY)
    with pytest.raises(ValueError, match="pull_policy"):
        _provider(tmp_path, pull_policy="sometimes")
    with pytest.raises(ValueError, match="target_service"):
        ComposeServiceTopology(
            target_service="agent",
            long_running_services=frozenset({"redis"}),
        )


@pytest.mark.parametrize("project_name", ["a", "0", "a-b", "a_b", "a0-b_1"])
def test_valid_project_names_are_accepted(tmp_path: Path, project_name: str) -> None:
    assert _provider(tmp_path, project_name=project_name).project_name == project_name


@pytest.mark.parametrize("project_name", ["foo.bar", "Upper", "bad/name"])
def test_invalid_project_names_are_rejected(tmp_path: Path, project_name: str) -> None:
    with pytest.raises(ValueError, match="project_name"):
        _provider(tmp_path, project_name=project_name)
