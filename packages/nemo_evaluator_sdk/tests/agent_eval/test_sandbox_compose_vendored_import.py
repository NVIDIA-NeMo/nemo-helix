# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import compatibility tests for the vendored Compose sandbox provider."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_vendored_compose_public_imports_are_constructible_without_docker(tmp_path: Path) -> None:
    """The vendored public Compose façade remains importable without Docker."""
    compose = importlib.import_module("nemo_helix.beta.evaluator.agent_eval.runtimes.sandbox.providers.compose")
    ComposeCleanupError = compose.ComposeCleanupError
    ComposeCommandResult = compose.ComposeCommandResult
    ComposeServiceTopology = compose.ComposeServiceTopology
    ComposeTeardownContext = compose.ComposeTeardownContext
    DockerComposeSandboxProvider = compose.DockerComposeSandboxProvider

    for cls in (
        ComposeCleanupError,
        ComposeCommandResult,
        ComposeServiceTopology,
        ComposeTeardownContext,
        DockerComposeSandboxProvider,
    ):
        assert getattr(importlib.import_module(cls.__module__), cls.__name__) is cls

    assert compose.ProgressCallback is not None
    assert compose.PullPolicy is not None
    assert compose.TeardownHook is not None

    topology = ComposeServiceTopology("agent", frozenset({"agent"}))
    command_result = ComposeCommandResult(("docker", "compose", "ps"), 0, "", "")
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text("services: {}\\n", encoding="utf-8")
    provider = DockerComposeSandboxProvider(
        compose_files=(compose_file,),
        service_topology=topology,
        lock_path=tmp_path / "compose.lock",
    )

    assert command_result.ok
    assert provider.service_topology is topology
