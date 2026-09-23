# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-gate tests for the shared customization submit scaffold.

Covers :func:`require_container_runtime` (automodel / unsloth) and
:func:`require_distributed_runtime` (rl). We patch the platform config and the
Docker capability probe so the checks are exercised without a real runtime.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from nemo_helix_plugin.capabilities import ProbeResult
from nemo_helix_plugin.config import Runtime
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nhx.customization_common.contributor import jobs as jobs_mod
from nhx.customization_common.contributor.jobs import (
    require_container_runtime,
    require_distributed_runtime,
)


@pytest.fixture
def _patch_runtime(monkeypatch: pytest.MonkeyPatch):
    """Return a helper that pins the platform runtime and Docker availability."""

    def _apply(runtime: Runtime, *, docker_available: bool = True) -> None:
        monkeypatch.setattr(
            jobs_mod.NemoHelixConfig,
            "get",
            classmethod(lambda cls: SimpleNamespace(runtime=runtime)),
        )
        monkeypatch.setattr(
            jobs_mod,
            "probe_docker",
            lambda **kwargs: ProbeResult(
                available=docker_available,
                detail=None if docker_available else "Docker daemon unreachable (test)",
            ),
        )

    return _apply


class TestRequireContainerRuntime:
    def test_kubernetes_single_node_ok(self, _patch_runtime) -> None:
        _patch_runtime(Runtime.KUBERNETES)
        require_container_runtime("Automodel")  # no raise

    def test_kubernetes_multi_node_ok(self, _patch_runtime) -> None:
        _patch_runtime(Runtime.KUBERNETES)
        require_container_runtime("Automodel", num_nodes=2)  # no raise

    def test_docker_single_node_ok(self, _patch_runtime) -> None:
        _patch_runtime(Runtime.DOCKER, docker_available=True)
        require_container_runtime("Automodel")  # no raise

    def test_docker_daemon_unavailable_raises(self, _patch_runtime) -> None:
        _patch_runtime(Runtime.DOCKER, docker_available=False)
        with pytest.raises(HelixJobCompilationError, match="reachable Docker daemon"):
            require_container_runtime("Automodel")

    def test_docker_multi_node_requires_kubernetes(self, _patch_runtime) -> None:
        _patch_runtime(Runtime.DOCKER, docker_available=True)
        with pytest.raises(HelixJobCompilationError, match="multi-node training .* requires"):
            require_container_runtime("Automodel", num_nodes=2)

    def test_none_runtime_with_docker_ok(self, _patch_runtime) -> None:
        """Soft-downgraded NONE still allows compile when Docker is reachable."""
        _patch_runtime(Runtime.NONE, docker_available=True)
        require_container_runtime("Automodel")  # no raise

    def test_none_runtime_without_docker_raises(self, _patch_runtime) -> None:
        _patch_runtime(Runtime.NONE, docker_available=False)
        with pytest.raises(HelixJobCompilationError, match="reachable Docker daemon"):
            require_container_runtime("Automodel")

    def test_none_runtime_multi_node_raises_kubernetes_message(self, _patch_runtime) -> None:
        _patch_runtime(Runtime.NONE)
        with pytest.raises(HelixJobCompilationError, match="multi-node training .* requires"):
            require_container_runtime("Automodel", num_nodes=4)


class TestRequireDistributedRuntime:
    def test_kubernetes_ok(self, _patch_runtime) -> None:
        _patch_runtime(Runtime.KUBERNETES)
        require_distributed_runtime("NeMo-RL")  # no raise

    def test_docker_raises(self, _patch_runtime) -> None:
        _patch_runtime(Runtime.DOCKER)
        with pytest.raises(HelixJobCompilationError, match="requires platform.runtime: kubernetes"):
            require_distributed_runtime("NeMo-RL")
