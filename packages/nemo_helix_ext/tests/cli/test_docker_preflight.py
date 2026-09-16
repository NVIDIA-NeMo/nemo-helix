# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for config-aware Docker preflight (NVBug 6537617)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from nemo_helix_ext.cli.docker_preflight import (
    DOCKER_PREFLIGHT_MESSAGE,
    default_local_needs_docker,
    require_docker_for_default_local,
)
from nemo_helix_plugin.capabilities import ProbeResult
from nhx.platform_runner.config import PlatformAppConfig


@pytest.fixture
def stock_local_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "local.yaml"
    path.write_text(
        """
platform:
  runtime: docker
deployments:
  executors:
    - name: local-docker
      backend: docker
      config: {}
  default_executor: local-docker
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def k8s_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "k8s.yaml"
    path.write_text(
        """
platform:
  runtime: kubernetes
deployments:
  executors:
    - name: local-k8s
      backend: kubernetes
      config: {}
  default_executor: local-k8s
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def non_docker_default_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "subprocess.yaml"
    path.write_text(
        """
platform:
  runtime: docker
deployments:
  executors:
    - name: local-sandbox
      backend: sandbox
      config: {}
  default_executor: local-sandbox
""",
        encoding="utf-8",
    )
    return path


def test_default_local_needs_docker_for_stock_full_platform(stock_local_yaml: Path) -> None:
    cfg = PlatformAppConfig(config_path=str(stock_local_yaml))
    with patch(
        "nemo_helix_ext.cli.docker_preflight.resolve_run_configuration",
        return_value=MagicMock(
            services={"deployments", "entities"},
            controllers={"deployments"},
            config_path=str(stock_local_yaml),
        ),
    ):
        assert default_local_needs_docker(cfg) is True


def test_default_local_skips_when_deployments_not_selected(stock_local_yaml: Path) -> None:
    cfg = PlatformAppConfig(config_path=str(stock_local_yaml), services=["entities"], service_group=None)
    with patch(
        "nemo_helix_ext.cli.docker_preflight.resolve_run_configuration",
        return_value=MagicMock(
            services={"entities"},
            controllers=set(),
            config_path=str(stock_local_yaml),
        ),
    ):
        assert default_local_needs_docker(cfg) is False


def test_default_local_skips_kubernetes_runtime(k8s_yaml: Path) -> None:
    cfg = PlatformAppConfig(config_path=str(k8s_yaml))
    with patch(
        "nemo_helix_ext.cli.docker_preflight.resolve_run_configuration",
        return_value=MagicMock(
            services={"deployments"},
            controllers={"deployments"},
            config_path=str(k8s_yaml),
        ),
    ):
        assert default_local_needs_docker(cfg) is False


def test_default_local_skips_non_docker_default_executor(non_docker_default_yaml: Path) -> None:
    cfg = PlatformAppConfig(config_path=str(non_docker_default_yaml))
    with patch(
        "nemo_helix_ext.cli.docker_preflight.resolve_run_configuration",
        return_value=MagicMock(
            services={"deployments"},
            controllers=set(),
            config_path=str(non_docker_default_yaml),
        ),
    ):
        assert default_local_needs_docker(cfg) is False


def test_require_docker_exits_without_spawn_when_probe_false(stock_local_yaml: Path) -> None:
    cfg = PlatformAppConfig(config_path=str(stock_local_yaml))
    with (
        patch(
            "nemo_helix_ext.cli.docker_preflight.resolve_run_configuration",
            return_value=MagicMock(
                services={"deployments"},
                controllers={"deployments"},
                config_path=str(stock_local_yaml),
            ),
        ),
        patch(
            "nemo_helix_ext.cli.docker_preflight.probe_docker",
            return_value=ProbeResult(available=False, detail="down"),
        ) as probe,
        pytest.raises(typer.Exit) as exc,
    ):
        require_docker_for_default_local(cfg)
    assert exc.value.exit_code == 1
    probe.assert_called_once_with(docker_host=None, use_cache=False)


def test_require_docker_noop_when_probe_true(stock_local_yaml: Path) -> None:
    cfg = PlatformAppConfig(config_path=str(stock_local_yaml))
    with (
        patch(
            "nemo_helix_ext.cli.docker_preflight.resolve_run_configuration",
            return_value=MagicMock(
                services={"deployments"},
                controllers={"deployments"},
                config_path=str(stock_local_yaml),
            ),
        ),
        patch(
            "nemo_helix_ext.cli.docker_preflight.probe_docker",
            return_value=ProbeResult(available=True),
        ),
    ):
        require_docker_for_default_local(cfg)


def test_require_docker_probes_executor_docker_host(tmp_path: Path) -> None:
    path = tmp_path / "remote-docker.yaml"
    path.write_text(
        """
platform:
  runtime: docker
deployments:
  executors:
    - name: remote-docker
      backend: docker
      config:
        docker_host: tcp://docker.example:2375
  default_executor: remote-docker
""",
        encoding="utf-8",
    )
    cfg = PlatformAppConfig(config_path=str(path))
    with (
        patch(
            "nemo_helix_ext.cli.docker_preflight.resolve_run_configuration",
            return_value=MagicMock(
                services={"deployments"},
                controllers={"deployments"},
                config_path=str(path),
            ),
        ),
        patch(
            "nemo_helix_ext.cli.docker_preflight.probe_docker",
            return_value=ProbeResult(available=True),
        ) as probe,
    ):
        require_docker_for_default_local(cfg)
    probe.assert_called_once_with(docker_host="tcp://docker.example:2375", use_cache=False)


def test_preflight_message_names_docker() -> None:
    assert "Docker" in DOCKER_PREFLIGHT_MESSAGE
    assert "default local" in DOCKER_PREFLIGHT_MESSAGE.lower() or "default_executor" in DOCKER_PREFLIGHT_MESSAGE
