# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the docker DeploymentConfig -> orchestration plan builder."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from backends.docker.docker_helpers import lora_config, sample_config
from nemo_deployments_plugin.backends.docker.containers import (
    DeploymentConfigError,
    build_docker_plan,
    validate_config_for_docker,
)
from nemo_deployments_plugin.entities import Container, ContainerPort, DeploymentConfig
from nemo_helix_plugin.auth import AuthContext

_AUTH_PROXY_MOD = "nemo_deployments_plugin.auth_proxy"


def test_single_container_plan_has_no_init_or_sidecars() -> None:
    plan = build_docker_plan(sample_config())
    assert plan.primary.name == "main"
    assert plan.init_containers == []
    assert plan.sidecars == []
    assert plan.is_multi_container is False


def test_auth_proxy_injected_as_docker_sidecar_when_auth_on() -> None:
    config = sample_config()
    config = config.model_copy(update={"auth_proxy_sidecar": True, "auth_proxy_sidecar_identity": "agents"})
    with (
        patch(f"{_AUTH_PROXY_MOD}.platform_auth_enabled", return_value=True),
        patch(f"{_AUTH_PROXY_MOD}.get_qualified_image", return_value="my-registry/nhx-api:local"),
        patch(f"{_AUTH_PROXY_MOD}._upstream_base_url", return_value="http://host.docker.internal:8080"),
    ):
        plan = build_docker_plan(config, auth_context=AuthContext(principal_id="user:alice"))
    # Injected as a sidecar (shares primary netns), not the primary.
    assert plan.primary.name == "main"
    proxy = next(c for c in plan.sidecars if c.name == "auth-proxy")
    env = {e.name: e.value for e in proxy.env}
    assert env["NHX_AUTH_PROXY_PRINCIPAL"] == "agents"
    assert env["NHX_BASE_URL"] == "http://host.docker.internal:8080"
    # Sidecar must not declare ports (shares netns).
    assert proxy.ports == []


def test_auth_proxy_not_injected_when_auth_off() -> None:
    config = sample_config().model_copy(update={"auth_proxy_sidecar": True, "auth_proxy_sidecar_identity": "agents"})
    with patch(f"{_AUTH_PROXY_MOD}.platform_auth_enabled", return_value=False):
        plan = build_docker_plan(config)
    assert plan.sidecars == []


def test_docker_plan_preserves_captured_creator_context() -> None:
    creator = AuthContext(principal_id="user:alice", principal_groups=["research"])
    config = sample_config().model_copy(
        update={
            "auth_proxy_sidecar": True,
            "auth_proxy_sidecar_identity": "models",
            "auth_proxy_sidecar_on_behalf_of": "user:alice",
        }
    )
    with patch(f"{_AUTH_PROXY_MOD}.platform_auth_enabled", return_value=True):
        plan = build_docker_plan(config, auth_context=creator)

    proxy = next(container for container in plan.sidecars if container.name == "auth-proxy")
    env = {item.name: item.value for item in proxy.env}
    raw_context = env["NHX_AUTH_PROXY_AUTH_CONTEXT"]
    assert raw_context is not None
    assert AuthContext.model_validate_json(raw_context) == creator


def test_auth_proxy_docker_upstream_rewrites_loopback() -> None:
    # docker=True + loopback base_url -> host.docker.internal substitution.
    # _upstream_base_url imports these lazily from nemo_helix_plugin.config.
    from nemo_deployments_plugin.auth_proxy import _upstream_base_url

    with (
        patch("nemo_helix_plugin.config.determine_loopback_override", return_value="host.docker.internal"),
        patch("nemo_helix_plugin.config.get_platform_config") as get_cfg,
    ):
        get_cfg.return_value.base_url = "http://localhost:8080"
        assert _upstream_base_url(docker=True) == "http://host.docker.internal:8080"
        # k8s path leaves it verbatim.
        assert _upstream_base_url(docker=False) == "http://localhost:8080"


def test_lora_plan_splits_init_primary_and_sidecar() -> None:
    plan = build_docker_plan(lora_config())
    assert plan.primary.name == "server"
    assert [c.name for c in plan.init_containers] == ["lora-cache-init"]
    assert [c.name for c in plan.sidecars] == ["lora-adapters"]
    assert plan.is_multi_container is True


def test_plan_rejects_sidecar_with_ports() -> None:
    config = DeploymentConfig(
        name="cfg1",
        workspace="default",
        containers=[
            Container(name="server", image="img", ports=[ContainerPort(name="http", containerPort=8000)]),
            Container(name="sidecar", image="img2", ports=[ContainerPort(name="x", containerPort=9000)]),
        ],
    )
    with pytest.raises(DeploymentConfigError, match="may not declare ports"):
        build_docker_plan(config)


def test_plan_rejects_empty_config() -> None:
    config = DeploymentConfig(name="cfg1", workspace="default", containers=[])
    with pytest.raises(DeploymentConfigError, match="at least one container"):
        build_docker_plan(config)


def test_validate_config_returns_primary_for_back_compat() -> None:
    assert validate_config_for_docker(sample_config()).name == "main"
    assert validate_config_for_docker(lora_config()).name == "server"
