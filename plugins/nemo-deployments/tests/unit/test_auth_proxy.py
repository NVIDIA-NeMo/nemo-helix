# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auth-proxy sidecar compilation in the deployments plugin."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from nemo_deployments_plugin.auth_proxy import AUTH_PROXY_CONTAINER_NAME, build_auth_proxy_container
from nemo_deployments_plugin.entities import DeploymentConfig
from pydantic import ValidationError

_MOD = "nemo_deployments_plugin.auth_proxy"


def _config(**kwargs) -> DeploymentConfig:
    return DeploymentConfig(name="dep", workspace="default", **kwargs)


def test_no_sidecar_when_not_requested() -> None:
    with patch(f"{_MOD}.platform_auth_enabled", return_value=True):
        assert build_auth_proxy_container(_config(auth_proxy_sidecar=False)) is None


def test_no_sidecar_when_auth_disabled() -> None:
    # Requested but auth off -> no-op.
    with patch(f"{_MOD}.platform_auth_enabled", return_value=False):
        assert (
            build_auth_proxy_container(_config(auth_proxy_sidecar=True, auth_proxy_sidecar_identity="agents")) is None
        )


def test_builds_sidecar_when_requested_and_auth_on() -> None:
    with (
        patch(f"{_MOD}.platform_auth_enabled", return_value=True),
        patch(f"{_MOD}.get_qualified_image", return_value="my-registry/nhx-api:local"),
        patch(f"{_MOD}._upstream_base_url", return_value="http://nemo-helix-api:8080"),
    ):
        container = build_auth_proxy_container(_config(auth_proxy_sidecar=True, auth_proxy_sidecar_identity="agents"))
    assert container is not None
    assert container.name == AUTH_PROXY_CONTAINER_NAME
    assert container.image == "my-registry/nhx-api:local"
    assert container.command == ["nemo", "services", "run", "--sidecars", "auth-proxy"]
    assert container.restart_policy == "Always"
    env = {e.name: e.value for e in container.env}
    assert env["NHX_AUTH_PROXY_PRINCIPAL"] == "agents"
    assert env["NHX_BASE_URL"] == "http://nemo-helix-api:8080"
    # Loopback exec probe (proxy binds 127.0.0.1, so pod-IP httpGet would be refused).
    assert container.readiness_probe is not None
    assert container.readiness_probe.exec_action is not None
    assert "127.0.0.1" in " ".join(container.readiness_probe.exec_action.command)


def test_sidecar_stamps_on_behalf_of_when_set() -> None:
    with (
        patch(f"{_MOD}.platform_auth_enabled", return_value=True),
        patch(f"{_MOD}.get_qualified_image", return_value="my-registry/nhx-api:local"),
        patch(f"{_MOD}._upstream_base_url", return_value="http://nemo-helix-api:8080"),
    ):
        container = build_auth_proxy_container(
            _config(
                auth_proxy_sidecar=True,
                auth_proxy_sidecar_identity="agents",
                auth_proxy_sidecar_on_behalf_of="user:alice",
            )
        )
    assert container is not None
    env = {e.name: e.value for e in container.env}
    assert env["NHX_AUTH_PROXY_PRINCIPAL"] == "agents"
    assert env["NHX_AUTH_PROXY_ON_BEHALF_OF"] == "user:alice"


def test_sidecar_omits_on_behalf_of_when_unset() -> None:
    with (
        patch(f"{_MOD}.platform_auth_enabled", return_value=True),
        patch(f"{_MOD}.get_qualified_image", return_value="my-registry/nhx-api:local"),
        patch(f"{_MOD}._upstream_base_url", return_value="http://nemo-helix-api:8080"),
    ):
        container = build_auth_proxy_container(_config(auth_proxy_sidecar=True, auth_proxy_sidecar_identity="agents"))
    assert container is not None
    env = {e.name: e.value for e in container.env}
    assert "NHX_AUTH_PROXY_ON_BEHALF_OF" not in env


def test_sidecar_without_identity_is_rejected() -> None:
    # No default identity: a config requesting the sidecar without an identity
    # is invalid and fails validation (surfaced as a 4xx at the create endpoint).
    with pytest.raises(ValidationError, match="auth_proxy_sidecar_identity is required"):
        _config(auth_proxy_sidecar=True)
