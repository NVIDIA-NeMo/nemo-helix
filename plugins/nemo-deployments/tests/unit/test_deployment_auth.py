# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

from nemo_deployments_plugin.deployment_auth import plan_deployment_auth
from nemo_helix_plugin.auth import AuthContext

_MOD = "nemo_deployments_plugin.deployment_auth"


def _auth_context() -> AuthContext:
    return AuthContext(principal_id="user:alice", principal_groups=["research"])


def test_auth_off_omits_auth_proxy_but_preserves_workload_identity_request() -> None:
    with (
        patch(f"{_MOD}.platform_auth_enabled", return_value=False),
        patch(f"{_MOD}.is_workload_identity_token_exchange_enabled", return_value=True),
    ):
        plan = plan_deployment_auth(
            service_identity="agents",
            on_behalf_of="user:alice",
            auth_context=_auth_context(),
            workload_name="hello-dep",
            workload_label="Agent deployment",
        )

    assert plan.auth_proxy_base_url is None
    assert plan.auth_proxy_identity is None
    assert plan.auth_proxy_on_behalf_of is None
    assert plan.workload_identity_enabled is True
    assert plan.error is None


def test_trusted_header_auth_uses_creator_on_behalf_of() -> None:
    with (
        patch(f"{_MOD}.platform_auth_enabled", return_value=True),
        patch(f"{_MOD}.is_workload_identity_token_exchange_enabled", return_value=False),
        patch(f"{_MOD}.auth_proxy_port", return_value=8090),
    ):
        plan = plan_deployment_auth(
            service_identity="agents",
            on_behalf_of="user:alice",
            auth_context=None,
            workload_name="hello-dep",
            workload_label="Agent deployment",
        )

    assert plan.auth_proxy_base_url == "http://127.0.0.1:8090"
    assert plan.auth_proxy_identity == "agents"
    assert plan.auth_proxy_on_behalf_of == "user:alice"
    assert plan.workload_identity_enabled is False
    assert plan.error is None


def test_token_exchange_requires_auth_context_when_platform_auth_is_on() -> None:
    with (
        patch(f"{_MOD}.platform_auth_enabled", return_value=True),
        patch(f"{_MOD}.is_workload_identity_token_exchange_enabled", return_value=True),
    ):
        plan = plan_deployment_auth(
            service_identity="agents",
            on_behalf_of="user:alice",
            auth_context=None,
            workload_name="hello-dep",
            workload_label="Agent deployment",
        )

    assert plan.error is not None
    assert "creator auth context" in plan.error
    assert plan.auth_proxy_identity is None
    assert plan.workload_identity_enabled is False


def test_token_exchange_uses_workload_identity_without_trusted_on_behalf_of() -> None:
    with (
        patch(f"{_MOD}.platform_auth_enabled", return_value=True),
        patch(f"{_MOD}.is_workload_identity_token_exchange_enabled", return_value=True),
        patch(f"{_MOD}.auth_proxy_port", return_value=8090),
    ):
        plan = plan_deployment_auth(
            service_identity="agents",
            on_behalf_of="user:alice",
            auth_context=_auth_context(),
            workload_name="hello-dep",
            workload_label="Agent deployment",
        )

    assert plan.auth_proxy_base_url == "http://127.0.0.1:8090"
    assert plan.auth_proxy_identity == "agents"
    assert plan.auth_proxy_on_behalf_of is None
    assert plan.workload_identity_enabled is True
    assert plan.error is None
