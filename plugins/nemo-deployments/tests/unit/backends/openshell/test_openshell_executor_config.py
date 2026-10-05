# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenShellExecutorConfig."""

from __future__ import annotations

import pytest
from nemo_deployments_plugin.backends.openshell.config import OpenShellExecutorConfig
from pydantic import ValidationError


def test_defaults_target_local_plaintext() -> None:
    config = OpenShellExecutorConfig()
    assert config.grpc_target() == "127.0.0.1:17670"
    assert config.use_insecure() is True


def test_https_endpoint_implies_tls() -> None:
    config = OpenShellExecutorConfig(gateway_endpoint="https://gw.example:443")
    assert config.grpc_target() == "gw.example:443"
    assert config.use_insecure() is False


def test_insecure_override_forces_plaintext() -> None:
    config = OpenShellExecutorConfig(gateway_endpoint="https://gw.example:443", insecure=True)
    assert config.use_insecure() is True


def test_invalid_endpoint_rejected() -> None:
    with pytest.raises(ValidationError):
        OpenShellExecutorConfig(gateway_endpoint="not-a-url")


def test_platform_egress_defaults_to_local_docker() -> None:
    config = OpenShellExecutorConfig()
    assert config.platform_egress is not None
    assert config.platform_egress.host == "host.docker.internal"
    assert config.platform_egress.port == 8080
    assert config.platform_egress.binaries == []  # empty -> backend uses its default set


def test_platform_egress_is_overridable() -> None:
    config = OpenShellExecutorConfig.model_validate(
        {"platform_egress": {"host": "10.0.0.5", "port": 9000, "binaries": ["/opt/py/bin/python3.13"]}}
    )
    assert config.platform_egress is not None
    assert config.platform_egress.host == "10.0.0.5"
    assert config.platform_egress.port == 9000
    assert config.platform_egress.binaries == ["/opt/py/bin/python3.13"]


def test_platform_egress_can_be_disabled() -> None:
    # null platform_egress -> the sandbox gets no direct egress at all, correct for
    # gateway-managed inference (inference.local), which the supervisor brokers outside
    # the policy's egress rules.
    config = OpenShellExecutorConfig(platform_egress=None)
    assert config.platform_egress is None


def test_unknown_config_key_is_rejected() -> None:
    # extra="forbid": a typo'd key (platform_egres) becomes a loud error instead of silently
    # falling back to the default egress, which would be a security-relevant divergence.
    with pytest.raises(ValidationError):
        OpenShellExecutorConfig.model_validate({"gateway_endpoint": "http://127.0.0.1:17670", "platform_egres": None})


def test_platform_egress_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        OpenShellExecutorConfig.model_validate({"platform_egress": {"host": "h", "prtocol": "rest"}})


def test_platform_egress_accepts_proto_value_sets() -> None:
    # Value sets follow OpenShell's authored policy schema, not the shorter summaries:
    # "graphql"/"" protocol and "skip" tls are valid and must not be rejected.
    config = OpenShellExecutorConfig.model_validate(
        {"platform_egress": {"protocol": "graphql", "tls": "skip", "access": "read-only"}}
    )
    assert config.platform_egress is not None
    assert (config.platform_egress.protocol, config.platform_egress.tls, config.platform_egress.access) == (
        "graphql",
        "skip",
        "read-only",
    )


@pytest.mark.parametrize("tls", ["terminate", "passthrough"])
def test_platform_egress_rejects_tls_modes_openshell_removed(tls: str) -> None:
    with pytest.raises(ValidationError):
        OpenShellExecutorConfig.model_validate({"platform_egress": {"tls": tls}})


def test_platform_egress_rejects_invalid_protocol() -> None:
    with pytest.raises(ValidationError):
        OpenShellExecutorConfig.model_validate({"platform_egress": {"protocol": "ftp"}})


def test_serve_path_defaults_to_venv_bin_prepended_to_system_path() -> None:
    config = OpenShellExecutorConfig()
    assert config.serve_path == "/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def test_serve_path_is_overridable() -> None:
    config = OpenShellExecutorConfig.model_validate({"serve_path": "/opt/bin:/usr/bin"})
    assert config.serve_path == "/opt/bin:/usr/bin"


def test_serve_virtual_env_defaults_to_the_packaged_venv() -> None:
    config = OpenShellExecutorConfig.model_validate({})
    assert config.serve_virtual_env == "/workspace/.venv"


def test_serve_virtual_env_is_overridable_and_can_be_disabled() -> None:
    assert OpenShellExecutorConfig.model_validate({"serve_virtual_env": "/opt/venv"}).serve_virtual_env == "/opt/venv"
    assert OpenShellExecutorConfig.model_validate({"serve_virtual_env": ""}).serve_virtual_env == ""
