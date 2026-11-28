# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenSandboxExecutorConfig."""

from __future__ import annotations

from typing import Any

import pytest
from nemo_deployments_plugin.backends.opensandbox.config import OpenSandboxExecutorConfig
from nemo_deployments_plugin.backends.opensandbox.endpoint import OpenSandboxEndpointTransport
from pydantic import ValidationError

VALID: dict[str, Any] = {
    "domain": "opensandbox-server.opensandbox-system.svc.cluster.local",
    "gateway_address": "opensandbox-ingress-gateway.opensandbox-system.svc.cluster.local:80",
}


def test_service_transport_uses_the_gateway_and_server_addresses() -> None:
    config = OpenSandboxExecutorConfig(**VALID, protocol="https", api_key_env="OSB_KEY")
    assert config.service_transport() == OpenSandboxEndpointTransport(
        connect_base="https://opensandbox-ingress-gateway.opensandbox-system.svc.cluster.local:80",
        server_base="https://opensandbox-server.opensandbox-system.svc.cluster.local",
        api_key_env="OSB_KEY",
    )


@pytest.mark.parametrize("address", ["localhost:8080", "10.96.0.12", "osb", "a.b-c.d:1"])
def test_host_port_forms_are_accepted(address: str) -> None:
    config = OpenSandboxExecutorConfig(domain=address, gateway_address=address)
    assert (config.domain, config.gateway_address) == (address, address)


@pytest.mark.parametrize(
    "override",
    [
        {"domain": "http://osb.svc"},
        {"domain": "osb.svc/v1"},
        {"domain": ""},
        {"domain": "osb.svc:port"},
        {"gateway_address": "http://gateway.svc"},
        {"gateway_address": "gateway.svc/prefix"},
        {"gateway_address": "gateway.svc?x=1"},
        {"gateway_address": "user@gateway.svc"},
        {"protocol": "grpc"},
        {"api_key_env": ""},
        {"api_key_env": "9KEY"},
        {"api_key_env": "OPEN-SANDBOX-KEY"},
        {"api_key": "plaintext"},
    ],
)
def test_invalid_settings_are_rejected(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        OpenSandboxExecutorConfig(**{**VALID, **override})
