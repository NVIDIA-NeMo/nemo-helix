# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from urllib.parse import urlunsplit

from nhx.common.config import Runtime
from nhx.core.models.controllers.backends.deployments_plugin.resolve import rewrite_loopback_for_docker_container


def test_rewrites_loopback_url_for_docker_with_configured_address() -> None:
    result = rewrite_loopback_for_docker_container(
        "http://localhost:32827/apis/files/v2/hf",
        runtime=Runtime.DOCKER,
        loopback_address="host.docker.internal",
    )

    assert result == "http://host.docker.internal:32827/apis/files/v2/hf"


def test_rewrites_ipv6_loopback_with_valid_bracket_formatting() -> None:
    result = rewrite_loopback_for_docker_container(
        "http://[::1]:32827/apis/files/v2/hf",
        runtime=Runtime.DOCKER,
        loopback_address="fd00::10",
    )

    assert result == "http://[fd00::10]:32827/apis/files/v2/hf"


def test_preserves_url_when_loopback_text_is_not_hostname() -> None:
    url = "http://api.localhost.example:32827/apis/files/v2/hf?next=http://127.0.0.1:8080"

    result = rewrite_loopback_for_docker_container(
        url,
        runtime=Runtime.DOCKER,
        loopback_address="host.docker.internal",
    )

    assert result == url


def test_preserves_loopback_url_for_kubernetes() -> None:
    url = "http://localhost:32827/apis/files/v2/hf"

    result = rewrite_loopback_for_docker_container(
        url,
        runtime=Runtime.KUBERNETES,
        loopback_address="host.docker.internal",
    )

    assert result == url


def test_uses_detected_loopback_override_when_config_is_unset() -> None:
    with patch(
        "nhx.core.models.controllers.backends.deployments_plugin.resolve.determine_loopback_override",
        return_value="nemo-gateway",
    ):
        result = rewrite_loopback_for_docker_container(
            "http://127.0.0.1:32827/apis/files/v2/hf",
            runtime=Runtime.DOCKER,
            loopback_address=None,
        )

    assert result == "http://nemo-gateway:32827/apis/files/v2/hf"


def test_falls_back_to_host_docker_internal_for_docker_loopback() -> None:
    with patch(
        "nhx.core.models.controllers.backends.deployments_plugin.resolve.determine_loopback_override",
        return_value=None,
    ):
        result = rewrite_loopback_for_docker_container(
            "http://localhost:32827/apis/files/v2/hf",
            runtime=Runtime.DOCKER,
            loopback_address=None,
        )

    assert result == "http://host.docker.internal:32827/apis/files/v2/hf"


def test_rewrites_loopback_hostname_with_userinfo_from_parsed_url() -> None:
    url = urlunsplit(("http", "user:pass@localhost:32827", "/apis/files/v2/hf", "", ""))
    expected = urlunsplit(("http", "user:pass@host.docker.internal:32827", "/apis/files/v2/hf", "", ""))

    result = rewrite_loopback_for_docker_container(
        url,
        runtime=Runtime.DOCKER,
        loopback_address="host.docker.internal",
    )

    assert result == expected
