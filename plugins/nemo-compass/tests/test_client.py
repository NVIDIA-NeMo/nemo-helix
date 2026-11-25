# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from nemo_compass_plugin.platform_client import make_client

REMOTE_URL = "https://nemo-helix.example.com"


def test_remote_uses_local_oauth_context() -> None:
    config_path = MagicMock()
    config_path.exists.return_value = True

    with (
        patch("nemo_compass_plugin.platform_client.Config.get_default_config_path", return_value=config_path),
        patch("nemo_compass_plugin.platform_client.build_async_nemo_client") as build,
    ):
        client = make_client(REMOTE_URL)

    build.assert_called_once_with(base_url=REMOTE_URL, config_path=config_path)
    assert client is build.return_value


def test_remote_rejects_http_before_credential_bootstrap() -> None:
    config_path = MagicMock()
    config_path.exists.return_value = True
    base_url = "http://nemo-helix.example.com"

    with (
        patch("nemo_compass_plugin.platform_client.Config.get_default_config_path", return_value=config_path),
        patch("nemo_compass_plugin.platform_client.build_async_nemo_client") as build,
        pytest.raises(ValueError, match="non-HTTPS remote URL"),
    ):
        make_client(base_url)

    build.assert_not_called()


def test_remote_http_without_config_uses_direct_client() -> None:
    config_path = MagicMock()
    config_path.exists.return_value = False
    base_url = "http://nemo-helix.example.com"

    with (
        patch("nemo_compass_plugin.platform_client.Config.get_default_config_path", return_value=config_path),
        patch("nemo_compass_plugin.platform_client.build_direct_async_nemo_client") as build_direct,
    ):
        client = make_client(base_url)

    build_direct.assert_called_once_with(base_url=base_url)
    assert client is build_direct.return_value
