# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from nhx.guardrails.app.utils.cli_utils import get_args
from nhx.guardrails.config import GuardrailsServiceConfig


def test_configuration_loading():
    test_args = ["--config-store", "/test/config/path"]
    with patch("sys.argv", ["script_name"] + test_args):
        settings = GuardrailsServiceConfig()
        settings.configure(config_path=get_args().config_store)
        expected_config_path = "/test/config/path"
        actual_config_path = settings.config_sources[0]["config_path"]
        assert actual_config_path == expected_config_path, (
            f"Expected config path to be {expected_config_path}, but got {actual_config_path}."
        )


def test_error_handling_invalid_path():
    with patch("sys.argv", ["script_name", "--config-store", "/non/existent/path"]):
        settings = GuardrailsServiceConfig()
        settings.configure(config_path=get_args().config_store)
        assert settings.config_sources[0]["config_path"] == "/non/existent/path", (
            "Settings should handle invalid paths gracefully."
        )
