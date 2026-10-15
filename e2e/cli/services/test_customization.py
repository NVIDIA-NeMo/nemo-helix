# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for customization CLI commands."""

from __future__ import annotations

import json

import pytest
from nhx.testing import NemoRun, assert_exit_0

pytestmark = [pytest.mark.timeout(600)]


@pytest.mark.parametrize(
    ("customization_backend", "endpoint_suffix"),
    [
        ("automodel", "/automodel/jobs"),
        ("unsloth", "/unsloth/jobs"),
    ],
)
def test_customization_backend_cli_explain(
    nemo_run: NemoRun,
    customization_backend: str,
    endpoint_suffix: str,
) -> None:
    """Verify plugin backend CLIs are mounted under the customization root."""
    help_result = nemo_run("customization", "--help")
    assert_exit_0(help_result, "customization help failed")
    if customization_backend not in help_result.stdout:
        pytest.skip(f"customization {customization_backend} CLI is not available in this build")

    result = nemo_run("customization", customization_backend, "explain")
    assert_exit_0(result, f"customization {customization_backend} explain failed")
    payload = json.loads(result.stdout)
    assert "input_spec_schema" in payload
    assert "spec_schema" in payload
    assert payload["endpoint"].endswith(endpoint_suffix)
