# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "dockerfile",
    [
        "docker/Dockerfile.nhx-api",
        "docker/Dockerfile.nhx-core",
    ],
)
def test_embedded_policy_requires_no_build_artifact(dockerfile: str) -> None:
    contents = (REPO_ROOT / dockerfile).read_text()

    assert "policy-wasm" not in contents
    assert "policy.wasm" not in contents
    assert "ensure_embedded_policy_wasm" not in contents


@pytest.mark.parametrize(
    "dockerfile",
    [
        "docker/Dockerfile.nhx-api",
        "docker/Dockerfile.nhx-core",
    ],
)
def test_rego_policy_is_compiled_with_installed_venv_after_workspace_install(dockerfile: str) -> None:
    contents = (REPO_ROOT / dockerfile).read_text()

    workspace_install = contents.index("uv sync --frozen")
    policy_validation = contents.index(
        "/app/.venv/bin/python -c 'from nhx.core.auth.app.embedded_pdp import get_policy; get_policy().set_data({})'"
    )

    assert workspace_install < policy_validation
