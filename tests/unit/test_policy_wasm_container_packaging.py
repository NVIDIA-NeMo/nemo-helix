# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_WASM_DESTINATION = "/app/services/core/auth/src/nhx/core/auth/assets/policy.wasm"


@pytest.mark.parametrize(
    "dockerfile",
    [
        "docker/Dockerfile.nhx-api",
        "docker/Dockerfile.nhx-core",
    ],
)
def test_policy_wasm_is_added_before_workspace_install(dockerfile: str) -> None:
    contents = (REPO_ROOT / dockerfile).read_text()

    policy_copy = contents.index(f"COPY --from=policy-wasm-artifacts /artifacts/policy.wasm {POLICY_WASM_DESTINATION}")
    workspace_install = contents.index("uv sync --frozen")

    assert policy_copy < workspace_install
    assert "site-packages/nhx/core/auth/assets/policy.wasm" not in contents
    assert "ensure_embedded_policy_wasm(auto_build=False)" in contents


@pytest.mark.parametrize(
    "dockerfile",
    [
        "docker/Dockerfile.nhx-api",
        "docker/Dockerfile.nhx-core",
    ],
)
def test_policy_wasm_is_validated_with_installed_venv_after_workspace_install(dockerfile: str) -> None:
    contents = (REPO_ROOT / dockerfile).read_text()

    workspace_install = contents.index("uv sync --frozen")
    policy_validation = contents.index(
        "/app/.venv/bin/python -c 'from nhx.core.auth.app.embedded_pdp.policy_wasm "
        "import ensure_embedded_policy_wasm; ensure_embedded_policy_wasm(auto_build=False)'"
    )

    assert workspace_install < policy_validation
