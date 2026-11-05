# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
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
def test_native_opa_checks_installed_sources_without_starting_server(dockerfile: str) -> None:
    contents = (REPO_ROOT / dockerfile).read_text()
    image = re.search(r"^FROM openpolicyagent/opa:([\d.]+)-static@sha256:[0-9a-f]{64} AS opa$", contents, re.MULTILINE)
    assert image is not None, "Use the digest-pinned official multi-platform image"
    resolver = REPO_ROOT / "services/core/auth/src/nhx/core/auth/app/embedded_pdp/opa_binary.py"
    assert f'OPA_VERSION = "{image.group(1)}"' in resolver.read_text()
    # Provision both the source-validation stage and the final runtime image.
    assert contents.count("COPY --from=opa /opa /usr/local/bin/opa") == 2
    workspace_install = contents.index("uv sync --frozen")
    policy_validation = contents.index("RUN /usr/local/bin/opa check ")
    assert workspace_install < policy_validation
    assert "get_policy()" not in contents
    assert "opa run" not in contents
