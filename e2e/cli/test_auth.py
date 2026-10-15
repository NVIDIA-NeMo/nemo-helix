# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for auth CLI commands.

Tests NeMo CLI auth status (and optionally token) when a cluster is available.
Does not test full login flow (requires OIDC); asserts commands run and produce expected output.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nhx.testing import NemoRun, assert_exit_0

pytestmark = [pytest.mark.timeout(120)]

_MINIMAL_CONFIG_YAML = """
current_context: default
clusters:
  - name: default
    base_url: https://example.com
users:
  - name: default
    type: no-auth
contexts:
  - name: default
    cluster: default
    user: default
    workspace: default
""".strip()


def test_auth_lifecycle(nemo_run: NemoRun, tmp_path: Path) -> None:
    """Full auth lifecycle using unsigned JWT (no OIDC required).

    login --unsigned-token → token → status (shows email) → refresh → logout
    All steps use an isolated temp config file so the user's real config is untouched.
    """
    config_file = tmp_path / "nemo-config.yaml"
    config_file.write_text(_MINIMAL_CONFIG_YAML)
    env = {"NHX_CONFIG_FILE": str(config_file)}

    email = "lifecycle-test@example.com"

    result = nemo_run("auth", "login", "--unsigned-token", "--email", email, env_extra=env)
    assert_exit_0(result, "auth login failed")
    out = result.stdout + result.stderr
    assert "saved" in out.lower() or "token" in out.lower()

    result = nemo_run("auth", "token", env_extra=env)
    assert_exit_0(result, "auth token failed")
    token_value = result.stdout.strip()
    assert len(token_value) > 20, "expected a non-trivial JWT string"

    result = nemo_run("auth", "status", env_extra=env)
    assert_exit_0(result, "auth status failed")
    out = result.stdout + result.stderr
    assert email in out or "disabled" in out.lower(), (
        f"expected email {email!r} or 'disabled' in auth status output:\n{out}"
    )

    result = nemo_run("auth", "refresh", env_extra=env)
    assert_exit_0(result, "auth refresh failed")
    out = result.stdout + result.stderr
    assert "refreshed" in out.lower()

    # logout: clears credentials or reports auth disabled — both are valid exit-0 responses
    result = nemo_run("auth", "logout", env_extra=env)
    assert_exit_0(result, "auth logout failed")
    out = result.stdout + result.stderr
    assert "logged out" in out.lower() or "disabled" in out.lower() or "nothing" in out.lower()
