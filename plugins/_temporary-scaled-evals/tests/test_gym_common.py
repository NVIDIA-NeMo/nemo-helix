# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared Gym dispatch helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("scaled_evals")
import scaled_evals.dispatch.gym.common as gym_common
from scaled_evals.dispatch.gym.common import resolve_env_file_path


def test_resolve_env_file_path_uses_harness_mount_in_compose(monkeypatch, tmp_path: Path) -> None:
    """Relative examples/ paths resolve via /harness when not under repo_root."""
    fake_repo = tmp_path / "site-packages-root"
    fake_repo.mkdir()
    harness_root = tmp_path / "harness"
    harness_env = harness_root / "gym-sandbox-daytona" / "targets" / "daytona.env"
    harness_env.parent.mkdir(parents=True)
    harness_env.write_text("GYM_AGENT_NAME=mini_swe_agent_2\n", encoding="utf-8")

    monkeypatch.setattr(gym_common, "repo_root", lambda: fake_repo)
    monkeypatch.setattr(gym_common, "CONTAINER_HARNESS_ROOT", harness_root)

    resolved = resolve_env_file_path("examples/gym-sandbox-daytona/targets/daytona.env")
    assert resolved == harness_env.resolve()


def test_resolve_env_file_path_prefers_repo_root_when_present(monkeypatch, tmp_path: Path) -> None:
    repo_env = tmp_path / "examples" / "gym-sandbox-daytona" / "targets" / "daytona.env"
    repo_env.parent.mkdir(parents=True)
    repo_env.write_text("GYM_AGENT_NAME=mini_swe_agent_2\n", encoding="utf-8")

    monkeypatch.setattr(gym_common, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(gym_common, "CONTAINER_HARNESS_ROOT", tmp_path / "harness")

    resolved = resolve_env_file_path("examples/gym-sandbox-daytona/targets/daytona.env")
    assert resolved == repo_env.resolve()
