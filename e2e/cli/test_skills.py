# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for skills CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from nhx.testing import assert_exit_0, run_nemo_local

pytestmark = [pytest.mark.timeout(60)]


def test_skills_lifecycle(tmp_path: Path) -> None:
    """Skills lifecycle: show (generic) → show (agent) → invalid agent → install → verify."""
    result = run_nemo_local("skills", "show", "inference")
    assert_exit_0(result, "skills show failed")
    assert "nemo" in result.stdout.lower()

    result = run_nemo_local("skills", "show", "--agent", "claude", "inference")
    assert_exit_0(result, "skills show --agent claude failed")
    assert "nemo" in result.stdout.lower()

    result = run_nemo_local("skills", "show", "--agent", "nonexistent-agent", "inference")
    assert result.returncode == 1, "skills show with unknown agent should exit 1"

    # Create a fake .git marker so _find_project_root() resolves to tmp_path
    # instead of the real repo root, keeping installed files out of the working tree.
    (tmp_path / ".git").mkdir()
    skill_path = tmp_path / ".claude/skills/nemo-inference/SKILL.md"

    result = run_nemo_local("skills", "install", "--agent", "claude", "--skill", "inference", cwd=tmp_path)
    assert_exit_0(result, "skills install failed")
    assert skill_path.exists(), f"skill file not created at {skill_path}"
    assert "nemo" in skill_path.read_text().lower()
