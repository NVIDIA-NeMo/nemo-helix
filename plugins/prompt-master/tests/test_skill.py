# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from prompt_master_plugin.runner import SKILLS_DIR


def test_bundles_prompt_master_skill_and_references() -> None:
    skill = SKILLS_DIR / "prompt-master"

    skill_text = (skill / "SKILL.md").read_text(encoding="utf-8")
    assert "name: prompt-master" in skill_text
    assert "version: 1.8.0" in skill_text
    assert (skill / "references" / "templates.md").is_file()
    assert (skill / "references" / "patterns.md").is_file()


def test_bundled_skill_retains_upstream_license() -> None:
    assert "MIT License" in (SKILLS_DIR / "prompt-master" / "LICENSE").read_text(encoding="utf-8")
