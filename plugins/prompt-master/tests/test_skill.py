# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from prompt_master_plugin.skills import skills_dir


def test_bundles_prompt_master_skill_and_references() -> None:
    skill = skills_dir() / "prompt-master"

    skill_text = (skill / "SKILL.md").read_text(encoding="utf-8")
    assert "name: prompt-master" in skill_text
    assert "version: 1.8.0" in skill_text
    assert (skill / "references" / "templates.md").is_file()
    assert (skill / "references" / "patterns.md").is_file()


def test_bundled_skill_retains_upstream_license_and_revision() -> None:
    skill = skills_dir() / "prompt-master"

    assert "MIT License" in (skill / "LICENSE").read_text(encoding="utf-8")
    provenance = (skill / "UPSTREAM.md").read_text(encoding="utf-8")
    assert "nidhinjs/prompt-master" in provenance
    assert "2bd92518e26bf659e21e3d9ab90573fcf3ddeccb" in provenance
