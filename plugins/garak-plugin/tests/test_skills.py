# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the garak_plugin plugin skills surface."""

import yaml
from garak_plugin.skills import get_skills_path


def test_get_skills_path_exists():
    path = get_skills_path()
    assert path.exists(), f"Skills directory does not exist: {path}"


def test_get_skills_path_is_directory():
    path = get_skills_path()
    assert path.is_dir(), f"Skills path is not a directory: {path}"


def test_skills_directory_has_at_least_one_skill():
    root = get_skills_path()
    skill_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]
    assert len(skill_dirs) >= 1, f"No skill subdirectories with SKILL.md found in {root}"


def test_skill_md_has_valid_frontmatter():
    root = get_skills_path()
    for skill_dir in sorted(root.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue

        text = skill_file.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"Missing frontmatter delimiter in {skill_file}"

        end = text.find("\n---\n", 4)
        assert end != -1, f"Unterminated frontmatter in {skill_file}"

        raw_fm = text[4:end]
        metadata = yaml.safe_load(raw_fm)
        assert isinstance(metadata, dict), f"Frontmatter is not a mapping in {skill_file}"
        assert "name" in metadata, f"Frontmatter missing 'name' in {skill_file}"
        assert "description" in metadata, f"Frontmatter missing 'description' in {skill_file}"
