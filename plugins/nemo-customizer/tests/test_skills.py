# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nemo_customizer.skills import get_skills_path


def test_get_skills_path_exists() -> None:
    path = get_skills_path()
    assert path.is_dir()


def test_nemo_customizer_skill_present() -> None:
    skill_dir = get_skills_path() / "nemo-customizer"
    skill = skill_dir / "SKILL.md"
    tests = skill_dir / "tests.json"
    assert skill.is_file()
    assert tests.is_file()
    text = skill.read_text()
    assert "name: nemo-customizer" in text
    assert "nemo customization automodel --job-json" in text
    assert "nemo customization unsloth --job-json" in text
    assert "run --venv" not in text
