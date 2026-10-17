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
    sft_fast_path = skill_dir / "references" / "fast-path-sft.md"
    assert skill.is_file()
    assert tests.is_file()
    assert sft_fast_path.is_file()
    text = skill.read_text()
    assert "name: nemo-customizer" in text
    assert "nemo customization automodel --job-json" in text
    # The unsloth submit command lives in the SFT fast path that SKILL.md points to.
    assert "references/fast-path-sft.md" in text
    assert "nemo customization unsloth --job-json" in sft_fast_path.read_text()
    for doc in [skill, *sorted((skill_dir / "references").glob("*.md"))]:
        assert "run --venv" not in doc.read_text(), doc.name
