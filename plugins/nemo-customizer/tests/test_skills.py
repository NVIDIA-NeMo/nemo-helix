# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pytest
import yaml
from nemo_customizer.skills import get_skills_path

SKILLS = ["nemo-customizer", "nemo-gym-environments"]


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


def test_nemo_gym_environments_skill_present() -> None:
    skill_dir = get_skills_path() / "nemo-gym-environments"
    assert (skill_dir / "tests.json").is_file()
    assert "name: nemo-gym-environments" in (skill_dir / "SKILL.md").read_text()


def _frontmatter(skill: str) -> dict:
    text = (get_skills_path() / skill / "SKILL.md").read_text()
    return yaml.safe_load(text.split("---\n")[1])


def _skill_docs(skill: str) -> list[Path]:
    skill_dir = get_skills_path() / skill
    return [skill_dir / "SKILL.md", *sorted((skill_dir / "references").glob("*.md"))]


@pytest.mark.parametrize("skill", SKILLS)
def test_skill_paths_resolve_inside_the_installed_skill(skill: str) -> None:
    """An installed skill is only its own directory, so a repo path would not resolve."""
    skill_dir = get_skills_path() / skill
    paths = set(re.findall(r"`((?:references|scripts)/[\w./-]+)`", (skill_dir / "SKILL.md").read_text()))
    assert paths
    assert [path for path in sorted(paths) if not (skill_dir / path).exists()] == []


@pytest.mark.parametrize("skill", SKILLS)
def test_cross_skill_paths_resolve(skill: str) -> None:
    """Installers put skills side by side, so `../<skill>/…` paths resolve once installed."""
    missing = [
        f"{doc.name}: {path}"
        for doc in _skill_docs(skill)
        for path in re.findall(r"`((?:\.\./)+nemo-[\w-]+/[\w./-]+)`", doc.read_text())
        if not (doc.parent / path).exists()
    ]
    assert missing == []


@pytest.mark.parametrize("skill", SKILLS)
def test_description_is_short_enough_to_route_on(skill: str) -> None:
    # The installer keeps only name and description, so the description is the routing signal.
    assert len(_frontmatter(skill)["description"].split()) <= 50


def test_customizer_description_names_the_retrieval_skill() -> None:
    assert "nemo-retrieval-recipes" in _frontmatter("nemo-customizer")["description"]


@pytest.mark.parametrize("skill", SKILLS)
def test_skills_never_run_model_repo_code(skill: str) -> None:
    for doc in _skill_docs(skill):
        assert "trust_remote_code=True" not in doc.read_text(), doc.name


def test_no_reference_says_rl_has_no_lora() -> None:
    """DPO is full-weight only; GRPO trains LoRA adapters too."""
    for doc in _skill_docs("nemo-customizer"):
        assert "full-weight (no LoRA)" not in doc.read_text(), doc.name
