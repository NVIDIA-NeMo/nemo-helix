# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the skills base module."""

from pathlib import Path

import pytest
from nemo_helix_ext.cli.commands.skills.base import Scope, Skill, installed_skill_name


def test_scope_enum_has_project_and_user():
    assert Scope.PROJECT.value == "project"
    assert Scope.USER.value == "user"


def test_scope_enum_members():
    assert set(Scope) == {Scope.PROJECT, Scope.USER}


def test_installed_skill_name_adds_nemo_prefix_once():
    assert installed_skill_name("inference") == "nemo-inference"
    assert installed_skill_name("nemo-files") == "nemo-files"


@pytest.mark.parametrize("name", ["../target", "nemo-../../target", "bad/name", "bad\\name", ""])
def test_installed_skill_name_rejects_path_like_names(name: str):
    with pytest.raises(ValueError, match="Invalid skill name"):
        installed_skill_name(name)


def test_skill_preserves_source_dir_positional_compatibility():
    source_dir = Path("/tmp/source")

    skill = Skill("name", "description", "0.1", "content", "raw", source_dir)

    assert skill.source_dir == source_dir
    assert skill.preconditions == []
