# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=9.0.3,<10",
#   "pyyaml>=6.0.2",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from check_skill_frontmatter import (  # noqa: E402
    MalformedFrontmatterError,
    check_file,
    is_canonical_skill,
    main,
)

VALID_FRONTMATTER = """---
name: example-skill
description: An example skill.
metadata:
  author: someone@example.com
allowed-tools: Read, Bash
---

# Example Skill
"""

MISSING_AUTHOR_FRONTMATTER = """---
name: example-skill
description: An example skill.
allowed-tools: Read
---

# Example Skill
"""

MISSING_TOOLS_FRONTMATTER = """---
name: example-skill
description: An example skill.
metadata:
  author: someone@example.com
allowed-tools: ""
---

# Example Skill
"""

INVALID_TOOLS_FRONTMATTER = """---
name: example-skill
description: An example skill.
metadata:
  author: someone@example.com
allowed-tools: Bash Read
---

# Example Skill
"""

MALFORMED_YAML_FRONTMATTER = """---
name: example-skill
description: [unterminated
---

# Example Skill
"""

NO_FRONTMATTER = "# Example Skill\n\nNo frontmatter here.\n"


def test_is_canonical_skill_matches_top_level_skills_dir():
    assert is_canonical_skill(Path("skills/nemo-evaluator-plugin/SKILL.md"))


def test_is_canonical_skill_matches_nemo_platform_ext():
    path = Path("packages/nemo_platform_ext/src/nemo_platform_ext/skills/nemo-status/SKILL.md")
    assert is_canonical_skill(path)


def test_is_canonical_skill_matches_plugin_skills():
    assert is_canonical_skill(Path("plugins/nemo-auditor/src/nemo_auditor/skills/auditor/SKILL.md"))


def test_is_canonical_skill_matches_framework_skills():
    path = Path("plugins/nemo-experimentalist/framework-skills/langchain-framework/SKILL.md")
    assert is_canonical_skill(path)


def test_is_canonical_skill_ignores_dev_only_dirs():
    assert not is_canonical_skill(Path(".cursor/rules/nemo-nemo-status/SKILL.md"))
    assert not is_canonical_skill(Path("agents/some-agent-ethos/skills/SKILL.md"))


def test_check_file_passes_for_valid_frontmatter(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(VALID_FRONTMATTER)
    assert check_file(skill) == []


def test_check_file_flags_missing_author(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(MISSING_AUTHOR_FRONTMATTER)
    violations = check_file(skill)
    assert len(violations) == 1
    assert "metadata.author" in violations[0].issue


@pytest.mark.parametrize(
    "author_yaml",
    [
        "42",
        "true",
        "[team@example.com]",
        "{email: team@example.com}",
        '""',
        '"   "',
    ],
)
def test_check_file_flags_non_string_or_empty_author(tmp_path, author_yaml):
    skill = tmp_path / "SKILL.md"
    skill.write_text(VALID_FRONTMATTER.replace("someone@example.com", author_yaml))
    violations = check_file(skill)
    assert len(violations) == 1
    assert "metadata.author" in violations[0].issue


def test_check_file_flags_missing_tools(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(MISSING_TOOLS_FRONTMATTER)
    violations = check_file(skill)
    assert len(violations) == 1
    assert "allowed-tools" in violations[0].issue


def test_check_file_flags_no_frontmatter(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(NO_FRONTMATTER)
    violations = check_file(skill)
    assert len(violations) == 1
    assert "frontmatter" in violations[0].issue


def test_check_file_flags_invalid_tools_grammar(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(INVALID_TOOLS_FRONTMATTER)
    violations = check_file(skill)
    assert len(violations) == 1
    assert "allowed-tools" in violations[0].issue


def test_check_file_raises_on_malformed_yaml(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(MALFORMED_YAML_FRONTMATTER)
    with pytest.raises(MalformedFrontmatterError):
        check_file(skill)


def test_main_exits_2_on_malformed_yaml(tmp_path, monkeypatch, capsys):
    skill_dir = tmp_path / "skills" / "broken-skill"
    skill_dir.mkdir(parents=True)
    skill = skill_dir / "SKILL.md"
    skill.write_text(MALFORMED_YAML_FRONTMATTER)

    monkeypatch.setattr(
        sys,
        "argv",
        ["check_skill_frontmatter.py", "--root", str(tmp_path), str(skill)],
    )
    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "not valid YAML" in captured.err


def test_main_github_format_uses_relative_paths(tmp_path, monkeypatch, capsys):
    skill_dir = tmp_path / "skills" / "broken-skill"
    skill_dir.mkdir(parents=True)
    skill = skill_dir / "SKILL.md"
    skill.write_text(MISSING_AUTHOR_FRONTMATTER)

    monkeypatch.setattr(
        sys,
        "argv",
        ["check_skill_frontmatter.py", "--root", str(tmp_path), "--format", "github", str(skill)],
    )
    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "::error file=skills/broken-skill/SKILL.md::" in captured.out
    assert str(tmp_path) not in captured.out


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-q",
                "-c",
                os.devnull,
                "-p",
                "no:cacheprovider",
                "-W",
                "ignore::SyntaxWarning",
                "--confcutdir",
                str(Path(__file__).parent),
            ]
        )
    )
