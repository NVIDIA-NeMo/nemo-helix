# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

import pytest

ENVIRONMENT = (
    Path(__file__).resolve().parents[1] / "examples/harbor_evals_skill/dataset/nemo-evals-plugin-001/environment"
)


def test_removal_validates_all_directories_before_deleting(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    skills.mkdir()
    valid = skills / "remove-me"
    valid.mkdir()
    (valid / "SKILL.md").touch()
    escaped = skills / "escape"
    escaped.symlink_to(valid)
    remove = runpy.run_path(str(ENVIRONMENT / "image_setup.py"))["remove_skills"]
    with pytest.raises(ValueError):
        remove([str(valid), str(escaped)], roots=[skills])
    assert valid.exists()
    remove([str(valid)], roots=[skills])
    assert not valid.exists()


def test_codex_version_passes_through_and_only_exec_records_probe(tmp_path: Path) -> None:
    run = runpy.run_path(str(ENVIRONMENT / "codex"))["run"]
    log = tmp_path / "scope-probe.json"
    commands = []

    def launch(binary: str, argv: list[str]) -> None:
        commands.append((binary, argv))

    run(["--version"], {"NHX_SCOPE_PROBE": "1"}, log, launch=launch)
    assert not log.exists()
    run(["exec", "--json", "--", "test"], {}, log, launch=launch)
    assert log.read_text().strip() == '{"enabled": false}'
    assert commands == [
        ("/opt/codex/bin/codex", ["codex", "--version"]),
        ("/opt/codex/bin/codex", ["codex", "exec", "--json", "--", "test"]),
    ]
