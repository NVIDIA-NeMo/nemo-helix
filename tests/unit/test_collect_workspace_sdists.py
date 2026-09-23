# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "docker" / "scripts" / "collect-workspace-sdists.py"


def load_module():
    spec = spec_from_file_location("collect_workspace_sdists", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_invalid_projects_are_recorded_as_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_module()
    valid_project = tmp_path / "valid-project"
    valid_project.mkdir()
    (valid_project / "pyproject.toml").write_text("[project]\nname = 'valid-project'\n", encoding="utf-8")
    directory_without_pyproject = tmp_path / "not-a-project"
    directory_without_pyproject.mkdir()
    nonexistent_project = tmp_path / "missing-project"
    output_dir = tmp_path / "out"
    build_commands: list[list[str]] = []

    def fake_run_text(command: list[str]) -> str:
        build_commands.append(command)
        return "built valid-project"

    monkeypatch.setattr(module, "run_text", fake_run_text)
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")
    monkeypatch.delenv("UV_BIN", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--output",
            str(output_dir),
            "--project",
            str(valid_project),
            "--project",
            str(directory_without_pyproject),
            "--project",
            str(nonexistent_project),
        ],
    )

    assert module.main() == 0

    assert build_commands == [
        [
            "uv",
            "build",
            "--sdist",
            "--out-dir",
            str(output_dir / "workspace"),
            str(valid_project),
        ]
    ]
    assert (output_dir / "manifests" / "built-workspace-sdists.txt").read_text(encoding="utf-8") == (
        f"{valid_project}\tbuilt valid-project\n"
    )
    assert (output_dir / "manifests" / "missing-workspace-sdists.txt").read_text(encoding="utf-8") == (
        f"{directory_without_pyproject}\tmissing pyproject.toml\n{nonexistent_project}\tproject path does not exist\n"
    )


def test_uv_startup_errors_are_recorded_as_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_module()
    valid_project = tmp_path / "valid-project"
    valid_project.mkdir()
    (valid_project / "pyproject.toml").write_text("[project]\nname = 'valid-project'\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    def fake_run_text(command: list[str]) -> str:
        raise OSError("uv unavailable")

    monkeypatch.setattr(module, "run_text", fake_run_text)
    monkeypatch.setenv("NHX_COLLECT_SOURCES", "1")
    monkeypatch.delenv("UV_BIN", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--output",
            str(output_dir),
            "--project",
            str(valid_project),
        ],
    )

    assert module.main() == 0

    assert (output_dir / "manifests" / "built-workspace-sdists.txt").read_text(encoding="utf-8") == ""
    assert (output_dir / "manifests" / "missing-workspace-sdists.txt").read_text(encoding="utf-8") == (
        f"{valid_project}\tuv unavailable\n"
    )
