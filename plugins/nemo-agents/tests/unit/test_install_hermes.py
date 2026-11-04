# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes bootstrap preserves isolation and only accepts upstream lock overrides."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from nemo_agents_plugin.container import install_hermes


@pytest.mark.parametrize("unexpected", [[], ["unrelated conflict"]])
def test_install_checks_upstream_override_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unexpected: list[str]
) -> None:
    interpreter = tmp_path / "venv" / "bin" / "python"
    checker = Mock(return_value=unexpected)
    monkeypatch.setattr(
        install_hermes.runpy,
        "run_path",
        lambda _: {"override_requirements": lambda _: ["Pillow==13.0.0"], "unexpected_incompatibilities": checker},
    )
    monkeypatch.setattr(install_hermes, "version", lambda _: "0.6.0")

    def run(command, **kwargs):
        assert command[command.index("--python") + 1] == str(interpreter)
        if "install" in command:
            assert Path(command[command.index("--overrides") + 1]).read_text() == "Pillow==13.0.0\n"
            assert "nemo-fabric==0.6.0" in command
            assert "nemo-fabric-adapters-hermes[relay]==0.6.0" in command
            assert "nemo-relay==0.10.0" in command
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 1, stdout="report", stderr="")

    monkeypatch.setattr(install_hermes.subprocess, "run", run)
    if unexpected == []:
        install_hermes.install_hermes(tmp_path, interpreter)
    else:
        with pytest.raises(subprocess.CalledProcessError):
            install_hermes.install_hermes(tmp_path, interpreter)
    checker.assert_called_once_with("report", tmp_path / "uv.lock")


def test_main_preserves_virtual_environment_interpreter_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_python = tmp_path / "base-python"
    base_python.touch()
    interpreter = tmp_path / "venv-python"
    interpreter.symlink_to(base_python)
    install = Mock()
    monkeypatch.setattr(install_hermes, "install_hermes", install)
    monkeypatch.setattr(sys, "argv", ["install-hermes", "--checkout", str(tmp_path), "--python", str(interpreter)])
    install_hermes.main()
    install.assert_called_once_with(tmp_path.resolve(), interpreter)
