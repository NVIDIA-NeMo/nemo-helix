# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import csv
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "update_python_trust", Path(__file__).parents[1] / "update-python-trust.py"
)
assert SPEC is not None and SPEC.loader is not None
trust = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trust)


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_replace_binary_detaches_shared_files_and_updates_record(tmp_path: Path, link: str) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"old Go executable")
    binary = tmp_path / "wandb-core"
    if link == "symlink":
        binary.symlink_to(original)
    else:
        os.link(original, binary)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"patched Go executable")
    record = tmp_path / "RECORD"
    record.write_text("wandb/bin/wandb-core,old-hash,17\nwandb/__init__.py,unchanged,123\n")

    trust.replace_binary(binary, replacement, record)

    assert original.read_bytes() == b"old Go executable"
    assert binary.read_bytes() == replacement.read_bytes()
    assert not binary.is_symlink()
    assert binary.stat().st_mode & 0o111 == 0o111
    digest = base64.urlsafe_b64encode(hashlib.sha256(replacement.read_bytes()).digest()).decode().rstrip("=")
    with record.open() as handle:
        assert list(csv.reader(handle)) == [
            ["wandb/bin/wandb-core", f"sha256={digest}", str(replacement.stat().st_size)],
            ["wandb/__init__.py", "unchanged", "123"],
        ]


def test_discovers_nested_gym_environments_without_resolving_python_symlinks(tmp_path: Path, monkeypatch) -> None:
    interpreter = tmp_path / "system-python"
    interpreter.touch()
    monkeypatch.setattr(trust.shutil, "which", lambda _: str(interpreter))
    configs = [tmp_path / "gym/agents/a/.venv", tmp_path / "gym/resources/b/.venv"]
    for directory in configs:
        (directory / "bin").mkdir(parents=True)
        (directory / "pyvenv.cfg").touch()
        (directory / "bin/python").symlink_to(interpreter)

    found = trust.python_environments([tmp_path / "gym"])

    assert found == sorted([interpreter, *(directory / "bin/python" for directory in configs)])


def test_updates_bootstrap_wheel_as_well_as_ensurepip_version(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "ensurepip/__init__.py"
    module.parent.mkdir()
    module.write_text('_PIP_VERSION = "26.2.1"\n')
    bundled = module.parent / "_bundled"
    bundled.mkdir()
    old = bundled / "pip-26.2.1-py3-none-any.whl"
    old.write_bytes(b"old certificate bundle")
    monkeypatch.setattr(trust.subprocess, "check_output", lambda *args, **kwargs: str(module))
    monkeypatch.setattr(trust, "fixed_pip_wheel", lambda: b"verified fixed wheel")

    trust.update_ensurepip(tmp_path / "python")

    assert module.read_text() == '_PIP_VERSION = "26.2.2"\n'
    assert not old.exists()
    assert (bundled / "pip-26.2.2-py3-none-any.whl").read_bytes() == b"verified fixed wheel"
