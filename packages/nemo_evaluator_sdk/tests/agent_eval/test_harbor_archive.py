# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-owned capture validates packages without importing platform code."""

import os
import subprocess
import sys

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes import harbor_archive
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_archive import (
    capture_validated_task,
    local_task_identity,
    private_directory,
    validate_native_task_inputs,
)


def _task(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    for name, content in {
        "task.toml": '[task]\nname="suite/task"\n',
        "instruction.md": "Do it",
        "environment/Dockerfile": "FROM ubuntu",
        "tests/test.sh": "exit 0",
    }.items():
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(content)
    return root


_EXPECTED_ERRORS = {
    "absolute_symlink": "Symlink escapes",
    "escaping_symlink": "Symlink escapes",
    "dot_escape": "Symlink escapes",
    "self_name_escape": "Symlink escapes",
    "dangling": "Symlink escapes",
    "loop": "Symlink escapes",
    "cycle": "Symlink escapes",
    "metadata_link": "Metadata must be a regular file",
}


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "absolute_symlink",
        "escaping_symlink",
        "dot_escape",
        "self_name_escape",
        "dangling",
        "loop",
        "cycle",
        "metadata_link",
        "environment",
        "config",
        "traversal",
    ],
)
def test_capture_and_cleanup(tmp_path, invalid):
    root = _task(tmp_path)
    if invalid == "absolute_symlink":
        (root / "link").symlink_to(root / "instruction.md")
    elif invalid == "escaping_symlink":
        (tmp_path / "outside").write_text("secret")
        (root / "link").symlink_to("../outside")
    elif invalid == "dot_escape":
        (root / "a").symlink_to(".")
        (root / "b").symlink_to("a/..")
    elif invalid == "self_name_escape":
        (root / "link").symlink_to("../task/instruction.md")
    elif invalid == "dangling":
        (root / "link").symlink_to("missing")
    elif invalid == "loop":
        (root / "a").symlink_to("a")
    elif invalid == "cycle":
        (root / "a").symlink_to("b")
        (root / "b").symlink_to("a")
    elif invalid == "metadata_link":
        (root / "other.md").write_text("Other")
        (root / "instruction.md").unlink()
        (root / "instruction.md").symlink_to("other.md")
    elif invalid == "environment":
        (root / "environment/Dockerfile").unlink()
        (root / "environment").rmdir()
    elif invalid == "config":
        (root / "task.toml").write_text("invalid [")
    elif invalid == "traversal":
        (root / "task.toml").write_text('[[steps]]\nname="../escape"\n')
    with private_directory() as staging:
        if invalid:
            with pytest.raises(ValueError, match=_EXPECTED_ERRORS.get(invalid, "")):
                capture_validated_task(root, staging)
        else:
            snapshot, native = capture_validated_task(root, staging)
            assert snapshot != root
            assert native.task_id == "suite/task"
            assert native.instruction == "Do it"
    assert not staging.exists()


def test_sdk_import_does_not_require_harbor_or_plugin():
    code = """
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"harbor", "nemo_evaluator"}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("config", ["", '[[steps]]\nname = "one"\n'])
def test_missing_metadata_is_a_value_error(tmp_path, config):
    root = tmp_path / "task"
    (root / "steps" / "one").mkdir(parents=True)
    if config:
        (root / "task.toml").write_text(config)
    with pytest.raises(ValueError, match="Missing or unreadable metadata"):
        validate_native_task_inputs(root)


def test_non_table_task_section_is_a_value_error(tmp_path):
    (tmp_path / "task.toml").write_text('task = "x"\n')
    with pytest.raises(ValueError, match="Invalid native task identity"):
        local_task_identity(tmp_path)


def test_saved_trial_dirs_treat_case_only_duplicates_as_stale(tmp_path):
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _task_dirs_for
    from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask

    for folder, name in (("a", "Foo"), ("b", "foo")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "task.toml").write_text(f'[task]\nname = "{name}"\n')
    task = AgentEvalTask(id="Foo", intent="Foo", inputs={})
    assert _task_dirs_for(tmp_path, [task]) == {"Foo": None}


def test_capture_preserves_internal_symlinks(tmp_path):
    root = _task(tmp_path)
    (root / "tests/data").mkdir()
    (root / "tests/data/input.txt").write_text("x")
    links = {"tests/run.sh": "test.sh", "tests/fixtures": "data", "link": "tests/test.sh"}
    for name, target in links.items():
        (root / name).symlink_to(target)
    with private_directory() as staging:
        snapshot, _ = capture_validated_task(root, staging)
        for name, target in links.items():
            assert (snapshot / name).is_symlink()
            assert os.readlink(snapshot / name) == target
        assert (snapshot / "tests/fixtures/input.txt").read_text() == "x"


@pytest.mark.parametrize(("name", "target"), [("a", "."), ("tests/up", "..")])
def test_capture_rejects_links_to_ancestor_directories(tmp_path, name, target):
    root = _task(tmp_path)
    (root / name).symlink_to(target)
    with private_directory() as staging:
        with pytest.raises(ValueError, match="Symlink escapes"):
            capture_validated_task(root, staging)


def test_capture_rechecks_links_changed_after_inventory(tmp_path, monkeypatch):
    root = _task(tmp_path)
    (root / "link").symlink_to("instruction.md")
    (tmp_path / "outside").write_text("secret")
    inventory = harbor_archive.inventory_task_entries

    def swap_after_inventory(task_root):
        entries = inventory(task_root)
        (root / "link").unlink()
        (root / "link").symlink_to("../outside")
        return entries

    monkeypatch.setattr(harbor_archive, "inventory_task_entries", swap_after_inventory)
    with private_directory() as staging:
        with pytest.raises(ValueError, match="Symlink escapes"):
            capture_validated_task(root, staging)
    assert not staging.exists()


@pytest.mark.parametrize("name", [".gitignore", "instruction.md"])
def test_dangling_metadata_link_is_rejected(tmp_path, name):
    root = _task(tmp_path)
    (root / name).unlink(missing_ok=True)
    (root / name).symlink_to("missing")
    with pytest.raises(ValueError, match="Metadata must be a regular file"):
        validate_native_task_inputs(root)
