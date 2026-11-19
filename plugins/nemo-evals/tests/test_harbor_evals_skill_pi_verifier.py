# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from nemo_evals.harbor.archive import pack_task
from nemo_evals.harbor.publication import describe_harbor_task_archive
from nemo_helix_plugin.client.errors import InternalServerError, NotFoundError

TASK = Path(__file__).resolve().parents[1] / "examples/harbor_evals_skill/dataset/nemo-evals-plugin-003"
TASK_PIN = "default/pi-digit#" + "a" * 64
TASKSET_PIN = "default/pi-digit-suite#" + "b" * 64


@pytest.fixture
def submission(tmp_path: Path):
    workspace = tmp_path / "workspace"
    task = workspace / "pi-digit"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('name = "pi-digit"\nversion = "1.0"\n')
    (task / "instruction.md").write_text("Write the 100th fractional digit to pi_digit.txt.\n")
    (task / "environment/Dockerfile").write_text("FROM python:3.13-slim\n")
    (task / "tests/test.sh").write_text("#!/bin/sh\ntrue\n")
    (workspace / "taskset.json").write_text(json.dumps({"taskset_ref": TASKSET_PIN, "tasks": [TASK_PIN]}))
    (workspace / "pi_eval.py").write_text("closed_book = True\n")
    definition = describe_harbor_task_archive(task, fileset_ref="default/harbor-tasks")
    archive = tmp_path / "stored.tar.gz"
    assert pack_task(task, archive) == definition.source.files_hash
    saved = SimpleNamespace(spec=definition, name="pi-digit", workspace="default")
    taskset = SimpleNamespace(tasks=[SimpleNamespace(root=TASK_PIN)], workspace="default", name="pi-digit-suite")

    def retrieve_task(name: str, *, revision: str, workspace: str):
        assert (name, revision, workspace) == ("pi-digit", "a" * 64, "default")
        return saved

    def retrieve_taskset(name: str, *, revision: str, workspace: str):
        assert (name, revision, workspace) == ("pi-digit-suite", "b" * 64, "default")
        return taskset

    evaluator = SimpleNamespace(
        tasks=SimpleNamespace(retrieve=retrieve_task), tasksets=SimpleNamespace(retrieve=retrieve_taskset)
    )
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    grader = runpy.run_path(str(TASK / "tests/grade.py"))["grade"]

    def download(_, ref: str, output, *, limit: int, expected_digest: str) -> None:
        assert ref == definition.source.fileset_ref and expected_digest == definition.source.files_hash
        assert limit > 0
        output.write(archive.read_bytes())

    grader.__globals__["download_verified"] = download
    return workspace, evidence, evaluator, taskset, grader, archive, definition


def _judge(metric: str, text: str) -> int:
    assert metric in {"task_instruction", "script_correct"}
    return int("100th" in text if metric == "task_instruction" else "closed_book = True" in text)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("valid", (1, 1, 1, 1)),
        ("missing", (0, 0, 0, 1)),
        ("wrong_member", (0, 1, 0, 1)),
        ("changed_archive", (0, 1, 0, 1)),
        ("open_book", (0, 1, 1, 0)),
        ("invalid_script", (0, 1, 1, 0)),
        ("symlink_script", (0, 1, 1, 0)),
    ],
)
def test_shared_verifier_reads_pinned_records_and_grades_each_deliverable(submission, case: str, expected: tuple):
    workspace, evidence, evaluator, taskset, grade, _, _ = submission
    script = workspace / "pi_eval.py"
    if case == "missing":
        (workspace / "taskset.json").unlink()
    elif case == "wrong_member":
        taskset.tasks[0].root = "default/wrong#" + "d" * 64
    elif case == "changed_archive":
        (workspace / "pi-digit/tests/test.sh").write_text("#!/bin/sh\nfalse\n")
    elif case == "open_book":
        script.write_text("closed_book = False\n")
    elif case == "invalid_script":
        script.write_text("def\n")
    elif case == "symlink_script":
        script.rename(evidence / "hidden.py")
        script.symlink_to(evidence / "hidden.py")

    assert grade(workspace, evidence, evaluator, None, _judge) == dict(
        zip(("reward", "task_instruction", "taskset_persisted", "script_correct"), expected, strict=True)
    )
    assert (evidence / "pi-digit.tar.gz").exists() == bool(expected[2])


def test_shared_verifier_accepts_an_equivalent_published_archive(submission):
    workspace, evidence, evaluator, _, grade, archive, definition = submission
    repacked = gzip.compress(gzip.decompress(archive.read_bytes()), compresslevel=1, mtime=0)
    digest = hashlib.sha256(repacked).hexdigest()
    assert digest != definition.source.files_hash
    definition.source.files_hash = digest
    archive.write_bytes(repacked)

    assert grade(workspace, evidence, evaluator, None, _judge)["taskset_persisted"] == 1
    assert (evidence / "pi-digit.tar.gz").read_bytes() == repacked


def test_missing_revision_scores_zero_and_service_failure_propagates(submission):
    workspace, evidence, evaluator, _, grade, _, _ = submission

    def missing(*args, **kwargs):
        raise NotFoundError(httpx.Response(404, json={"detail": "missing task"}))

    def unavailable(*args, **kwargs):
        raise InternalServerError(httpx.Response(503, json={"detail": "unavailable"}))

    evaluator.tasks.retrieve = missing
    assert grade(workspace, evidence, evaluator, None, _judge)["task_instruction"] == 0
    evaluator.tasks.retrieve = unavailable
    with pytest.raises(InternalServerError):
        grade(workspace, evidence, evaluator, None, _judge)


def test_judge_failure_is_a_verifier_error(submission):
    workspace, evidence, evaluator, _, grade, _, _ = submission

    def unavailable(*args):
        raise RuntimeError("judge unavailable")

    with pytest.raises(RuntimeError, match="judge unavailable"):
        grade(workspace, evidence, evaluator, None, unavailable)
