# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor exports retain task identity in both supported staging modes."""

from pathlib import Path

import pytest
import yaml

pytest.importorskip("scaled_evals")

from scaled_evals.dispatch.sandbox_k8s import (
    make_sandbox_k8s_docker_submitter,
    make_sandbox_k8s_submitter,
)
from test_evaluations import (
    _TASKS_CONFIG,
    NOW,
    _dispatcher,
    _eval_row,
    _fake_download,
    _FakeBackend,
    _make_task_pack,
    _spec_with_tarball,
    _worker_conn,
)


@pytest.mark.parametrize("docker", [False, True])
@pytest.mark.parametrize("slug", [None, "smoke-named"])
def test_task_staging_preserves_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, docker: bool, slug: str | None
) -> None:
    config = tmp_path / "oracle.yaml"
    config.write_text(_TASKS_CONFIG.replace("BROKEN_PYTHON_TASK_PATH", "TASK_PATH"))
    env_file = tmp_path / "sandbox.env"
    env_file.write_text("SANDBOX_NAMESPACE=ns\nTASK_IMAGE=reg/static:old\nVERIFY_SSL=true\n")
    pack = _make_task_pack(tmp_path / "pack.tar.gz")
    monkeypatch.setattr("scaled_evals.api.s3.download_object", _fake_download(pack))
    work = tmp_path / "work"
    submit = (
        make_sandbox_k8s_docker_submitter(
            config_path=str(config),
            env_file=str(env_file),
            work_dir=str(work),
            runner=lambda *args: None,
            work_volume="archive-test-work",
            image="registry.example/harbor:test",
            harbor_dir=str(tmp_path / "harbor"),
            harbor_jobs_dir=str(tmp_path / "jobs"),
            kube_config_dir=str(tmp_path / "kube"),
        )
        if docker
        else make_sandbox_k8s_submitter(
            config_path=str(config),
            env_file=str(env_file),
            work_dir=str(work),
            runner=lambda *args: None,
            harbor_dir=str(tmp_path / "harbor"),
        )
    )
    spec = _spec_with_tarball("task_abc/rev/2/tarball.tar.gz").model_copy(update={"task_slug": slug})
    submit(spec)
    name = slug or "task"
    staged = work / "ev_test123" / name
    assert (staged / "task.toml").is_file()
    rendered = yaml.safe_load((work / "ev_test123.yaml").read_text())
    assert rendered["tasks"][0]["path"] == (f"/work/ev_test123/{name}" if docker else str(staged))


@pytest.mark.parametrize("slug", ["../escape", "/absolute", ".", "two/tasks"])
def test_staged_task_slug_rejects_unsafe_paths(slug: str) -> None:
    from scaled_evals.dispatch.sandbox_k8s import _staged_task_name

    spec = _spec_with_tarball("task/rev/1/tarball.tar.gz").model_copy(update={"task_slug": slug})
    with pytest.raises(ValueError, match="safe directory name"):
        _staged_task_name(spec)


def test_dispatch_uses_frozen_task_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeBackend()
    snapshot = {
        "schema_version": "scaled-evals-execution-inputs-v1",
        "captured_at": NOW.isoformat(),
        "evaluation": {"framework": "harbor", "framework_version": None, "runtime": "gym_daytona"},
        "task": {"slug": "frozen-task"},
        "profiles": {},
        "credentials": {},
        "submission_identity": {},
    }
    conn, executed = _worker_conn(
        _eval_row(
            status="queued",
            runtime="gym_daytona",
            task_slug="renamed-task",
            execution_snapshot=snapshot,
            image_ref="",
        )
    )
    worker = _dispatcher(backend, conn)
    for method in ("_sync_artifacts_warn", "_build_archive_warn", "_write_provenance_warn"):
        monkeypatch.setattr(worker, method, lambda *args, **kwargs: None)
    worker.run("ev_test123")
    assert backend.launched, executed
    assert backend.launched[0].task_slug == "frozen-task"
