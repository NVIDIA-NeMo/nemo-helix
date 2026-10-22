# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cover correctness fixes identified during review."""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

try:
    import click
    import httpx
    from scaled_evals.api.build import buildkit
    from scaled_evals.api.build.errors import BuildError
    from scaled_evals.api.build.image_builder_service import _post_resolve
    from scaled_evals.api.evaluation_store import EvaluationStore
    from scaled_evals.api.routers import evaluations as evaluations_router
    from scaled_evals.api.task_store import TaskStore
    from scaled_evals.cli.client import download_artifact, make_client, request, upload_file
    from scaled_evals.dispatch import detached_runner
    from scaled_evals.dispatch.credentials import write_env_file
    from scaled_evals.dispatch.gym.docker import make_gym_docker_submitter
    from scaled_evals.dispatch.kubectl import execute_kubectl
    from scaled_evals.dispatch.runtime_backend import LaunchSpec
    from task_fixtures import seed_evaluation, stored_revision, stored_task
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_external_commands_and_http_redirects_fail_bounded() -> None:
    runner_kwargs: dict[str, Any] = {}

    def timeout_runner(args: list[str], **kwargs: Any) -> None:
        runner_kwargs.update(kwargs)
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    result = execute_kubectl(["kubectl", "get", "pods"], runner=timeout_runner)
    assert runner_kwargs["timeout"] == 120.0
    assert result.returncode == 124
    assert result.stdout == ""
    assert "timed out" in result.stderr

    transport = httpx.MockTransport(
        lambda _request: httpx.Response(302, headers={"location": "https://example.invalid"})
    )
    with make_client("https://api.example.invalid", None, transport=transport) as client:
        with pytest.raises(click.ClickException, match="HTTP 302"):
            request(client, "GET", "/tasks")


def test_p1_cancellation_persists_before_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_events: list[str] = []
    row = {"id": "ev-cancel"}

    class _Evaluations:
        def cancel(self, _evaluation_id: str) -> tuple[dict[str, str], bool]:
            request_events.append("cancel")
            return row, True

    class _Database:
        evaluations = _Evaluations()

    def teardown(_db: Any, cancelled: dict[str, str]) -> dict[str, str]:
        request_events.append("teardown")
        return cancelled

    monkeypatch.setattr(evaluations_router, "teardown_cancelled_evaluation", teardown)
    monkeypatch.setattr(evaluations_router, "_response", lambda value: value)
    assert evaluations_router.cancel_evaluation("ev-cancel", cast(Any, _Database())) == row
    assert request_events == ["cancel", "teardown"]


def test_p1_kubernetes_registry_policy_is_narrow_and_reapply_safe() -> None:
    plugin_root = Path(__file__).parents[1]
    registry_auth = (plugin_root / "deploy/k8s/registry-auth.yaml").read_text()
    settings_env = (plugin_root / "deploy/k8s/settings.env").read_text()

    assert "\nkind: Secret\n" not in registry_auth
    assert "kind: CronJob" in registry_auth
    assert "TASK_IMAGE_ALLOWED_REPOSITORIES=${SE_TASK_IMAGE_REGISTRY}/*" in settings_env


def test_task_pack_extraction_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "pack.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("one", "two"):
            content = b"ab"
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))

    monkeypatch.setattr(buildkit.settings, "task_pack_max_members", 1)
    monkeypatch.setattr(buildkit.settings, "task_pack_max_extracted_size_bytes", 100)
    with pytest.raises(BuildError, match="member limit"):
        buildkit._extract_tarball(archive, tmp_path / "members")

    monkeypatch.setattr(buildkit.settings, "task_pack_max_members", 10)
    monkeypatch.setattr(buildkit.settings, "task_pack_max_extracted_size_bytes", 3)
    with pytest.raises(BuildError, match="extracted-size limit"):
        buildkit._extract_tarball(archive, tmp_path / "size")


async def test_buildkit_timeout_terminates_and_reaps_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _HangingProcess:
        returncode = None
        communicate_calls = 0
        terminated = False

        async def communicate(self) -> tuple[bytes, None]:
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                await asyncio.Future()
            return b"", None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            raise AssertionError("terminate should have been enough")

    process = _HangingProcess()

    async def create_process(*_args: Any, **_kwargs: Any) -> _HangingProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(buildkit, "_buildctl_args", lambda *_args: ["buildctl"])
    monkeypatch.setattr(buildkit, "_buildctl_env", lambda *_args: {})
    monkeypatch.setattr(buildkit.settings, "buildkit_timeout_seconds", 0.001)

    with pytest.raises(BuildError, match="timed out"):
        await buildkit._run_buildctl(
            tmp_path / "context",
            "registry.example.invalid/task:rev1",
            tmp_path / "metadata.json",
            tmp_path,
        )
    assert process.terminated is True
    assert process.communicate_calls == 2


def test_repair_and_cleanup_preserve_recoverability() -> None:
    store = TaskStore()
    store.create_with_initial_revision(
        "task", name="task", slug="task", description=None, visibility="private", object_key="k1", owner_id="owner"
    )
    for revision in (1, 2):
        if revision == 2:
            store.create_next_revision("task")
        store.mark_latest_revision_building("task", build_backend="buildkit")
        store.record_success("task", revision, image_ref=f"image:{revision}", image_digest="sha256:" + "a" * 64)
    object_key = stored_revision("task", 2)["tarball_object_key"]
    assert store.mark_task_pack_missing(
        owner_id="owner", task_id="task", revision=2, object_key=object_key, previous_status="ready"
    )
    assert stored_revision("task", 2)["status"] == "failed"
    assert stored_task("task")["current_revision"] == 1
    now = datetime.now(UTC)
    seed_evaluation(
        {
            "id": "ev",
            "task_id": "task",
            "task_revision": 1,
            "status": "running",
            "current_execution": 2,
            "backend_handle": '{"external_id": "old"}',
            "dispatch_job_name": "job",
            "dispatch_reconcile_claimed_by": "worker",
            "max_infrastructure_retries": 1,
            "cleanups": [
                {
                    "execution_number": 2,
                    "runtime": "sandbox_k8s",
                    "backend_handle": '{"external_id": "older"}',
                    "dispatch_job_name": "job",
                    "failure_code": "JobFailed",
                    "failure_detail": "earlier",
                    "retry_after_cleanup": False,
                    "status": "deleted",
                    "next_attempt_at": now,
                    "created_at": now,
                    "updated_at": now,
                }
            ],
        }
    )
    evaluations = EvaluationStore()
    result = evaluations.record_dispatch_job_infrastructure_failure(
        "ev",
        execution_number=2,
        dispatch_job_name="job",
        reconcile_worker_id="worker",
        failure_code="JobFailed",
        detail="failed",
        retry_delay_seconds=1,
    )
    assert result == {"action": "cleanup", "retry": True}
    # A deleted cleanup of the same execution is replaced by a pending one.
    [cleanup] = evaluations.cleanups_for("ev")
    assert cleanup["status"] == "pending"


def test_detached_spawn_failure_is_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pid_path = tmp_path / "runner.pid"
    exit_path = tmp_path / "runner.exit.json"
    token = "token"
    pid_path.write_text(json.dumps({"pid": os.getpid(), "token": token}))

    def fail_spawn(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError("runner missing")

    monkeypatch.setattr(detached_runner.subprocess, "run", fail_spawn)

    assert detached_runner.run(pid_path, exit_path, token, ["missing"]) == 127
    assert not pid_path.exists()
    terminal = json.loads(exit_path.read_text())
    assert terminal["exit_code"] == 127
    assert "runner missing" in terminal["error"]


def test_cleartext_and_external_targets_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="must use HTTPS"):
        _post_resolve(
            data={},
            files={"context": ("context.tar.gz", b"x", "application/gzip")},
            oc_token="secret",
            service_url="http://builder.example.test",
        )
    with pytest.raises(click.ClickException, match="must use HTTPS"):
        make_client("http://localhost:8080", "secret")
    with make_client("http://localhost:8080", None) as client:
        assert client.base_url == httpx.URL("http://localhost:8080/v1/")

    def redirect(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            307,
            headers={"location": "http://storage.example.test/archive"},
        )

    source = tmp_path / "archive.tar.gz"
    source.write_bytes(b"archive")
    with make_client(
        "https://api.example.test",
        "secret",
        transport=httpx.MockTransport(redirect),
    ) as client:
        # The broker upload target is derived from the (HTTPS-validated) client base_url
        # rather than a client-supplied URL, so the cleartext guard now lives in make_client
        # (asserted above). upload_file instead rejects a malformed/empty upload block.
        with pytest.raises(click.ClickException, match="missing workspace/fileset/path"):
            upload_file(client, {"url": "http://storage.example.test/archive"}, source)
        with pytest.raises(click.ClickException, match="must use HTTPS"):
            download_artifact(client, "/artifacts/archive", tmp_path / "download")


def test_gym_paths_are_created_exclusively(tmp_path: Path) -> None:
    target = tmp_path / "target.env"
    target.write_text("attacker-controlled")
    with pytest.raises(FileExistsError):
        write_env_file(target, {"TOKEN": "secret"})
    target.unlink()
    target.symlink_to(tmp_path / "captured")
    with pytest.raises(FileExistsError):
        write_env_file(target, {"TOKEN": "secret"})
    assert not (tmp_path / "captured").exists()

    env_file = tmp_path / "daytona.env"
    env_file.write_text("GYM_AGENT_NAME=test\n")
    work = tmp_path / "work"
    (work / "ev-x2").mkdir(parents=True)
    submit = make_gym_docker_submitter(
        backend_name="gym_sandbox_daytona",
        image=None,
        env_file=str(env_file),
        work_dir=str(work),
        work_volume="gym-work",
        runner=lambda *_args: None,
    )
    with pytest.raises(FileExistsError):
        submit(
            LaunchSpec(
                evaluation_id="ev-x2",
                name="test",
                framework="nemo_gym",
                runner_image_ref="registry.example/gym:frozen",
                image_ref="task:tag",
                parallelism=1,
            )
        )
