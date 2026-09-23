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
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

try:
    import click
    import httpx
    from nemo_scaled_evals_plugin import migrations
    from scaled_evals.api.build import buildkit
    from scaled_evals.api.build.errors import BuildError
    from scaled_evals.api.build.image_builder_service import _post_resolve
    from scaled_evals.api.repositories.build_repository import TaskBuildRepository
    from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
    from scaled_evals.api.repositories.runtime_resource_repository import (
        RuntimeResourceRepository,
    )
    from scaled_evals.api.repositories.switchyard_campaign_repository import (
        SwitchyardCampaignRepository,
    )
    from scaled_evals.api.repositories.task_repository import TaskRepository
    from scaled_evals.api.routers import evaluations as evaluations_router
    from scaled_evals.cli.client import download_artifact, make_client, request, upload_file
    from scaled_evals.dispatch import detached_runner
    from scaled_evals.dispatch import worker as worker_module
    from scaled_evals.dispatch.credentials import write_env_file
    from scaled_evals.dispatch.gym.docker import make_gym_docker_submitter
    from scaled_evals.dispatch.kubectl import execute_kubectl
    from scaled_evals.dispatch.runtime_backend import LaunchSpec
    from scaled_evals.dispatch.worker import Dispatcher
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


class _MigrationResult:
    def fetchone(self) -> tuple[None]:
        return (None,)


class _MigrationConnection:
    def __init__(self) -> None:
        self.autocommit = True
        self.commits = 0
        self.rollbacks = 0

    def execute(self, *_args: Any, **_kwargs: Any) -> _MigrationResult:
        return _MigrationResult()

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def test_build_success_only_advances_a_transitioned_revision() -> None:
    connection = MagicMock()
    connection.transaction.return_value = nullcontext()
    cursor = connection.cursor.return_value.__enter__.return_value
    repository = TaskBuildRepository(connection)

    cursor.rowcount = 0
    repository.record_success("task", 2, image_ref="image:2", image_digest="sha256:" + "a" * 64)
    assert cursor.execute.call_count == 1

    cursor.reset_mock()
    cursor.rowcount = 1
    repository.record_success("task", 2, image_ref="image:2", image_digest="sha256:" + "a" * 64)
    assert cursor.execute.call_count == 2


def test_fresh_schema_is_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    schema_files = [Path("schema/00.sql"), Path("schema/01.sql")]
    migration_files = [Path("migrations/001.sql")]
    connection = _MigrationConnection()
    events: list[tuple[str, bool]] = []

    monkeypatch.setattr(migrations, "_connect", lambda *_args: nullcontext(connection))
    monkeypatch.setattr(migrations, "sql_root", lambda: Path("."))
    monkeypatch.setattr(
        migrations,
        "_sql_files",
        lambda path: migration_files if path.name == "migrations" else schema_files,
    )
    monkeypatch.setattr(
        migrations,
        "_run_file",
        lambda conn, path: events.append((path.as_posix(), conn.autocommit)),
    )

    assert migrations.apply_sql("unused") == (2, 1)
    assert events == [
        ("schema/00.sql", False),
        ("schema/01.sql", False),
        ("migrations/001.sql", True),
    ]
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.autocommit is True

    connection = _MigrationConnection()
    monkeypatch.setattr(migrations, "_connect", lambda *_args: nullcontext(connection))

    def fail_second_schema(_conn: _MigrationConnection, path: Path) -> None:
        if path == schema_files[1]:
            raise RuntimeError("interrupted schema load")

    monkeypatch.setattr(migrations, "_run_file", fail_second_schema)
    with pytest.raises(RuntimeError, match="interrupted schema load"):
        migrations.apply_sql("unused")
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.autocommit is True


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


def test_p1_cleanup_recovery_and_cancellation_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = MagicMock()
    connection = MagicMock()
    connection.transaction.return_value = nullcontext()
    connection.cursor.return_value.__enter__.return_value = cursor

    RuntimeResourceRepository(connection).claim_due_switchyard_teardown(
        claim_timeout=30,
        worker_id="worker-a",
    )
    claim_sql = cursor.execute.call_args.args[0]
    assert "r.status IN ('draining', 'deleting', 'delete_failed')" in claim_sql

    cursor.reset_mock()
    SwitchyardCampaignRepository(connection).mark_delete_unavailable(
        "bmr-missing",
        worker_id="worker-a",
        detail="lease missing after 5 attempts",
    )
    terminal_sql, terminal_params = cursor.execute.call_args.args
    assert "SET status = 'deleted'" in terminal_sql
    assert "resource_name IS NOT NULL" not in terminal_sql
    assert terminal_params == ("lease missing after 5 attempts", "bmr-missing", "worker-a")

    campaign_events: list[str] = []

    class _CampaignRepository:
        def __init__(self, _connection: Any) -> None:
            pass

        def mark_delete_failed(self, *_args: Any, **_kwargs: Any) -> None:
            campaign_events.append("retry")

        def mark_delete_unavailable(self, *_args: Any, **_kwargs: Any) -> None:
            campaign_events.append("terminal")

    monkeypatch.setattr(worker_module, "SwitchyardCampaignRepository", _CampaignRepository)
    dispatcher = Dispatcher(
        connect=cast(Any, lambda: nullcontext(connection)),
        switchyard=MagicMock(),
        worker_id="worker-a",
    )
    campaign = {
        "benchmark_run_id": "bmr-missing",
        "status": "deleting",
        "resource_name": None,
        "metadata": {},
    }
    dispatcher.delete_switchyard_campaign({**campaign, "claim_attempt": 4})
    dispatcher.delete_switchyard_campaign({**campaign, "claim_attempt": 5})
    assert campaign_events == ["retry", "terminal"]

    request_events: list[str] = []
    row = {"id": "ev-cancel"}

    class _Evaluations:
        def cancel(self, _evaluation_id: str) -> tuple[dict[str, str], bool]:
            return row, True

    class _Database:
        evaluations = _Evaluations()

        def commit(self) -> None:
            request_events.append("commit")

    def teardown(_db: Any, cancelled: dict[str, str]) -> dict[str, str]:
        request_events.append("teardown")
        return cancelled

    monkeypatch.setattr(evaluations_router, "teardown_cancelled_evaluation", teardown)
    monkeypatch.setattr(evaluations_router, "_response", lambda value: value)
    assert evaluations_router.cancel_evaluation("ev-cancel", cast(Any, _Database())) == row
    assert request_events == ["commit", "teardown"]


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


def test_repair_and_cleanup_sql_preserve_recoverability() -> None:
    connection = MagicMock()
    connection.transaction.return_value = nullcontext()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.rowcount = 1

    assert TaskRepository(connection).mark_task_pack_missing(
        owner_id="owner",
        task_id="task",
        revision=3,
        object_key="tasks/task/rev/3.tar.gz",
        previous_status="ready",
    )
    repair_sql = cursor.execute.call_args_list[1].args[0]
    assert "latest.revision IS NOT NULL" in repair_sql

    cursor.reset_mock()
    cursor.fetchone.side_effect = [
        {
            "id": "ev",
            "status": "running",
            "runtime": "sandbox_k8s",
            "backend_handle": {"external_id": "old"},
            "benchmark_run_id": None,
            "infrastructure_retries": 0,
            "max_infrastructure_retries": 1,
        },
        {"benchmark_available": True},
        {"id": 7},
    ]
    result = EvaluationRepository(connection).record_dispatch_job_infrastructure_failure(
        "ev",
        execution_number=2,
        dispatch_job_name="job",
        reconcile_worker_id="worker",
        failure_code="JobFailed",
        detail="failed",
        retry_delay_seconds=1,
    )
    insert_sql = next(
        call.args[0]
        for call in cursor.execute.call_args_list
        if "INSERT INTO evaluation_execution_cleanups" in call.args[0]
    )
    assert result == {"action": "cleanup", "retry": True}
    assert "WHERE evaluation_execution_cleanups.status = 'deleted'" in insert_sql
    assert "status = 'pending'" in insert_sql


def test_expired_runtime_claim_and_worker_namespace_are_portable() -> None:
    connection = MagicMock()
    connection.transaction.return_value = nullcontext()
    cursor = connection.cursor.return_value.__enter__.return_value

    RuntimeResourceRepository(connection).claim_due_switchyard_teardown(
        claim_timeout=30,
        worker_id="worker",
    )
    sql, params = cursor.execute.call_args.args
    assert "r.status = 'deleting'" in sql
    assert "r.drain_until IS NULL" in sql
    assert params == (30, 30, "worker")

    workers = (Path(__file__).parents[1] / "deploy/k8s/workers.yaml").read_text()
    assert 'namespace="$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)"' in workers
    assert "namespace: ${namespace}" in workers
    assert "namespace: nemo-helix-scaled-evals" not in workers


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
