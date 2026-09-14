# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined Harbor exports: real archive bytes, durable queue, and API behavior."""

import hashlib
import io
import json
import tarfile
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("scaled_evals")

from api_test_fixture import app, v1
from scaled_evals.api import s3
from scaled_evals.api.db import Database, get_db
from scaled_evals.api.repositories.base_repository import Conflict, NotFound
from scaled_evals.api.repositories.benchmark_archive_repository import BenchmarkArchiveRepository
from scaled_evals.api.settings import settings
from scaled_evals.benchmark_archive import BenchmarkArchiveError, build_benchmark_archive
from scaled_evals.dispatch.switchyard_archive import check_campaign_evidence
from scaled_evals.dispatch.worker import Dispatcher

NOW = datetime(2026, 9, 11, tzinfo=UTC)


def member(evaluation_id="ev_1", **changes):
    return {
        "id": evaluation_id,
        "task_id": "task_1",
        "task_revision": 1,
        "status": "succeeded",
        "current_execution": 1,
        "archive_object_key": f"evaluations/{evaluation_id}/results.tar.gz",
        "archive_built_at": NOW.isoformat(),
        "archive_size_bytes": 1,
        **changes,
    }


def job(**changes):
    return {
        "benchmark_run_id": "bmr_1",
        "generation": "gen",
        "claim_token": "claim",
        "status": "building",
        "members": [member()],
        **changes,
    }


def tar_bytes(files, *, unsafe=None):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, value in files.items():
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            info = tarfile.TarInfo(f"artifacts/{name}")
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
        if unsafe:
            info = tarfile.TarInfo(unsafe)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            archive.addfile(info)
    return output.getvalue()


def harbor_files(*, reward=1, error=None, cost=0.5):
    config = {
        "trial_name": "task__same",
        "task": {"path": "/tasks/task"},
        "agent": {"name": "oracle"},
    }
    trial = {
        "id": "c4c19ebd-d192-4d2f-a433-3b123c707916",
        "task_name": "task",
        "trial_name": "task__same",
        "trial_uri": "file:///jobs/old/task__same",
        "task_id": {"path": "/tasks/task"},
        "task_checksum": "hash",
        "config": config,
        "agent_info": {"name": "oracle", "version": "1"},
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
        "agent_result": {"n_input_tokens": 10, "n_output_tokens": 2, "cost_usd": cost},
        "exception_info": (
            {
                "exception_type": error,
                "exception_message": "failed",
                "exception_traceback": "",
                "occurred_at": NOW.isoformat(),
            }
            if error
            else None
        ),
    }
    return {
        "config.json": {
            "job_name": "old",
            "agents": [{"name": "oracle"}],
            "tasks": [{"path": "/tasks/task"}],
        },
        "result.json": {
            "n_total_trials": 1,
            "started_at": NOW.isoformat(),
            "finished_at": NOW.isoformat(),
            "stats": {"n_retries": 2},
        },
        "task__same/config.json": config,
        "task__same/result.json": trial,
        "task__same/agent/trajectory.json": {"steps": ["evidence"]},
        "task__same/verifier/reward.txt": b"1",
        "scaled-evals-provenance.json": {"evaluation_id": "source"},
        "scaled-evals-sbom.cdx.json": {"bomFormat": "CycloneDX"},
    }


def build(monkeypatch, tmp_path, sources, *, benchmark_artifacts=None, listing=None):
    members = [
        member(f"ev_{i}", task_id=f"task_{i}", task_slug=f"smoke-{i}", archive_size_bytes=len(data))
        for i, data in enumerate(sources, 1)
    ]
    blobs = {m["archive_object_key"]: data for m, data in zip(members, sources, strict=True)}
    blobs.update(benchmark_artifacts or {})
    objects = [
        {"key": key, "size_bytes": len(body), "updated_at": NOW.isoformat()}
        for key, body in (benchmark_artifacts or {}).items()
    ]
    monkeypatch.setattr(s3, "list_objects", listing or (lambda prefix: objects))
    monkeypatch.setattr(s3, "stream_object", lambda key: iter([blobs[key]]))
    captured = {}

    def upload(path, key, *, content_type):
        captured.update(key=key, content_type=content_type)
        destination = tmp_path / "result.tar.gz"
        destination.write_bytes(path.read_bytes())
        return destination.stat().st_size

    monkeypatch.setattr(s3, "upload_file", upload)
    built = build_benchmark_archive(
        job(members=members), check_claim=lambda: None, evidence_checks=(check_campaign_evidence,)
    )
    assert built["size_bytes"] > 0
    assert built["sha256"] == hashlib.sha256((tmp_path / "result.tar.gz").read_bytes()).hexdigest()
    assert captured["key"] == "benchmark-runs/bmr_1/archives/gen/claim.tar.gz"
    files = {}
    with tarfile.open(tmp_path / "result.tar.gz") as archive:
        for item in archive:
            if item.isfile():
                body = archive.extractfile(item)
                assert body is not None
                with body:
                    files[item.name] = body.read()
    return files


def test_export_combines_trials_counts_costs_and_preserves_member_evidence(monkeypatch, tmp_path):
    files = build(
        monkeypatch,
        tmp_path,
        [
            tar_bytes(harbor_files()),
            tar_bytes(harbor_files(reward=None, error="AgentError", cost=1.5)),
        ],
    )
    result = json.loads(files["bmr_1/result.json"])
    assert result["n_total_trials"] == 2
    stats = result["stats"]
    assert stats["n_completed_trials"] == 2
    assert stats["n_errored_trials"] == 1
    assert stats["n_input_tokens"] == 20
    assert stats["cost_usd"] == 2.0
    assert stats["n_retries"] == 4
    assert stats["evals"]["oracle__adhoc"]["reward_stats"]["reward"]["1"] == ["ev_1__task__same"]
    assert stats["evals"]["oracle__adhoc"]["exception_stats"] == {"AgentError": ["ev_2__task__same"]}
    for index in (1, 2):
        trial = f"ev_{index}__task__same"
        assert f"bmr_1/{trial}/agent/trajectory.json" in files
        data = json.loads(files[f"bmr_1/{trial}/result.json"])
        assert data["trial_name"] == data["config"]["trial_name"] == trial
        assert data["trial_uri"] == f"bmr_1/{trial}"
        metadata = f"bmr_1/_scaled_evals/evaluations/ev_{index}"
        assert f"{metadata}/scaled-evals-provenance.json" in files
        assert f"{metadata}/scaled-evals-sbom.cdx.json" in files
        assert json.loads(files[f"{metadata}/task__same/result.json"])["trial_name"] == "task__same"
    manifest = json.loads(files["bmr_1/scaled-evals-benchmark-archive.json"])
    assert not manifest["partial"]
    assert len(manifest["members"]) == 2
    assert len(manifest["members"][0]["archive_sha256"]) == 64


def test_export_reports_members_without_trials(monkeypatch, tmp_path):
    files = build(monkeypatch, tmp_path, [tar_bytes(harbor_files()), tar_bytes({"error.log": b"failed"})])
    manifest = json.loads(files["bmr_1/scaled-evals-benchmark-archive.json"])
    assert manifest["partial"] is True
    assert manifest["members"][1]["missing"] == ["config.json", "result.json", "trial directories"]
    assert files["bmr_1/_scaled_evals/evaluations/ev_2/error.log"] == b"failed"


@pytest.mark.parametrize("unsafe", ["artifacts/link", "artifacts/../../escape", "/absolute"])
def test_export_rejects_unsafe_tar_members(monkeypatch, tmp_path, unsafe):
    with pytest.raises(BenchmarkArchiveError, match="unsafe|regular"):
        build(monkeypatch, tmp_path, [tar_bytes(harbor_files(), unsafe=unsafe)])
    assert not (tmp_path / "result.tar.gz").exists()


def test_export_enforces_aggregate_limits(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "benchmark_archive_max_files", 10)
    with pytest.raises(BenchmarkArchiveError, match="limits"):
        build(monkeypatch, tmp_path, [tar_bytes(harbor_files()), tar_bytes(harbor_files())])


def test_export_rejects_changed_source_size(monkeypatch):
    monkeypatch.setattr(s3, "stream_object", lambda key: iter([b"wrong"]))
    with pytest.raises(BenchmarkArchiveError, match="size changed"):
        build_benchmark_archive(job(), check_claim=lambda: None)


@pytest.fixture
def api_db():
    db = MagicMock(spec=Database)
    db.benchmark_runs.exists.return_value = True
    v1.dependency_overrides[get_db] = lambda: db
    yield db
    v1.dependency_overrides.pop(get_db)


def test_api_queues_and_reports_snapshot_without_private_storage_fields(api_db):
    api_db.benchmark_archives.request.return_value = job(status="queued")
    client = TestClient(app)
    response = client.post("/v1/benchmark-runs/bmr_1/archive", json={})
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert data["members"][0]["current_execution"] == 1
    assert "claim_token" not in data
    assert data["download"] is None
    api_db.commit.assert_called_once()


@pytest.mark.parametrize(
    "exception, status",
    [
        (Conflict("run_not_terminal", "all members must finish"), 409),
        (NotFound("benchmark_run", "benchmark run not found"), 404),
    ],
)
def test_api_archive_rejects_unavailable_run(api_db, exception, status):
    api_db.benchmark_archives.request.side_effect = exception
    client = TestClient(app)
    response = client.post("/v1/benchmark-runs/bmr_1/archive", json={})
    assert response.status_code == status
    api_db.commit.assert_not_called()


def test_api_missing_and_ready_archive(api_db, monkeypatch):
    client = TestClient(app)
    api_db.benchmark_archives.get.return_value = None
    assert client.get("/v1/benchmark-runs/bmr_1/archive").json()["status"] == "missing"
    assert client.get("/v1/benchmark-runs/bmr_1/archive/download").status_code == 404
    api_db.benchmark_archives.get.return_value = job(status="ready", object_key="private/key", sha256="a" * 64)
    data = client.get("/v1/benchmark-runs/bmr_1/archive").json()
    assert data["download"] == "/benchmark-runs/bmr_1/archive/download"
    assert data["sha256"] == "a" * 64
    assert "object_key" not in data
    monkeypatch.setattr(s3, "stream_object", lambda key: iter([b"archive"]))
    response = client.get("/v1/benchmark-runs/bmr_1/archive/download")
    assert response.content == b"archive"
    assert "bmr_1-results.tar.gz" in response.headers["content-disposition"]


def test_worker_records_build_and_failure(monkeypatch):
    repo = MagicMock(spec=BenchmarkArchiveRepository)
    monkeypatch.setattr("scaled_evals.dispatch.worker.BenchmarkArchiveRepository", lambda conn: repo)

    @contextmanager
    def connect():
        yield MagicMock()

    worker = Dispatcher(connect=connect)
    builder = MagicMock(return_value={"object_key": "key", "size_bytes": 123})
    monkeypatch.setattr("scaled_evals.dispatch.worker.build_benchmark_archive", builder)
    worker.build_benchmark_archive(job())
    repo.validate_members.assert_called_once_with("bmr_1", [member()])
    repo.finish.assert_called_once_with(job(), object_key="key", size_bytes=123)
    repo.fail.assert_not_called()
    builder.side_effect = ValueError("missing archive")
    worker.build_benchmark_archive(job())
    repo.fail.assert_called_once_with(job(), "missing archive")


def test_export_groups_equivalent_integer_and_float_rewards(monkeypatch, tmp_path):
    files = build(
        monkeypatch,
        tmp_path,
        [
            tar_bytes(harbor_files(reward=1)),
            tar_bytes(harbor_files(reward=1.0)),
        ],
    )
    result = json.loads(files["bmr_1/result.json"])
    distribution = result["stats"]["evals"]["oracle__adhoc"]["reward_stats"]["reward"]
    assert distribution == {"1": ["ev_1__task__same", "ev_2__task__same"]}


def test_export_retains_two_task_groups_with_two_attempts_each(monkeypatch, tmp_path):
    source = harbor_files()
    for key, value in list(source.items()):
        if key.startswith("task__same/"):
            source[key.replace("task__same/", "task__second/")] = deepcopy(value)
    source["result.json"]["n_total_trials"] = 2
    files = build(monkeypatch, tmp_path, [tar_bytes(source), tar_bytes(source)])
    trials = [
        json.loads(value)
        for key, value in files.items()
        if key.startswith("bmr_1/ev_") and key.endswith("/result.json")
    ]
    assert Counter(t["task_name"] for t in trials) == {"smoke-1": 2, "smoke-2": 2}
    for trial in trials:
        name = trial["task_name"]
        assert trial["config"]["task"]["path"] == f"/tasks/{name}"
        assert trial["task_id"] == {"path": f"/tasks/{name}"}
        assert "name" not in trial["config"]["task"]  # Not a Harbor package reference.
        config = json.loads(files[f"bmr_1/{trial['trial_name']}/config.json"])
        assert config["task"] == trial["config"]["task"]
    config = json.loads(files["bmr_1/config.json"])
    assert {task["path"] for task in config["tasks"]} == {"/tasks/smoke-1", "/tasks/smoke-2"}
    original = json.loads(files["bmr_1/_scaled_evals/evaluations/ev_1/task__same/result.json"])
    assert original["task_name"] == "task"
    assert original["config"]["task"]["path"] == "/tasks/task"


def test_export_keeps_distinct_local_tasks_within_member(monkeypatch, tmp_path):
    source = harbor_files()
    other = deepcopy(harbor_files())
    other["task__same/config.json"]["task"]["path"] = "/other/task"
    for key, value in other.items():
        if key.startswith("task__same/"):
            source[key.replace("task__same/", "other__trial/")] = value
    source["result.json"]["n_total_trials"] = 2
    files = build(monkeypatch, tmp_path, [tar_bytes(source)])
    names = {
        json.loads(value)["task_name"]
        for key, value in files.items()
        if key.startswith("bmr_1/ev_") and key.endswith("/result.json")
    }
    assert len(names) == 2
    assert all(name.startswith("smoke-1__task__") for name in names)


def test_export_preserves_package_task_identity(monkeypatch, tmp_path):
    source = harbor_files()
    task = {"name": "harbor/hello-world", "ref": "1"}
    source["task__same/config.json"]["task"] = task
    source["task__same/result.json"]["task_name"] = "harbor/hello-world"
    source["task__same/result.json"]["task_id"] = {
        "org": "harbor",
        "name": "hello-world",
        "ref": "1",
    }
    source["config.json"]["tasks"] = [task]
    files = build(monkeypatch, tmp_path, [tar_bytes(source)])
    result = json.loads(files["bmr_1/ev_1__task__same/result.json"])
    assert result["task_name"] == "harbor/hello-world"
    assert result["config"]["task"] == task


@pytest.mark.parametrize("digest_prefix", ["", "sha256:"])
def test_export_captures_shared_benchmark_artifacts_once(monkeypatch, tmp_path, digest_prefix):
    key = "benchmark-runs/bmr_1/artifacts/switchyard/routing_stats_final.json"
    stats = b'{"requests": 4, "cost_usd": 0.25}'
    source = harbor_files()
    source["switchyard/campaign_evidence.json"] = {
        "status": "ready",
        "routing_stats_object_key": key,
        "routing_stats_sha256": digest_prefix + hashlib.sha256(stats).hexdigest(),
    }
    binary = b"\x00\xffbenchmark evidence"
    artifacts = {key: stats, "benchmark-runs/bmr_1/artifacts/trace.bin": binary}
    files = build(
        monkeypatch,
        tmp_path,
        [tar_bytes(source), tar_bytes(source)],
        benchmark_artifacts=artifacts,
    )
    manifest = json.loads(files["bmr_1/scaled-evals-benchmark-archive.json"])
    assert manifest["partial"] is False
    assert manifest["unavailable_benchmark_evidence"] == []
    assert len(manifest["benchmark_artifacts"]) == 2
    for entry in manifest["benchmark_artifacts"]:
        body = artifacts[entry["object_key"]]
        assert files[f"bmr_1/{entry['path']}"] == body
        assert entry["sha256"] == hashlib.sha256(body).hexdigest()
        assert entry["size_bytes"] == len(body)
    assert files["bmr_1/_scaled_evals/benchmark/artifacts/switchyard/routing_stats_final.json"] == stats
    for member in manifest["members"]:
        path = f"bmr_1/_scaled_evals/evaluations/{member['id']}/switchyard/campaign_evidence.json"
        assert json.loads(files[path]) == source["switchyard/campaign_evidence.json"]


@pytest.mark.parametrize("suffix", ["../escape", "/absolute", "a/../escape", "a//b", "a\\b", "."])
def test_export_rejects_unsafe_benchmark_artifact_paths(monkeypatch, tmp_path, suffix):
    with pytest.raises(BenchmarkArchiveError, match="unsafe"):
        build(
            monkeypatch,
            tmp_path,
            [tar_bytes(harbor_files())],
            benchmark_artifacts={f"benchmark-runs/bmr_1/artifacts/{suffix}": b"bad"},
        )


def test_export_rejects_benchmark_artifacts_outside_run(monkeypatch, tmp_path):
    with pytest.raises(BenchmarkArchiveError, match="outside"):
        build(
            monkeypatch,
            tmp_path,
            [tar_bytes(harbor_files())],
            benchmark_artifacts={"benchmark-runs/bmr_other/artifacts/stats.json": b"bad"},
        )


@pytest.mark.parametrize("changed", ["size", "listing", "duplicate"])
def test_export_rejects_changed_or_duplicate_benchmark_artifacts(monkeypatch, tmp_path, changed):
    key = "benchmark-runs/bmr_1/artifacts/stats.json"
    item = {"key": key, "size_bytes": 3, "updated_at": NOW.isoformat()}
    pages = iter([[item], [{**item, "updated_at": "later"}]])

    def listing(prefix):
        if changed == "size":
            return [{**item, "size_bytes": 2}]
        if changed == "duplicate":
            return [item, item]
        return next(pages)

    with pytest.raises(BenchmarkArchiveError, match="changed|duplicate"):
        build(
            monkeypatch,
            tmp_path,
            [tar_bytes(harbor_files())],
            benchmark_artifacts={key: b"abc"},
            listing=listing,
        )


@pytest.mark.parametrize("limit", ["files", "bytes"])
def test_benchmark_artifacts_share_member_resource_limits(monkeypatch, tmp_path, limit):
    source = harbor_files()
    if limit == "files":
        monkeypatch.setattr(settings, "benchmark_archive_max_files", len(source))
        body = b"small"
    else:
        monkeypatch.setattr(settings, "benchmark_archive_max_source_bytes", 10000)
        body = b"x" * 10001
    with pytest.raises(BenchmarkArchiveError, match="limits"):
        build(
            monkeypatch,
            tmp_path,
            [tar_bytes(source)],
            benchmark_artifacts={"benchmark-runs/bmr_1/artifacts/trace.bin": body},
        )


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("digest_prefix", ["", "sha256:"])
def test_export_rejects_missing_or_mismatched_campaign_evidence(monkeypatch, tmp_path, present, digest_prefix):
    key = "benchmark-runs/bmr_1/artifacts/switchyard/routing_stats_final.json"
    source = harbor_files()
    source["switchyard/campaign_evidence.json"] = {
        "status": "ready",
        "routing_stats_object_key": key,
        "routing_stats_sha256": digest_prefix + "0" * 64,
    }
    with pytest.raises(BenchmarkArchiveError, match="missing or changed"):
        build(
            monkeypatch,
            tmp_path,
            [tar_bytes(source)],
            benchmark_artifacts={key: b"changed"} if present else {},
        )


def test_export_marks_unavailable_campaign_evidence_partial(monkeypatch, tmp_path):
    source = harbor_files()
    source["switchyard/campaign_evidence.json"] = {"status": "unavailable"}
    files = build(monkeypatch, tmp_path, [tar_bytes(source)])
    manifest = json.loads(files["bmr_1/scaled-evals-benchmark-archive.json"])
    assert manifest["partial"] is True
    assert manifest["unavailable_benchmark_evidence"] == [{"evaluation_id": "ev_1", "kind": "switchyard_campaign"}]


def test_mounted_openapi_includes_benchmark_archive_contract():
    schema = app.openapi()
    operations = schema["paths"]["/v1/benchmark-runs/{run_id}/archive"]
    assert "202" in operations["post"]["responses"]
    assert "200" in operations["get"]["responses"]
    assert "get" in schema["paths"]["/v1/benchmark-runs/{run_id}/archive/download"]
    fields = schema["components"]["schemas"]["BenchmarkArchiveResponse"]["properties"]
    assert {"members", "download", "generation", "partial", "sha256"} <= fields.keys()
    assert not {"claim_token", "object_key", "attempts"} & fields.keys()


@pytest.mark.parametrize("status", [None, "building", "failed"])
def test_idle_dispatcher_processes_benchmark_archive_queue(monkeypatch, status):
    repo = MagicMock(spec=BenchmarkArchiveRepository)
    repo.claim.return_value = job(status=status) if status else None
    monkeypatch.setattr("scaled_evals.dispatch.worker.BenchmarkArchiveRepository", lambda conn: repo)
    monkeypatch.setattr(settings, "dispatch_kubernetes_jobs_enabled", False)

    @contextmanager
    def connect():
        yield MagicMock()

    worker = Dispatcher(connect=connect)
    for method in (
        "claim_next_switchyard_teardown",
        "claim_next_switchyard_campaign_cleanup",
        "claim_next_switchyard_campaign_finalization",
        "claim_next_switchyard_campaign_deletion",
        "claim_next",
        "claim_next_evidence",
        "claim_next_archive",
    ):
        monkeypatch.setattr(worker, method, lambda: None)
    builder = MagicMock()
    monkeypatch.setattr(worker, "build_benchmark_archive", builder)
    assert worker.work_once() is (status is not None)
    repo.claim.assert_called_once_with(claim_timeout=worker.claim_timeout)
    assert builder.call_count == (1 if status == "building" else 0)


@pytest.mark.parametrize("ownership_lost", [False, True])
def test_archive_heartbeat_distinguishes_transient_errors_from_lost_ownership(monkeypatch, ownership_lost):
    import threading
    from types import SimpleNamespace

    repo = MagicMock(spec=BenchmarkArchiveRepository)
    repo.heartbeat.side_effect = [False] if ownership_lost else [RuntimeError("temporary DB failure"), True]
    monkeypatch.setattr("scaled_evals.dispatch.worker.BenchmarkArchiveRepository", lambda conn: repo)
    stop = MagicMock()
    stop.wait.side_effect = [False, False, True]
    lost = threading.Event()
    events = iter([stop, lost])

    def make_thread(*, target, daemon):
        thread = MagicMock()
        # Drive the heartbeat deterministically without sleeps or a real thread.
        thread.start.side_effect = target
        return thread

    monkeypatch.setattr(
        "scaled_evals.dispatch.worker.threading",
        SimpleNamespace(Event=lambda: next(events), Thread=make_thread),
    )

    @contextmanager
    def connect():
        yield MagicMock()

    def build_archive(job, *, check_claim, evidence_checks):
        assert evidence_checks == (check_campaign_evidence,)
        check_claim()
        return {"object_key": "archive", "size_bytes": 123}

    monkeypatch.setattr("scaled_evals.dispatch.worker.build_benchmark_archive", build_archive)
    Dispatcher(connect=connect).build_benchmark_archive(job())
    if ownership_lost:
        repo.finish.assert_not_called()
        repo.fail.assert_called_once_with(job(), "benchmark archive worker lease lost")
    else:
        assert repo.heartbeat.call_count == 2
        repo.fail.assert_not_called()
        repo.finish.assert_called_once_with(job(), object_key="archive", size_bytes=123)
