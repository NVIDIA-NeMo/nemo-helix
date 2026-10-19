# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from datetime import UTC, datetime
from subprocess import CompletedProcess
from unittest.mock import MagicMock

import pytest

try:
    from scaled_evals.api.repositories.resource_usage_repository import ResourceUsageRepository
    from scaled_evals.dispatch import sandbox_k8s
    from scaled_evals.dispatch import worker as worker_module
    from scaled_evals.dispatch.worker import Dispatcher
    from scaled_evals.models.resource_usage import ResourceUsageSample
    from scaled_evals.models.runtime import LaunchHandle
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("250m", 0.25), ("1000000n", 0.001), ("2", 2.0), ("bad", None)],
)
def test_parse_cpu_cores(value: str, expected: float | None) -> None:
    assert sandbox_k8s._parse_cpu_cores(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("512Ki", 512 * 1024), ("2Gi", 2 * 1024**3), ("10M", 10_000_000), ("bad", None)],
)
def test_parse_memory_bytes(value: str, expected: int | None) -> None:
    assert sandbox_k8s._parse_memory_bytes(value) == expected


def test_kubernetes_sampler_combines_actual_usage_and_workload_sizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pods = """{"items":[{"spec":{"containers":[
      {"resources":{"requests":{"cpu":"250m","memory":"512Mi","nvidia.com/gpu":"1"},
                     "limits":{"cpu":"1","memory":"1Gi","nvidia.com/gpu":"1"}}},
      {"resources":{"requests":{"cpu":"100m","memory":"64Mi"}}}
    ]}}]}"""
    results = iter(
        [
            CompletedProcess([], 0, stdout=pods, stderr=""),
            CompletedProcess([], 0, stdout="pod-a main 125m 256Mi\npod-a helper 25m 32Mi\n", stderr=""),
        ]
    )
    monkeypatch.setattr(sandbox_k8s.shutil, "which", lambda _name: "/usr/bin/kubectl")
    monkeypatch.setattr(sandbox_k8s, "execute_kubectl", lambda *_args, **_kwargs: next(results))
    handle = LaunchHandle(
        backend="sandbox_k8s",
        external_id="ev_1",
        raw={
            "cleanup": {
                "selector": "scaled-evals.nvidia.com/evaluation-id=ev_1",
                "namespace": "evals",
                "verify_ssl": True,
            }
        },
    )

    sample = sandbox_k8s.sample_sandbox_k8s_resources(handle)[0]

    assert sample.source == "kubernetes_metrics_api"
    assert sample.cpu_usage_cores == pytest.approx(0.15)
    assert sample.memory_usage_bytes == 288 * 1024**2
    assert sample.cpu_request_cores == pytest.approx(0.35)
    assert sample.cpu_limit_cores == 1
    assert sample.memory_request_bytes == 576 * 1024**2
    assert sample.memory_limit_bytes == 1024**3
    assert sample.gpu_request == 1


def test_kubernetes_sampler_keeps_requests_when_metrics_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        [
            CompletedProcess(
                [],
                0,
                stdout='{"items":[{"spec":{"containers":[{"resources":{"requests":{"cpu":"1"}}}]}}]}',
                stderr="",
            ),
            CompletedProcess([], 1, stdout="", stderr="metrics unavailable"),
        ]
    )
    monkeypatch.setattr(sandbox_k8s.shutil, "which", lambda _name: "/usr/bin/kubectl")
    monkeypatch.setattr(sandbox_k8s, "execute_kubectl", lambda *_args, **_kwargs: next(results))
    handle = LaunchHandle(
        backend="sandbox_k8s",
        external_id="ev_1",
        raw={"cleanup": {"selector": "evaluation=ev_1", "namespace": "default"}},
    )

    sample = sandbox_k8s.sample_sandbox_k8s_resources(handle)[0]

    assert sample.cpu_request_cores == 1
    assert sample.cpu_usage_cores is None
    assert sample.memory_usage_bytes is None
    assert sample.collection_status == "metrics_unavailable"
    assert sample.collection_error == "metrics unavailable"


def test_resource_usage_repository_upserts_bounded_aggregates() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    observed_at = datetime(2026, 8, 7, tzinfo=UTC)
    sample = ResourceUsageSample(
        source="kubernetes_metrics_api",
        observed_at=observed_at,
        cpu_usage_cores=0.25,
        memory_usage_bytes=1024,
        cpu_request_cores=1,
    )

    ResourceUsageRepository(conn).record_samples("ev_1", execution_number=2, samples=[sample])

    sql, rows = cur.executemany.call_args.args
    assert "ON CONFLICT (evaluation_id, execution_number, component) DO UPDATE" in sql
    assert "sample_count = evaluation_resource_usage.sample_count + 1" in sql
    assert rows[0][:8] == (
        "ev_1",
        2,
        "sandbox",
        "kubernetes_metrics_api",
        "sampled",
        None,
        observed_at,
        observed_at,
    )
    assert rows[0][8:11] == (1, 0.25, 0.25)


def test_resource_usage_repository_lists_attempt_aware_averages() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [{"execution_number": 2, "avg_cpu_cores": 0.25}]

    rows = ResourceUsageRepository(conn).list_for_evaluation("ev_1")

    sql, params = cur.execute.call_args.args
    assert "cpu_usage_cores_sum / NULLIF(cpu_sample_count, 0)" in sql
    assert "ORDER BY execution_number ASC, component ASC" in sql
    assert params == ("ev_1",)
    assert rows == [{"execution_number": 2, "avg_cpu_cores": 0.25}]


def test_dispatcher_persists_optional_backend_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = MagicMock()

    @contextmanager
    def connect():
        yield conn

    repository = MagicMock()
    monkeypatch.setattr(worker_module, "ResourceUsageRepository", lambda _conn: repository)
    sample = ResourceUsageSample(source="test", cpu_usage_cores=0.5)
    backend = MagicMock()
    backend.sample_resources.return_value = [sample]
    handle = LaunchHandle(backend="test", external_id="run-1")

    Dispatcher(connect=connect)._capture_resource_usage("ev_1", execution_number=3, backend=backend, handle=handle)

    backend.sample_resources.assert_called_once_with(handle)
    repository.record_samples.assert_called_once_with("ev_1", execution_number=3, samples=[sample])
