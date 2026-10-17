# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from subprocess import CompletedProcess
from unittest.mock import MagicMock

import pytest

try:
    from scaled_evals.api.evaluation_store import EvaluationStore
    from scaled_evals.dispatch import sandbox_k8s
    from scaled_evals.dispatch import worker as worker_module
    from scaled_evals.dispatch.worker import Dispatcher
    from scaled_evals.models.resource_usage import ResourceUsageSample
    from scaled_evals.models.runtime import LaunchHandle
    from task_fixtures import seed_evaluation, seed_task_revision
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


def test_resource_samples_fold_into_attempt_aware_aggregates() -> None:
    seed_task_revision("task_x")
    seed_evaluation({"id": "ev_1"})
    first, last = datetime(2026, 8, 7, tzinfo=UTC), datetime(2026, 8, 7, 0, 1, tzinfo=UTC)
    store = EvaluationStore()

    store.record_resource_samples(
        "ev_1",
        execution_number=2,
        samples=[
            ResourceUsageSample(
                source="kubernetes_metrics_api", observed_at=last, cpu_usage_cores=0.25, cpu_request_cores=1
            ),
            ResourceUsageSample(source="kubernetes_metrics_api", observed_at=first, cpu_usage_cores=0.75),
        ],
    )
    store.record_resource_samples("ev_1", execution_number=1, samples=[ResourceUsageSample(source="test")])

    first_execution, second_execution = store.resource_usage_for("ev_1")
    assert (first_execution["execution_number"], first_execution["avg_cpu_cores"]) == (1, None)
    assert (second_execution["component"], second_execution["sample_count"]) == ("sandbox", 2)
    assert (second_execution["first_observed_at"], second_execution["last_observed_at"]) == (first, last)
    assert (second_execution["avg_cpu_cores"], second_execution["peak_cpu_cores"]) == (0.5, 0.75)
    assert second_execution["cpu_request_cores"] == 1


def test_dispatcher_persists_optional_backend_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MagicMock()
    monkeypatch.setattr(worker_module, "EvaluationStore", lambda *_: store)
    sample = ResourceUsageSample(source="test", cpu_usage_cores=0.5)
    backend = MagicMock()
    backend.sample_resources.return_value = [sample]
    handle = LaunchHandle(backend="test", external_id="run-1")

    Dispatcher()._capture_resource_usage("ev_1", execution_number=3, backend=backend, handle=handle)

    backend.sample_resources.assert_called_once_with(handle)
    store.record_resource_samples.assert_called_once_with("ev_1", execution_number=3, samples=[sample])
