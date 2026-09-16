# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import app, client, v1
from cryptography.fernet import Fernet
from pydantic import ValidationError
from scaled_evals.api import auth
from scaled_evals.api.db import get_db
from scaled_evals.api.routers import ops
from scaled_evals.api.settings import Settings, settings


@pytest.fixture(autouse=True)
def _user_db_override() -> Iterator[MagicMock]:
    db = MagicMock()
    db.users.quota_usage.return_value = {
        "evaluations_active": 0,
        "sandbox_slots_active": 0,
        "tasks_owned": 0,
    }
    app.dependency_overrides[get_db] = lambda: db
    v1.dependency_overrides[get_db] = lambda: db
    yield db
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(auth.current_principal, None)
    v1.dependency_overrides.pop(get_db, None)
    v1.dependency_overrides.pop(auth.current_principal, None)


def _clear_http_metrics() -> None:
    ops._HTTP_REQUESTS.clear()
    ops._HTTP_REQUEST_DURATION_SECONDS.clear()
    ops._HTTP_REQUEST_DURATION_COUNT.clear()


@pytest.mark.parametrize(
    ("path", "repository"),
    [
        ("/v1/tasks", "tasks"),
        ("/v1/credentials", "credentials"),
        ("/v1/config-profiles", "config_profiles"),
        ("/v1/benchmarks", "benchmarks"),
        ("/v1/benchmark-runs", "benchmark_runs"),
        ("/v1/evaluations", "evaluations"),
    ],
)
def test_resource_list_search_reaches_repository(
    path: str,
    repository: str,
    _user_db_override: MagicMock,
) -> None:
    repo = getattr(_user_db_override, repository)
    repo.list.return_value = []

    response = client.get(path, params={"q": "needle"})

    assert response.status_code == 200
    assert repo.list.call_args.kwargs["q"] == "needle"


def test_healthz() -> None:
    response = client.get("/v1/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        ops,
        "_evaluation_status_counts",
        lambda: dict.fromkeys(ops._KNOWN_EVALUATION_STATUSES, 0),
    )
    monkeypatch.setattr(ops, "_fleet_totals", lambda: {})
    monkeypatch.setattr(
        ops,
        "_dispatch_observability_snapshot",
        lambda: {
            "oldest_queued_seconds": 0.0,
            "unclaimed_queued": 0,
            "live_workers": 0,
            "stale_workers": 0,
            "oldest_worker_lease_seconds": 0.0,
            "stuck_jobs": [],
            "backend_failures": [],
            "switchyard_teardown": {},
        },
    )
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    assert "scaled_evals_up 1" in response.text


def test_metrics_scrape_does_not_probe_dependencies(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        ops,
        "_evaluation_status_counts",
        lambda: dict.fromkeys(ops._KNOWN_EVALUATION_STATUSES, 0),
    )
    ops._READINESS_SNAPSHOT.checks = {"postgres": "ok"}

    def fail_probe() -> None:
        raise AssertionError("metrics must not run live dependency probes")

    monkeypatch.setattr(ops, "_postgres_probe", fail_probe)
    monkeypatch.setattr(ops.s3, "check_bucket", fail_probe)
    monkeypatch.setattr(ops.buildkit, "check_buildkit", fail_probe)
    monkeypatch.setattr(ops.registry, "check_registry", fail_probe)

    response = client.get("/v1/metrics")

    assert response.status_code == 200
    assert 'scaled_evals_dependency_ready{dependency="postgres"} 1' in response.text


def test_dependency_checks_skip_disabled_build_services(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    monkeypatch.setattr(settings, "registry_enabled", False)
    monkeypatch.setattr(settings, "build_worker_required", False)
    monkeypatch.setattr(ops, "_postgres_probe", lambda: None)
    monkeypatch.setattr(ops, "_schema_probe", lambda: None)
    monkeypatch.setattr(ops.s3, "check_bucket", lambda: None)

    def fail_probe() -> None:
        raise AssertionError("disabled dependencies should not be probed")

    monkeypatch.setattr(ops.buildkit, "check_buildkit", fail_probe)
    monkeypatch.setattr(ops.registry, "check_registry", fail_probe)

    checks, required_ok = ops._run_dependency_checks()

    assert required_ok is True
    assert checks["buildkit"] == "skipped: disabled"
    assert checks["registry"] == "skipped: disabled"
    assert checks["build_worker"] == "skipped: disabled"


def test_dependency_checks_require_fresh_build_worker(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    monkeypatch.setattr(settings, "registry_enabled", False)
    monkeypatch.setattr(settings, "build_worker_required", True)
    monkeypatch.setattr(ops, "_postgres_probe", lambda: None)
    monkeypatch.setattr(ops, "_schema_probe", lambda: None)
    monkeypatch.setattr(ops.s3, "check_bucket", lambda: None)

    def stale_worker() -> None:
        raise RuntimeError("no fresh build worker heartbeat")

    monkeypatch.setattr(ops, "_build_worker_probe", stale_worker)

    checks, required_ok = ops._run_dependency_checks()

    assert required_ok is False
    assert checks["build_worker"] == "fail: RuntimeError"


def test_dependency_checks_require_platform_jobs_controller(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "platform_build_jobs_enabled", True)
    monkeypatch.setattr(settings, "platform_evaluation_jobs_enabled", False)
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    monkeypatch.setattr(settings, "registry_enabled", False)
    monkeypatch.setattr(settings, "build_worker_required", False)
    monkeypatch.setattr(ops, "_postgres_probe", lambda: None)
    monkeypatch.setattr(ops, "_schema_probe", lambda: None)
    monkeypatch.setattr(ops.s3, "check_bucket", lambda: None)

    def stale_controller() -> None:
        raise RuntimeError("no fresh Platform Jobs controller heartbeat")

    monkeypatch.setattr(ops, "_platform_jobs_controller_probe", stale_controller)

    checks, required_ok = ops._run_dependency_checks()

    assert required_ok is False
    assert checks["platform_jobs_controller"] == "fail: RuntimeError"
    assert checks["build_worker"] == "skipped: disabled"


def test_dependency_checks_require_compatible_schema(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    monkeypatch.setattr(settings, "registry_enabled", False)
    monkeypatch.setattr(settings, "build_worker_required", False)
    monkeypatch.setattr(ops, "_postgres_probe", lambda: None)
    monkeypatch.setattr(ops.s3, "check_bucket", lambda: None)

    def drifted_schema() -> None:
        raise RuntimeError('column "current_execution" does not exist')

    monkeypatch.setattr(ops, "_schema_probe", drifted_schema)

    checks, required_ok = ops._run_dependency_checks()

    assert required_ok is False
    assert checks["schema"] == "fail: RuntimeError"


def test_readyz_reports_dependency_checks(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        ops,
        "_run_dependency_checks",
        lambda: (
            {
                "postgres": "ok",
                "object_store": "ok",
                "build_worker": "ok",
                "buildkit": "skipped: disabled",
                "registry": "ok",
                "gym_dispatch": "skipped: disabled",
                "sandbox_k8s_dispatch": "ok",
            },
            True,
        ),
    )
    response = client.get("/v1/readyz")
    body = response.json()
    assert "checks" in body
    assert "postgres" in body["checks"]
    assert "object_store" in body["checks"]
    assert "build_worker" in body["checks"]
    assert "buildkit" in body["checks"]
    assert "registry" in body["checks"]
    assert "gym_dispatch" in body["checks"]
    assert "sandbox_k8s_dispatch" in body["checks"]
    assert "stub" not in body
    assert body["status"] in {"ok", "degraded"}
    assert response.status_code in {200, 503}


def test_dispatch_worker_health_endpoint_is_required_when_configured(monkeypatch) -> None:  # noqa: ANN001
    class HealthyResponse:
        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        settings,
        "dispatch_worker_health_url",
        "http://scaled-evals-mr-588-dispatch-worker:8081/",
    )
    monkeypatch.setattr(ops.httpx, "get", lambda url, timeout: HealthyResponse())

    assert ops._dependency_is_required("dispatch_worker") is True
    ops._dispatch_worker_probe()


def test_users_me_reports_owner_backed_capacity(
    monkeypatch: pytest.MonkeyPatch,
    _user_db_override: MagicMock,
) -> None:
    monkeypatch.setattr(settings, "control_plane_per_user_run_limit", 50)
    _user_db_override.users.quota_usage.return_value = {
        "evaluations_active": 72,
        "sandbox_slots_active": 49,
        "tasks_owned": 11,
    }

    response = client.get("/v1/users/me")

    assert response.status_code == 200
    assert response.json()["quotas"] == {
        "evaluations_active_max": 50,
        "evaluations_active": 72,
        "tasks_owned": 11,
        "sandbox_slots_max": 50,
        "sandbox_slots_active": 49,
    }
    _user_db_override.users.quota_usage.assert_called_once_with("dev")


def test_admin_usage_reports_actor_counts(
    monkeypatch: pytest.MonkeyPatch,
    _user_db_override: MagicMock,
) -> None:
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "dev")
    v1.dependency_overrides[auth.current_principal] = lambda: auth.CurrentPrincipal(
        owner_type="USER", owner_id="dev", source="starfleet_jwt"
    )
    _user_db_override.users.usage_by_actor.return_value = {
        "total_runs": 3,
        "total_tasks": 12,
        "total_tasks_run": 9,
        "total_evaluation_jobs": 34,
        "total_executions": 39,
        "total_trials": 81,
        "total_benchmark_runs": 4,
        "queued_runs": 1,
        "active_runs": 1,
        "succeeded_runs": 1,
        "failed_runs": 0,
        "cancelled_runs": 0,
        "total_parallelism": 5,
        "avg_runtime_seconds": 12.5,
        "max_runtime_seconds": 12.5,
        "actors": [
            {
                "owner_id": "dev",
                "email": "dev@example.com",
                "username": "dev",
                "display_name": "Dev User",
                "total_runs": 3,
                "queued_runs": 1,
                "active_runs": 1,
                "succeeded_runs": 1,
                "failed_runs": 0,
                "cancelled_runs": 0,
                "total_parallelism": 5,
                "avg_runtime_seconds": 12.5,
                "max_runtime_seconds": 12.5,
                "last_run_at": "2026-07-10T00:00:00Z",
            }
        ],
    }

    response = client.get("/v1/admin/usage")

    assert response.status_code == 200, response.text
    assert response.json()["total_tasks"] == 12
    assert response.json()["total_tasks_run"] == 9
    assert response.json()["total_evaluation_jobs"] == 34
    assert response.json()["total_executions"] == 39
    assert response.json()["total_trials"] == 81
    assert response.json()["total_benchmark_runs"] == 4
    assert response.json()["actors"][0]["owner_id"] == "dev"
    _user_db_override.users.usage_by_actor.assert_called_once_with(limit=20)


def test_admin_failures_reports_categorized_examples(
    monkeypatch: pytest.MonkeyPatch,
    _user_db_override: MagicMock,
) -> None:
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "dev")
    v1.dependency_overrides[auth.current_principal] = lambda: auth.CurrentPrincipal(
        owner_type="USER", owner_id="dev", source="starfleet_jwt"
    )
    _user_db_override.users.failure_summary.return_value = {
        "window_days": 7,
        "window_start": "2026-08-01T00:00:00Z",
        "window_end": "2026-08-08T00:00:00Z",
        "total_failures": 2,
        "timeline": [
            {
                "date": "2026-08-07",
                "total": 2,
                "counts": {"inference_http_504": 2},
                "codes": {"inference_http_504": {"APIStatusError": 2}},
            }
        ],
        "categories": [
            {
                "key": "inference_http_504",
                "count": 2,
                "examples": [
                    {
                        "evaluation_id": "ev_504",
                        "evaluation_name": "gateway failure",
                        "task_id": "task_1",
                        "owner_id": "dev",
                        "owner_label": "Dev User",
                        "runtime": "sandbox_k8s",
                        "failure_code": "APIStatusError",
                        "detail": "inference request returned HTTP 504",
                        "occurred_at": "2026-08-07T00:00:00Z",
                    }
                ],
            }
        ],
    }

    response = client.get("/v1/admin/failures?days=7&examples=1")

    assert response.status_code == 200, response.text
    assert response.json()["total_failures"] == 2
    assert response.json()["categories"][0]["label"] == "Inference HTTP 504"
    assert response.json()["timeline"][0]["counts"] == {"inference_http_504": 2}
    assert response.json()["timeline"][0]["codes"] == {"inference_http_504": {"APIStatusError": 2}}
    assert response.json()["categories"][0]["examples"][0]["evaluation_id"] == "ev_504"
    call = _user_db_override.users.failure_summary.call_args.kwargs
    assert call["window_days"] == 7
    assert call["examples_per_category"] == 1
    assert call["window_start"] < call["window_end"]


def test_admin_compute_reports_coverage_and_actual_usage(
    monkeypatch: pytest.MonkeyPatch,
    _user_db_override: MagicMock,
) -> None:
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "dev")
    v1.dependency_overrides[auth.current_principal] = lambda: auth.CurrentPrincipal(
        owner_type="USER", owner_id="dev", source="starfleet_jwt"
    )
    _user_db_override.users.compute_summary.return_value = {
        "runtime": "all",
        "window_days": 7,
        "window_start": "2026-08-01T00:00:00Z",
        "window_end": "2026-08-08T00:00:00Z",
        "evaluations": 10,
        "sampled_evaluations": 4,
        "samples": 52,
        "avg_cpu_cores": 0.25,
        "peak_cpu_cores": 1.5,
        "avg_cpu_request_cores": 1.0,
        "avg_cpu_limit_cores": 2.0,
        "avg_cpu_request_utilization_percent": 25.0,
        "avg_memory_bytes": 268435456.0,
        "peak_memory_bytes": 536870912,
        "avg_memory_request_bytes": 1073741824.0,
        "avg_memory_limit_bytes": 2147483648.0,
        "avg_memory_request_utilization_percent": 25.0,
        "requested_gpus": 2.0,
        "gpu_utilization_available": False,
        "runtimes": [],
        "timeline": [],
    }

    response = client.get("/v1/admin/compute?days=7")

    assert response.status_code == 200, response.text
    assert response.json()["sampled_evaluations"] == 4
    assert response.json()["avg_cpu_cores"] == 0.25
    assert response.json()["gpu_utilization_available"] is False
    call = _user_db_override.users.compute_summary.call_args.kwargs
    assert call["window_days"] == 7
    assert call["window_start"] < call["window_end"]


@pytest.mark.parametrize("query", ["days=0", "days=91"])
def test_admin_compute_validates_window(query: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "dev")

    response = client.get(f"/v1/admin/compute?{query}")

    assert response.status_code == 422


@pytest.mark.parametrize("query", ["days=0", "days=91", "examples=0", "examples=11"])
def test_admin_failures_validates_window_and_example_limits(
    query: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "dev")

    response = client.get(f"/v1/admin/failures?{query}")

    assert response.status_code == 422


@pytest.mark.parametrize(
    "query",
    [
        "from=2026-08-01T00%3A00%3A00Z",
        "from=2026-08-02T00%3A00%3A00Z&to=2026-08-01T00%3A00%3A00Z",
        "from=2026-01-01T00%3A00%3A00Z&to=2026-08-01T00%3A00%3A00Z",
        "from=2026-08-01T00%3A00%3A00&to=2026-08-02T00%3A00%3A00",
    ],
)
def test_admin_failures_validates_custom_datetime_window(
    query: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "dev")

    response = client.get(f"/v1/admin/failures?{query}")

    assert response.status_code == 422


def test_hosted_task_image_settings_require_tag_runtime_references() -> None:
    with pytest.raises(
        ValidationError,
        match="SANDBOX_K8S_TASK_IMAGE_REFERENCE_MODE=tag",
    ):
        Settings.model_validate(
            {
                "credentials_encryption_key": Fernet.generate_key().decode(),
                "task_image_hosted_mode": True,
                "task_image_validation_mode": "resolve",
                "task_image_allowed_registries": "artifactory.nvidia.com",
                "sandbox_k8s_task_image_reference_mode": "digest",
            }
        )


def test_hosted_task_image_settings_accept_tag_references() -> None:
    configured = Settings.model_validate(
        {
            "credentials_encryption_key": Fernet.generate_key().decode(),
            "task_image_hosted_mode": True,
            "task_image_validation_mode": "resolve",
            "task_image_allowed_registries": "artifactory.nvidia.com",
            "sandbox_k8s_task_image_reference_mode": "tag",
        }
    )

    assert configured.task_image_hosted_mode is True
