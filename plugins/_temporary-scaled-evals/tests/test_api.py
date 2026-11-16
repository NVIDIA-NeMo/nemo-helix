# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Iterator
from unittest.mock import MagicMock

import anyio
import httpx
import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import app, client, v1
from pydantic import ValidationError
from scaled_evals.api import auth
from scaled_evals.api.db import get_db, get_request_db
from scaled_evals.api.routers import ops
from scaled_evals.api.settings import Settings, settings


@pytest.fixture(autouse=True)
def _user_db_override() -> Iterator[MagicMock]:
    db = MagicMock()
    db.evaluations.quota_usage.return_value = {
        "evaluations_active": 0,
        "sandbox_slots_active": 0,
        "tasks_owned": 0,
    }
    app.dependency_overrides[get_db] = lambda: db
    v1.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_request_db] = lambda: db
    v1.dependency_overrides[get_request_db] = lambda: db
    ops._evaluation_metrics_cache = None
    yield db
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_request_db, None)
    app.dependency_overrides.pop(auth.current_principal, None)
    v1.dependency_overrides.pop(get_db, None)
    v1.dependency_overrides.pop(get_request_db, None)
    v1.dependency_overrides.pop(auth.current_principal, None)


def _clear_http_metrics() -> None:
    ops._HTTP_REQUESTS.clear()
    ops._HTTP_REQUEST_DURATION_SECONDS.clear()
    ops._HTTP_REQUEST_DURATION_COUNT.clear()


@pytest.mark.parametrize(
    ("path", "repository"),
    [
        ("/v1/tasks", "tasks"),
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


def test_healthz_responds_when_worker_thread_capacity_is_exhausted() -> None:
    async def probe() -> None:
        limiter = anyio.to_thread.current_default_thread_limiter()  # ty: ignore[unresolved-attribute]
        original_capacity = limiter.total_tokens
        occupied_worker = object()
        limiter.total_tokens = 1
        await limiter.acquire_on_behalf_of(occupied_worker)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as connection:
                with anyio.fail_after(1):
                    response = await connection.get("/v1/healthz")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
        finally:
            limiter.release_on_behalf_of(occupied_worker)
            limiter.total_tokens = original_capacity

    anyio.run(probe)


def test_metrics(monkeypatch) -> None:  # noqa: ANN001
    scans: list[int] = []

    def status_counts() -> dict[str, int]:
        scans.append(1)
        return dict.fromkeys(ops._KNOWN_EVALUATION_STATUSES, 0)

    monkeypatch.setattr(ops, "_evaluation_status_counts", status_counts)
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
        },
    )
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    assert "scaled_evals_up 1" in response.text
    # A second scrape inside the cache window does not rescan every evaluation.
    assert client.get("/v1/metrics").status_code == 200
    assert len(scans) == 1


def test_metrics_scrape_does_not_probe_dependencies(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        ops,
        "_evaluation_status_counts",
        lambda: dict.fromkeys(ops._KNOWN_EVALUATION_STATUSES, 0),
    )
    ops._READINESS_SNAPSHOT.checks = {"object_store": "ok"}

    def fail_probe() -> None:
        raise AssertionError("metrics must not run live dependency probes")

    monkeypatch.setattr(ops.artifacts, "check_bucket", fail_probe)
    monkeypatch.setattr(ops.buildkit, "check_buildkit", fail_probe)
    monkeypatch.setattr(ops.registry, "check_registry", fail_probe)

    response = client.get("/v1/metrics")

    assert response.status_code == 200
    assert 'scaled_evals_dependency_ready{dependency="object_store"} 1' in response.text


def test_dependency_checks_skip_disabled_build_services(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    monkeypatch.setattr(settings, "registry_enabled", False)
    monkeypatch.setattr(ops.artifacts, "check_bucket", lambda: None)

    def fail_probe() -> None:
        raise AssertionError("disabled dependencies should not be probed")

    monkeypatch.setattr(ops.buildkit, "check_buildkit", fail_probe)
    monkeypatch.setattr(ops.registry, "check_registry", fail_probe)

    checks, required_ok = ops._run_dependency_checks()

    assert required_ok is True
    assert checks["buildkit"] == "skipped: disabled"
    assert checks["registry"] == "skipped: disabled"
    assert checks.keys().isdisjoint({"postgres", "schema", "platform_jobs_controller"})
    assert checks["entity_store"] == "ok"

    # Every route keeps its state in Entity Store, so readiness fails without it.
    def store_down(*_args: object, **_kwargs: object) -> None:
        raise ConnectionError("entities unavailable")

    monkeypatch.setattr(ops.entity_store.entity_client(), "list", store_down)
    checks, required_ok = ops._run_dependency_checks()
    assert required_ok is False and checks["entity_store"] == "fail: ConnectionError"
    assert ops._dependency_is_required("entity_store")
    assert client.get("/v1/readyz").status_code == 503


def test_readyz_reports_dependency_checks(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        ops,
        "_run_dependency_checks",
        lambda: (
            {
                "object_store": "ok",
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
    assert "object_store" in body["checks"]
    assert "buildkit" in body["checks"]
    assert "registry" in body["checks"]
    assert "gym_dispatch" in body["checks"]
    assert "sandbox_k8s_dispatch" in body["checks"]
    assert "stub" not in body
    assert body["status"] in {"ok", "degraded"}
    assert response.status_code in {200, 503}


def test_users_me_reports_owner_backed_capacity(
    monkeypatch: pytest.MonkeyPatch,
    _user_db_override: MagicMock,
) -> None:
    monkeypatch.setattr(settings, "control_plane_per_user_run_limit", 50)
    _user_db_override.evaluations.quota_usage.return_value = {
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
    _user_db_override.evaluations.quota_usage.assert_called_once_with("dev")


@pytest.mark.parametrize("query", ["days=0", "days=91"])
def test_admin_compute_validates_window(query: str) -> None:
    response = client.get(f"/v1/admin/compute?{query}")

    assert response.status_code == 422


@pytest.mark.parametrize("query", ["days=0", "days=91", "examples=0", "examples=11"])
def test_admin_failures_validates_window_and_example_limits(query: str) -> None:
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
def test_admin_failures_validates_custom_datetime_window(query: str) -> None:
    response = client.get(f"/v1/admin/failures?{query}")

    assert response.status_code == 422


def test_hosted_task_image_settings_require_tag_runtime_references() -> None:
    with pytest.raises(
        ValidationError,
        match="SANDBOX_K8S_TASK_IMAGE_REFERENCE_MODE=tag",
    ):
        Settings.model_validate(
            {
                "task_image_hosted_mode": True,
                "task_image_validation_mode": "resolve",
                "task_image_allowed_registries": "artifactory.nvidia.com",
                "sandbox_k8s_task_image_reference_mode": "digest",
            }
        )


def test_hosted_task_image_settings_accept_tag_references() -> None:
    configured = Settings.model_validate(
        {
            "task_image_hosted_mode": True,
            "task_image_validation_mode": "resolve",
            "task_image_allowed_registries": "artifactory.nvidia.com",
            "sandbox_k8s_task_image_reference_mode": "tag",
        }
    )

    assert configured.task_image_hosted_mode is True
