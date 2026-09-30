# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any
from unittest.mock import MagicMock

import pytest

try:
    import httpx
    from click.testing import CliRunner
    from scaled_evals.api.auth import CurrentPrincipal
    from scaled_evals.api.repositories.benchmark_run_repository import BenchmarkRunRepository
    from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
    from scaled_evals.api.routers import benchmark_runs as benchmark_runs_router
    from scaled_evals.api.routers.benchmark_runs import _create_command, _reproduce_request
    from scaled_evals.api.runnability import (
        BlockedPreflight,
        _append_reference_checks,
        _reference_shape_error,
        preflight_benchmark_run,
    )
    from scaled_evals.api.schemas.benchmark_runs import CreateBenchmarkRunRequest
    from scaled_evals.api.settings import settings
    from scaled_evals.cli import client as client_module
    from scaled_evals.cli.main import cli
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_member_profile_references_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        _reference_shape_error(
            CreateBenchmarkRunRequest(
                name="suite",
                benchmark_id="bm_suite",
                member_framework_profile_ids={"outside": "cfg_member"},
            )
        )
        == "member_framework_profile_ids key must be a task_ id: outside"
    )
    assert (
        _reference_shape_error(
            CreateBenchmarkRunRequest(
                name="suite",
                benchmark_id="bm_suite",
                member_framework_profile_ids={"task_0": "invalid"},
            )
        )
        == "member_framework_profile_ids[task_0] must be a cfg_ id"
    )

    evaluations = MagicMock()
    db = MagicMock(evaluations=evaluations)
    body = CreateBenchmarkRunRequest(
        name="suite",
        benchmark_id="bm_suite",
        framework_profile_id="cfg_default",
        member_framework_profile_ids={"task_0": "cfg_member"},
    )
    assert (
        _append_reference_checks(
            db,
            body,
            CurrentPrincipal(owner_type="DEV", owner_id="dev"),
            [],
        )
        is None
    )
    evaluations.validate_profile_references.assert_called_once_with(
        [("cfg_default", "harbor"), ("cfg_member", "harbor")]
    )

    monkeypatch.setattr("scaled_evals.api.runnability._append_reference_checks", lambda *_args: None)
    monkeypatch.setattr(
        "scaled_evals.api.runnability._append_runtime_checks",
        lambda *_args: (MagicMock(metadata={}), None),
    )
    db.benchmark_runs.benchmark_revision_for_run.return_value = {"revision": 1}
    db.benchmark_runs.load_members.return_value = [
        {"task_id": "task_0", "task_revision": 1, "revision_status": "ready"}
    ]
    result = preflight_benchmark_run(
        db,
        CreateBenchmarkRunRequest(
            name="suite",
            benchmark_id="bm_suite",
            member_framework_profile_ids={"task_1": "cfg_member"},
        ),
        CurrentPrincipal(owner_type="DEV", owner_id="dev"),
        object_exists=lambda _key: True,
        resolve_bundle=MagicMock(),
    )
    assert isinstance(result, BlockedPreflight)
    assert result.blocker.status_code == 422
    assert "outside the benchmark: task_1" in result.blocker.message


def test_member_dataset_profiles_receive_image_and_builder_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluations = MagicMock()
    db = MagicMock(evaluations=evaluations)
    body = CreateBenchmarkRunRequest(
        name="suite",
        benchmark_id="bm_suite",
        framework_profile_id="cfg_default",
        member_framework_profile_ids={"task_0": "cfg_member"},
    )
    profiles = {
        "cfg_default": {"config": {}},
        "cfg_member": {
            "config": {
                "dataset_only": True,
                "dataset_image_mode": "direct",
            }
        },
    }
    evaluations.load_framework_profile.side_effect = profiles.get

    blocker = _append_reference_checks(
        db,
        body,
        CurrentPrincipal(owner_type="DEV", owner_id="dev"),
        [],
    )
    assert blocker is not None
    assert blocker.code == "managed_dataset_images_required"

    profiles["cfg_member"]["config"]["dataset_image_mode"] = "managed"
    monkeypatch.setattr(settings, "image_builder_service_url", "")
    monkeypatch.setattr(settings, "cloud_build_enabled", False)
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    blocker = _append_reference_checks(
        db,
        body,
        CurrentPrincipal(owner_type="DEV", owner_id="dev"),
        [],
    )
    assert blocker is not None
    assert blocker.code == "managed_dataset_builder_unavailable"
    assert {call.args[0] for call in evaluations.load_framework_profile.call_args_list[-2:]} == {
        "cfg_default",
        "cfg_member",
    }


def test_member_profile_selection_and_metadata_are_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, Any]] = []
    monkeypatch.setattr(
        EvaluationRepository,
        "create",
        lambda _self, _evaluation_id, **kwargs: created.append(kwargs),
    )
    connection = MagicMock()
    connection.transaction.return_value = nullcontext()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = {"id": "bmr_1"}
    metadata = {"member_framework_profile_ids": {"task_0": "cfg_member"}}

    BenchmarkRunRepository(connection).create_run(
        "bmr_1",
        name="suite",
        framework="harbor",
        requested_framework_version=None,
        framework_version="1",
        runner_image_ref=None,
        runner_image_digest=None,
        framework_adapter_version=None,
        sandbox_k8s_version=None,
        runner_metadata=metadata,
        benchmark_id="bm_suite",
        benchmark_revision=1,
        members=[
            {
                "id": "ev_0",
                "task_id": "task_0",
                "task_revision": 1,
                "framework_profile_id": "cfg_member",
            },
            {"id": "ev_1", "task_id": "task_1", "task_revision": 1},
        ],
        framework_profile_id="cfg_default",
        harbor_profile_id="cfg_default",
        intake_profile_id=None,
        credentials={},
        runtime="sandbox_k8s",
        network_policy="unrestricted",
        network_policy_config={},
        parallelism=1,
        visibility="private",
        owner_id="dev",
    )

    insert = next(call for call in cursor.execute.call_args_list if "INSERT INTO benchmark_runs" in call.args[0])
    assert insert.args[1][10].obj == metadata
    assert [(member["framework_profile_id"], member["harbor_profile_id"]) for member in created] == [
        ("cfg_member", "cfg_member"),
        ("cfg_default", "cfg_default"),
    ]


def test_create_routes_overrides_into_member_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    runner = MagicMock(
        requested_version=None,
        version="1",
        image_ref=None,
        image_digest=None,
        adapter_version=None,
        sandbox_k8s_version=None,
    )
    preflight = MagicMock(
        runner=runner,
        revision=1,
        members=[
            {"task_id": "task_0", "task_revision": 1, "task_slug": "zero"},
            {"task_id": "task_1", "task_revision": 1, "task_slug": "one"},
        ],
        runner_metadata={"portable": True},
    )
    monkeypatch.setattr(benchmark_runs_router, "preflight_benchmark_run", lambda *_args, **_kwargs: preflight)
    monkeypatch.setattr(benchmark_runs_router, "_response", lambda _db, row: row)
    db = MagicMock()
    db.benchmark_runs.create_run.side_effect = lambda _run_id, **kwargs: captured.update(kwargs) or kwargs

    result = benchmark_runs_router.create_benchmark_run(
        CreateBenchmarkRunRequest(
            name="suite",
            benchmark_id="bm_suite",
            framework_profile_id="cfg_default",
            member_framework_profile_ids={"task_0": "cfg_member"},
        ),
        db,
        CurrentPrincipal(owner_type="DEV", owner_id="dev"),
    )

    assert result["runner_metadata"] == {
        "portable": True,
        "member_framework_profile_ids": {"task_0": "cfg_member"},
    }
    assert [member["framework_profile_id"] for member in captured["members"]] == [
        "cfg_member",
        None,
    ]
    db.commit.assert_called_once()


def test_reproduction_preserves_member_profiles_in_stable_order() -> None:
    body = _reproduce_request(
        {
            "name": "suite",
            "benchmark_id": "bm_suite",
            "benchmark_revision": 1,
            "framework": "harbor",
            "framework_profile_id": "cfg_default",
            "runner_metadata": {
                "member_framework_profile_ids": {
                    "task_1": "cfg_second",
                    "task_0": "cfg_first",
                }
            },
            "runtime": "sandbox_k8s",
            "parallelism": 1,
        },
        {},
    )

    assert body.member_framework_profile_ids == {
        "task_1": "cfg_second",
        "task_0": "cfg_first",
    }
    command = _create_command(body)
    options = [command[index + 1] for index, value in enumerate(command) if value == "--member-framework-profile"]
    assert options == ["task_0=cfg_first", "task_1=cfg_second"]


def test_cli_accepts_repeatable_member_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={
                "id": "bmr_1",
                "name": "suite",
                "status": "running",
                "benchmark_id": "bm_suite",
                "benchmark_revision": 1,
            },
        )

    real_make_client = client_module.make_client

    def make_client(base_url: str, token: str | None, **kwargs: Any) -> httpx.Client:
        return real_make_client(
            base_url,
            token,
            transport=httpx.MockTransport(handler),
            **kwargs,
        )

    monkeypatch.setattr("scaled_evals.cli.main.make_client", make_client)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark-run",
            "create",
            "--name",
            "suite",
            "--benchmark-id",
            "bm_suite",
            "--member-framework-profile",
            "task_0=cfg_first",
            "--member-framework-profile",
            "task_1=cfg_second",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == {
        "path": "/v1/benchmark-runs",
        "body": {
            "name": "suite",
            "benchmark_id": "bm_suite",
            "member_framework_profile_ids": {
                "task_0": "cfg_first",
                "task_1": "cfg_second",
            },
        },
    }
