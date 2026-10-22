# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ASE-792: metadata-only benchmark variants + agent timeout floor."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client, v1
from scaled_evals.api.auth import current_principal
from scaled_evals.api.benchmark_store import BenchmarkStore, TaskRef
from scaled_evals.api.evaluation_store import EvaluationStore
from scaled_evals.dispatch.sandbox_k8s import apply_agent_timeout_floor
from scaled_evals.dispatch.worker import (
    Dispatcher,
    assert_lifecycle_covers_agent_floor,
    evaluation_run_budget_seconds,
    snapshot_agent_timeout_floor,
)
from scaled_evals.models.execution_snapshot import EXECUTION_SNAPSHOT_SCHEMA_VERSION
from task_fixtures import seed_benchmark, seed_benchmark_run, seed_task_revision


@pytest.fixture(autouse=True)
def _override_db():
    yield
    v1.dependency_overrides.pop(current_principal, None)


def _seed_base_revision_2() -> None:
    seed_benchmark("bm_base", (("task_a", 1),))
    BenchmarkStore().create_next_revision(
        "bm_base", description=None, tasks=[TaskRef("task_a", 3), TaskRef("task_b", None)]
    )


def _snapshot_row(floor_sec: int | None, **overrides) -> dict:
    evaluation: dict = {"id": "ev_1"}
    if floor_sec is not None:
        evaluation["benchmark_variant"] = {
            "derived_from": {"benchmark_id": "bm_base", "revision": 2},
            "operational_policy": {"agent_timeout_floor_sec": floor_sec},
        }
    return {
        "execution_snapshot": {
            "schema_version": EXECUTION_SNAPSHOT_SCHEMA_VERSION,
            "evaluation": evaluation,
            "task": {},
            "profiles": {},
            "credentials": {},
            "submission_identity": {},
        },
        **overrides,
    }


def test_variant_create_and_timeout_floor_cover_both_workflows(tmp_path: Path) -> None:
    """Create a metadata-only variant, then raise the staged agent timeout from it."""
    # --- workflow 1: create metadata-only variant ---
    seed_task_revision("task_a", 3)
    seed_task_revision("task_b")
    _seed_base_revision_2()
    response = client.post(
        "/v1/benchmarks/bm_base/variants",
        json={
            "name": "Suite AA",
            "slug": "suite-aa",
            "from_revision": 2,
            "operational_policy": {"agent_timeout_floor_sec": 7200},
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["derived_from"] == {"benchmark_id": "bm_base", "revision": 2}
    assert body["operational_policy"] == {"agent_timeout_floor_sec": 7200}
    assert body["links"]["base"] == "/benchmarks/bm_base"
    assert BenchmarkStore().members(body["id"], 1) == [
        {"task_id": "task_a", "task_revision": 3, "position": 0},
        {"task_id": "task_b", "task_revision": None, "position": 1},
    ]

    # --- workflow 2a: staged task.toml agent timeout floor ---
    tree = tmp_path / "task"
    tree.mkdir()
    (tree / "task.toml").write_text(
        'version = "1.0"\n\n[agent]\ntimeout_sec = 1800.0\n\n[verifier]\ntimeout_sec = 60.0\n',
        encoding="utf-8",
    )
    agent = apply_agent_timeout_floor(tree, body["operational_policy"]["agent_timeout_floor_sec"])
    assert agent == {"original": 1800.0, "effective": 7200.0}
    text = (tree / "task.toml").read_text(encoding="utf-8")
    assert "timeout_sec = 7200" in text
    assert "timeout_sec = 60.0" in text  # verifier untouched

    # A commented section header still counts as [agent], and a key that merely
    # starts with "timeout_sec" is left alone.
    quirky = tmp_path / "quirky"
    quirky.mkdir()
    (quirky / "task.toml").write_text(
        "[agent]  # tuned\ntimeout_seconds = 30\ntimeout_sec = 1800\n",
        encoding="utf-8",
    )
    assert apply_agent_timeout_floor(quirky, 7200) == {"original": 1800.0, "effective": 7200.0}
    quirky_text = (quirky / "task.toml").read_text(encoding="utf-8")
    assert "timeout_seconds = 30" in quirky_text
    assert quirky_text.count("[agent]") == 1

    # Longer originals stay put.
    long_tree = tmp_path / "long"
    long_tree.mkdir()
    (long_tree / "task.toml").write_text(
        'version = "1.0"\n\n[agent]\ntimeout_sec = 9000\n',
        encoding="utf-8",
    )
    assert apply_agent_timeout_floor(long_tree, 7200) == {
        "original": 9000.0,
        "effective": 9000.0,
    }

    # --- workflow 2b: an unappliable floor fails the launch instead of silently passing ---
    empty_tree = tmp_path / "empty"
    empty_tree.mkdir()
    for missing in (empty_tree, None):
        with pytest.raises(RuntimeError, match="no staged task.toml"):
            apply_agent_timeout_floor(missing, 7200)


def test_snapshotted_floor_drives_validation_and_job_deadline() -> None:
    """The floor is read from the submission snapshot, never re-read from the DB."""
    assert snapshot_agent_timeout_floor(_snapshot_row(7200)) == 7200
    assert snapshot_agent_timeout_floor(_snapshot_row(None)) is None
    assert snapshot_agent_timeout_floor({"execution_snapshot": None}) is None

    # A variant raises the agent budget but must not rewrite the sandbox hard stop,
    # so a profile that would truncate the agent is rejected before launch.
    short = _snapshot_row(
        7200,
        runtime="sandbox_k8s",
        framework_config={"environment": {"kwargs": {"lifecycle_timeout": 3600}}},
    )
    with pytest.raises(RuntimeError, match="lifecycle_timeout to at least 7500"):
        assert_lifecycle_covers_agent_floor(short, 7200)
    with pytest.raises(RuntimeError, match="stops the sandbox after 3600s"):
        # sandbox_k8s stops the sandbox on its own plugin default when unset.
        assert_lifecycle_covers_agent_floor(_snapshot_row(7200, runtime="sandbox_k8s"), 7200)
    assert_lifecycle_covers_agent_floor(
        _snapshot_row(
            7200,
            runtime="sandbox_k8s",
            framework_config={"environment": {"kwargs": {"lifecycle_timeout": 8100}}},
        ),
        7200,
    )

    # Other runtimes have no implicit hard stop, so a silent profile is fine and
    # only a declared lifecycle_timeout is held to the floor.
    assert_lifecycle_covers_agent_floor(_snapshot_row(7200, runtime="gym_daytona"), 7200)
    with pytest.raises(RuntimeError, match="stops the sandbox after 3600s"):
        assert_lifecycle_covers_agent_floor(
            _snapshot_row(
                7200,
                runtime="gym_daytona",
                framework_config={"environment": {"kwargs": {"lifecycle_timeout": 3600}}},
            ),
            7200,
        )

    # The run budget sizes both the Job deadline and the dispatcher's poll cap.
    # Without profile config, the frozen floor keeps the Job alive past the agent budget.
    assert evaluation_run_budget_seconds(_snapshot_row(7200), floor=3600) == 7500  # floor + 300 sandbox grace
    assert evaluation_run_budget_seconds(_snapshot_row(None), floor=3600) == 3600
    # Every sequential trial wave gets the full sandbox lifecycle.
    waves = _snapshot_row(None, framework_config={"environment": {"kwargs": {"lifecycle_timeout": 3600}}})
    waves["execution_snapshot"]["evaluation"].update(n_attempts=10, parallelism=4)
    assert evaluation_run_budget_seconds(waves, floor=3600) == 3 * 3900
    assert Dispatcher(poll_interval=10, max_polls=360)._terminal_max_polls(waves) == 1170


def test_benchmark_variant_snapshot_freezes_lineage_and_policy() -> None:
    """Submission freezes the variant so later policy edits cannot change the run."""
    seed_task_revision("task_a", 3)
    seed_task_revision("task_b")
    _seed_base_revision_2()
    BenchmarkStore().create_variant(
        "bm_variant",
        base_benchmark_id="bm_base",
        from_revision=2,
        name="Suite AA",
        slug="suite-aa",
        description=None,
        visibility="private",
        operational_policy={"agent_timeout_floor_sec": 7200},
    )
    run = {"framework": "harbor", "runtime": "sandbox_k8s", "parallelism": 1}
    seed_benchmark_run({**run, "id": "bmr_variant", "name": "v", "benchmark_id": "bm_variant", "benchmark_revision": 1})
    seed_benchmark_run({**run, "id": "bmr_base", "name": "b", "benchmark_id": "bm_base", "benchmark_revision": 2})
    assert EvaluationStore._benchmark_variant_snapshot("bmr_variant") == {
        "derived_from": {"benchmark_id": "bm_base", "revision": 2},
        "operational_policy": {"agent_timeout_floor_sec": 7200},
    }
    assert EvaluationStore._benchmark_variant_snapshot("bmr_base") is None
    assert EvaluationStore._benchmark_variant_snapshot("bmr_missing") is None
    assert EvaluationStore._benchmark_variant_snapshot(None) is None


def test_create_variant_404_when_base_missing() -> None:
    response = client.post(
        "/v1/benchmarks/bm_missing/variants",
        json={
            "name": "Suite AA",
            "operational_policy": {"agent_timeout_floor_sec": 7200},
        },
    )
    assert response.status_code == 404


def test_create_variant_rejects_unknown_policy_key() -> None:
    response = client.post(
        "/v1/benchmarks/bm_base/variants",
        json={
            "name": "Suite AA",
            "operational_policy": {
                "agent_timeout_floor_sec": 7200,
                "instruction": "nope",
            },
        },
    )
    assert response.status_code == 422
