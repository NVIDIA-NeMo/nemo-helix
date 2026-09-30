# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ASE-792: metadata-only benchmark variants + agent timeout floor."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client, v1
from scaled_evals.api.auth import current_principal
from scaled_evals.api.db import get_conn
from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
from scaled_evals.dispatch.sandbox_k8s import apply_agent_timeout_floor
from scaled_evals.dispatch.worker import (
    assert_lifecycle_covers_agent_floor,
    snapshot_agent_timeout_floor,
)
from scaled_evals.models.execution_snapshot import EXECUTION_SNAPSHOT_SCHEMA_VERSION


def _conn_returning(fetchones: list, fetchall=None) -> MagicMock:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = list(fetchones)
    if fetchall is not None:
        cur.fetchall.return_value = fetchall
    return conn


def _use_conn(conn: MagicMock) -> None:
    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen


def _benchmark_row(**overrides) -> dict:
    row = {
        "id": "bm_variant",
        "name": "Suite AA",
        "slug": "suite-aa",
        "description": None,
        "visibility": "private",
        "qualification_status": "registered",
        "qualification_evidence": {},
        "qualified_at": None,
        "qualified_by": None,
        "current_revision": 1,
        "created_at": "2026-06-26T00:00:00Z",
        "updated_at": "2026-06-26T00:00:00Z",
    }
    row.update(overrides)
    return row


@pytest.fixture(autouse=True)
def _override_db():
    def _empty() -> Iterator[MagicMock]:
        conn = MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = []
        cur.fetchone.return_value = None
        yield conn

    v1.dependency_overrides[get_conn] = _empty
    yield
    v1.dependency_overrides.pop(get_conn, None)
    v1.dependency_overrides.pop(current_principal, None)


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
    _use_conn(
        _conn_returning(
            [
                {"current_revision": 2},
                {"?column?": 1},
                _benchmark_row(),
                {"?column?": 1},
                {"?column?": 1},
                {"?column?": 1},
                {"?column?": 1},
            ],
            fetchall=[
                {"task_id": "task_a", "task_revision": 3, "position": 0},
                {"task_id": "task_b", "task_revision": 1, "position": 1},
            ],
        )
    )
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


def test_benchmark_variant_snapshot_freezes_lineage_and_policy() -> None:
    """Submission freezes the variant so later policy edits cannot change the run."""
    cur = MagicMock()
    cur.fetchone.return_value = {
        "derived_from_benchmark_id": "bm_base",
        "derived_from_revision": 2,
        "operational_policy": {"agent_timeout_floor_sec": 7200},
    }
    assert EvaluationRepository._benchmark_variant_snapshot(cur, "bmr_1") == {
        "derived_from": {"benchmark_id": "bm_base", "revision": 2},
        "operational_policy": {"agent_timeout_floor_sec": 7200},
    }

    cur.fetchone.return_value = {
        "derived_from_benchmark_id": None,
        "derived_from_revision": None,
        "operational_policy": {},
    }
    assert EvaluationRepository._benchmark_variant_snapshot(cur, "bmr_1") is None
    assert EvaluationRepository._benchmark_variant_snapshot(cur, None) is None


def test_create_variant_404_when_base_missing() -> None:
    _use_conn(_conn_returning([None]))
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
