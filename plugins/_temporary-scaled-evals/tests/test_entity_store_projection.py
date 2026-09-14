# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

pytest.importorskip("scaled_evals")
pytest.importorskip("nemo_scaled_evals_plugin")

from nemo_platform_plugin.entities.base import EntityNotFoundError
from nemo_scaled_evals_plugin.entities import ScaledEvaluation
from nemo_scaled_evals_plugin.projection import (
    EvaluationProjectionReader,
    EvaluationProjectionWriter,
    entity_to_row,
    parity_report,
    row_to_entity,
)
from scaled_evals.api.routers.evaluations import _response
from scaled_evals.api.schemas.common import encode_cursor
from scaled_evals.api.schemas.evaluations import Evaluation

WORKSPACE = "default"
CREATED = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)


def _row(**overrides: Any) -> dict[str, Any]:
    """Return a row shaped like EVALUATION_DETAIL_COLUMNS."""
    row = {
        "id": "eval_0123456789abcdef0123456789",
        "owner_id": "user_1",
        "name": "nightly",
        "framework": "harbor",
        "requested_framework_version": None,
        "framework_version": "0.20.0",
        "runner_image_ref": "registry.example/runner:1",
        "runner_image_digest": "sha256:abc",
        "framework_adapter_version": "1.2.3",
        "sandbox_k8s_version": "0.4.0",
        "runner_metadata": {"agent_bundle": "bundle_1"},
        "task_id": "task_1",
        "task_revision": 2,
        "benchmark_run_id": None,
        "framework_profile_id": "prof_1",
        "harbor_profile_id": None,
        "switchyard_profile_id": None,
        "intake_profile_id": None,
        "credentials": {"anthropic": "cred_1"},
        "extra_skill_object_keys": ["skills/a.md"],
        "instruction_prefix": None,
        "instruction_postfix": None,
        "initial_user_turns": [],
        "runtime": "sandbox_k8s",
        "network_policy": "unrestricted",
        "network_policy_config": {},
        "n_attempts": 1,
        "parallelism": 1,
        "visibility": "private",
        "status": "succeeded",
        "status_detail": "done",
        "cancel_teardown_status": "not_requested",
        "cancel_teardown_error": None,
        "cancel_teardown_updated_at": None,
        "backend_handle": "sandbox-1",
        "dispatch_job_name": "scaled-evals-evaluation-eval_1-e1",
        "dispatch_job_uid": "uid-1",
        "current_execution": 1,
        "max_executions": 3,
        "infrastructure_retries": 0,
        "max_infrastructure_retries": 2,
        "next_retry_at": None,
        "last_failure_code": None,
        "last_failure_category": None,
        "reward": 1.0,
        "reward_value": 1.0,
        "n_trials": 1,
        "n_completed": 1,
        "n_errored": 0,
        "n_failed_solve": 0,
        "exception_counts": {},
        "finished_at": CREATED + timedelta(minutes=5),
        "created_at": CREATED,
        "updated_at": CREATED + timedelta(minutes=5),
        "result": {"reward": 1.0},
        "evidence_status": "ready",
        "evidence_error": None,
        "archive_status": "ready",
        "archive_error": None,
        "image_ref": "registry.example/task:2",
        "image_digest": "sha256:def",
        "deleted_at": None,
    }
    row.update(overrides)
    return row


class FakeEntityClient:
    """Minimal in-memory stand-in for SyncEntityClient."""

    def __init__(self) -> None:
        self.stored: dict[str, ScaledEvaluation] = {}
        self.list_calls: list[dict[str, Any]] = []
        self.page: list[ScaledEvaluation] = []
        self._next_id = 0

    def create(self, entity: ScaledEvaluation) -> ScaledEvaluation:
        self._next_id += 1
        # `id` is a computed view over a private attr, exactly as the real store
        # assigns it; a writer that assigns `entity.id` would raise.
        entity._id = f"entity-{self._next_id}"
        self.stored[entity.name] = entity
        return entity

    def update(self, entity: ScaledEvaluation, *, original_name: str | None = None) -> ScaledEvaluation:
        self.stored[entity.name] = entity
        return entity

    def get(self, entity_type: Any, name: str, *, workspace: str | None = None, **_: Any) -> ScaledEvaluation:
        if name not in self.stored:
            raise EntityNotFoundError(f"{name} not found")
        return self.stored[name]

    def list(self, entity_type: Any, **kwargs: Any) -> Any:
        self.list_calls.append(kwargs)
        filter_operation = kwargs.get("filter_operation")
        return type(
            "Page",
            (),
            {"data": self.page, "pagination": None, "filter": filter_operation},
        )()


def test_projection_round_trips_into_the_existing_response_schemas() -> None:
    row = _row()

    # Parity is measured against a real read-back, so the check would catch the
    # store mangling the payload; comparing the mapping to itself would not.
    client = FakeEntityClient()
    EvaluationProjectionWriter(client, workspace=WORKSPACE).project(row)
    read_back = EvaluationProjectionReader(client, workspace=WORKSPACE).get(row["id"])
    assert read_back is not None
    assert parity_report(row, read_back) == []

    entity = row_to_entity(row, workspace=WORKSPACE)
    assert entity.name == row["id"]
    assert (entity.standalone, entity.deleted) == (True, False)
    assert entity.row_created_at == CREATED
    # `q` searches six columns with ILIKE; the blob is lowercased so `$like`
    # matches case-insensitively whatever collation the store uses.
    assert "nightly" in entity.search_blob and "sandbox_k8s" in entity.search_blob
    # Timestamps survive as ISO strings, not repr() output Pydantic can't parse.
    assert entity.detail["created_at"] == CREATED.isoformat()
    # Only what the responses read is copied into the second store. Prompt
    # content and soft-delete bookkeeping stay in Postgres.
    for withheld in (
        "deleted_at",
        "instruction_prefix",
        "instruction_postfix",
        "initial_user_turns",
        "extra_skill_object_keys",
        "backend_handle",
        "archive_status",
        "image_ref",
    ):
        assert withheld not in entity.detail

    rebuilt = Evaluation(**entity_to_row(entity))
    assert rebuilt.model_dump() == Evaluation(**row).model_dump()
    assert rebuilt.outcome.category == "completed"
    # The detail read adds `result` and links on top of the list item. Going
    # through the router's own builder proves the projection feeds the real
    # response path, not just the bare model.
    assert _response(entity_to_row(entity)).model_dump() == _response(row).model_dump()

    # A soft-deleted row still projects, so reads can 404 from the projection.
    deleted = row_to_entity(_row(deleted_at=CREATED), workspace=WORKSPACE)
    assert deleted.deleted
    # A benchmark member is not standalone, which is what hides it from the
    # default listing.
    assert not row_to_entity(_row(benchmark_run_id="run_1"), workspace=WORKSPACE).standalone


def test_reader_reproduces_the_sql_list_predicates_and_ordering() -> None:
    client = FakeEntityClient()
    reader = EvaluationProjectionReader(client, workspace=WORKSPACE)
    older = row_to_entity(_row(id="eval_a", created_at=CREATED - timedelta(hours=1)), workspace=WORKSPACE)
    same_a = row_to_entity(_row(id="eval_b"), workspace=WORKSPACE)
    same_b = row_to_entity(_row(id="eval_c"), workspace=WORKSPACE)
    client.page = [same_a, older, same_b]

    rows = reader.list(
        limit=20,
        cursor=encode_cursor(CREATED + timedelta(hours=1), "eval_z"),
        order="desc",
        status="succeeded",
        task_id="task_1",
        shared=True,
        owner_id="user_1",
        q="Nightly",
    )

    call = client.list_calls[0]
    # limit + 1 mirrors the SQL fetch-one-extra that drives next_cursor.
    assert call["page_size"] == 21
    assert call["sort"] == "-row_created_at"
    spec = json.dumps(call["filter_operation"].to_dict())
    for expected in (
        '"data.deleted": {"$eq": false}',
        '"data.standalone": {"$eq": true}',
        '"data.status": {"$eq": "succeeded"}',
        '"data.task_id": {"$eq": "task_1"}',
        '"data.owner_id": {"$eq": "user_1"}',
        '"data.visibility": {"$nin": ["private"]}',
        '"data.search_blob": {"$like": "%nightly%"}',
    ):
        assert expected in spec
    # The keyset must be the row comparison (created_at, id) < (C, I), not a
    # bare timestamp filter, or equal timestamps would repeat or skip a page.
    assert '"$or"' in spec and '"data.row_created_at": {"$lt"' in spec
    assert '"data.evaluation_id": {"$lt": "eval_z"}' in spec

    # The store sorts on one field, so the id tiebreaker is reapplied locally.
    assert [row["id"] for row in rows] == ["eval_c", "eval_b", "eval_a"]

    # Default listing hides benchmark members; drilling in targets the run.
    reader.list(limit=5, cursor=None, order="asc", status=None, task_id=None, shared=False, benchmark_run_id="run_1")
    drill = json.dumps(client.list_calls[1]["filter_operation"].to_dict())
    assert '"data.benchmark_run_id": {"$eq": "run_1"}' in drill
    assert '"data.standalone"' not in drill
    assert client.list_calls[1]["sort"] == "row_created_at"


def test_reader_get_hides_missing_and_soft_deleted_rows() -> None:
    client = FakeEntityClient()
    reader = EvaluationProjectionReader(client, workspace=WORKSPACE)
    assert reader.get("eval_missing") is None

    client.create(row_to_entity(_row(), workspace=WORKSPACE))
    found = reader.get(_row()["id"])
    assert found is not None
    assert Evaluation(**found).id == _row()["id"]

    client.stored.clear()
    client.create(row_to_entity(_row(deleted_at=CREATED), workspace=WORKSPACE))
    assert reader.get(_row()["id"]) is None


def test_writer_upserts_with_compare_and_swap_and_resumes_from_the_watermark() -> None:
    client = FakeEntityClient()
    writer = EvaluationProjectionWriter(client, workspace=WORKSPACE)

    # Nothing projected yet: the watermark is empty so the first pass replays
    # from the beginning rather than skipping history.
    assert writer.watermark() is None

    writer.project(_row())
    created = client.stored[_row()["id"]]
    assert created.id == "entity-1"

    # Re-projecting the same evaluation must overwrite in place, carrying the
    # store's id and version so the write is a compare-and-swap, not a create
    # that would conflict forever.
    created._db_version = 7
    writer.project(_row(status="failed"))
    updated = client.stored[_row()["id"]]
    assert (updated.id, updated.db_version, updated.status) == ("entity-1", 7, "failed")
    assert len(client.stored) == 1

    client.page = [updated]
    assert writer.watermark() == updated.row_updated_at
    assert client.list_calls[-1]["sort"] == "-row_updated_at"


@pytest.mark.asyncio
async def test_controller_projects_changed_rows_and_advances_the_watermark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nemo_scaled_evals_plugin.controller as controller_module
    from nemo_scaled_evals_plugin.controller import ScaledEvalsJobsController

    client = FakeEntityClient()
    controller = ScaledEvalsJobsController()
    controller._projection = EvaluationProjectionWriter(client, workspace=WORKSPACE)
    first = _row(id="eval_first", updated_at=CREATED)
    second = _row(id="eval_second", updated_at=CREATED + timedelta(minutes=1))
    batches = [[first, second], []]
    seen: list[datetime | None] = []

    def _changed(updated_after: datetime | None, limit: int) -> list[dict[str, Any]]:
        seen.append(updated_after)
        return batches.pop(0)

    monkeypatch.setattr(controller, "_changed_evaluations", _changed)

    await controller._project_evaluations()
    assert sorted(client.stored) == ["eval_first", "eval_second"]
    # The watermark advances to the newest row actually written, so the next
    # pass resumes instead of replaying the whole table.
    assert controller._projection_watermark == second["updated_at"]

    await controller._project_evaluations()
    assert seen == [None, second["updated_at"]]

    # The phase is registered only when projection is enabled.
    monkeypatch.setattr(
        controller_module,
        "settings",
        type("S", (), {"platform_build_jobs_enabled": False, "platform_evaluation_jobs_enabled": False})(),
    )
    assert [name for name, _ in controller._phases()] == ["project_evaluations"]
    controller._projection = None
    assert controller._phases() == []
