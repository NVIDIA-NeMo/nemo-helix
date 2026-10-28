# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run contracts and persisted JSON, including the complete transition matrix."""

import json
from datetime import datetime, timezone
from itertools import product

import pytest
from nemo_evals.api.taskset_run_schemas import HarborTaskJobsExecutionInput
from nemo_evals.taskset_runs.entities import (
    SETTLED_MEMBER_PHASES,
    TERMINAL_RUN_PHASES,
    MemberPhase,
    ResultState,
    TasksetRunEntity,
    TasksetRunMemberEntity,
    TasksetRunPhase,
    assert_member_transition,
    assert_run_transition,
)
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_helix_plugin.client.types import RESPONSE_VALIDATION_CONTEXT
from pydantic import ValidationError

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


def test_enum_contracts() -> None:
    """Pin lifecycle strings and terminal sets so persisted wire values stay stable.

    With no fixture inputs, assertions compare the public enums and phase sets
    against the contract expected by run readers and future controllers.
    """
    assert [p.value for p in TasksetRunPhase] == ["accepted", "running", "completed", "failed", "cancelled", "expired"]
    assert [p.value for p in ResultState] == ["pending", "ready", "failed", "none"]
    assert [p.value for p in MemberPhase] == [
        "image_preparing",
        "image_ready",
        "registered",
        "succeeded",
        "failed",
        "cancelled",
        "expired",
    ]
    assert TERMINAL_RUN_PHASES == frozenset(
        {TasksetRunPhase(p) for p in ["completed", "failed", "cancelled", "expired"]}
    )
    assert SETTLED_MEMBER_PHASES == frozenset({MemberPhase(p) for p in ["succeeded", "failed", "cancelled", "expired"]})


@pytest.mark.parametrize(("old", "new"), list(product(TasksetRunPhase, repeat=2)))
def test_transition_matrix(old: TasksetRunPhase, new: TasksetRunPhase) -> None:
    """Check each parametrized old/new pair against the complete lifecycle contract.

    The guard must return None for the eight allowed changes and raise ValueError
    for every other pair, preventing self-transitions and reopening terminal runs.
    """
    allowed = {
        ("accepted", "running"),
        ("accepted", "failed"),
        ("accepted", "cancelled"),
        ("accepted", "expired"),
        ("running", "completed"),
        ("running", "failed"),
        ("running", "cancelled"),
        ("running", "expired"),
    }
    if (old.value, new.value) in allowed:
        assert assert_run_transition(old, new) is None
    else:
        with pytest.raises(ValueError):
            assert_run_transition(old, new)


def test_record_defaults_and_json_roundtrip() -> None:
    """Prove run/member data and response metadata survive the persisted JSON boundary.

    Locally constructed records supply timestamps and optional fields; assertions
    check defaults, restored data, and Entity Store metadata after JSON encoding
    and response-context validation, rather than relying on an in-memory copy.
    """
    run = TasksetRunEntity(
        name="run-123",
        workspace="default",
        plan_ref="default/plan#plan.json",
        plan_digest="a" * 64,
        request_digest="b" * 64,
        taskset_ref="default/suite#3",
        accepted_at=NOW,
        timeout_seconds=3600,
        execution_deadline_at=NOW,
        finalization_deadline_at=NOW,
    )
    assert run.__entity_type__ == "taskset_run"
    assert run.phase is TasksetRunPhase.ACCEPTED
    assert run.result_state is ResultState.PENDING
    assert run.idempotency_key is run.cancel_requested_at is run.error is None
    run.idempotency_key = "key"
    run.cancel_requested_at = NOW
    run.error = "failure"
    member = TasksetRunMemberEntity(
        name="task-7",
        workspace="default",
        task_index=7,
        phase=MemberPhase.REGISTERED,
        task_job="default/job-7",
        error="failure",
    )
    assert member.__entity_type__ == "taskset_run_member"
    for entity in (run, member):
        data = json.loads(
            json.dumps(entity.model_dump(exclude=entity.__base_fields__, exclude_computed_fields=True, mode="json"))
        )
        restored = type(entity).model_validate(
            {
                "name": entity.name,
                "workspace": entity.workspace,
                "id": "entity-id",
                "parent": "run-id",
                "created_by": "user:alice",
                "created_at": NOW.isoformat(),
                "db_version": 3,
                **data,
            },
            context=RESPONSE_VALIDATION_CONTEXT,
        )
        assert restored.model_dump(exclude=entity.__base_fields__, exclude_computed_fields=True) == entity.model_dump(
            exclude=entity.__base_fields__, exclude_computed_fields=True
        )
        assert (restored.id, restored.parent, restored.created_by, restored.db_version) == (
            "entity-id",
            "run-id",
            "user:alice",
            3,
        )
        assert restored.created_at == NOW


@pytest.mark.parametrize("entity_type", [TasksetRunEntity, TasksetRunMemberEntity])
def test_record_required_fields(entity_type) -> None:
    """Require validation failure when the parametrized record lacks its domain fields."""
    with pytest.raises(ValidationError):
        entity_type.model_validate({"name": "x", "workspace": "default"})


@pytest.mark.parametrize(("old", "new"), list(product(MemberPhase, repeat=2)))
def test_member_transition_matrix(old: MemberPhase, new: MemberPhase) -> None:
    """A member leaves the queue once and a settled member never changes; same-phase writes stay open."""
    settled = {"succeeded", "failed", "cancelled", "expired"}
    allowed = {("image_ready", p) for p in {"image_ready", "registered", *(settled - {"succeeded"})}} | {
        ("registered", p) for p in {"registered", *settled}
    }
    if (old.value, new.value) in allowed:
        assert assert_member_transition(old, new) is None
    else:
        with pytest.raises(ValueError):
            assert_member_transition(old, new)


def test_member_defaults_and_index() -> None:
    """Assert absent Job/error defaults and reject a negative task index in local records."""
    member = TasksetRunMemberEntity(name="task-0", workspace="default", task_index=0, phase=MemberPhase.IMAGE_READY)
    assert member.task_job is member.error is member.bundle_ref is member.job_status is member.create_sent_at is None
    with pytest.raises(ValidationError):
        TasksetRunMemberEntity(name="task-0", workspace="default", task_index=-1, phase=MemberPhase.IMAGE_READY)


def test_frozen_plan_roundtrip(plan: TasksetRunPlan) -> None:
    """Verify the plan fixture round-trips while retaining its strict frozen schema.

    Assertions pin the ten stored fields and expect validation errors for
    rebinding, extras, an invalid cap, and a missing nullable cap field,
    protecting the resolved-input contract of stored plans.
    """
    assert TasksetRunPlan.model_validate_json(plan.model_dump_json()) == plan
    assert set(TasksetRunPlan.model_fields) == {
        "run_id",
        "workspace",
        "taskset_ref",
        "spec",
        "max_concurrent_task_jobs",
        "sandbox_policy",
        "dispatcher_image",
        "dispatcher_profile",
        "task_environments",
        "intake",
    }
    with pytest.raises(ValidationError, match="frozen"):
        setattr(plan, "run_id", "changed")  # setattr: ty already rejects a direct write to a frozen field
    for field, value in [
        ("schema_version", 1),
        ("max_failed_tasks", None),
        ("max_concurrent_task_jobs", 0),
    ]:
        with pytest.raises(ValidationError):
            TasksetRunPlan.model_validate({**plan.model_dump(), field: value})
    data = plan.model_dump()
    del data["max_concurrent_task_jobs"]
    with pytest.raises(ValidationError):
        TasksetRunPlan.model_validate(data)
    # One frozen environment per task: a retry reads images from here, never prepares them again.
    with pytest.raises(ValidationError, match="task_environments has 2 entries for 1 tasks"):
        TasksetRunPlan.model_validate(
            {**plan.model_dump(), "task_environments": plan.model_dump()["task_environments"] * 2}
        )


def test_execution_input() -> None:
    """Verify caller configuration defaults and reject unsupported execution inputs.

    Locally built sandbox settings must apply documented defaults; an invalid
    limit, unknown fields/kinds (including the removed max_failed_tasks and
    sandbox_policy) must raise ValidationError before submit resolution.
    """
    input_model = HarborTaskJobsExecutionInput()
    assert input_model.kind == "harbor-task-jobs"
    assert input_model.dispatcher_profile is None  # submit resolves the operator's first profile
    assert input_model.sandbox is None  # submit resolves the operator's default sandbox profile
    assert input_model.max_concurrent_task_jobs is None
    for changes in (
        {"max_failed_tasks": 1},
        {"missing_prebuilt_image": "fail"},
        {"max_concurrent_task_jobs": 0},
        {"unexpected": True},
        {"kind": "other"},
        {"sandbox_policy": "harbor-standard"},
        {"sandbox": {"profile": ""}},
        {"sandbox": {"gpus": 1}},
        {"sandbox": {"root_template": "harbor-cpu-root"}},
    ):
        with pytest.raises(ValidationError):
            HarborTaskJobsExecutionInput.model_validate({**input_model.model_dump(), **changes})
    selected = HarborTaskJobsExecutionInput.model_validate({"sandbox": {"profile": "harbor-open"}}).sandbox
    assert selected is not None and selected.profile == "harbor-open"
