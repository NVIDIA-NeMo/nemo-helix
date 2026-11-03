# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes request validation, logical identity and public run projection."""

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import pytest
from nemo_evals.api.fields import TasksetRef
from nemo_evals.api.taskset_run_schemas import (
    _INTERNAL_FIELDS,
    TasksetRunInput,
    TasksetRunStatus,
    request_digest,
    to_status,
)
from nemo_evals.jobs.agent_spec import AgentEvalInputSpec, AgentEvalSpec, HarborRunnerTarget
from nemo_evals.jobs.publication import PublicationOutcome
from nemo_evals.jobs.runner_targets import runner_to_target
from nemo_evals.taskset_runs.entities import (
    MemberPhase,
    ResultState,
    TasksetRunEntity,
    TasksetRunMemberEntity,
    TasksetRunPhase,
)
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborAgentTaskRunner, HarborRuntimeConfig
from pydantic import ValidationError


@pytest.fixture
def run_body() -> dict[str, Any]:
    return {
        "spec": {"tasks": "default/sort-suite", "target": {"kind": "harbor"}},
        "execution": {},
    }


@pytest.mark.parametrize(
    "ref", ["sort-suite", "default/sort-suite", "default/sort-suite#stable", "default/sort-suite#" + "a" * 64]
)
def test_run_accepts_stored_tasksets(run_body, ref):
    run_body["spec"]["tasks"] = ref
    request = TasksetRunInput.model_validate(run_body)
    assert request.spec.tasks == TasksetRef(ref)
    assert request.idempotency_key is None
    assert request.timeout_seconds is None


def test_run_preserves_scoring_options_and_explicit_budgets(run_body):
    run_body["spec"].update(max_concurrent_tasks=8, fail_fast=True)
    run_body.update(idempotency_key="x" * 256, timeout_seconds=3600)
    request = TasksetRunInput.model_validate(run_body)
    assert request.spec.max_concurrent_tasks == 8
    assert request.spec.fail_fast is True
    assert request.idempotency_key == "x" * 256
    assert request.timeout_seconds == 3600


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown": 1},
        {"idempotency_key": ""},
        {"idempotency_key": "x" * 257},
        {"timeout_seconds": 0},
        {"timeout_seconds": -1},
    ],
)
def test_run_rejects_invalid_top_level_fields(run_body, overrides):
    run_body.update(overrides)
    with pytest.raises(ValidationError):
        TasksetRunInput.model_validate(run_body)


def test_run_requires_execution(run_body):
    del run_body["execution"]
    with pytest.raises(ValidationError):
        TasksetRunInput.model_validate(run_body)


@pytest.mark.parametrize(
    "execution",
    [
        {"sandbox_policy": "harbor-standard"},
        {"kind": "docker"},
        {"sandbox": {"profile": ""}},
        {"sandbox": {"gpus": 1}},
        {"max_concurrent_task_jobs": 0},
        {"max_failed_tasks": 1},
        {"sandbox": {"cpus": 0}},
        {"sandbox": {"memory_mb": -1}},
        {"sandbox": {"cpu": 4}},
    ],
)
def test_run_preserves_execution_validation(run_body, execution):
    run_body["execution"] = execution
    with pytest.raises(ValidationError):
        TasksetRunInput.model_validate(run_body)


@pytest.mark.parametrize("as_json", [False, True])
def test_run_validation_error_hides_rejected_credentials(run_body: dict[str, Any], as_json: bool) -> None:
    fake_key = "sk-not-a-real-key-0123456789"
    run_body["spec"]["target"]["agent_kwargs"] = {"api_key": fake_key}
    with pytest.raises(ValidationError) as excinfo:
        if as_json:
            TasksetRunInput.model_validate_json(json.dumps(run_body))
        else:
            TasksetRunInput.model_validate(run_body)

    message = str(excinfo.value)
    assert "agent_kwargs" in message
    assert "env_secrets" in message
    assert fake_key not in message


@pytest.mark.parametrize(
    "tasks", [["default/task-1"], [{"id": "task-1", "intent": "Answer", "inputs": {}, "metrics": []}]]
)
def test_run_rejects_lists_with_publication_guidance(run_body, tasks):
    run_body["spec"]["tasks"] = tasks
    AgentEvalInputSpec.model_validate(run_body["spec"])
    with pytest.raises(ValidationError) as exc:
        TasksetRunInput.model_validate(run_body)
    assert (
        exc.value.errors()[0]["msg"]
        == "Value error, Kubernetes runs take a stored taskset reference in spec.tasks; publish the tasks as a taskset first"
    )


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "fabric", "source": {"config": {"metadata": {"name": "agent"}}}},
        {"kind": "model", "model": {"url": "http://model.test/v1/chat/completions", "name": "model"}},
        {
            "kind": "agent",
            "agent": {
                "url": "http://agent.test",
                "name": "agent",
                "format": "generic",
                "body": {"question": "{{item.prompt}}"},
                "response_path": "$.answer",
            },
        },
        {"kind": "gym", "source": {"component": "simple_agent", "config": "agent.yaml"}, "resources_server": "mcqa"},
        None,
    ],
)
def test_run_requires_harbor_target(run_body, target):
    run_body["spec"]["target"] = target
    if target is None:
        run_body["spec"]["trials"] = []
    AgentEvalInputSpec.model_validate(run_body["spec"])
    with pytest.raises(ValidationError) as exc:
        TasksetRunInput.model_validate(run_body)
    assert (
        exc.value.errors()[0]["msg"]
        == 'Value error, execution requires a Harbor runner target (spec.target.kind = "harbor")'
    )


@pytest.mark.parametrize("value", [1, 4, 8])
def test_run_rejects_explicit_trial_concurrency(run_body, value):
    run_body["spec"]["target"]["n_concurrent_trials"] = value
    with pytest.raises(ValidationError) as exc:
        TasksetRunInput.model_validate(run_body)
    assert (
        exc.value.errors()[0]["msg"]
        == "Value error, Kubernetes runs trials one at a time in v1; cap the run with execution.max_concurrent_task_jobs"
    )


def test_run_accepts_null_trial_concurrency_after_default_dump(run_body):
    target = runner_to_target(HarborAgentTaskRunner(config=HarborRuntimeConfig(agent_name="oracle")))
    run_body["spec"]["target"] = target.model_dump(mode="json")
    request = TasksetRunInput.model_validate(run_body)
    assert isinstance(request.spec.target, HarborRunnerTarget)
    assert request.spec.target.n_concurrent_trials is None
    assert TasksetRunInput.model_validate_json(request.model_dump_json()) == request


def test_run_rejects_trial_concurrency_after_runtime_config_dump(run_body):
    config = HarborRuntimeConfig.model_validate(HarborRuntimeConfig(agent_name="oracle").model_dump())
    run_body["spec"]["target"] = runner_to_target(HarborAgentTaskRunner(config=config)).model_dump(mode="json")
    with pytest.raises(ValidationError, match="Kubernetes runs trials one at a time"):
        TasksetRunInput.model_validate(run_body)


@pytest.mark.parametrize(
    "spec",
    [
        {"tasks": "suite"},
        {"tasks": "suite", "target": {"kind": "harbor"}, "trials": []},
        {"tasks": "suite", "trials": [], "publication": {"intake": {"evaluation_id": "evaluation-1"}}},
    ],
)
def test_nested_spec_validation_runs_before_kubernetes_rules(run_body, spec):
    message = "publication.intake.agent_name" if "publication" in spec else "provide exactly one of"
    run_body["spec"] = spec
    with pytest.raises(ValidationError, match=re.escape(message)) as exc:
        TasksetRunInput.model_validate(run_body)
    assert exc.value.errors()[0]["loc"] == ("spec",)


@pytest.mark.parametrize("spec_type", [AgentEvalInputSpec, AgentEvalSpec])
def test_shared_scoring_schema_has_correct_scope(spec_type):
    schema = spec_type.model_json_schema()["properties"]
    assert schema["max_concurrent_tasks"]["default"] == 4
    assert schema["max_concurrent_tasks"]["minimum"] == 1
    assert schema["max_concurrent_tasks"]["description"] == (
        "Concurrency for this evaluation's SDK work. Harbor and Gym: concurrent scoring calls only "
        "(trials are bounded by the runner). Other runners: concurrent trials and scoring calls. "
        "For model and agent targets, target.params.parallelism bounds concurrent inference requests "
        "during trial generation. "
        "On Kubernetes runs it applies per task Job; the task-Job cap is execution.max_concurrent_task_jobs."
    )
    assert schema["fail_fast"]["default"] is False
    assert schema["fail_fast"]["description"] == (
        "Fail on the first metric error instead of recording an errored score. "
        "On Kubernetes runs it fails that task's Job."
    )


def test_run_schema_has_only_the_five_request_fields():
    schema = TasksetRunInput.model_json_schema()
    assert set(schema["properties"]) == {"spec", "execution", "publication", "idempotency_key", "timeout_seconds"}
    assert set(schema["required"]) == {"spec", "execution"}
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"idempotency_key": "retry-1"},
        {"idempotency_key": "retry-2", "timeout_seconds": None},
        {
            "spec": {
                "tasks": "default/sort-suite",
                "target": {"kind": "harbor", "n_attempts": 1, "n_concurrent_trials": None},
            }
        },
    ],
)
def test_digest_ignores_key_and_fills_accepted_defaults(run_body, overrides):
    other = deepcopy(run_body)
    other.update(overrides)
    assert request_digest(TasksetRunInput.model_validate(run_body)) == request_digest(
        TasksetRunInput.model_validate(other)
    )


def test_digest_fills_omitted_sandbox_overrides(run_body):
    run_body["execution"]["sandbox"] = {"cpus": 4}
    other = deepcopy(run_body)
    other["execution"]["sandbox"] = {"cpus": 4, "profile": None}
    assert request_digest(TasksetRunInput.model_validate(run_body)) == request_digest(
        TasksetRunInput.model_validate(other)
    )


def test_digest_sorts_nested_dictionary_keys(run_body):
    run_body["spec"]["target"]["env_vars"] = {"LANG": "C", "DEBUG": "1"}
    other = deepcopy(run_body)
    other["spec"]["target"]["env_vars"] = {"DEBUG": "1", "LANG": "C"}
    assert request_digest(TasksetRunInput.model_validate(run_body)) == request_digest(
        TasksetRunInput.model_validate(other)
    )


@pytest.mark.parametrize(
    "path,left,right",
    [
        (("spec", "tasks"), "default/sort-suite", "sort-suite"),
        (("spec", "target", "source"), {"name": "oracle"}, {"name": "codex"}),
        (("timeout_seconds",), None, 3600),
        (("execution", "sandbox"), {"profile": "harbor-standard"}, {"profile": "another-profile"}),
        (("execution", "dispatcher_profile"), "local-k8s", "remote-k8s"),
        (("spec", "target", "artifacts"), ["/a", "/b"], ["/b", "/a"]),
        (("execution", "max_concurrent_task_jobs"), None, 10),
        (("execution", "sandbox"), {"cpus": 2}, {"cpus": 4}),
        (("spec", "target", "agent_kwargs"), {"value": 1}, {"value": 1.0}),
        (("spec", "target", "closed_book"), False, True),
    ],
)
def test_digest_preserves_meaningful_request_differences(run_body, path, left, right):
    requests = []
    for value in (left, right):
        body = deepcopy(run_body)
        node = body
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value
        requests.append(TasksetRunInput.model_validate(body))
    assert request_digest(requests[0]) != request_digest(requests[1])


def test_spec_publication_is_rejected_with_guidance(run_body):
    run_body["spec"]["publication"] = {"intake": {"evaluation_id": "e", "agent_name": "a"}}
    with pytest.raises(ValidationError, match="set publication.intake.experiment"):
        TasksetRunInput.model_validate(run_body)


def test_publication_is_part_of_the_request_identity(run_body):
    """The same idempotency_key with another Experiment is another request (409), not a replay."""
    plain = request_digest(TasksetRunInput.model_validate(run_body))
    digests = {
        experiment: request_digest(
            TasksetRunInput.model_validate(run_body | {"publication": {"intake": {"experiment": experiment}}})
        )
        for experiment in ("swe", "other")
    }
    assert len({plain, *digests.values()}) == 3


def test_digest_hashes_the_exact_compact_json_projection(run_body):
    request = TasksetRunInput.model_validate(run_body)
    expected_body = {
        "spec": {
            "tasks": "default/sort-suite",
            "target": {
                "kind": "harbor",
                "source": {"name": "oracle", "model": None},
                "agent_kwargs": {},
                "env_secrets": {},
                "env_vars": {},
                "n_attempts": 1,
                "n_concurrent_trials": None,
                "max_retries": 0,
                "artifacts": [],
                "trace_dir": None,
                "reward_key": "reward",
                "agent_setup_timeout_multiplier": None,
                "agent_timeout_multiplier": None,
            },
            "trials": None,
            "max_concurrent_tasks": 4,
            "fail_fast": False,
            "labels": {},
            "publication": None,
        },
        "execution": {
            "kind": "harbor-task-jobs",
            "dispatcher_profile": None,
            "max_concurrent_task_jobs": None,
            "sandbox": None,
        },
        "timeout_seconds": None,
    }
    canonical = json.dumps(expected_body, sort_keys=True, separators=(",", ":"))
    digest = request_digest(request)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest == hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("phase", TasksetRunPhase)
@pytest.mark.parametrize("result_state", ResultState)
@pytest.mark.parametrize("populated", [False, True])
def test_status_projects_all_domain_fields_without_persistence_details(phase, result_state, populated):
    accepted = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
    execution = datetime(2026, 10, 12, 13, tzinfo=timezone.utc)
    finalization = datetime(2026, 10, 12, 13, 10, tzinfo=timezone.utc)
    run = TasksetRunEntity(
        name="run-123",
        workspace="default",
        project="private-project",
        taskset_ref="default/sort-suite#" + "a" * 64,
        plan_ref="default/private-plan#plan.json",
        plan_digest="b" * 64,
        request_digest="c" * 64,
        phase=phase,
        result_state=result_state,
        accepted_at=accepted,
        timeout_seconds=3600,
        started_at=accepted if populated else None,
        execution_deadline_at=execution,
        finalization_deadline_at=finalization,
        idempotency_key="retry-1" if populated else None,
        cancel_requested_at=execution if populated else None,
        error="failed task" if populated else None,
    )
    run._created_by = "user:alice" if populated else None
    run._id = "private-entity-id"
    run._parent = "private-parent"
    run._db_version = 7
    run._created_at = accepted
    run._updated_at = execution
    status = to_status(run)
    assert status.model_dump() == {
        "run_id": "run-123",
        "workspace": "default",
        "taskset_ref": "default/sort-suite#" + "a" * 64,
        "phase": phase,
        "result_state": result_state,
        "accepted_at": accepted,
        "timeout_seconds": 3600,
        "started_at": accepted if populated else None,
        "execution_deadline_at": execution,
        "finalization_deadline_at": finalization,
        "idempotency_key": "retry-1" if populated else None,
        "cancel_requested_at": execution if populated else None,
        "error": "failed task" if populated else None,
        "created_by": "user:alice" if populated else None,
        "progress": None,  # set by GET /runs/{id} only
    }
    wire = json.loads(status.model_dump_json())
    assert wire["accepted_at"] == "2026-10-12T12:00:00Z"
    assert not set(wire) & {
        "warnings",
        "plan_ref",
        "plan_digest",
        "request_digest",
        "id",
        "entity_id",
        "project",
        "parent",
        "db_version",
        "created_at",
        "updated_at",
    }


def test_status_schema_covers_every_run_domain_field():
    domain_fields = set(TasksetRunEntity.model_fields) - TasksetRunEntity.__base_fields__
    assert (
        domain_fields
        == (set(TasksetRunStatus.model_fields) - {"run_id", "workspace", "created_by", "progress"}) | _INTERNAL_FIELDS
    )
    assert _INTERNAL_FIELDS == {"plan_ref", "plan_digest", "request_digest"}


def test_status_rejects_unmapped_future_domain_fields():
    class FutureRun(TasksetRunEntity):
        future_domain_field: int = 1

    now = datetime(2026, 10, 12, tzinfo=timezone.utc)
    run = FutureRun(
        name="run-1",
        workspace="default",
        taskset_ref="default/suite",
        plan_ref="default/plan#plan.json",
        plan_digest="a" * 64,
        request_digest="b" * 64,
        accepted_at=now,
        timeout_seconds=60,
        execution_deadline_at=now,
        finalization_deadline_at=now,
    )
    with pytest.raises(ValidationError, match="future_domain_field"):
        to_status(run)


@pytest.mark.parametrize("phase", list(MemberPhase))
@pytest.mark.parametrize("with_details", [False, True])
def test_to_task_preserves_stored_fields_without_internal_metadata(phase, with_details):
    """Task DTOs rename task_job and expose only the seven stored public fields."""
    from nemo_evals.api.taskset_run_schemas import to_task

    member = TasksetRunMemberEntity(
        name="task-10",
        workspace="default",
        task_index=10,
        phase=phase,
        task_job="run-selected-task-10" if with_details else None,
        job_status=HelixJobStatus.ACTIVE if with_details else None,
        error="verifier exited 1\nfull diagnostic" if with_details else None,
        bundle_ref="default/run-selected-task-10#results" if with_details else None,
        create_sent_at=datetime(2026, 10, 12, tzinfo=timezone.utc) if with_details else None,
        publication=PublicationOutcome(status=HelixJobStatus.COMPLETED, evaluation_id="e", trial_count=2)
        if with_details
        else None,
    )
    member._parent = "taskset_run-parent"
    member._id = "taskset_run_member-identifier"
    task = to_task(member)
    assert task.model_dump() == {
        "task": "task-10",
        "task_index": 10,
        "phase": phase,
        "job": "run-selected-task-10" if with_details else None,
        "job_status": HelixJobStatus.ACTIVE if with_details else None,
        "error": "verifier exited 1\nfull diagnostic" if with_details else None,
        "publication": {
            "status": HelixJobStatus.COMPLETED,
            "evaluation_id": "e",
            "trial_count": 2,
            "evaluator_result_count": 0,
            "skipped": [],
            "error": None,
        }
        if with_details
        else None,
    }
