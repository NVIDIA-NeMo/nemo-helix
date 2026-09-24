# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved definitions carry validated origins through the persisted job contract."""

import pytest
from nemo_evaluator.api.task_definitions.evaluator import EvaluatorTaskDefinition, ResolvedEvaluatorTaskDefinition
from nemo_evaluator.api.task_definitions.harbor import HarborTaskDefinition, ResolvedHarborTaskDefinition
from nemo_evaluator.jobs.agent_spec import ResolvedTask, validate_task_collection
from pydantic import ValidationError


@pytest.fixture
def harbor_definition():
    return {
        "kind": "harbor",
        "native_task_id": "native/task",
        "source": {"fileset_ref": "default/files#archive", "files_hash": "b" * 64},
        "harbor_hash": {"digest": "c" * 64, "harbor_version": "test"},
        "provenance": {"entity_name": "default/stored", "revision_digest": "a" * 64},
    }


@pytest.mark.parametrize("missing", [True, False])
def test_harbor_definition_requires_stored_origin(harbor_definition, missing):
    if missing:
        del harbor_definition["provenance"]
    else:
        harbor_definition["provenance"] = None
    with pytest.raises(ValidationError, match="provenance"):
        ResolvedHarborTaskDefinition.model_validate(harbor_definition)


@pytest.mark.parametrize("kind", ["evaluator", "harbor"])
def test_stored_origin_round_trips_inside_definition(harbor_definition, kind):
    definition = (
        harbor_definition
        if kind == "harbor"
        else {"kind": "evaluator", "intent": "Answer", "provenance": harbor_definition["provenance"]}
    )
    task = ResolvedTask.model_validate({"id": "native/task", "spec": definition})
    payload = task.model_dump(mode="json")
    assert "provenance" not in payload
    assert payload["spec"]["provenance"] == {"entity_name": "default/stored", "revision_digest": "a" * 64}
    assert ResolvedTask.model_validate_json(task.model_dump_json()) == task
    payload["provenance"] = payload["spec"].pop("provenance")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResolvedTask.model_validate(payload)


def test_inline_evaluator_definition_has_no_stored_origin():
    definition = ResolvedEvaluatorTaskDefinition(kind="evaluator", intent="Answer")
    assert definition.provenance is None
    assert ResolvedEvaluatorTaskDefinition.model_validate_json(definition.model_dump_json()) == definition


@pytest.mark.parametrize(
    "field,value",
    [
        ("entity_name", "unqualified"),
        ("entity_name", "default/stored#latest"),
        ("revision_digest", "latest"),
        ("revision_digest", "g" * 64),
    ],
)
def test_resolved_origin_rejects_invalid_identity_and_digest(harbor_definition, field, value):
    harbor_definition["provenance"][field] = value
    with pytest.raises(ValidationError) as exc:
        ResolvedHarborTaskDefinition.model_validate(harbor_definition)
    assert exc.value.errors()[0]["loc"] == ("provenance", field)


def test_nested_origin_still_rejects_duplicate_entities():
    tasks = [
        ResolvedTask.model_validate(
            {
                "id": task_id,
                "spec": {
                    "kind": "evaluator",
                    "intent": "Answer",
                    "provenance": {"entity_name": "default/stored", "revision_digest": digest * 64},
                },
            }
        )
        for task_id, digest in [("first", "a"), ("second", "b")]
    ]
    with pytest.raises(ValueError, match="Duplicate task identity"):
        validate_task_collection(tasks)


def test_nested_harbor_origin_preserves_runtime_id_invariant(harbor_definition):
    with pytest.raises(ValidationError, match="id must equal spec.native_task_id"):
        ResolvedTask.model_validate({"id": "different", "spec": harbor_definition})


@pytest.mark.parametrize(
    "model,payload",
    [
        (EvaluatorTaskDefinition, {"kind": "evaluator", "intent": "Answer"}),
        (
            HarborTaskDefinition,
            {
                "kind": "harbor",
                "native_task_id": "native/task",
                "source": {"fileset_ref": "default/files#archive", "files_hash": "b" * 64},
                "harbor_hash": {"digest": "c" * 64, "harbor_version": "test"},
            },
        ),
    ],
)
def test_published_definitions_do_not_accept_snapshot_origin(model, payload):
    model.model_validate(payload)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate({**payload, "provenance": {"entity_name": "default/stored", "revision_digest": "a" * 64}})
