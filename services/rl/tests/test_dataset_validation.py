# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GRPO Gym dataset schema: agent-agnostic at submit time, strict on the convert path."""

import json
from pathlib import Path

import jsonschema
import pytest
from nmp.rl.entities.values import TrainingType
from nmp.rl.schemas.environment import GymDatasetRow, GymVerifiersDatasetRow
from nmp.rl.tasks.training.datasets import validation as validation_mod
from nmp.rl.tasks.training.datasets.validation import GRPO_SCHEMA, DatasetValidator

VERIFIERS_ROW = {
    "task_idx": 0,
    "vf_env_id": "acereason-math",
    "responses_create_params": {"input": [{"role": "user", "content": "2+2?"}]},
    "answer": "4",
    "agent_ref": {"type": "responses_api_agents", "name": "verifiers_agent"},
}

# math_with_judge targets simple_agent, carries expected_answer, and has no vf_env_id.
RESOURCES_SERVER_ROW = {
    "task_idx": 0,
    "responses_create_params": {"input": [{"role": "user", "content": "2+2?"}]},
    "question": "2+2?",
    "expected_answer": "4",
    "agent_ref": {"type": "responses_api_agents", "name": "math_with_judge_simple_agent"},
}


@pytest.mark.parametrize("row", [VERIFIERS_ROW, RESOURCES_SERVER_ROW], ids=["verifiers", "resources_server"])
def test_schema_accepts_any_gym_agent(row: dict) -> None:
    """vf_env_id belongs to verifiers_agent; requiring it rejects every other Gym agent."""
    jsonschema.validate(row, GRPO_SCHEMA())


def test_schema_requires_only_what_nemo_rl_reads() -> None:
    assert set(GRPO_SCHEMA()["required"]) == {"responses_create_params", "agent_ref"}


@pytest.mark.parametrize("missing", ["responses_create_params", "agent_ref"], ids=["no_params", "no_agent_ref"])
def test_schema_still_rejects_rows_nemo_rl_cannot_run(missing: str) -> None:
    """NeMo-RL reads both unconditionally, so a row without them fails at rollout."""
    row = {key: value for key, value in VERIFIERS_ROW.items() if key != missing}
    with pytest.raises(jsonschema.exceptions.ValidationError, match=missing):
        jsonschema.validate(row, GRPO_SCHEMA())


def test_convert_path_still_pins_the_verifiers_shape() -> None:
    """pi-to-gym-conversion knows it emits verifiers rows, so it keeps validating them."""
    GymVerifiersDatasetRow.model_validate(VERIFIERS_ROW)
    with pytest.raises(ValueError, match="vf_env_id"):
        GymVerifiersDatasetRow.model_validate(RESOURCES_SERVER_ROW)


def test_agent_specific_fields_are_preserved() -> None:
    """Unknown keys pass through: the resources server reads expected_answer itself."""
    parsed = GymDatasetRow.model_validate(RESOURCES_SERVER_ROW)
    assert parsed.model_dump()["expected_answer"] == "4"


def test_validate_dataset_compiles_schema_once(tmp_path: Path, mocker) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(VERIFIERS_ROW) + "\n" for _ in range(50)))
    compile_spy = mocker.spy(validation_mod, "_compile_validator")
    DatasetValidator(training_type=TrainingType.GRPO).validate_dataset(str(path))
    assert compile_spy.call_count == 1
