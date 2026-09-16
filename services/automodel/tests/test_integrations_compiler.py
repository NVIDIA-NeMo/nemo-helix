# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import pytest
from nemo_helix_plugin.integrations import IntegrationsSpec
from nemo_helix_plugin.models.types import ModelEntity
from nhx.automodel.api.v2.jobs.schemas import (
    CustomizationJobOutput,
    LoRAParams,
    OutputResponse,
    SFTTraining,
)
from nhx.automodel.app.jobs.training.compiler import compile_training_step
from nhx.automodel.entities.values import OutputNameType
from nhx.common.entities.utils import get_random_id


def _make_model_entity() -> ModelEntity:
    return ModelEntity(
        id=get_random_id("model"),
        workspace="default",
        name="test-target",
        fileset="default/base-model",
        trust_remote_code=False,
        finetuning_type=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


def _job_spec_with_integrations() -> CustomizationJobOutput:
    return CustomizationJobOutput(
        model="default/test-target",
        dataset="default/train",
        training=SFTTraining(
            peft=LoRAParams(rank=8, alpha=32, merge=False),
            learning_rate=1e-4,
            batch_size=4,
            micro_batch_size=1,
            max_seq_length=2048,
        ),
        output=OutputResponse(name="out", type=OutputNameType.ADAPTER, fileset="out-fs"),
        integrations=IntegrationsSpec.model_validate(
            {
                "wandb": {
                    "project": "my-project",
                    "api_key_secret": "default/wandb-key",
                },
            },
        ),
    )


def test_compile_training_step_injects_wandb_secret() -> None:
    step = compile_training_step(_job_spec_with_integrations(), base_env=[], me=_make_model_entity())

    env = step["environment"] if isinstance(step, dict) else step.environment
    assert env is not None
    assert {"name": "WANDB_API_KEY", "from_secret": {"name": "default/wandb-key"}} in env


def test_compile_training_step_no_integrations() -> None:
    spec = _job_spec_with_integrations().model_copy(update={"integrations": None})
    step = compile_training_step(spec, base_env=[], me=_make_model_entity())

    env = step["environment"] if isinstance(step, dict) else step.environment
    assert env is not None
    secret_envs = [item for item in env if item.get("name") == "WANDB_API_KEY"]
    assert secret_envs == []


def test_compile_training_step_warns_on_incomplete_wandb(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spec = _job_spec_with_integrations().model_copy(
        update={
            "integrations": IntegrationsSpec.model_validate({"wandb": {"project": "my-project"}}),
        },
    )
    caplog.set_level("WARNING")

    compile_training_step(spec, base_env=[], me=_make_model_entity())

    assert "api_key_secret is missing" in caplog.text
