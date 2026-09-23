# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.unsloth.app.jobs.training.compiler import compile_training_step
from nhx.unsloth.schemas import (
    DatasetSpec,
    LoRAParams,
    ModelLoadSpec,
    OutputResponse,
    TrainingSpec,
    UnslothJobOutput,
)


def _job_spec_with_integrations() -> UnslothJobOutput:
    return UnslothJobOutput(
        model=ModelLoadSpec(name="default/base"),
        dataset=DatasetSpec(path="default/train"),
        training=TrainingSpec(lora=LoRAParams()),
        output=OutputResponse(
            name="out",
            type="adapter",
            save_method="lora",
            fileset="out",
        ),
        integrations={
            "wandb": {
                "project": "my-project",
                "api_key_secret": "default/wandb-key",
            },
        },
    )


def test_compile_training_step_injects_wandb_secret() -> None:
    step = compile_training_step(_job_spec_with_integrations(), base_env=[])

    assert step["environment"] == [
        {"name": "WANDB_API_KEY", "from_secret": {"name": "default/wandb-key"}},
    ]


def test_compile_training_step_no_integrations() -> None:
    spec = _job_spec_with_integrations()
    spec = spec.model_copy(update={"integrations": None})
    step = compile_training_step(spec, base_env=[])

    assert step["environment"] == []


def test_compile_training_step_warns_on_incomplete_wandb(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from nemo_helix_plugin.integrations import IntegrationsSpec

    spec = _job_spec_with_integrations().model_copy(
        update={
            "integrations": IntegrationsSpec.model_validate({"wandb": {"project": "my-project"}}),
        },
    )
    caplog.set_level("WARNING")

    compile_training_step(spec, base_env=[])

    assert "api_key_secret is missing" in caplog.text
