# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from nemo_platform_plugin.models.types import ModelEntity
from nmp.automodel.adapter import automodel_spec_to_compiler_output
from nmp.automodel.api.v2.jobs.schemas import (
    CustomizationJobOutput,
    DistillationTraining,
    EmbeddingParams,
    LoRAParams,
    OutputResponse,
    SFTTraining,
)
from nmp.automodel.app.jobs.compiler import _build_file_download_config
from nmp.automodel.compile import platform_job_config_compiler
from nmp.automodel.entities.values import OutputNameType
from nmp.automodel.images import get_tasks_image, get_training_image
from nmp.common.entities.utils import get_random_id
from nmp.common.jobs.exceptions import PlatformJobCompilationError
from nmp.customization_common.service.platform_client import AsyncCustomizationPlatformClients


def _make_mock_model_entity(
    workspace: str = "default",
    name: str = "test-target",
    fileset: str | None = "default/base-model",
) -> ModelEntity:
    return ModelEntity(
        id=get_random_id("model"),
        workspace=workspace,
        name=name,
        fileset=fileset,
        trust_remote_code=False,
        finetuning_type=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def platform_clients() -> AsyncCustomizationPlatformClients:
    return AsyncCustomizationPlatformClients(files=AsyncMock(), models=AsyncMock())


def _output(*, output_type: OutputNameType = OutputNameType.ADAPTER) -> OutputResponse:
    return OutputResponse(name="out", type=output_type, fileset="out-fs")


def _executor_container(step: Any) -> Any:
    """CPU/GPU executors have a container; subprocess does not. Tests only compile the former."""
    executor = step.executor if hasattr(step, "executor") else step["executor"]
    container = getattr(executor, "container", None)
    if container is None and isinstance(executor, dict):
        container = executor.get("container")
    assert container is not None, "expected a container-backed execution provider"
    return container


def _make_job_output() -> CustomizationJobOutput:
    return CustomizationJobOutput(
        model="default/test-target",
        dataset="default/my-dataset",
        training=SFTTraining(
            peft=LoRAParams(rank=8, alpha=32, merge=False),
            learning_rate=1e-4,
            batch_size=4,
            micro_batch_size=1,
            max_seq_length=2048,
        ),
        output=_output(),
    )


def test_build_file_download_config_rejects_missing_model_fileset() -> None:
    with pytest.raises(PlatformJobCompilationError, match="has no fileset"):
        _build_file_download_config(_make_job_output(), _make_mock_model_entity(fileset=None))


def test_compile_training_step_carries_pass2_fields() -> None:
    """Pass-2 hyperparameters on the v2 SFTTraining reach the internal TrainingStepConfig."""
    from nmp.automodel.app.jobs.training.compiler import compile_training_step

    job_output = CustomizationJobOutput(
        model="default/test-target",
        dataset="default/my-dataset",
        training=SFTTraining(
            peft=LoRAParams(rank=8, alpha=32, merge=False, exclude_modules=["*.out_proj"], use_triton=False),
            learning_rate=1e-4,
            adam_eps=1e-6,
            optimizer="AdamW",
            lr_decay_style="linear",
            attn_implementation="flash_attention_2",
            batch_size=4,
            micro_batch_size=1,
            sequence_packing=True,
            sequence_packing_max_samples=256,
            max_seq_length=2048,
        ),
        output=_output(),
    )
    step = compile_training_step(job_output, base_env=[], me=_make_mock_model_entity())
    cfg = step.config if hasattr(step, "config") else step["config"]

    assert cfg["optimizer"]["optimizer_name"] == "AdamW"
    assert cfg["optimizer"]["lr_decay_style"] == "linear"
    assert cfg["optimizer"]["eps"] == 1e-6
    assert cfg["model"]["attn_implementation"] == "flash_attention_2"
    assert cfg["batch"]["sequence_packing_max_samples"] == 256
    assert cfg["training"]["lora"]["exclude_modules"] == ["*.out_proj"]
    assert cfg["training"]["lora"]["use_triton"] is False


def test_compile_training_step_carries_explicit_cross_encoder_recipe() -> None:
    from nmp.automodel.app.jobs.training.compiler import compile_training_step

    job_output = CustomizationJobOutput(
        model="default/test-target",
        dataset="default/my-dataset",
        training=SFTTraining(
            recipe="cross_encoder",
            peft=None,
            batch_size=4,
            micro_batch_size=1,
        ),
        output=_output(output_type=OutputNameType.MODEL),
    )

    step = compile_training_step(job_output, base_env=[], me=_make_mock_model_entity())
    cfg = step.config if hasattr(step, "config") else step["config"]

    assert cfg["training"]["recipe"] == "cross_encoder"


def test_compile_training_step_carries_embedding_config() -> None:
    from nmp.automodel.app.jobs.training.compiler import compile_training_step

    job_output = CustomizationJobOutput(
        model="default/test-target",
        dataset="default/my-dataset",
        training=SFTTraining(
            recipe="bi_encoder",
            peft=None,
            batch_size=4,
            micro_batch_size=1,
            embedding=EmbeddingParams(
                train_n_passages=7,
                query_prefix="query: ",
                passage_prefix="passage: ",
                query_max_length=256,
            ),
        ),
        output=_output(output_type=OutputNameType.MODEL),
    )

    step = compile_training_step(job_output, base_env=[], me=_make_mock_model_entity())
    cfg = step.config if hasattr(step, "config") else step["config"]

    assert cfg["embedding"]["train_n_passages"] == 7
    assert cfg["embedding"]["query_prefix"] == "query: "
    assert cfg["embedding"]["passage_prefix"] == "passage: "
    assert cfg["embedding"]["query_max_length"] == 256


def test_sft_training_applies_nemotron_defaults_for_encoder_recipes() -> None:
    embed = SFTTraining.model_validate({"recipe": "bi_encoder"}).with_resolved_recipe("bi_encoder")
    rerank = SFTTraining.model_validate({"recipe": "cross_encoder"}).with_resolved_recipe("cross_encoder")
    sft = SFTTraining.model_validate({"recipe": "sft"}).with_resolved_recipe("sft")

    assert embed.batch_size == 128
    assert embed.micro_batch_size == 4
    assert embed.learning_rate == 1e-5
    assert embed.warmup_steps == 5
    assert rerank.learning_rate == 3e-6
    assert rerank.warmup_steps == 100
    assert sft.batch_size == 32
    assert sft.learning_rate == 1e-4
    assert sft.warmup_steps == 0


def test_sft_training_encoder_defaults_do_not_override_explicit_hparams() -> None:
    training = SFTTraining.model_validate(
        {"recipe": "bi_encoder", "batch_size": 16, "learning_rate": 2e-5, "warmup_steps": 1}
    ).with_resolved_recipe("bi_encoder")
    assert training.batch_size == 16
    assert training.learning_rate == 2e-5
    assert training.warmup_steps == 1


def test_distillation_with_resolved_recipe_rejects_encoder_recipes() -> None:
    training = DistillationTraining.model_validate({"recipe": "auto", "teacher_model": "default/teacher"})
    with pytest.raises(ValueError, match="only supports the sft recipe"):
        training.with_resolved_recipe("bi_encoder")
    resolved = training.with_resolved_recipe("sft")
    assert resolved.recipe == "sft"


def test_resolve_training_recipe_auto_uses_cross_encoder_head() -> None:
    from nmp.automodel.app.jobs.training.compiler import _resolve_training_recipe
    from nmp.automodel.app.jobs.training.schemas import TrainingRecipe

    me = Mock()
    me.spec.model_fields_set = {"head_type"}
    me.spec.head_type = "cross_encoder"
    assert _resolve_training_recipe(me, "auto") == TrainingRecipe.CROSS_ENCODER


def test_compile_training_step_applies_retrieval_defaults_after_auto_resolution() -> None:
    from nmp.automodel.app.jobs.training.compiler import compile_training_step

    me = _make_mock_model_entity()
    me.spec = Mock()
    me.spec.model_fields_set = {"head_type"}
    me.spec.head_type = "embedding"
    me.spec.chat_template = None
    me.spec.checkpoint_model_name = None
    me.spec.moe_config = None
    me.spec.family = None
    me.spec.is_chat = False

    job_output = CustomizationJobOutput(
        model="default/test-target",
        dataset="default/my-dataset",
        training=SFTTraining(recipe="auto"),
        output=_output(output_type=OutputNameType.MODEL),
    )
    step = compile_training_step(job_output, base_env=[], me=me)
    cfg = step.config if hasattr(step, "config") else step["config"]
    assert cfg["training"]["recipe"] == "bi_encoder"
    assert cfg["batch"]["global_batch_size"] == 128
    assert cfg["optimizer"]["learning_rate"] == 1e-5
    assert cfg["optimizer"]["warmup_steps"] == 5


def test_compile_training_step_auto_defaults_keep_explicit_lr() -> None:
    from nmp.automodel.app.jobs.training.compiler import compile_training_step

    me = _make_mock_model_entity()
    me.spec = Mock()
    me.spec.model_fields_set = {"head_type"}
    me.spec.head_type = "embedding"
    me.spec.chat_template = None
    me.spec.checkpoint_model_name = None
    me.spec.moe_config = None
    me.spec.family = None
    me.spec.is_chat = False

    job_output = CustomizationJobOutput(
        model="default/test-target",
        dataset="default/my-dataset",
        training=SFTTraining(recipe="auto", learning_rate=2e-5),
        output=_output(output_type=OutputNameType.MODEL),
    )
    step = compile_training_step(job_output, base_env=[], me=me)
    cfg = step.config if hasattr(step, "config") else step["config"]
    assert cfg["optimizer"]["learning_rate"] == 2e-5
    assert cfg["batch"]["global_batch_size"] == 128


@pytest.mark.asyncio
async def test_platform_job_config_compiler_rejects_unmerged_lora_for_encoders(
    platform_clients: AsyncCustomizationPlatformClients,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nmp.automodel.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_mock_model_entity()),
    )
    job = CustomizationJobOutput(
        model="default/test-target",
        dataset="default/my-dataset",
        training=SFTTraining(
            recipe="cross_encoder",
            peft=LoRAParams(rank=8, alpha=32, merge=False),
            batch_size=4,
            micro_batch_size=1,
        ),
        output=_output(),
    )
    with pytest.raises(PlatformJobCompilationError, match="unmerged LoRA"):
        await platform_job_config_compiler(job, "default", platform_clients)


def test_the_reporting_budget_reaches_the_training_step_config() -> None:
    """One more hop in a chain that is long enough to break quietly.

    plugin ScheduleSpec -> adapter -> CustomizationJobOutput.training ->
    TrainingStepConfig.ScheduleConfig -> the recipe config -> the callback. Every
    link defaults, so a broken one reports at 200 rather than failing, which is
    exactly the kind of regression nothing else here would notice.
    """
    from nmp.automodel.app.jobs.training.compiler import compile_training_step
    from nmp.customization_common.training.reporting import ProgressReportingConfig

    job_output = CustomizationJobOutput(
        model="default/test-target",
        dataset="default/my-dataset",
        training=SFTTraining(progress_reporting=ProgressReportingConfig(time_series_metrics=["*_loss"])),
        output=_output(),
    )
    step = compile_training_step(job_output, base_env=[], me=_make_mock_model_entity())
    cfg = step.config if hasattr(step, "config") else step["config"]

    assert cfg["schedule"]["progress_reporting"]["time_series_metrics"] == ["*_loss"]


def test_a_spec_still_carrying_log_every_n_steps_compiles() -> None:
    """The removed field was inert, and removing it has to stay invisible.

    It described itself as controlling how often training metrics are logged and
    controlled nothing: nothing read it, it never reached the recipe config, and
    it was in neither the submitter-facing plugin schema nor any generated spec.
    What makes deleting it safe rather than breaking is that this model ignores
    extras -- so a stored spec that still carries the key parses as it always
    did. Pinned because a later `extra="forbid"` here would turn that silent
    tolerance into a hard failure for exactly those specs.
    """
    from nmp.automodel.app.jobs.training.compiler import compile_training_step

    training = SFTTraining.model_validate({"learning_rate": 1e-4, "log_every_n_steps": 10})
    assert not hasattr(training, "log_every_n_steps")

    job_output = CustomizationJobOutput(
        model="default/test-target",
        dataset="default/my-dataset",
        training=training,
        output=_output(),
    )
    step = compile_training_step(job_output, base_env=[], me=_make_mock_model_entity())
    cfg = step.config if hasattr(step, "config") else step["config"]

    assert cfg["optimizer"]["learning_rate"] == 1e-4, "the spec compiles, key and all"
    assert "log_every_n_steps" not in cfg["schedule"]


def test_the_reporting_budget_survives_the_plugin_adapter() -> None:
    """The adapter flattens the plugin's schedule block and is easy to drop a field from."""
    from nmp.automodel.adapter import automodel_spec_to_compiler_output

    spec = {
        "model": "default/test-target",
        "dataset": {"training": "default/my-dataset"},
        "training": {"training_type": "sft", "finetuning_type": "lora"},
        "schedule": {"epochs": 1, "progress_reporting": {"time_series_metrics": ["*_loss"]}},
        "output": {"name": "out", "type": "adapter", "fileset": "out-fs"},
    }
    reporting = automodel_spec_to_compiler_output(spec).training.progress_reporting

    assert reporting.time_series_metrics == ["*_loss"]


def test_a_plugin_spec_without_a_schedule_block_still_compiles() -> None:
    """`schedule` is optional in the plugin shape, so the adapter must not index it."""
    from nmp.automodel.adapter import automodel_spec_to_compiler_output

    spec = {
        "model": "default/test-target",
        "dataset": {"training": "default/my-dataset"},
        "training": {"training_type": "sft", "finetuning_type": "lora"},
        "output": {"name": "out", "type": "adapter", "fileset": "out-fs"},
    }
    reporting = automodel_spec_to_compiler_output(spec).training.progress_reporting

    assert reporting.time_series_metrics is None


@pytest.mark.asyncio
async def test_platform_job_config_compiler_sft_lora(
    platform_clients: AsyncCustomizationPlatformClients,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nmp.automodel.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_mock_model_entity()),
    )
    contract_dir = Path(__file__).resolve().parent / "contract" / "input_configs"
    input_path = contract_dir / "llama-3.2-1b" / "llama_3_2_1b_lora.json"
    if not input_path.exists():
        pytest.skip("contract configs not present")

    raw = json.loads(input_path.read_text())
    plugin_shape = {
        "model": raw["model"]["path"],
        "dataset": {"training": "default/train-data"},
        "training": {
            "training_type": "sft",
            "finetuning_type": "lora",
            "lora": {
                "rank": raw["training"]["lora"]["rank"],
                "alpha": raw["training"]["lora"]["alpha"],
                "merge": False,
            },
            "max_seq_length": raw["model"]["max_seq_length"],
        },
        "schedule": {
            "epochs": raw["schedule"]["epochs"],
            "max_steps": raw["schedule"]["max_steps"],
        },
        "batch": {
            "global_batch_size": raw["batch"]["global_batch_size"],
            "micro_batch_size": raw["batch"]["micro_batch_size"],
        },
        "optimizer": {"learning_rate": raw["optimizer"]["learning_rate"]},
        "parallelism": {
            "num_nodes": raw["parallelism"]["num_nodes"],
            "num_gpus_per_node": raw["parallelism"]["num_gpus_per_node"],
            "tensor_parallel_size": raw["parallelism"]["tensor_parallel_size"],
        },
        "output": {"name": "test-out", "type": "adapter", "fileset": "test-out-fs"},
    }
    compiler_spec = automodel_spec_to_compiler_output(plugin_shape)
    spec = await platform_job_config_compiler(compiler_spec, "default", platform_clients)

    steps = spec.steps if hasattr(spec, "steps") else spec["steps"]
    assert len(steps) == 4
    training_step = steps[1]
    training_name = training_step.name if hasattr(training_step, "name") else training_step["name"]
    assert training_name == "training"
    training_cmd = _executor_container(training_step).command
    assert "nmp.automodel.tasks.training" in " ".join(training_cmd)
    download_cmd = _executor_container(steps[0]).command
    assert download_cmd == [
        "-m",
        "nmp.customization_common.tasks.file_io",
        "--service-source",
        "automodel",
        "--service-name",
        "customizer",
    ]
    download_entrypoint = _executor_container(steps[0]).entrypoint
    assert download_entrypoint == ["/opt/venv/bin/python"]
    for cpu_step in (steps[0], steps[2], steps[3]):
        executor = cpu_step.executor if hasattr(cpu_step, "executor") else cpu_step["executor"]
        profile = getattr(executor, "profile", None)
        if profile is None and isinstance(executor, dict):
            profile = executor.get("profile")
        assert profile == "gpu"

    def _step_image(step: Any) -> str:
        return _executor_container(step).image

    assert _step_image(steps[0]) == get_tasks_image()
    assert _step_image(steps[1]) == get_training_image()
    assert _step_image(steps[2]) == get_tasks_image()
    assert _step_image(steps[3]) == get_tasks_image()


@pytest.mark.asyncio
async def test_platform_job_config_compiler_applies_profile_to_task_steps(
    platform_clients: AsyncCustomizationPlatformClients,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nmp.automodel.app.jobs.compiler.fetch_model_entity",
        AsyncMock(return_value=_make_mock_model_entity()),
    )

    spec = await platform_job_config_compiler(_make_job_output(), "default", platform_clients, profile="custom-gpu")

    assert [step.executor.profile for step in spec.steps] == ["custom-gpu"] * 4
