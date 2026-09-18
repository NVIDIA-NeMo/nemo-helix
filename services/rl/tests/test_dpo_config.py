# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the config builders in dpo_config.

Mostly the pure helpers — optimizer, scheduler, precision, data, logger, and the
inert Megatron block. ``compile_dpo_config`` itself needs a real on-disk dataset
to prepare and validate, so it is only exercised where nothing else can stand in:
the reporting budget's hop through the DPO block, which no other layer can see.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nmp.customization_common.service.context import NMPJobContext
from nmp.rl.app.jobs.training.schemas import (
    DPOConfig,
    ModelConfig,
    OptimizerType,
    TrainingStepConfig,
    TrainingType,
)
from nmp.rl.tasks.training.backends.nemo_rl import dpo_config
from nmp.rl.tasks.training.backends.nemo_rl.dpo_config import (
    _adapt_precision,
    _build_data_config,
    _build_logger_config,
    _build_optimizer_config,
    _build_scheduler_config,
    _build_tokenizer_config,
    _megatron_cfg_disabled,
)
from nmp.rl.tasks.training.datasets.preparation import PreparedDataset


def _make_step_config(
    *,
    optimizer: TrainingStepConfig.OptimizerConfig | None = None,
    schedule: TrainingStepConfig.ScheduleConfig | None = None,
    parallelism: TrainingStepConfig.ParallelismConfig | None = None,
    max_seq_length: int = 1024,
) -> TrainingStepConfig:
    return TrainingStepConfig(
        model=ModelConfig(path="/model", max_seq_length=max_seq_length),
        dataset=TrainingStepConfig.DatasetConfig(path="/data"),
        training=TrainingStepConfig.TrainingConfig(training_type=TrainingType.DPO, dpo=DPOConfig()),
        schedule=schedule or TrainingStepConfig.ScheduleConfig(),
        batch=TrainingStepConfig.BatchConfig(),
        optimizer=optimizer or TrainingStepConfig.OptimizerConfig(),
        parallelism=parallelism or TrainingStepConfig.ParallelismConfig(),
        output_model="out",
    )


def _write_preference_dataset(directory: Path, rows: int = 8) -> Path:
    """The minimum on-disk shape ``prepare_dataset`` and the validator accept."""
    import json

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "train.jsonl"
    path.write_text(
        "".join(
            json.dumps({"prompt": f"q{i}", "chosen": f"good{i}", "rejected": f"bad{i}"}) + "\n" for i in range(rows)
        ),
        encoding="utf-8",
    )
    return path


def _job_ctx(tmp_path: Path) -> NMPJobContext:
    return NMPJobContext(
        workspace="default",
        job_id="rl-test",
        attempt_id="attempt-1",
        step="dpo-training",
        task="task-1",
        jobs_url=None,
        files_url=None,
        storage_path=tmp_path,
        config_path=tmp_path / "config.json",
    )


# --------------------------------------------------------------------------- #
# _adapt_precision
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [
        ("bf16", "bfloat16"),
        ("bf16-mixed", "bfloat16"),
        ("fp16", "float16"),
        ("fp32", "float32"),
        (None, "bfloat16"),
        ("nonsense", "bfloat16"),  # unknown → safe default
    ],
)
def test_adapt_precision(value: str | None, expected: str) -> None:
    assert _adapt_precision(value) == expected


# --------------------------------------------------------------------------- #
# _build_optimizer_config
# --------------------------------------------------------------------------- #


def test_optimizer_config_adamw_default() -> None:
    opt = _build_optimizer_config(_make_step_config())  # optimizer_type None → AdamW
    assert opt["name"] == "torch.optim.AdamW"


@pytest.mark.parametrize(
    "opt_type",
    [OptimizerType.ADAM_WITH_COSINE_ANNEALING, OptimizerType.ADAM_WITH_FLAT_LR],
)
def test_optimizer_config_adam_variants(opt_type: OptimizerType) -> None:
    cfg = _make_step_config(optimizer=TrainingStepConfig.OptimizerConfig(optimizer_type=opt_type))
    assert _build_optimizer_config(cfg)["name"] == "torch.optim.Adam"


def test_optimizer_config_passes_through_kwargs() -> None:
    optimizer = TrainingStepConfig.OptimizerConfig(
        learning_rate=2e-5, weight_decay=0.05, beta1=0.8, beta2=0.95, eps=3e-7
    )
    kwargs = _build_optimizer_config(_make_step_config(optimizer=optimizer))["kwargs"]
    assert kwargs["lr"] == 2e-5
    assert kwargs["weight_decay"] == 0.05
    assert kwargs["betas"] == [0.8, 0.95]
    assert kwargs["eps"] == 3e-7  # the configurable knob actually flows through


# --------------------------------------------------------------------------- #
# _build_scheduler_config
# --------------------------------------------------------------------------- #


def test_scheduler_cosine_is_warmup_then_decay_chain() -> None:
    cfg = _make_step_config(
        optimizer=TrainingStepConfig.OptimizerConfig(
            optimizer_type=OptimizerType.ADAMW_WITH_COSINE_ANNEALING, warmup_steps=10
        )
    )
    sched = _build_scheduler_config(cfg, num_steps=100)
    assert isinstance(sched, list)
    assert sched[0]["name"] == "torch.optim.lr_scheduler.LinearLR"
    assert sched[1]["name"] == "torch.optim.lr_scheduler.CosineAnnealingLR"
    assert sched[2]["milestones"] == [10]


def test_scheduler_flat_is_constant_lr() -> None:
    """Flat LR still has to arrive as a list.

    NeMo-RL types policy.scheduler as a list of scheduler/milestone entries or a
    SchedulerMilestones mapping. A bare {"name", "kwargs"} dict is neither, and MasterConfig
    rejects it up front -- the job dies in the driver before the model loads.
    """
    cfg = _make_step_config(
        optimizer=TrainingStepConfig.OptimizerConfig(optimizer_type=OptimizerType.ADAMW_WITH_FLAT_LR)
    )
    sched = _build_scheduler_config(cfg, num_steps=100)
    assert isinstance(sched, list)
    assert sched[0]["name"] == "torch.optim.lr_scheduler.ConstantLR"
    assert sched[0]["kwargs"] == {"factor": 1.0, "total_iters": 100}
    assert sched[1]["milestones"] == []


@pytest.mark.parametrize(
    "optimizer_type",
    [
        OptimizerType.ADAMW_WITH_COSINE_ANNEALING,
        OptimizerType.ADAM_WITH_COSINE_ANNEALING,
        OptimizerType.ADAMW_WITH_FLAT_LR,
        OptimizerType.ADAM_WITH_FLAT_LR,
        None,
    ],
)
def test_scheduler_shape_is_accepted_by_nemo_rl(optimizer_type: OptimizerType | None) -> None:
    """Every optimizer_type must emit a shape NeMo-RL's MasterConfig accepts."""
    cfg = _make_step_config(optimizer=TrainingStepConfig.OptimizerConfig(optimizer_type=optimizer_type))
    sched = _build_scheduler_config(cfg, num_steps=100)

    if isinstance(sched, list):
        # list[SinglePytorchSchedulerConfig | SinglePytorchMilestonesConfig]
        assert all(("name" in entry and "kwargs" in entry) or "milestones" in entry for entry in sched)
    else:
        # SchedulerMilestones = dict[str, list[int]]
        assert all(isinstance(value, list) for value in sched.values())


# --------------------------------------------------------------------------- #
# _megatron_cfg_disabled (inert block, must still be fully populated)
# --------------------------------------------------------------------------- #


def test_megatron_cfg_is_inert_but_complete() -> None:
    mc = _megatron_cfg_disabled(precision="bfloat16", max_grad_norm=2.5)
    assert mc["enabled"] is False
    assert mc["pipeline_dtype"] == "bfloat16"  # tracks policy.precision
    assert mc["optimizer"]["clip_grad"] == 2.5  # tracks policy.max_grad_norm
    # All required sub-blocks present so NeMo-RL's schema validates.
    for key in ("peft", "optimizer", "scheduler", "distributed_data_parallel_config", "fp8_cfg"):
        assert key in mc
    assert mc["fp8_cfg"]["enabled"] is False
    assert mc["peft"]["enabled"] is False


# --------------------------------------------------------------------------- #
# _build_data_config
# --------------------------------------------------------------------------- #


def test_data_config_binary_preference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dpo_config, "detect_dpo_schema_name", lambda _path: "BinaryPreferenceDataset")
    prepared = PreparedDataset(
        merged_dir=tmp_path,
        train_file=tmp_path / "training.jsonl",
        validation_file=tmp_path / "validation.jsonl",
        train_samples=10,
        validation_samples=2,
    )
    data = _build_data_config(_make_step_config(max_seq_length=512), prepared)

    assert data["max_input_seq_length"] == 512
    assert data["shuffle"] is False  # deterministic ordering is an intentional override
    for split in ("train", "validation"):
        assert data[split]["dataset_name"] == "BinaryPreferenceDataset"
        assert data[split]["prompt_key"] == "prompt"
        assert data[split]["chosen_key"] == "chosen"
        assert data[split]["rejected_key"] == "rejected"
    assert data["train"]["data_path"] == str(prepared.train_file)


# --------------------------------------------------------------------------- #
# _build_logger_config
# --------------------------------------------------------------------------- #


def test_logger_config_has_all_subsections_when_integrations_disabled(tmp_path: Path) -> None:
    cfg = _build_logger_config(_make_step_config(), _job_ctx(tmp_path), tmp_path)

    assert cfg["wandb_enabled"] is False
    assert cfg["mlflow_enabled"] is False
    assert cfg["monitor_gpus"] is False
    assert cfg["log_dir"].endswith("logs")
    # Every backend subsection is present even when disabled (NeMo-RL expects them).
    for key in ("wandb", "swanlab", "tensorboard", "mlflow", "gpu_monitoring"):
        assert key in cfg


# --------------------------------------------------------------------------- #
# compile_dpo_config: the reporting budget's one hop through the DPO block
# --------------------------------------------------------------------------- #


def test_the_reporting_budget_rides_the_dpo_block(tmp_path: Path) -> None:
    """The compiled DPO config is the only channel from the job to the driver.

    ``max_progress_points`` is not a NeMo-RL field; it rides as an undeclared
    extra alongside ``steps_per_epoch``, and the driver reads it back with
    getattr to build the progress callback's gate. Nothing upstream of the
    driver would notice it being dropped -- the run would simply report at the
    default forever.
    """
    from nmp.customization_common.training.reporting import ProgressReportingConfig
    from nmp.rl.tasks.training.backends.nemo_rl.dpo_config import compile_dpo_config

    _write_preference_dataset(tmp_path / "data")
    step_config = _make_step_config(
        schedule=TrainingStepConfig.ScheduleConfig(
            progress_reporting=ProgressReportingConfig(time_series_metrics=["*_loss"]),
        ),
    )
    step_config.dataset.path = str(tmp_path / "data")
    step_config.workspace_path = str(tmp_path / "work")

    cfg = compile_dpo_config(step_config, _job_ctx(tmp_path))

    assert cfg["dpo"]["progress_time_series_metrics"] == ["*_loss"]


def test_the_dpo_block_carries_the_default_when_unstated(tmp_path: Path) -> None:
    from nmp.rl.tasks.training.backends.nemo_rl.dpo_config import compile_dpo_config

    _write_preference_dataset(tmp_path / "data")
    step_config = _make_step_config()
    step_config.dataset.path = str(tmp_path / "data")
    step_config.workspace_path = str(tmp_path / "work")

    cfg = compile_dpo_config(step_config, _job_ctx(tmp_path))

    assert cfg["dpo"]["progress_time_series_metrics"] is None, "absent means everything, not nothing"


def test_tokenizer_config_omits_chat_template_when_none() -> None:
    """NeMo-RL's TokenizerConfig has ``chat_template: NotRequired[str]``.

    Absent is valid; ``None`` is not. ``resolve_chat_template`` returns None for any
    model that ships no template and has no user override, so emitting the key
    unconditionally fails MasterConfig validation in the driver. Shared by DPO and
    GRPO, hence tested on the helper rather than in one backend.
    """
    tokenizer = _build_tokenizer_config("/var/run/scratch/job/model", None)

    assert "chat_template" not in tokenizer
    assert tokenizer["name"] == "/var/run/scratch/job/model"
    # NotRequired[dict | None] -- None is accepted here, unlike chat_template.
    assert tokenizer["chat_template_kwargs"] is None


def test_tokenizer_config_keeps_chat_template_when_present() -> None:
    tokenizer = _build_tokenizer_config("/model", "{{ bos_token }}")

    assert tokenizer["chat_template"] == "{{ bos_token }}"
