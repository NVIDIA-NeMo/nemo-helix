# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsloth SFT training driver.

Single entry point — :func:`train_sft` — invoked from two call-sites:

- In-process from the plugin's ``UnslothJob.run()``.
- Inside a container by ``nhx.unsloth.tasks.training.__main__`` (when
  container submit is wired).

All heavyweight imports (``unsloth``, ``torch``, ``transformers``,
``trl``, ``peft``, ``datasets``) live inside :func:`train_sft` so the
parent process can import this module for dispatch / type lookups
without dragging in ML dependencies.

``import unsloth`` MUST happen before ``transformers`` is imported —
Unsloth monkey-patches transformer modules at import time. Out-of-order
imports silently degrade performance.
"""

import logging
import os
from dataclasses import replace
from typing import Any, Literal

from nemo_helix_plugin.job_context import JobContext
from nhx.customization_common.service.context import NHXJobContext
from nhx.customization_common.training.reporting import DIAGNOSTIC_TIME_SERIES, ProgressReportingConfig
from nhx.unsloth.integrations.hf_bridge import apply_integrations_to_sft_config
from nhx.unsloth.schemas import UnslothJobOutput
from nhx.unsloth.tasks.training.backends.callbacks import TrainingProgressCallback
from nhx.unsloth.tasks.training.progress import JobsServiceProgressReporter

logger = logging.getLogger(__name__)


def compute_default_eval_steps(
    *,
    num_train_samples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int | None = None,
) -> int:
    """Default ``eval_steps`` when validation data is present but ``eval_steps`` is unset.

    Runs one validation pass per effective epoch at
    ``max(1, effective_steps - 1)``, where ``effective_steps`` is
    ``min(steps_per_epoch, max_steps)`` when ``max_steps`` is set (same
    cap automodel uses in ``compute_val_check_interval``).
    """
    effective_batch = per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, (num_train_samples + effective_batch - 1) // effective_batch)
    effective_steps = min(steps_per_epoch, max_steps) if max_steps is not None else steps_per_epoch
    return max(1, effective_steps - 1)


def build_model_load_kwargs(spec: UnslothJobOutput, resolved_model: str) -> dict[str, Any]:
    """Assemble ``FastLanguageModel.from_pretrained`` kwargs (sans torch dtype).

    Kept torch-free so it can be unit-tested without the heavy ML stack; the
    ``dtype`` literal → torch dtype mapping stays in :func:`train_sft`.

    For ``finetuning_type='all_weights'`` we pass ``full_finetuning=True``.
    Unsloth only routes through ``FastModel.from_pretrained`` (which marks every
    parameter trainable and sets up the optimizer/precision state for full FT)
    when this flag is set. Without it — even with 4-/8-bit disabled — Unsloth
    takes its default LoRA-optimized load path and warns that full finetuning
    was not requested, leaving the all-weights run mis-configured.
    """
    kwargs: dict[str, Any] = {
        "model_name": resolved_model,
        "max_seq_length": spec.model.max_seq_length,
        "load_in_4bit": spec.model.load_in_4bit,
        "load_in_8bit": spec.model.load_in_8bit,
        "full_finetuning": spec.training.finetuning_type == "all_weights",
        "trust_remote_code": spec.model.trust_remote_code,
        "device_map": spec.model.device_map if spec.model.device_map is not None else {"": 0},
    }
    # Only pass rope_scaling when set — None lets Unsloth use the model's native context length.
    if spec.model.rope_scaling is not None:
        kwargs["rope_scaling"] = spec.model.rope_scaling
    return kwargs


def build_peft_kwargs(spec: UnslothJobOutput, *, gradient_checkpointing: bool | str) -> dict[str, Any]:
    """Assemble ``FastLanguageModel.get_peft_model`` kwargs for a LoRA run.

    Torch-free (unit-testable). Caller resolves ``gradient_checkpointing`` from
    ``spec.training.use_gradient_checkpointing`` (the JSON literal → ``True`` /
    ``False`` / ``"unsloth"`` mapping). Optional knobs (``loftq_config``,
    ``modules_to_save``, ``layers_to_transform``, ``layer_replication``) are only
    emitted when set so PEFT/Unsloth see absence, not ``None``.
    """
    lora = spec.training.lora
    assert lora is not None  # guaranteed by TrainingSpec._enforce_lora_invariant
    kwargs: dict[str, Any] = {
        "r": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "target_modules": list(lora.target_modules),
        "bias": lora.bias,
        "use_rslora": lora.use_rslora,
        "random_state": lora.random_state,
        "use_dora": lora.use_dora,
        "init_lora_weights": lora.init_lora_weights,
        "use_gradient_checkpointing": gradient_checkpointing,
        "max_seq_length": spec.model.max_seq_length,
    }
    if lora.loftq_config is not None:
        kwargs["loftq_config"] = lora.loftq_config
    if lora.modules_to_save is not None:
        kwargs["modules_to_save"] = lora.modules_to_save
    if lora.layers_to_transform is not None:
        kwargs["layers_to_transform"] = lora.layers_to_transform
    if lora.layer_replication is not None:
        kwargs["layer_replication"] = lora.layer_replication
    return kwargs


def train_sft(
    spec: UnslothJobOutput,
    ctx: JobContext,
    *,
    model_path: str | None = None,
    dataset_path: str | None = None,
    validation_path: str | None = None,
    output_path: str | None = None,
    progress_callback: TrainingProgressCallback | None = None,
) -> dict[str, Any]:
    """Run SFT with Unsloth's FastLanguageModel + LoRA.

    Args:
        spec: Canonical job spec.
        ctx: Live job context (workspace + storage paths).
        model_path: Resolved local path to the model weights. When set,
            overrides ``spec.model.name`` for the call to
            ``FastLanguageModel.from_pretrained``. Pass ``None`` to use
            ``spec.model.name`` directly (legacy behavior, kept for
            tests that exercise raw HF ids).
        dataset_path: Resolved local path to the training dataset. When
            set, overrides ``spec.dataset.path``. ``None`` keeps the
            spec value verbatim.
        validation_path: Resolved local path to the validation dataset
            (optional).
        output_path: Resolved local path the saved checkpoint must
            land at. Set by the container entrypoint to the upload
            step's expected location. When ``None`` falls back to
            ``ctx.storage.persistent / spec.output.name`` (matches the
            historical local-run layout — kept for tests).
        progress_callback: Optional Jobs-service progress reporter.
            When ``None``, a callback is created from platform job
            environment variables (no-op when Jobs context is absent).

    Returns:
        A result dict with final training loss, step count, output
        name/type, the resolved local checkpoint path, and the
        ``CUDA_VISIBLE_DEVICES`` value the training process observed.

    Raises:
        RuntimeError: if the configured ``training.training_type`` is not
            ``"sft"`` (only SFT is implemented today).
    """
    # ── Heavy imports — local to this function ─────────────────────────
    # NB: `unsloth` MUST be imported BEFORE transformers/peft/trl. Do not
    # reorder.
    import unsloth  # noqa: F401  (import-side-effects required)
    from datasets import Dataset, load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    if spec.training.training_type != "sft":
        raise RuntimeError(
            f"Unsloth backend only supports training_type='sft', got {spec.training.training_type!r}",
        )

    # Resolve output path. Container submit passes ``output_path`` (the
    # upload step's expected location); falling back to
    # ``ctx.storage.persistent / spec.output.name`` matches the historic
    # local-run layout and keeps the legacy unit tests green.
    from pathlib import Path

    output_dir = Path(output_path) if output_path else ctx.storage.persistent / spec.output.name
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_rank = spec.training.lora.rank if spec.training.lora else None
    lora_alpha = spec.training.lora.alpha if spec.training.lora else None
    logger.info(
        f"Unsloth SFT: model={spec.model.name} max_seq_length={spec.model.max_seq_length} "
        f"steps={spec.schedule.max_steps} epochs={spec.schedule.epochs} "
        f"lora=(r={lora_rank}, alpha={lora_alpha}) "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )

    # ── Model loading ──────────────────────────────────────────────────
    resolved_model = model_path or spec.model.name
    model_kwargs = build_model_load_kwargs(spec, resolved_model)
    # Unsloth's `dtype` kwarg accepts `None` (auto) or a torch dtype. Map
    # the JSON-friendly literal to a torch dtype lazily.
    if spec.model.dtype != "auto":
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        model_kwargs["dtype"] = dtype_map[spec.model.dtype]

    model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)

    # ── Adapter ────────────────────────────────────────────────────────
    if spec.training.finetuning_type == "lora":
        assert spec.training.lora is not None  # validated by UnslothJobInput
        # use_gradient_checkpointing accepts True / False / "unsloth"; map
        # the JSON literal back.
        gc_value: bool | str
        if spec.training.use_gradient_checkpointing == "unsloth":
            gc_value = "unsloth"
        elif spec.training.use_gradient_checkpointing == "true":
            gc_value = True
        else:
            gc_value = False
        model = FastLanguageModel.get_peft_model(
            model,
            **build_peft_kwargs(spec, gradient_checkpointing=gc_value),
        )
    # All-weights FT: leave `model` as-is. `build_model_load_kwargs` passed
    # `full_finetuning=True`, so `from_pretrained` routed through Unsloth's
    # `FastModel.from_pretrained` and returned an un-wrapped HF model with every
    # parameter trainable (4-/8-bit were already rejected by the spec validator
    # for `finetuning_type='all_weights'`).

    # ── Dataset ────────────────────────────────────────────────────────
    resolved_train_path = dataset_path or spec.dataset.path
    train_ds = _load_training_dataset(
        path=resolved_train_path,
        text_field=spec.dataset.text_field,
        apply_chat_template=spec.dataset.apply_chat_template,
        tokenizer=tokenizer,
        load_dataset=load_dataset,
        Dataset=Dataset,
        split="train",
    )
    eval_ds = None
    resolved_validation_path = validation_path or spec.dataset.validation_path
    if resolved_validation_path:
        eval_ds = _load_training_dataset(
            path=resolved_validation_path,
            text_field=spec.dataset.text_field,
            apply_chat_template=spec.dataset.apply_chat_template,
            tokenizer=tokenizer,
            load_dataset=load_dataset,
            Dataset=Dataset,
            split="validation",
        )

    # ── SFTConfig ───────────────────────────────────────────────────
    bf16 = spec.hardware.precision == "bf16"
    fp16 = spec.hardware.precision == "fp16"

    # Prefer identifiers from the passed JobContext; the in-process UnslothJob.run()
    # path may not have the Job Controller env vars set. Fall back to env-derived
    # values for fields JobContext doesn't carry (step/task).
    job_ctx = NHXJobContext.from_env()
    if ctx.job_id:
        job_ctx = replace(job_ctx, job_id=ctx.job_id, workspace=ctx.workspace)

    report_to, integration_kwargs, integration_env = apply_integrations_to_sft_config(
        integrations=spec.integrations,
        job_ctx=job_ctx,
        output_name=spec.output.name,
        workspace_path=output_dir,
        model_name=spec.model.name,
    )
    for key, value in integration_env.items():
        os.environ[key] = value

    args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": spec.batch.per_device_train_batch_size,
        "gradient_accumulation_steps": spec.batch.gradient_accumulation_steps,
        "learning_rate": spec.optimizer.learning_rate,
        "weight_decay": spec.optimizer.weight_decay,
        "optim": spec.optimizer.optim,
        "adam_beta1": spec.optimizer.adam_beta1,
        "adam_beta2": spec.optimizer.adam_beta2,
        "adam_epsilon": spec.optimizer.adam_epsilon,
        "max_grad_norm": spec.optimizer.max_grad_norm,
        "label_smoothing_factor": spec.optimizer.label_smoothing_factor,
        "lr_scheduler_type": spec.schedule.lr_scheduler_type,
        "warmup_steps": spec.schedule.warmup_steps,
        "logging_steps": spec.schedule.logging_steps,
        "seed": spec.schedule.seed,
        "bf16": bf16,
        "fp16": fp16,
        "report_to": list(report_to),
        # SFT-specific — belong on SFTConfig in trl>=0.13, not on SFTTrainer.
        "dataset_text_field": spec.dataset.text_field,
        "max_length": spec.model.max_seq_length,
        "packing": spec.dataset.packing,
    }
    # Optional knobs: only set when provided so trl/transformers keep their defaults.
    if spec.optimizer.neftune_noise_alpha is not None:
        args_kwargs["neftune_noise_alpha"] = spec.optimizer.neftune_noise_alpha
    if spec.schedule.lr_scheduler_kwargs is not None:
        args_kwargs["lr_scheduler_kwargs"] = spec.schedule.lr_scheduler_kwargs
    if spec.schedule.warmup_ratio is not None:
        args_kwargs["warmup_ratio"] = spec.schedule.warmup_ratio
    # epochs always set (defaults to 1); max_steps, when present, caps/overrides it (trl semantics).
    args_kwargs["num_train_epochs"] = spec.schedule.epochs
    if spec.schedule.max_steps is not None:
        args_kwargs["max_steps"] = spec.schedule.max_steps
    if spec.schedule.save_steps is not None:
        args_kwargs["save_steps"] = spec.schedule.save_steps
        args_kwargs["save_strategy"] = "steps"
    else:
        args_kwargs["save_strategy"] = "epoch"
    eval_steps = spec.schedule.eval_steps
    if eval_ds is not None and eval_steps is None:
        eval_steps = compute_default_eval_steps(
            num_train_samples=len(train_ds),
            per_device_train_batch_size=spec.batch.per_device_train_batch_size,
            gradient_accumulation_steps=spec.batch.gradient_accumulation_steps,
            max_steps=spec.schedule.max_steps,
        )
        logger.info(
            "Default eval_steps=%s (validation data present, schedule.eval_steps unset)",
            eval_steps,
        )
    if eval_ds is not None and eval_steps is not None:
        args_kwargs["eval_steps"] = eval_steps
        args_kwargs["eval_strategy"] = "steps"

    args_kwargs.update(integration_kwargs)

    args = SFTConfig(**args_kwargs)

    progress = progress_callback or _create_progress_callback(spec.schedule.progress_reporting)
    from nhx.unsloth.tasks.training.backends.hf_trainer_callback import (
        create_hf_trainer_progress_callback,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        callbacks=[create_hf_trainer_progress_callback(progress)],
    )
    try:
        train_result = trainer.train()
    finally:
        progress.close()

    # ── Save ──────────────────────────────────────────────────────────
    saved_path = _save_model(model, tokenizer, output_dir, spec)

    return {
        "loss": float(train_result.training_loss),
        "model": spec.model.name,
        "model_path_used": resolved_model,
        "backend": "unsloth",
        "output_name": spec.output.name,
        "output_type": spec.output.type,
        "output_save_method": spec.output.save_method,
        "output_path": str(saved_path),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _create_progress_callback(reporting: ProgressReportingConfig) -> TrainingProgressCallback:
    """Build a Jobs-service progress callback from platform env vars.

    unsloth's default is the shared diagnostic set, and for this backend that is
    a no-op: `create_hf_trainer_progress_callback` reports `loss`, `lr` and
    `grad_norm` on the train path and `loss` on validation, so all four qualified
    names match and nothing is left out. Stated anyway rather than left as None,
    so the policy is written down in the same place as the other two backends'
    and a metric added to that hook later is a deliberate decision rather than a
    silent new curve.

    ``is None`` and not a falsy check: an explicit empty list means "record no
    series at all", which is a different request from "use the default".
    """
    return TrainingProgressCallback(
        JobsServiceProgressReporter(NHXJobContext.from_env()),
        time_series_metrics=(
            DIAGNOSTIC_TIME_SERIES if reporting.time_series_metrics is None else reporting.time_series_metrics
        ),
        min_report_interval_seconds=reporting.min_report_interval_seconds,
    )


TRAIN_FILE = "train.jsonl"
VAL_FILE = "validation.jsonl"


def _resolve_local_data_files(
    path: str,
    *,
    split: Literal["train", "validation"] | None = None,
) -> str | list[str]:
    """Resolve a local dataset path to the JSON/JSONL file(s) to load.

    A file path is returned unchanged. A directory (the file_io download
    step's output dir) is expanded to the sorted ``.jsonl`` files inside
    it, falling back to ``.json``. Both ``.json``/``.jsonl`` and nested
    layouts are handled via a recursive glob.

    When both ``train.jsonl`` and ``validation.jsonl`` live in the same
    directory, pass ``split`` so training and eval each load the right
    file instead of concatenating every JSONL in the folder.

    Raises:
        FileNotFoundError: if ``path`` is a directory with no JSON/JSONL
            files under it.
    """
    from pathlib import Path

    p = Path(path).expanduser()
    if not p.is_dir():
        return str(p)

    if split == "train":
        train_file = p / TRAIN_FILE
        if train_file.is_file():
            return str(train_file)
    elif split == "validation":
        val_file = p / VAL_FILE
        if val_file.is_file():
            return str(val_file)

    for pattern in ("*.jsonl", "*.json"):
        files = sorted(str(f) for f in p.rglob(pattern))
        if files:
            return files

    raise FileNotFoundError(
        f"No .jsonl/.json files found under dataset directory {path!r}. "
        "Ensure the dataset fileset contains a JSON or JSONL file.",
    )


def _load_training_dataset(
    *,
    path: str,
    text_field: str,
    apply_chat_template: bool,
    tokenizer: Any,
    load_dataset: Any,
    Dataset: Any,
    split: Literal["train", "validation"] | None = None,
) -> Any:
    """Load a JSONL or HF dataset; optionally apply the chat template.

    Heuristic for HF id vs local path: presence of a ``.jsonl`` /
    ``.json`` extension or a leading path-ish token (``/``, ``./``,
    ``~``) implies a local path. Anything else is routed through
    ``load_dataset`` as an HF dataset id.

    Container submit hands us the dataset *directory* the file_io
    download step populated (``DEFAULT_DATASET_PATH``), not a single
    file. ``Dataset.from_json`` does not expand a directory, so a local
    directory is resolved to the ``.jsonl`` / ``.json`` file(s) inside it
    before loading. A local file path is passed through unchanged (the
    in-process plugin run hands us a concrete file).
    """
    is_local = path.endswith((".jsonl", ".json")) or path.startswith(("/", "./", "~"))
    raw = Dataset.from_json(_resolve_local_data_files(path, split=split)) if is_local else load_dataset(path)

    if not apply_chat_template:
        return raw

    # Chat-template mode: rows must have a "messages" field; we render
    # each into the requested ``text_field`` so SFTTrainer can find it.
    def _render(example: dict[str, Any]) -> dict[str, Any]:
        rendered = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return {**example, text_field: rendered}

    return raw.map(_render)


def _save_model(model: Any, tokenizer: Any, output_dir: Any, spec: UnslothJobOutput) -> Any:
    """Dispatch on save_method; returns the path actually written to.

    Unsloth's recipes use three methods:
    - ``"lora"``         → ``model.save_pretrained(output_dir)`` (adapter only)
    - ``"merged_16bit"`` → ``model.save_pretrained_merged(..., save_method="merged_16bit")``
    - ``"merged_4bit"``  → ``model.save_pretrained_merged(..., save_method="merged_4bit")``
    """
    if spec.output.save_method == "lora":
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return output_dir

    # merged_16bit or merged_4bit
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method=spec.output.save_method,
    )
    return output_dir
