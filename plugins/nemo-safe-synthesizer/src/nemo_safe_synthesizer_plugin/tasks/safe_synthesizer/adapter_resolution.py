# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation-only reuse of prior NSS adapter artifacts."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from nemo_safe_synthesizer.cli.artifact_structure import RunName, Workdir
from nemo_safe_synthesizer.config.internal_results import SafeSynthesizerResults
from nemo_safe_synthesizer_plugin.job_config import DEFAULT_PRETRAINED_MODEL

if TYPE_CHECKING:
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
    from nemo_safe_synthesizer_plugin.job_config import SafeSynthesizerJobConfig

logger = logging.getLogger("safe_synthesizer")

EMBEDDED_RUN_CONFIG_NAME = "safe-synthesizer-config.json"


def is_peft_adapter_directory(path: Path) -> bool:
    """Return whether ``path`` looks like a PEFT LoRA adapter directory."""
    return path.is_dir() and (path / "adapter_config.json").is_file()


def embed_run_config_in_adapter(adapter_dir: Path) -> None:
    """Copy the training run config into an adapter directory for Files reuse."""
    if not is_peft_adapter_directory(adapter_dir):
        return
    sibling_config = adapter_dir.parent / EMBEDDED_RUN_CONFIG_NAME
    embedded_config = adapter_dir / EMBEDDED_RUN_CONFIG_NAME
    if sibling_config.is_file() and not embedded_config.exists():
        shutil.copy2(sibling_config, embedded_config)


def resolve_pretrained_model_path(pretrained_model: str) -> Path:
    """Resolve a configured path relative to the current working directory."""
    path = Path(pretrained_model).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path.resolve()


def is_adapter_reuse_requested(job_config: SafeSynthesizerJobConfig) -> bool:
    """Return whether the job spec requests generation-only reuse of a prior adapter."""
    if job_config.pretrained_model_job:
        return True

    pretrained_model = job_config.config.training.pretrained_model
    if not pretrained_model or pretrained_model == DEFAULT_PRETRAINED_MODEL:
        return False

    path = resolve_pretrained_model_path(pretrained_model)
    if not path.exists():
        return False
    if is_peft_adapter_directory(path):
        return True

    try:
        Workdir.from_path(path)
    except ValueError:
        return False
    return True


def resolve_prior_training_workdir(adapter_location: str | Path) -> Workdir:
    """Locate or materialize the prior NSS training ``Workdir`` for an adapter."""
    path = resolve_pretrained_model_path(str(adapter_location))

    if is_peft_adapter_directory(path):
        train_dir = path.parent
        if train_dir.name == "train" and (train_dir.parent / "dataset").is_dir():
            return Workdir.from_path(train_dir.parent)
        work_base = path.parent / "work"
        if work_base.is_dir():
            return Workdir.from_path(work_base)
        return _materialize_workdir_from_adapter(path)

    return Workdir.from_path(path)


def _materialize_workdir_from_adapter(adapter_dir: Path) -> Workdir:
    """Build an NSS run layout from a standalone adapter directory (for example from Files)."""
    if not is_peft_adapter_directory(adapter_dir):
        raise ValueError(f"Not a PEFT adapter directory: {adapter_dir}")

    metadata_path = adapter_dir / "metadata_v2.json"
    if not metadata_path.is_file():
        raise ValueError(
            f"Adapter at {adapter_dir} is missing metadata_v2.json, which is required for generation-only reuse."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    workdir_info = metadata.get("workdir") if isinstance(metadata.get("workdir"), dict) else {}
    config_name = workdir_info.get("config_name", "default")
    dataset_name = workdir_info.get("dataset_name", "data")
    run_name = workdir_info.get("run_name", "adapter-reuse")

    if adapter_dir.name == "adapter" and (adapter_dir.parent / "train").is_dir():
        return Workdir.from_path(adapter_dir.parent.parent)

    run_root = adapter_dir.parent / f"{config_name}---{dataset_name}" / run_name
    train_adapter = run_root / "train" / "adapter"
    if train_adapter.resolve() != adapter_dir.resolve():
        if train_adapter.exists():
            shutil.rmtree(train_adapter)
        train_adapter.parent.mkdir(parents=True, exist_ok=True)
        if adapter_dir.resolve().is_relative_to(run_root.resolve()):
            shutil.copytree(adapter_dir, train_adapter)
        else:
            shutil.copytree(adapter_dir, train_adapter)

    embedded_config = train_adapter / EMBEDDED_RUN_CONFIG_NAME
    train_config = run_root / "train" / EMBEDDED_RUN_CONFIG_NAME
    if embedded_config.is_file() and not train_config.is_file():
        shutil.copy2(embedded_config, train_config)

    if not train_config.is_file():
        raise ValueError(
            f"Adapter at {adapter_dir} is missing {EMBEDDED_RUN_CONFIG_NAME}. "
            "Re-run training with a current NSS plugin build so the config is embedded in the adapter artifact."
        )

    return Workdir.from_path(run_root)


def create_generation_workdir(parent_workdir: Workdir, save_path: Path) -> Workdir:
    """Create a child generation workdir that reads adapter/config/data from ``parent_workdir``."""
    return Workdir(
        base_path=save_path,
        config_name=parent_workdir.config_name,
        dataset_name=parent_workdir.dataset_name,
        run_name=RunName().to_string(),
        _current_phase="generate",
        _parent_workdir=parent_workdir,
    )


def _apply_generation_overrides(ss: SafeSynthesizer, job_config: SafeSynthesizerJobConfig) -> None:
    """Apply generation and evaluation overrides from the job spec onto a resumed run."""
    if ss._nss_config is None:
        raise RuntimeError("SafeSynthesizer config was not loaded from the prior run")
    ss._nss_config.generation = job_config.config.generation
    ss._nss_config.evaluation = job_config.config.evaluation


def _write_cached_training_split(parent_workdir: Workdir, data_source: pd.DataFrame) -> None:
    """Persist a training split when the reused adapter bundle does not include one."""
    training_path = parent_workdir.dataset.training
    if training_path.exists():
        return
    parent_workdir.dataset.path.mkdir(parents=True, exist_ok=True)
    data_source.to_csv(training_path, index=False)


def run_generation_from_prior_adapter(
    job_config: SafeSynthesizerJobConfig,
    data_source: pd.DataFrame | None,
    save_path: Path,
    *,
    adapter_location: str | None = None,
) -> tuple[SafeSynthesizerResults, Path | None]:
    """Generate synthetic data from a prior adapter without retraining."""
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    if adapter_location is None:
        if not job_config.config.training.pretrained_model:
            raise ValueError("Adapter reuse requires pretrained_model_job or config.training.pretrained_model.")
        adapter_location = job_config.config.training.pretrained_model

    parent_workdir = resolve_prior_training_workdir(adapter_location)
    if data_source is not None:
        _write_cached_training_split(parent_workdir, data_source)

    gen_workdir = create_generation_workdir(parent_workdir, save_path)
    gen_workdir.ensure_directories()

    logger.info(
        "Generation-only adapter reuse: loading adapter from %s into run %s",
        parent_workdir.adapter_path,
        gen_workdir.run_dir,
    )

    ss = SafeSynthesizer(config=job_config.config, workdir=gen_workdir)
    if data_source is not None:
        ss = ss.with_data_source(data_source)
    ss.load_from_save_path()
    _apply_generation_overrides(ss, job_config)
    ss.generate().evaluate()
    ss.save_results()
    return ss.results, parent_workdir.adapter_path
