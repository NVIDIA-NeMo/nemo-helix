# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge a downloaded LoRA adapter into its downloaded base model (``python -m nhx.automodel.tasks.merge_adapter``)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from nemo_helix_plugin.jobs.constants import NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR
from nhx.automodel.app.constants import DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_MODEL_PATH
from nhx.automodel.tasks.training.backends.registry import uses_custom_automodel_impl
from nhx.automodel.tasks.training.chat_templates import apply_chat_template_to_checkpoint
from nhx.common.jobs.constants import DEFAULT_JOB_STORAGE_PATH
from nhx.customization_common.training.lora import merge_lora_adapter
from pydantic import BaseModel, ConfigDict

ADAPTER_PATH = f"{DEFAULT_JOB_STORAGE_PATH}/adapter"


class MergeAdapterStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trust_remote_code: bool = False
    has_vision_tower: bool = False
    override_custom_impl: bool = False
    chat_template: str | None = None


def uses_builtin_model_class(base_model_path: Path, *, override_custom_impl: bool) -> bool:
    """Whether the adapter was trained on transformers' class for this architecture (Automodel's own
    implementations save adapters under its module names), as training decides it."""
    try:
        architectures = json.loads((base_model_path / "config.json").read_text()).get("architectures")
    except (OSError, ValueError):
        return False
    return not override_custom_impl and bool(architectures) and uses_custom_automodel_impl(architectures[0])


def run_merge(
    config: MergeAdapterStepConfig,
    *,
    base: Path = Path(DEFAULT_MODEL_PATH),
    adapter: Path = Path(ADAPTER_PATH),
    output: Path = Path(DEFAULT_OUTPUT_MODEL_PATH),
    merge=merge_lora_adapter,
) -> None:
    """*merge* is ``merge_lora_adapter`` in production; a test passes a stand-in that needs no torch."""
    merge(
        adapter_path=adapter,
        base_model_path=str(base),
        output_path=output,
        trust_remote_code=config.trust_remote_code,
        has_vision_tower=config.has_vision_tower,
        builtin_model_class=uses_builtin_model_class(base, override_custom_impl=config.override_custom_impl),
    )
    if config.chat_template:
        apply_chat_template_to_checkpoint(output, config.chat_template)


if __name__ == "__main__":
    with open(os.environ[NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR], encoding="utf-8") as handle:
        run_merge(MergeAdapterStepConfig.model_validate_json(handle.read()))
