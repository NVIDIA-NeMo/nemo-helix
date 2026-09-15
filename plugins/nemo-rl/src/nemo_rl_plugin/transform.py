# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the platform refs (model entity + dataset/environment filesets)
against the live SDK, resolve output naming, and return the
canonical :class:`~nmp.rl.schemas.RlJobOutput`. Only platform refs are
accepted — the container pipeline expects a real fileset to download from.
"""

from __future__ import annotations

from nmp.customization_common.contributor.transform import generated_output_name
from nmp.customization_common.schemas.values import OutputNameType
from nmp.customization_common.service.platform_client import (
    AsyncCustomizationPlatformClients,
    check_dataset_access,
    check_environment_access,
    check_gym_dataset_layout,
    fetch_model_entity,
)
from nmp.rl.entities.values import TrainingType
from nmp.rl.schemas import GRPOTraining, OutputResponse, RlJobOutput

from nemo_rl_plugin.environment import check_environment_package
from nemo_rl_plugin.schema import OutputRequest, RlJobInput


def _infer_output_type(input_spec: RlJobInput) -> OutputNameType:
    if input_spec.training.type == "grpo" and isinstance(input_spec.training, GRPOTraining):
        if input_spec.training.finetuning_type == "lora":
            return OutputNameType.ADAPTER
    return OutputNameType.MODEL


async def transform_input_to_output(
    input_spec: RlJobInput,
    workspace: str,
    platform: AsyncCustomizationPlatformClients,
) -> RlJobOutput:
    """Enrich submitter input into a canonical :class:`RlJobOutput`.

    Raises:
        ValueError: When the model entity or dataset fileset cannot be resolved.
        PermissionError: When access to the model or dataset is denied.
    """
    training_type = TrainingType(input_spec.training.type)

    model_entity = await fetch_model_entity(input_spec.model, workspace, platform)
    await check_dataset_access(platform, input_spec.dataset, workspace)

    if training_type == TrainingType.GRPO:
        if not input_spec.environment:
            raise ValueError("GRPO jobs require an environment fileset reference.")
        await check_environment_access(platform, input_spec.environment, workspace)
        await check_environment_package(platform, input_spec.environment, workspace)
        await check_gym_dataset_layout(platform, input_spec.dataset, workspace)

    model_spec = model_entity.spec
    head_type = getattr(model_spec, "head_type", None) if model_spec else None
    is_encoder = head_type in {"embedding", "cross_encoder"} or bool(
        model_spec and getattr(model_spec, "is_embedding_model", False)
    )
    if is_encoder:
        raise ValueError(
            f"{training_type.value.upper()} is not supported for encoder models. Use a causal LM model entity instead.",
        )

    output_request = input_spec.output or OutputRequest()
    out_name = output_request.name or generated_output_name(input_spec.model, input_spec.dataset, workspace)

    output = OutputResponse(
        name=out_name,
        type=_infer_output_type(input_spec),
        fileset=out_name,
    )

    return RlJobOutput(
        name=input_spec.name,
        model=input_spec.model,
        dataset=input_spec.dataset,
        environment=input_spec.environment,
        training=input_spec.training,
        integrations=input_spec.integrations,
        output=output,
    )
