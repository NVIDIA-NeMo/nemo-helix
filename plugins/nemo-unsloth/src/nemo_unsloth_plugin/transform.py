# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input → canonical spec transformation.

Mirrors the automodel pattern: validates the platform refs (model
entity + dataset fileset) against the live SDK, then resolves output
naming and the fileset name. :meth:`~nemo_unsloth_plugin.jobs.jobs.UnslothJob.compile`
turns the canonical spec into a 4-step container job that performs
download → train → upload → model_entity on the platform cluster.

Only platform refs are accepted today (per the strict-refs design
choice). Bare HF ids and arbitrary local paths are rejected before
submit because the container pipeline expects a real fileset to
download from.
"""

from __future__ import annotations

from typing import Literal

from nemo_platform_plugin.models.types import ModelSpec
from nmp.customization_common.contributor.transform import generated_output_name
from nmp.customization_common.service.platform_client import (
    AsyncCustomizationPlatformClients,
    check_dataset_access,
    fetch_model_entity,
)
from nmp.unsloth.schemas import OutputResponse, UnslothJobOutput

from nemo_unsloth_plugin.schema import OutputRequest, UnslothJobInput


def _infer_output_type(output_request: OutputRequest) -> Literal["adapter", "model"]:
    """Adapter when saving the LoRA, model otherwise (merged or full)."""
    if output_request.save_method == "lora":
        return "adapter"
    return "model"


def _is_encoder_model(model_spec: ModelSpec) -> bool:
    return model_spec.head_type in {"embedding", "cross_encoder"} or model_spec.is_embedding_model


async def transform_input_to_output(
    input_spec: UnslothJobInput,
    workspace: str,
    platform: AsyncCustomizationPlatformClients,
) -> UnslothJobOutput:
    """Enrich submitter input into a canonical :class:`UnslothJobOutput`.

    Args:
        input_spec: Submitter-facing input shape.
        workspace: The job's workspace; used as the default for any bare
            entity / fileset refs.
        platform: Typed async clients for validating platform references.

    Returns:
        Canonical :class:`UnslothJobOutput` with ``output.fileset``
        populated.

    Raises:
        ValueError: When the model entity or dataset fileset cannot be
            resolved.
        PermissionError: When access to the model or dataset is denied.
    """
    # Strict refs: both calls error if the entity / fileset is missing.
    model_entity = await fetch_model_entity(input_spec.model.name, workspace, platform)
    await check_dataset_access(platform, input_spec.dataset.path, workspace)
    if input_spec.dataset.validation_path:
        await check_dataset_access(platform, input_spec.dataset.validation_path, workspace)

    model_spec = model_entity.spec
    if model_spec is not None and _is_encoder_model(model_spec):
        raise ValueError(
            "Encoder-model SFT is not supported by the unsloth backend. Use a causal LM model entity instead.",
        )

    output_request = input_spec.output or OutputRequest()
    if output_request.name is None:
        out_name = generated_output_name(input_spec.model.name, input_spec.dataset.path, workspace)
    else:
        out_name = output_request.name

    output = OutputResponse(
        name=out_name,
        type=_infer_output_type(output_request),
        save_method=output_request.save_method,
        fileset=out_name,  # default the fileset to the entity name (mirrors automodel)
        description=output_request.description,
    )

    return UnslothJobOutput(
        name=input_spec.name,
        model=input_spec.model,
        dataset=input_spec.dataset,
        training=input_spec.training,
        schedule=input_spec.schedule,
        batch=input_spec.batch,
        optimizer=input_spec.optimizer,
        hardware=input_spec.hardware,
        integrations=input_spec.integrations,
        output=output,
        deployment_config=input_spec.deployment_config,
    )
