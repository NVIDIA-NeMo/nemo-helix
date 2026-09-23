# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification of previously uploaded Harbor definitions."""

from nemo_evaluator.api.task_definitions.harbor import HarborTaskDefinition
from nemo_helix_plugin.files.client import AsyncFilesClient


async def verified_harbor_definition(
    spec: HarborTaskDefinition, *, files_client: AsyncFilesClient
) -> HarborTaskDefinition:
    """Return the archive's verified projection for synchronous pinned publication.

    Args:
        spec: Definition identifying the archive; metrics are preserved, not normalized.
        files_client: Authenticated asynchronous Files client used to download the archive.

    Returns:
        A copy with verified instruction and config, including a nullable multistep instruction.

    Raises:
        ValueError: The archive is invalid or its native identity differs from the definition.
    """
    from nemo_evaluator.harbor.materialization import verify_definition

    native = await verify_definition(spec, files_client)
    if spec.native_task_id != native.task_id:
        raise ValueError("Harbor native_task_id does not match the verified archive")
    return spec.model_copy(update={"instruction": native.instruction, "config": native.config})
