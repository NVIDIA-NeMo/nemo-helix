# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async helpers for resolving model/dataset references against the platform.

Used by each plugin's ``transform.py`` (async, runs inside the FastAPI request
handler / ``to_spec`` flow) to validate that the submitter's ``model`` and
``dataset`` references exist before the job moves on to compile / run.
"""

from dataclasses import dataclass

from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.files.types import FilesetPurpose
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_platform_plugin.models.types import ModelEntity
from nmp.common.entities.utils import parse_entity_ref
from nmp.customization_common.schemas.file_io import FileSetRef


@dataclass(frozen=True, slots=True)
class AsyncCustomizationPlatformClients:
    """Typed async service clients needed while compiling customization jobs."""

    files: AsyncFilesClient
    models: AsyncModelsClient


def async_customization_platform_clients_from_platform(
    platform: AsyncNeMoPlatform,
) -> AsyncCustomizationPlatformClients:
    """Build the customization compile-time client bundle from the generated SDK."""
    return AsyncCustomizationPlatformClients(
        files=client_from_platform(platform, AsyncFilesClient),
        models=client_from_platform(platform, AsyncModelsClient),
    )


async def check_fileset_access(
    platform: AsyncCustomizationPlatformClients,
    fileset_uri: str,
    default_workspace: str,
    *,
    label: str = "fileset",
):
    """Verify the caller can access a fileset reference and return the fileset."""
    ref = FileSetRef.model_validate(fileset_uri)
    workspace = ref.workspace or default_workspace
    try:
        return await platform.files.get_fileset(workspace=workspace, name=ref.name)
    except PermissionDeniedError:
        raise PermissionError(f"Access denied to {label} fileset '{workspace}/{ref.name}'") from None
    except NotFoundError:
        raise ValueError(
            f"{label.capitalize()} fileset '{ref.name}' not found in workspace '{workspace}'. "
            "Verify the fileset exists."
        ) from None


async def check_dataset_access(
    platform: AsyncCustomizationPlatformClients,
    dataset_uri: str,
    default_workspace: str,
) -> None:
    """Verify the caller can access the dataset fileset.

    Raises:
        ValueError: If the fileset is not found.
        PermissionError: If access is denied.
    """
    await check_fileset_access(platform, dataset_uri, default_workspace, label="dataset")


async def check_environment_access(
    platform: AsyncCustomizationPlatformClients,
    environment_uri: str,
    default_workspace: str,
) -> None:
    """Verify the caller can access the environment fileset (GRPO)."""
    ref = FileSetRef.model_validate(environment_uri)
    workspace = ref.workspace or default_workspace
    response = await check_fileset_access(platform, environment_uri, default_workspace, label="environment")

    fs = response.data() if hasattr(response, "data") and callable(response.data) else response
    purpose = getattr(fs, "purpose", None)
    purpose_val = getattr(purpose, "value", purpose)
    if purpose_val is not None and purpose_val not in (
        FilesetPurpose.ENVIRONMENT.value,
        FilesetPurpose.GENERIC.value,
    ):
        raise ValueError(
            f"Environment fileset '{workspace}/{ref.name}' has purpose {purpose_val!r}; expected purpose='environment'."
        )


async def check_gym_dataset_layout(
    platform: AsyncCustomizationPlatformClients,
    dataset_uri: str,
    default_workspace: str,
) -> None:
    """Ensure a GRPO Gym dataset fileset contains training.jsonl."""
    ref = FileSetRef.model_validate(dataset_uri)
    workspace = ref.workspace or default_workspace
    try:
        listing = (await platform.files.list_files(workspace=workspace, name=ref.name)).data()
    except PermissionDeniedError:
        raise PermissionError(f"Access denied to dataset fileset '{workspace}/{ref.name}'") from None
    except NotFoundError:
        raise ValueError(
            f"Dataset fileset '{ref.name}' not found in workspace '{workspace}'. Verify the dataset exists."
        ) from None

    paths = {item.path for item in listing.data}
    if "training.jsonl" not in paths:
        raise ValueError(
            f"GRPO dataset fileset '{workspace}/{ref.name}' must contain training.jsonl "
            "(Gym JSONL rows, not DPO preference triples)."
        )


async def fetch_model_entity(
    model_ref: str,
    default_workspace: str,
    platform: AsyncCustomizationPlatformClients,
) -> ModelEntity:
    """Retrieve a model entity and verify its weights fileset is accessible."""
    resolved_ref = parse_entity_ref(model_ref, default_workspace)
    try:
        response = await platform.models.get_model(
            name=resolved_ref.name,
            workspace=resolved_ref.workspace,
            query_params={"verbose": True},
        )
        model = response.data()
    except PermissionDeniedError:
        raise PermissionError(f"Access denied to model '{resolved_ref.workspace}/{resolved_ref.name}'") from None
    except NotFoundError:
        raise ValueError(
            f"Model entity not found: '{resolved_ref.workspace}/{resolved_ref.name}'. Verify the model entity exists."
        ) from None

    if model.fileset:
        await check_fileset_access(
            platform,
            model.fileset,
            resolved_ref.workspace,
            label=f"weights for model '{resolved_ref.workspace}/{resolved_ref.name}'",
        )
    return model
