# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async helpers for resolving model/dataset references against the platform.

Used by each plugin's ``transform.py`` (async, runs inside the FastAPI request
handler / ``to_spec`` flow) to validate that the submitter's ``model`` and
``dataset`` references exist before the job moves on to compile / run.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import NemoClientError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import FilesetPurpose
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.types import ModelEntity
from nhx.common.entities.utils import parse_entity_ref
from nhx.customization_common.schemas.file_io import FileSetRef

#: ``source`` of every customization job (automodel, unsloth and rl) in the Jobs service.
CUSTOMIZATION_JOB_SOURCE = "customization"


@dataclass(frozen=True, slots=True)
class AsyncCustomizationHelixClients:
    """Typed async service clients needed while compiling customization jobs."""

    files: AsyncFilesClient
    models: AsyncModelsClient
    jobs: AsyncJobsClient


def async_customization_platform_clients_from_platform(
    platform: AsyncHelixClient,
) -> AsyncCustomizationHelixClients:
    """Build the customization compile-time client bundle from an async platform handle."""
    return AsyncCustomizationHelixClients(
        files=client_from_platform(platform, AsyncFilesClient),
        models=client_from_platform(platform, AsyncModelsClient),
        jobs=client_from_platform(platform, AsyncJobsClient),
    )


async def check_fileset_access(
    platform: AsyncCustomizationHelixClients,
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
    platform: AsyncCustomizationHelixClients,
    dataset_uri: str,
    default_workspace: str,
) -> None:
    """Verify the caller can access the dataset fileset and optional directory path.

    Raises:
        ValueError: If the fileset is not found.
        PermissionError: If access is denied.
    """
    await check_fileset_access(platform, dataset_uri, default_workspace, label="dataset")
    ref = FileSetRef.model_validate(dataset_uri)
    if ref.path is None:
        return

    workspace = ref.workspace or default_workspace
    try:
        listing = (await platform.files.list_files(workspace=workspace, name=ref.name)).data()
    except PermissionDeniedError:
        raise PermissionError(f"Access denied to dataset fileset '{workspace}/{ref.name}'") from None
    except NotFoundError:
        raise ValueError(
            f"Dataset fileset '{ref.name}' not found in workspace '{workspace}'. Verify the fileset exists."
        ) from None

    if not any(item.path.lstrip("/").startswith(ref.path) for item in listing.data):
        raise ValueError(f"Dataset path '{ref.path}' not found in fileset '{workspace}/{ref.name}'.")


async def check_environment_access(
    platform: AsyncCustomizationHelixClients,
    environment_uri: str,
    default_workspace: str,
) -> None:
    """Verify the caller can access the environment fileset (GRPO)."""
    ref = FileSetRef.model_validate(environment_uri)
    if ref.path is not None:
        raise ValueError("Environment fileset references must not include a '#path/' directory.")
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
    platform: AsyncCustomizationHelixClients,
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

    paths = {item.path.lstrip("/") for item in listing.data}
    expected_path = f"{ref.path or ''}training.jsonl"
    if expected_path not in paths:
        raise ValueError(
            f"GRPO dataset fileset '{workspace}/{ref.name}' must contain {expected_path} "
            "(Gym JSONL rows, not DPO preference triples)."
        )


async def fetch_model_entity(
    model_ref: str,
    default_workspace: str,
    platform: AsyncCustomizationHelixClients,
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


async def validate_adapter_base_model(
    adapter_name: str,
    base_model_ref: str,
    workspace: str,
    platform: AsyncCustomizationHelixClients,
) -> None:
    """Reject a LoRA output name that is already an adapter of a different base model.

    Adapter names are unique per workspace, so retraining an existing adapter name under a
    different base model conflicts on create and cannot be updated. Without this check the
    job trains to completion first and only fails in the model-entity step. Callers decide
    whether the job trains a LoRA adapter; this only compares base models.
    """
    try:
        existing = (await platform.models.get_adapter(name=adapter_name, workspace=workspace)).data()
    except NotFoundError:
        return

    base = parse_entity_ref(base_model_ref, workspace)
    expected = f"{base.workspace}/{base.name}"
    if existing.model is not None and existing.model != expected:
        raise HelixJobCompilationError(
            f"Adapter '{workspace}/{adapter_name}' already exists on base model '{existing.model}', "
            f"but this job trains against '{expected}'. Adapter names are unique per workspace, so "
            "the existing adapter cannot be re-parented. Choose a different output.name, or train "
            f"against '{existing.model}'."
        )


async def validate_output_name_not_in_flight(
    output_name: str,
    workspace: str,
    platform: AsyncCustomizationHelixClients,
) -> None:
    """Reject an output name that a pending or running customization job will also write.

    Every job uploads into a fileset named after its output and then registers the
    model or adapter under that name. Two jobs sharing a name therefore mix their
    checkpoints in one fileset, and the later one either overwrites the earlier
    entity or, for an adapter on a different base model, fails at registration
    after training has finished. Checking entities that already exist
    (``validate_adapter_base_model``) cannot see a job that has not registered yet.

    The new job is not stored until it compiles, so it is never among the results.
    Two submissions landing at the same instant can still both pass; this closes the
    hours-long window a training run leaves open, not that one.
    """
    in_flight = {
        "source": CUSTOMIZATION_JOB_SOURCE,
        "status": {"$in": [status.value for status in HelixJobStatus.non_terminals()]},
        "spec.output.name": output_name,
    }
    try:
        conflicting = [
            job
            async for job in platform.jobs.list(workspace=workspace, filter=in_flight, page_size=100)
            if _output_name(job.spec) == output_name
        ]
    except NemoClientError as exc:
        # Fail closed: the job would be created through the same Jobs service a moment later.
        raise HelixJobCompilationError(
            f"Could not check for running jobs that write output '{workspace}/{output_name}': {exc}"
        ) from exc

    if conflicting:
        job = conflicting[0]
        raise HelixJobCompilationError(
            f"Job '{job.name}' ({job.status.value}) is already producing output '{workspace}/{output_name}'. "
            "Wait for it to finish, or choose a different output.name."
        )


def _output_name(spec: Mapping[str, Any]) -> str | None:
    output = spec.get("output")
    return output.get("name") if isinstance(output, Mapping) else None
