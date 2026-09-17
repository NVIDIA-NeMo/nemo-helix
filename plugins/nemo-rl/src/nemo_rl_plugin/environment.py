# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit-time validation of the environment FileSet a GRPO job points at.

The job also validates the package after downloading it, but that happens inside the
training container: a malformed manifest surfaces minutes in, after the model download
and Ray startup. Everything decidable from the FileSet listing plus the manifest is
checked here instead, so the submitter gets it back from the API call.
"""

from __future__ import annotations

from nemo_platform_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_platform_plugin.client.errors import PermissionDeniedError as ClientPermissionDeniedError
from nmp.customization_common.schemas.file_io import FileSetRef
from nmp.customization_common.service.platform_client import AsyncCustomizationPlatformClients
from nmp.rl.tasks.environment.validate import (
    MANIFEST_FILENAME,
    EnvironmentPackageValidationError,
    parse_manifest,
    validate_manifest_against_listing,
)


async def check_environment_package(
    platform: AsyncCustomizationPlatformClients,
    environment_uri: str,
    default_workspace: str,
) -> None:
    """Validate the environment FileSet's manifest and layout.

    Raises:
        ValueError: When the fileset is missing, has no manifest, or the manifest does not
            describe the package that was uploaded.
        PermissionError: When access to the fileset is denied.
    """
    ref = FileSetRef.model_validate(environment_uri)
    workspace = ref.workspace or default_workspace

    try:
        listing = (await platform.files.list_files(workspace=workspace, name=ref.name)).data()
    except ClientPermissionDeniedError:
        raise PermissionError(f"Access denied to environment fileset '{workspace}/{ref.name}'") from None
    except ClientNotFoundError:
        raise ValueError(
            f"Environment fileset '{ref.name}' not found in workspace '{workspace}'. Verify the environment exists."
        ) from None

    paths = {item.path for item in listing.data}
    if MANIFEST_FILENAME not in paths:
        raise ValueError(
            f"Environment fileset '{workspace}/{ref.name}' has no {MANIFEST_FILENAME} at its root. "
            "An environment package declares its format, config_paths and metadata there."
        )

    try:
        raw = await (
            await platform.files.download_file(workspace=workspace, name=ref.name, path=MANIFEST_FILENAME)
        ).read()
    except ClientPermissionDeniedError:
        raise PermissionError(f"Access denied to environment fileset '{workspace}/{ref.name}'") from None

    try:
        manifest = parse_manifest(raw)
        validate_manifest_against_listing(manifest, paths)
    except EnvironmentPackageValidationError as exc:
        raise ValueError(f"Environment fileset '{workspace}/{ref.name}' is not a valid package: {exc}") from exc
