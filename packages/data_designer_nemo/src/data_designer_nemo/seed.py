# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import data_designer.config as dd
from data_designer.config.seed_source import SeedSource
from data_designer_nemo.errors import NDDInternalError, NDDInvalidConfigError
from data_designer_nemo.fileset_file_seed_source import FilesetFileSeedSource
from data_designer_nemo.secret_resolver import validate_secret
from filesets import FilesetPathError, build_fileset_ref, parse_fileset_ref
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import ListFilesQueryParams

logger = logging.getLogger(__name__)

_SUPPORTED_SEED_TYPES = {"directory", "file_contents", "hf", "nhx"}
_UNSUPPORTED_SEED_TYPES_MESSAGE = (
    "The NeMo Helix Data Designer service only supports seed data from HuggingFace "
    "or the NeMo Helix Files service (FilesetFile, Directory, or FileContents seed sources "
    "referencing fileset paths). Upload your data to the Files service, adjust your config, and try again."
)
_DATAFRAME_SEED_TYPE = "df"
LOCAL_DATAFRAME_SEED_ERROR_MESSAGE = (
    "Dataframe seed sources are not supported on the NeMo Helix. "
    "Save your data to a file or directory and update your config before trying again. "
    "If you intend to run this same workload remotely, upload the file or directory to the Files service."
)


async def validate_seed(
    dd_config: dd.DataDesignerConfig,
    workspace: str,
    sdk: AsyncHelixClient,
) -> str | None:
    if (seed_source := _get_seed_source(dd_config)) is None:
        return None

    _validate_seed_type_for_execution_context(seed_source.seed_type)

    if isinstance(seed_source, dd.HuggingFaceSeedSource):
        # In local execution context, a HF seed source token will always "resolve"
        # because the composite secret resolver includes a plaintext resolver.
        # In remote execution context, a HF seed source token must be a reference
        # to a Nemo Helix secret (if provided).
        if (token := seed_source.token) is not None:
            await validate_secret(sdk, token, workspace)
        return None

    if isinstance(seed_source, FilesetFileSeedSource | dd.DirectorySeedSource | dd.FileContentsSeedSource):
        return await _validate_seed_from_files_service(seed_source, workspace, sdk)


async def _validate_seed_from_files_service(
    seed_source: FilesetFileSeedSource | dd.DirectorySeedSource | dd.FileContentsSeedSource,
    workspace: str,
    sdk: AsyncHelixClient,
) -> str | None:
    try:
        workspace, fileset_name, fragment = parse_fileset_ref(seed_source.path, workspace_fallback=workspace)
    except FilesetPathError as e:
        raise NDDInvalidConfigError(
            f"The fileset reference in seed source path {seed_source.path!r} is formatted incorrectly"
        ) from e

    files = client_from_platform(sdk, AsyncFilesClient)
    try:
        await files.get_fileset(name=fileset_name, workspace=workspace)
    except NotFoundError as e:
        raise NDDInvalidConfigError(f"Could not find fileset {fileset_name!r} in workspace {workspace!r}") from e
    except PermissionDeniedError as e:
        raise NDDInvalidConfigError(f"Access denied to workspace {workspace!r}") from e
    except Exception as e:
        logger.exception("Error retrieving fileset", extra={"fileset_name": fileset_name, "workspace": workspace})
        raise NDDInternalError(
            f"An unexpected error occurred while retrieving fileset {fileset_name!r} in workspace {workspace!r}: {e}"
        ) from e

    canonical_root = build_fileset_ref(fragment, workspace=workspace, fileset=fileset_name)

    fully_qualified_fileset_name = f"{workspace}/{fileset_name}"
    query_params = ListFilesQueryParams(path=fragment) if fragment else None
    try:
        response = await files.list_files(
            workspace=workspace,
            name=fileset_name,
            query_params=query_params,
        )
    except NotFoundError as e:
        raise NDDInvalidConfigError(f"Path {fragment!r} not found in fileset {fully_qualified_fileset_name!r}") from e
    except PermissionDeniedError as e:
        raise NDDInvalidConfigError(f"Access denied to workspace {workspace!r}") from e
    except Exception as e:
        logger.exception(
            "Error listing fileset path",
            extra={"fileset_name": fileset_name, "workspace": workspace, "fragment": fragment},
        )
        raise NDDInternalError(
            f"An unexpected error occurred while listing path {fragment!r} in fileset {fully_qualified_fileset_name!r}: {e}"
        ) from e

    files_response = response.data()

    # A path that resolves to no files provides nothing to seed from, so it is
    # invalid regardless of whether the path itself exists. A FilesetFileSeedSource
    # points at a single file, while the directory-style sources enumerate files
    # under a directory; tailor the message accordingly.
    if not files_response.data:
        if isinstance(seed_source, FilesetFileSeedSource):
            # FilesetFileSeedSource already validates that fragment is present
            raise NDDInvalidConfigError(f"File {fragment!r} not found in fileset {fully_qualified_fileset_name!r}")
        raise NDDInvalidConfigError(_no_files_error_message(fully_qualified_fileset_name, fragment))

    return canonical_root


def _no_files_error_message(fileset: str, fragment: str | None) -> str:
    msg = f"Fileset {fileset!r} contains no files to use as seed data"
    if fragment:
        msg += f" under path {fragment!r}"

    return msg


def validate_seed_source_for_execution_context(data: Any) -> None:
    """Raises if a raw request seed source is unsupported for the execution context.

    This function is used in Pydantic validators defined on the preview and job request models,
    both of which carry a `config: dd.DataDesignerConfig` field.

    This function is used in "before"-style Pydantic validators, where the data argument is typed
    as Any. We run in the before context to preempt less-useful error messages from the DD library:
    - missing dataframe field (we don't serialize dataframes over the wire)
    - file does not exist (the client's local fs != the service's local fs)

    The validators using this function only care about preventing unsupported seed types. All the
    other standard Pydantic validation will get applied by FastAPI parsing the request; this does
    not bypass that. So, we can safely ignore all Exceptions (most commonly KeyError, on requests
    that don't include a seed_config at all) and index our way straight to the deeply nested field
    we care about for this particular validation.

    Per the Pydantic v2 contract, "before"-mode validators may raise ``ValueError``,
    ``AssertionError``, or ``PydanticCustomError`` — anything else (including our
    ``NDDInvalidConfigError``) propagates raw out of ``model_validate`` and is not wrapped in
    ``pydantic.ValidationError``. That breaks ``except ValidationError`` clauses in CLI / framework
    code that turn validation problems into clean user-facing messages. To keep those code paths
    working *and* keep ``NDDInvalidConfigError`` as the canonical error class for non-Pydantic
    callers, we translate at this boundary: catch the plugin's error class and re-raise as a
    ``ValueError`` carrying the same message.
    """
    seed_type = _get_raw_seed_type(data)
    if seed_type is None:
        return

    try:
        _validate_seed_type_for_execution_context(seed_type)
    except NDDInvalidConfigError as exc:
        raise ValueError(str(exc)) from exc


def _validate_seed_type_for_execution_context(seed_type: str) -> None:
    """Raises if a seed source type is unsupported in this execution context."""
    if seed_type not in _SUPPORTED_SEED_TYPES:
        raise NDDInvalidConfigError(_UNSUPPORTED_SEED_TYPES_MESSAGE)


def _get_seed_source(dd_config: dd.DataDesignerConfig) -> SeedSource | None:
    return dd_config.seed_config.source if dd_config.seed_config else None


def _get_raw_seed_type(data: Any) -> str | None:
    try:
        seed_type = data["config"]["seed_config"]["source"]["seed_type"]
    except Exception:
        return None

    return seed_type if isinstance(seed_type, str) else None
