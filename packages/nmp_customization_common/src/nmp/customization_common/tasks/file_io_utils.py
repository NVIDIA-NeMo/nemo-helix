# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fileset path/IO + error-handling helpers for the file_io task."""

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from nemo_platform_plugin.client.errors import (
    AuthenticationError,
    NemoHTTPError,
    NemoTransportError,
    PermissionDeniedError,
)
from nmp.customization_common.schemas.file_io import (
    FileDownloadError,
    FileIOTaskConfig,
    FileUploadError,
    PathTraversalError,
    ProgressReportError,
)
from nmp.customization_common.service.path_utils import remap_job_storage_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalFileInfo:
    """A local file entry for upload/download progress accounting."""

    path: str
    size: int


def list_local_files(src_path: Path) -> list[LocalFileInfo]:
    """List files under *src_path* for progress totals.

    Skips individual unreadable or transient paths (e.g. W&B log files deleted
    mid-walk) instead of failing the entire listing.
    """
    if not src_path.exists():
        logger.warning(f"Failed to list local files. Source path does not exist: {src_path}")
        return []

    def _on_walk_error(err: OSError) -> None:
        logger.warning(
            f"Skipping inaccessible path during upload listing: {getattr(err, 'filename', src_path)}. Error: {err}"
        )

    try:
        if src_path.is_file():
            size = src_path.stat().st_size
            logger.info(f"Found 1 file: {src_path.name}")
            return [LocalFileInfo(path=src_path.name, size=size)]

        files: list[LocalFileInfo] = []
        for root, _, filenames in os.walk(src_path, onerror=_on_walk_error):
            for filename in filenames:
                full_path = Path(root) / filename
                try:
                    if not full_path.is_file():
                        continue
                    relative_path = full_path.relative_to(src_path)
                    files.append(
                        LocalFileInfo(path=str(relative_path), size=full_path.stat().st_size),
                    )
                except OSError as e:
                    logger.warning(f"Skipping unreadable file during upload listing: {full_path}. Error: {e}")
        logger.info(f"Found {len(files)} files in {src_path}")
        return files
    except Exception as e:
        logger.warning(f"Failed to list local files. Source path: {src_path}. Error: {e}")
        return []


@contextmanager
def filesystem_sdk_error_handler(
    error_class: type[FileDownloadError | FileUploadError | ProgressReportError],
    operation: str,
    passthrough: tuple[type[BaseException], ...] = (),
) -> Iterator[None]:
    """Context manager for consistent FilesetFileSystem error handling.

    Catches FilesetFileSystem-specific exceptions and re-raises them as the
    specified error class with a consistent message format.
    """
    try:
        yield
    except passthrough:
        raise
    except FileNotFoundError as e:
        raise error_class(f"Failed to {operation} due to file not found error. Error: {e}") from e
    except PermissionError as e:
        raise error_class(f"Failed to {operation} due to permission denied error. Error: {e}") from e
    except httpx.TimeoutException as e:
        raise error_class(f"Failed to {operation} due to request timeout. Error: {e}") from e
    except httpx.ConnectError as e:
        raise error_class(f"Failed to {operation} due to connection error. Error: {e}") from e
    except Exception as e:
        raise error_class(f"Failed to {operation} due to unexpected error {type(e).__name__}: {e}") from e


@contextmanager
def sdk_error_handler(
    error_class: type[FileDownloadError | FileUploadError | ProgressReportError],
    operation: str,
    passthrough: tuple[type[BaseException], ...] = (),
) -> Iterator[None]:
    """Context manager for consistent SDK error handling.

    Catches SDK-specific exceptions and re-raises them as the specified error
    class with a consistent message format.
    """
    try:
        yield
    except passthrough:
        raise
    except NemoTransportError as e:
        raise error_class(f"Failed to {operation} due to connection error. Cause: {e.error}. Error: {e}") from e
    # AuthenticationError / PermissionDeniedError are subclasses of NemoHTTPError,
    # so they must be caught before NemoHTTPError.
    except AuthenticationError as e:
        raise error_class(f"Failed to {operation} due to authentication error. Error: {e}") from e
    except PermissionDeniedError as e:
        raise error_class(f"Failed to {operation} due to permission denied error. Error: {e}") from e
    except NemoHTTPError as e:
        raise error_class(f"Failed to {operation} due to API error. Status code: {e.status_code}. Error: {e}") from e
    except Exception as e:
        raise error_class(f"Failed to {operation} due to unexpected error {type(e).__name__}: {e}") from e


def get_config(config_path: Path) -> FileIOTaskConfig:
    """Load and validate the file_io step config from disk."""
    with open(config_path) as f:
        return FileIOTaskConfig.model_validate(json.load(f))


def validate_storage_path(storage_path: Path) -> Path:
    """Validate that a storage path exists and is a directory."""
    if not storage_path.exists() or not storage_path.is_dir():
        raise FileUploadError(
            f"Storage path does not exist: {storage_path}. Ensure the storage path exists and is a directory.",
        )
    return storage_path


def validate_safe_path(base_path: Path, user_path: str) -> Path:
    """Validate that a user-provided path stays within the base directory.

    Resolves both paths to canonical absolute form and verifies the result
    is under the base path. Prevents path traversal via ``..`` etc.

    Raises:
        PathTraversalError: If the resolved path would escape base_path.
    """
    resolved_base = base_path.resolve()
    candidate_path = remap_job_storage_path(base_path, user_path)
    if not candidate_path.is_absolute():
        candidate_path = base_path / candidate_path
    resolved_path = candidate_path.resolve()

    if not resolved_path.is_relative_to(resolved_base):
        raise PathTraversalError(
            f"Path '{user_path}' resolves outside of the base directory. "
            "This may indicate a path traversal attack. "
            "Ensure that paths such as ../.. are not used in the download destination path.",
        )

    return resolved_path
