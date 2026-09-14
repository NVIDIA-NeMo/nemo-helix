# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-source handling for the Anonymizer plugin."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from anonymizer.config.anonymizer_config import AnonymizerInput
from anyio import to_thread
from filesets import AsyncFilesetFileSystem, FilesetFileSystem, FilesetPathError, build_fileset_ref, parse_fileset_ref
from nemo_anonymizer_plugin.app.errors import AnonymizerInvalidConfigError
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.jobs.file_manager import AsyncFilesetFileManager, FilesetFileManager, TmpDirPath
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

SourceKind = Literal["http", "fileset", "local"]
_SUPPORTED_FILE_SUFFIXES = {".csv", ".parquet"}


class AnonymizerInputSpec(BaseModel):
    """Plugin boundary input spec.

    The upstream ``AnonymizerInput`` validates source locations at model
    construction time. The plugin keeps this looser shape at the API boundary
    so HTTP(S) URLs and fileset refs can be accepted and materialized before
    the upstream model is constructed.
    """

    source: str = Field(description="HTTP(S) URL or fileset reference for a CSV/Parquet input file.")
    text_column: str = Field(default="text", min_length=1, description="Column containing text to anonymize.")
    id_column: str | None = Field(default=None, description="Optional column to use as record identifier.")
    data_summary: str | None = Field(default=None, description="Short description of the data.")


@dataclass
class PreparedAnonymizerInput:
    """Upstream input plus any temp-file cleanup needed after execution."""

    input: AnonymizerInput
    tmp_dir_path: TmpDirPath | None = None

    def cleanup(self) -> None:
        if self.tmp_dir_path is None:
            return
        _cleanup_tmp_dir_path(self.tmp_dir_path)


def classify_input_source(source: str) -> SourceKind:
    if _is_http_source(source):
        return "http"
    if is_fileset_source(source):
        return "fileset"
    _raise_for_unsupported_scheme(source)
    return "local"


def is_fileset_source(source: str) -> bool:
    if source.startswith("fileset://"):
        return True
    if _is_http_source(source):
        return False
    if "://" in source:
        return False
    if "#" not in source:
        return False
    if _looks_like_local_hash_path(source):
        return False
    return _looks_like_fileset_hash_ref(source)


def validate_anonymizer_input_source(
    data: AnonymizerInputSpec,
    *,
    workspace: str,
    allow_local_paths: bool,
) -> None:
    source_kind = classify_input_source(data.source)
    if source_kind == "local" and not allow_local_paths:
        raise AnonymizerInvalidConfigError(
            f"Input source {data.source!r} is a local path. Remote anonymizer execution only supports "
            "http(s) URLs or fileset references."
        )
    if source_kind == "fileset":
        _parse_fileset_input_ref(data.source, workspace=workspace)


async def prepare_anonymizer_input_async(
    data: AnonymizerInputSpec,
    *,
    sdk: AsyncNeMoPlatform | NeMoPlatform | None,
    workspace: str,
    allow_local_paths: bool,
) -> PreparedAnonymizerInput:
    validate_anonymizer_input_source(data, workspace=workspace, allow_local_paths=allow_local_paths)
    source_kind = classify_input_source(data.source)
    if source_kind == "fileset":
        if sdk is None:
            raise AnonymizerInvalidConfigError("Fileset input requires a NeMo Platform SDK.")
        if isinstance(sdk, NeMoPlatform):
            return await to_thread.run_sync(
                partial(
                    prepare_anonymizer_input,
                    data,
                    sdk=sdk,
                    workspace=workspace,
                    allow_local_paths=allow_local_paths,
                )
            )
        tmp_dir_path = await _download_fileset_input_async(data.source, sdk=sdk, workspace=workspace)
        return _make_prepared_fileset_input(data, tmp_dir_path)
    return PreparedAnonymizerInput(input=_make_upstream_input(data, source=data.source))


def prepare_anonymizer_input(
    data: AnonymizerInputSpec,
    *,
    sdk: NeMoPlatform | None,
    workspace: str,
    allow_local_paths: bool,
) -> PreparedAnonymizerInput:
    validate_anonymizer_input_source(data, workspace=workspace, allow_local_paths=allow_local_paths)
    source_kind = classify_input_source(data.source)
    if source_kind == "fileset":
        if sdk is None:
            raise AnonymizerInvalidConfigError("Fileset input requires a NeMo Platform SDK.")
        tmp_dir_path = _download_fileset_input(data.source, sdk=sdk, workspace=workspace)
        return _make_prepared_fileset_input(data, tmp_dir_path)
    return PreparedAnonymizerInput(input=_make_upstream_input(data, source=data.source))


async def _download_fileset_input_async(
    source: str,
    *,
    sdk: AsyncNeMoPlatform,
    workspace: str,
) -> TmpDirPath:
    workspace_name, fileset_name, file_path = _parse_fileset_input_ref(source, workspace=workspace)
    manager = AsyncFilesetFileManager(
        workspace=workspace_name,
        fileset_name=fileset_name,
        filesystem=AsyncFilesetFileSystem(client=client_from_platform(sdk, AsyncFilesClient)),
        ensure_fileset_exists=False,
    )
    return await _download_fileset_input_async_inner(manager, workspace_name, fileset_name, file_path)


async def _download_fileset_input_async_inner(
    manager: AsyncFilesetFileManager,
    workspace_name: str,
    fileset_name: str,
    file_path: str,
) -> TmpDirPath:
    fileset_ref = build_fileset_ref(file_path, workspace=workspace_name, fileset=fileset_name)
    try:
        tmp_dir_path = await manager.download_from_url(fileset_ref)
    except Exception as exc:
        raise AnonymizerInvalidConfigError(f"Failed to download fileset input {fileset_ref!r}: {exc}") from exc
    try:
        _validate_downloaded_input(tmp_dir_path.path, fileset_ref)
    except Exception:
        _cleanup_tmp_dir_path(tmp_dir_path)
        raise
    return tmp_dir_path


def _download_fileset_input(
    source: str,
    *,
    sdk: NeMoPlatform,
    workspace: str,
) -> TmpDirPath:
    workspace_name, fileset_name, file_path = _parse_fileset_input_ref(source, workspace=workspace)
    manager = FilesetFileManager(
        workspace=workspace_name,
        fileset_name=fileset_name,
        filesystem=FilesetFileSystem(client=client_from_platform(sdk, FilesClient)),
        ensure_fileset_exists=False,
    )
    fileset_ref = build_fileset_ref(file_path, workspace=workspace_name, fileset=fileset_name)
    try:
        tmp_dir_path = manager.download_from_url(fileset_ref)
    except Exception as exc:
        raise AnonymizerInvalidConfigError(f"Failed to download fileset input {fileset_ref!r}: {exc}") from exc
    try:
        _validate_downloaded_input(tmp_dir_path.path, fileset_ref)
    except Exception:
        _cleanup_tmp_dir_path(tmp_dir_path)
        raise
    return tmp_dir_path


def _parse_fileset_input_ref(source: str, *, workspace: str) -> tuple[str, str, str]:
    try:
        workspace_name, fileset_name, file_path = parse_fileset_ref(source, workspace_fallback=workspace)
    except FilesetPathError as exc:
        raise AnonymizerInvalidConfigError(f"Invalid fileset input source {source!r}: {exc}") from exc
    if not file_path:
        raise AnonymizerInvalidConfigError(
            f"Fileset input source {source!r} must point to a CSV or Parquet file using a #path fragment."
        )
    return workspace_name, fileset_name, file_path


def _validate_downloaded_input(path: Path, source: str) -> None:
    if not path.is_file():
        raise AnonymizerInvalidConfigError(f"Fileset input {source!r} resolved to a directory, not a file.")
    if path.suffix.lower() not in _SUPPORTED_FILE_SUFFIXES:
        raise AnonymizerInvalidConfigError(
            f"Fileset input {source!r} must resolve to a .csv or .parquet file, got {path.name!r}."
        )


def _cleanup_tmp_dir_path(tmp_dir_path: TmpDirPath) -> None:
    try:
        tmp_dir_path.cleanup_tmp_dir()
    except FileNotFoundError:
        logger.debug("Temporary anonymizer input directory was already removed.", exc_info=True)
    except Exception:
        logger.warning("Failed to remove temporary anonymizer input directory.", exc_info=True)


def _make_prepared_fileset_input(data: AnonymizerInputSpec, tmp_dir_path: TmpDirPath) -> PreparedAnonymizerInput:
    try:
        upstream_input = _make_upstream_input(data, source=str(tmp_dir_path.path))
    except Exception:
        _cleanup_tmp_dir_path(tmp_dir_path)
        raise
    return PreparedAnonymizerInput(input=upstream_input, tmp_dir_path=tmp_dir_path)


def _make_upstream_input(data: AnonymizerInputSpec, *, source: str) -> AnonymizerInput:
    try:
        return AnonymizerInput(
            source=source,
            text_column=data.text_column,
            id_column=data.id_column,
            data_summary=data.data_summary,
        )
    except ValidationError as exc:
        raise AnonymizerInvalidConfigError(str(exc)) from exc


def _is_http_source(source: str) -> bool:
    return urlparse(source).scheme in {"http", "https"}


def _looks_like_local_hash_path(source: str) -> bool:
    local_path = source.split("#", 1)[0]
    if Path(local_path).is_absolute():
        return True
    if source.startswith(("./", "../", "~/")) or local_path in {".", "..", "~"}:
        return True
    try:
        return Path(source).exists()
    except OSError:
        return False


def _looks_like_fileset_hash_ref(source: str) -> bool:
    fileset_part = source.split("#", 1)[0]
    parts = fileset_part.split("/")
    return 1 <= len(parts) <= 2 and all(parts)


def _raise_for_unsupported_scheme(source: str) -> None:
    parsed = urlparse(source)
    if parsed.scheme and "://" in source:
        raise AnonymizerInvalidConfigError(
            f"Input source {source!r} uses unsupported scheme {parsed.scheme!r}. "
            "Use an http(s) URL or fileset reference."
        )
