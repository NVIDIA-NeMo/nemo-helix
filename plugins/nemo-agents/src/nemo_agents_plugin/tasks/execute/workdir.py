# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Working directory config validation and materialization for agent invocation tasks."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path, PurePosixPath

from filesets import FilesetFileSystem, FilesetPathError, build_fileset_ref, parse_fileset_ref
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.files.types import ListFilesQueryParams
from nemo_platform_plugin.jobs.file_manager import FilesetFileManager, TmpDirPath
from pydantic import BaseModel, Field, field_validator, model_validator


class AgentWorkdirArtifactMount(BaseModel):
    ref: str
    mount_path: str

    @field_validator("mount_path")
    @classmethod
    def validate_mount_path(cls, value: str) -> str:
        return _normalize_relative_path(value, field="mount_path")


class AgentWorkdir(BaseModel):
    base_workdir: str | None = Field(
        default=None,
        description="Optional Files reference for the initial working directory.",
    )
    artifact_mounts: list[AgentWorkdirArtifactMount] = Field(
        default_factory=list,
        description="Optional fileset artifacts to layer into the working directory.",
    )

    @model_validator(mode="after")
    def validate_non_overlapping_mounts(self) -> "AgentWorkdir":
        mount_parts = [(mount.mount_path, PurePosixPath(mount.mount_path).parts) for mount in self.artifact_mounts]
        for index, (left_path, left_parts) in enumerate(mount_parts):
            for right_path, right_parts in mount_parts[index + 1 :]:
                if _paths_overlap(left_parts, right_parts):
                    raise ValueError(
                        f"artifact_mounts contain overlapping mount paths: {left_path!r} and {right_path!r}"
                    )
        return self


async def validate_agent_workdir(
    workdir: AgentWorkdir,
    files_client: AsyncFilesClient,
    *,
    default_workspace: str,
) -> AgentWorkdir | None:
    base_workdir = None
    if workdir.base_workdir is not None:
        base_workdir = await _validate_ref(
            workdir.base_workdir,
            files_client,
            default_workspace=default_workspace,
            field="workdir.base_workdir",
            directory_like=True,
        )

    artifact_mounts = []
    for mount in workdir.artifact_mounts:
        ref = await _validate_ref(
            mount.ref,
            files_client,
            default_workspace=default_workspace,
            field="workdir.artifact_mounts.ref",
            directory_like=False,
        )
        artifact_mounts.append(AgentWorkdirArtifactMount(ref=ref, mount_path=mount.mount_path))

    if not base_workdir and not artifact_mounts:
        return None

    return AgentWorkdir(base_workdir=base_workdir, artifact_mounts=artifact_mounts)


def materialize_agent_workdir(spec: AgentWorkdir, files_client: FilesClient, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if spec.base_workdir is not None:
        _download_fileset_ref(files_client, spec.base_workdir, local_dir=target_dir)

    for mount in spec.artifact_mounts:
        mount_local_path = target_dir / mount.mount_path
        with tempfile.TemporaryDirectory(prefix=".nemo-agent-workdir-mount-") as download_dir:
            downloaded = _download_fileset_ref(files_client, mount.ref, local_dir=Path(download_dir))
            _replace_mount_path(downloaded.path, mount_local_path)


async def _validate_ref(
    ref: str,
    files_client: AsyncFilesClient,
    *,
    default_workspace: str,
    field: str,
    directory_like: bool,
) -> str:
    canonical_ref = _canonical_files_ref(
        ref,
        default_workspace=default_workspace,
        field=field,
        directory_like=directory_like,
    )
    ws, fileset, path = parse_fileset_ref(canonical_ref, workspace_fallback=default_workspace)
    query_params: ListFilesQueryParams = {"path": path} if path else {}
    response = await files_client.list_files(
        workspace=ws,
        name=fileset,
        query_params=query_params or None,
    )
    if not response.data().data:
        if directory_like:
            raise ValueError(f"{field} must point to a non-empty directory or fileset root.")
        raise ValueError(f"{field} must point to an existing fileset artifact.")
    return canonical_ref


def _download_fileset_ref(files_client: FilesClient, ref: str, *, local_dir: Path) -> TmpDirPath:
    ws, fileset, _ = parse_fileset_ref(ref, workspace_fallback="")
    manager = FilesetFileManager(
        workspace=ws,
        fileset_name=fileset,
        filesystem=FilesetFileSystem(client=files_client),
        ensure_fileset_exists=False,
    )
    return manager.download_from_url(ref, local_dir=local_dir)


def _replace_mount_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir() and not destination.is_symlink():
        shutil.rmtree(destination)
    elif destination.exists() or destination.is_symlink():
        destination.unlink()

    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _canonical_files_ref(ref: str, *, default_workspace: str, field: str, directory_like: bool) -> str:
    raw = ref.strip()
    if not raw:
        raise ValueError(f"{field} must not be empty.")
    if raw.startswith("fileset://"):
        raise ValueError(f"{field} must use 'workspace/fileset#path' or 'fileset#path' syntax.")
    if raw.count("#") > 1:
        raise ValueError(f"{field} contains more than one '#'.")
    if "#" not in raw and len(raw.split("/")) > 2:
        raise ValueError(f"{field} must use '#' before fileset paths.")

    path_supplied = "#" in raw
    fragment = raw.split("#", 1)[1] if path_supplied else ""
    if fragment.startswith("/"):
        raise ValueError(f"{field} fragment must be relative.")
    _validate_fragment(fragment, field=field)

    try:
        workspace, fileset, path = parse_fileset_ref(raw, workspace_fallback=default_workspace)
    except FilesetPathError as exc:
        raise ValueError(str(exc)) from exc

    if not fileset:
        raise ValueError(f"{field} must include a fileset name.")
    if directory_like and path_supplied and path:
        path = f"{path.rstrip('/')}/"
    if not path:
        return f"{workspace}/{fileset}#"
    return build_fileset_ref(path, workspace=workspace, fileset=fileset)


def _validate_fragment(fragment: str, *, field: str) -> None:
    if not fragment:
        return
    if "\\" in fragment:
        raise ValueError(f"{field} fragment must use '/' separators.")
    path = PurePosixPath(fragment)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} fragment must not contain path escapes.")


def _normalize_relative_path(value: str, *, field: str) -> str:
    raw = value.strip()
    if raw.startswith("/"):
        raise ValueError(f"{field} must be a relative path and must not contain path escapes.")
    normalized = raw.rstrip("/")
    if not normalized or normalized == ".":
        raise ValueError(f"{field} must be a non-empty relative path.")
    if "\\" in normalized:
        raise ValueError(f"{field} must use '/' separators.")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a relative path and must not contain path escapes.")
    return path.as_posix()


def _paths_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    shortest = min(len(left), len(right))
    return left[:shortest] == right[:shortest]
