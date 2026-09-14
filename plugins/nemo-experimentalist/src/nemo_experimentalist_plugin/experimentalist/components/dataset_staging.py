# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment-local staging and hydration for Eval Author inputs."""

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from filesets import FilesetPathError, parse_fileset_ref
from nemo_experimentalist_plugin.entities import Dataset, DatasetRef, local_path_from_uri
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.files.types import ListFilesQueryParams


@dataclass(frozen=True, slots=True)
class _StagedEvalAuthorInputs:
    """Staged Eval Author references returned by this module's factory."""

    train_dataset: DatasetRef
    validation_dataset: DatasetRef
    task_template: DatasetRef


def distribute_insight_suite_tasks(
    insight_suite: Dataset,
    train_dataset: Dataset,
    validation_dataset: Dataset,
) -> None:
    """Assign Insight-suite tasks to validation/train at a deterministic 30/70 split.

    Eval Author retains the materialized suite as its provenance artifact. The
    optimizer consumes its tasks through the train and validation datasets,
    reserving the first 30 percent for validation and using the remaining 70
    percent for training feedback.
    """
    tasks = list(insight_suite.list_tasks())
    validation_count = (3 * len(tasks) + 9) // 10
    validation_dataset.add_tasks(tasks[:validation_count])
    train_dataset.add_tasks(tasks[validation_count:])


def _parse_fileset_uri(uri: str, *, workspace: str) -> tuple[str, str, str]:
    if urlparse(uri).scheme != "fileset":
        raise ValueError(f"Not a fileset URI: {uri}")
    try:
        return parse_fileset_ref(uri, workspace_fallback=workspace)
    except FilesetPathError as exc:
        raise ValueError(f"Invalid fileset URI {uri!r}: {exc}") from exc


def _write_downloaded_file(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _relative_download_path(remote_path: str, prefix: str) -> str:
    clean_remote_path = remote_path.strip("/")
    if not clean_remote_path:
        return ""
    if not prefix:
        return clean_remote_path

    clean_prefix = prefix.strip("/")
    if clean_remote_path == clean_prefix:
        return Path(clean_remote_path).name
    if not clean_remote_path.startswith(f"{clean_prefix}/"):
        raise ValueError(f"Fileset listing returned path outside requested prefix {clean_prefix!r}: {remote_path!r}")
    return clean_remote_path[len(clean_prefix) + 1 :]


def _safe_download_target(destination: Path, relative_path: str) -> Path:
    resolved_destination = destination.resolve()
    target = (resolved_destination / relative_path).resolve()
    if not target.is_relative_to(resolved_destination):
        raise ValueError(f"Fileset path escapes staging destination: {relative_path!r}")
    return target


async def _download_fileset_tree(
    client: AsyncNemoClient,
    *,
    uri: str,
    workspace: str,
    destination: Path,
) -> None:
    files_workspace, fileset, prefix = _parse_fileset_uri(uri, workspace=workspace)
    files = AsyncFilesClient.from_client(client)
    query_params: ListFilesQueryParams | None = {"path": prefix} if prefix else None
    listing = (await files.list_files(workspace=files_workspace, name=fileset, query_params=query_params)).data()
    for item in listing.data:
        remote_path = item.path.strip("/")
        relative_path = _relative_download_path(remote_path, prefix)
        if not relative_path:
            continue
        target = _safe_download_target(destination, relative_path)
        response = await files.download_file(workspace=files_workspace, name=fileset, path=remote_path)
        await asyncio.to_thread(_write_downloaded_file, target, await response.read())


def _local_directory(ref: DatasetRef) -> Path:
    path = local_path_from_uri(ref.uri, context="Eval Author input").resolve()
    if not path.is_dir():
        raise ValueError(f"Eval Author input is not a directory: {path}")
    return path


def _stage(ref: DatasetRef, destination: Path) -> DatasetRef:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(_local_directory(ref), destination)
    return ref.model_copy(update={"uri": str(destination)})


async def stage_task_template(
    experiment_dir: Path,
    task_template: DatasetRef,
    *,
    client: AsyncNemoClient,
    workspace: str,
) -> DatasetRef:
    """Refresh a local or Fileset-backed task template in experiment-local staging."""
    destination = experiment_dir.resolve() / "dataset" / "task-template"
    source = None
    if urlparse(task_template.uri).scheme != "fileset":
        source = _local_directory(task_template)
        if source == destination:
            return task_template.model_copy(update={"uri": str(destination)})

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Eval Author task template staging path is not a directory: {destination}")
        shutil.rmtree(destination)

    if urlparse(task_template.uri).scheme == "fileset":
        try:
            await _download_fileset_tree(
                client,
                uri=task_template.uri,
                workspace=workspace,
                destination=destination,
            )
        except BaseException:
            if destination.exists():
                shutil.rmtree(destination)
            raise
        if not destination.is_dir() or not any(path.is_file() for path in destination.rglob("*")):
            if destination.exists():
                shutil.rmtree(destination)
            raise ValueError(f"Eval Author Fileset task template contains no files: {task_template.uri}")
    else:
        assert source is not None
        shutil.copytree(source, destination)

    return task_template.model_copy(update={"uri": str(destination)})


async def stage_eval_author_inputs(
    experiment_dir: Path,
    *,
    train_dataset: DatasetRef,
    validation_dataset: DatasetRef,
    task_template: DatasetRef,
    client: AsyncNemoClient,
    workspace: str,
) -> _StagedEvalAuthorInputs:
    """Stage mutable Eval Author inputs beneath the experiment directory."""
    dataset_dir = experiment_dir.resolve() / "dataset"
    return _StagedEvalAuthorInputs(
        train_dataset=_stage(train_dataset, dataset_dir / "train"),
        validation_dataset=_stage(validation_dataset, dataset_dir / "validation"),
        task_template=await stage_task_template(
            experiment_dir,
            task_template,
            client=client,
            workspace=workspace,
        ),
    )
