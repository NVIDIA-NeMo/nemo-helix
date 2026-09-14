# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated, verified, invocation-local Harbor Fileset tree materialization."""

import asyncio
import io
import shutil
import tempfile
import tomllib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from nemo_evaluator.harbor.manifest import (
    DOWNLOAD_CONCURRENCY,
    MAX_MANIFEST_BYTES,
    HarborTreeManifest,
    TreeDirectory,
    TreeFile,
)
from nemo_evaluator.harbor.tasks import StoredHarborTask
from nemo_evaluator.harbor.tree_io import download_verified, download_verified_async
from nemo_evaluator_sdk.agent_eval.taskset_sources import digest_harbor_tree
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient


@dataclass(frozen=True)
class MaterializedHarborMember:
    source: StoredHarborTask
    task_id: str
    task_dir: Path


@dataclass(frozen=True)
class MaterializedHarborTasks:
    dataset_root: Path
    members: tuple[MaterializedHarborMember, ...]
    materialization_digest: str


@contextmanager
def _staging(members: Sequence[StoredHarborTask], destination: Path) -> Iterator[Path]:
    if not members:
        raise ValueError("Expected at least one stored Harbor member")
    destination.mkdir(parents=True, exist_ok=True)
    owned = Path(tempfile.mkdtemp(prefix="harbor-", dir=destination.resolve()))
    try:
        (owned / "staging").mkdir()
        yield owned
    except BaseException:
        shutil.rmtree(owned)
        raise


def _prepare_tree(manifest: HarborTreeManifest, owned: Path, index: int) -> Path:
    root = owned / f"tree-{index}"
    root.mkdir()
    for entry in manifest.entries:
        if isinstance(entry, TreeDirectory):
            (root / entry.path).mkdir(mode=0o755)
    return root


def _member(source: StoredHarborTask, root: Path, folder: str, owned: Path) -> tuple[str, Path]:
    from harbor.models.task.task import Task

    try:
        if digest_harbor_tree(root) != source.definition.tree.tree_digest:
            raise ValueError("Task tree checksum mismatch")
        if not (root / "environment").is_dir():
            raise ValueError("Missing environment directory")
        # Construction supplies detailed configuration/artifact errors; is_valid_dir
        # additionally checks native package structure without starting a runtime.
        Task(root)
        if not Task.is_valid_dir(root):
            raise ValueError("Invalid native Harbor package or required artifact")
        config = tomllib.loads((root / "task.toml").read_text())
        task_id = config.get("task", {}).get("name", folder)
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Invalid task identity")
        if folder in {"task_template", "", ".", ".."}:
            raise ValueError(f"Invalid or reserved task folder: {folder}")
        target = owned / "staging" / folder
        if target.exists():
            raise ValueError(f"Duplicate physical task folder: {folder}")
        root.rename(target)
        return task_id, target
    except Exception as exc:
        raise ValueError(f"Harbor member {source.entity_name}: {exc}") from exc


def _publish(owned: Path, members: list[MaterializedHarborMember]) -> MaterializedHarborTasks:
    ids = [member.task_id for member in members]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate Harbor task IDs")
    staging = owned / "staging"
    digest = digest_harbor_tree(staging)
    final = owned / "dataset"
    staging.rename(final)
    return MaterializedHarborTasks(
        final,
        tuple(
            MaterializedHarborMember(member.source, member.task_id, final / member.task_dir.name) for member in members
        ),
        digest,
    )


async def materialize_harbor_tasks(
    members: Sequence[StoredHarborTask], *, files_client: AsyncFilesClient, destination_root: Path
) -> MaterializedHarborTasks:
    with _staging(members, destination_root) as owned:
        results = []
        for index, source in enumerate(members):
            tree = source.definition.tree
            buffer = io.BytesIO()
            await download_verified_async(files_client, tree.manifest_ref, buffer, limit=MAX_MANIFEST_BYTES)
            manifest = HarborTreeManifest.from_bytes(buffer.getvalue(), tree.manifest_digest)
            root = _prepare_tree(manifest, owned, index)
            entries = iter(entry for entry in manifest.entries if isinstance(entry, TreeFile))

            async def download_files() -> None:
                # Fixed workers avoid allocating one asyncio task per file. TaskGroup drains
                # and cancels all siblings before invocation-owned staging can be removed.
                for entry in entries:
                    path = root / entry.path
                    with path.open("xb") as output:
                        await download_verified_async(
                            files_client,
                            tree.root_ref + "/" + entry.path,
                            output,
                            limit=entry.size,
                            expected_size=entry.size,
                            expected_digest=entry.sha256,
                        )
                    path.chmod(0o644 | entry.executable)

            try:
                async with asyncio.TaskGroup() as group:
                    for _ in range(DOWNLOAD_CONCURRENCY):
                        group.create_task(download_files())
            except ExceptionGroup as exc:
                raise ValueError(
                    f"Harbor member {source.entity_name}: file download failed: {exc.exceptions[0]}"
                ) from exc
            task_id, task_dir = _member(source, root, manifest.task_dir, owned)
            results.append(MaterializedHarborMember(source, task_id, task_dir))
        return _publish(owned, results)


def materialize_harbor_tasks_sync(
    members: Sequence[StoredHarborTask], *, files_client: FilesClient, destination_root: Path
) -> MaterializedHarborTasks:
    with _staging(members, destination_root) as owned:
        results = []
        for index, source in enumerate(members):
            tree = source.definition.tree
            buffer = io.BytesIO()
            download_verified(files_client, tree.manifest_ref, buffer, limit=MAX_MANIFEST_BYTES)
            manifest = HarborTreeManifest.from_bytes(buffer.getvalue(), tree.manifest_digest)
            root = _prepare_tree(manifest, owned, index)
            for entry in manifest.entries:
                if isinstance(entry, TreeFile):
                    path = root / entry.path
                    with path.open("xb") as output:
                        download_verified(
                            files_client,
                            tree.root_ref + "/" + entry.path,
                            output,
                            limit=entry.size,
                            expected_size=entry.size,
                            expected_digest=entry.sha256,
                        )
                    path.chmod(0o644 | entry.executable)
            task_id, task_dir = _member(source, root, manifest.task_dir, owned)
            results.append(MaterializedHarborMember(source, task_id, task_dir))
        return _publish(owned, results)
