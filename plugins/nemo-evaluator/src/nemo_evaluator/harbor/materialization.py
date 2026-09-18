# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated archive verification and invocation-local Harbor materialization."""

import tarfile
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from nemo_evaluator.api.task_definitions.harbor import HarborTaskDefinition
from nemo_evaluator.harbor.archive import (
    MAX_ARCHIVE_BYTES,
    NativeTask,
    extract_task,
    private_directory,
    remove_owned_tree,
    run_validation,
)
from nemo_evaluator.harbor.archive_io import download_verified, download_verified_async
from nemo_evaluator.harbor.tasks import StoredHarborTask
from nemo_platform_plugin.client.errors import NemoHTTPError
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient


class HarborArtifactError(ValueError):
    """Safe API error for an inaccessible task archive."""

    def __init__(self, status_code: int):
        super().__init__("Harbor task archive is unavailable or access was denied")
        self.status_code = status_code


@dataclass(frozen=True)
class MaterializedHarborMember:
    source: StoredHarborTask
    task_id: str
    task_dir: Path


@dataclass(frozen=True)
class MaterializedHarborTasks:
    dataset_root: Path
    members: tuple[MaterializedHarborMember, ...]


async def verify_definition(definition: HarborTaskDefinition, files_client: AsyncFilesClient) -> NativeTask:
    try:
        with private_directory() as owned:
            _, native = await _download_async(definition, files_client, owned)
            return native
    except NemoHTTPError as exc:
        raise HarborArtifactError(exc.status_code if exc.status_code in {401, 403, 404} else 503) from exc
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ValueError("Invalid or unreadable Harbor archive") from exc


async def _download_async(
    definition: HarborTaskDefinition, files_client: AsyncFilesClient, owned: Path
) -> tuple[Path, NativeTask]:
    archive_path = owned / "task_archive"
    with archive_path.open("xb") as output:
        await download_verified_async(
            files_client,
            definition.source.fileset_ref,
            output,
            limit=MAX_ARCHIVE_BYTES,
            expected_digest=definition.source.files_hash,
        )
    contents = owned / "contents"
    contents.mkdir(mode=0o700)
    root, native = await run_validation(extract_task, archive_path, contents)
    archive_path.unlink()
    return root, native


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
        remove_owned_tree(owned)
        raise


def _member(source: StoredHarborTask, root: Path, native: NativeTask, owned: Path) -> MaterializedHarborMember:
    target = owned / "staging" / root.name
    if any(path.name.casefold() == root.name.casefold() for path in target.parent.iterdir()):
        raise ValueError("Duplicate physical task folder")
    root.rename(target)
    return MaterializedHarborMember(source, native.task_id, target)


def _publish(owned: Path, members: list[MaterializedHarborMember]) -> MaterializedHarborTasks:
    ids = [member.task_id.casefold() for member in members]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate Harbor task IDs")
    final = owned / "dataset"
    (owned / "staging").rename(final)
    return MaterializedHarborTasks(
        final,
        tuple(
            MaterializedHarborMember(member.source, member.task_id, final / member.task_dir.name) for member in members
        ),
    )


async def materialize_harbor_tasks(
    members: Sequence[StoredHarborTask], *, files_client: AsyncFilesClient, destination_root: Path
) -> MaterializedHarborTasks:
    with _staging(members, destination_root) as owned:
        results = []
        for source in members:
            with private_directory(owned) as member_parent:
                definition = HarborTaskDefinition.model_validate(source.definition.model_dump())
                root, native = await _download_async(definition, files_client, member_parent)
                results.append(_member(source, root, native, owned))
        return _publish(owned, results)


def materialize_harbor_tasks_sync(
    members: Sequence[StoredHarborTask], *, files_client: FilesClient, destination_root: Path
) -> MaterializedHarborTasks:
    with _staging(members, destination_root) as owned:
        results = []
        for source in members:
            with private_directory(owned) as member_parent:
                definition = HarborTaskDefinition.model_validate(source.definition.model_dump())
                archive_path = member_parent / "task_archive"
                with archive_path.open("xb") as output:
                    download_verified(
                        files_client,
                        definition.source.fileset_ref,
                        output,
                        limit=MAX_ARCHIVE_BYTES,
                        expected_digest=definition.source.files_hash,
                    )
                contents = member_parent / "contents"
                contents.mkdir(mode=0o700)
                root, native = extract_task(archive_path, contents)
                results.append(_member(source, root, native, owned))
        return _publish(owned, results)
