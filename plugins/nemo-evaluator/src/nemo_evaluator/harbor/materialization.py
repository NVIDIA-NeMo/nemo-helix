# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify archives and prepare execution directories, publishing only complete local datasets."""

import tarfile
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from nemo_evaluator.api.task_definitions.harbor import HarborTaskDefinition
from nemo_evaluator.harbor.archive import MAX_ARCHIVE_BYTES, extract_task, run_blocking_archive_operation
from nemo_evaluator.harbor.archive_io import download_verified, download_verified_async
from nemo_evaluator.harbor.tasks import StoredHarborTask
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_archive import NativeTask, private_directory, remove_owned_tree
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient


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
    native: NativeTask


@dataclass(frozen=True)
class MaterializedHarborTasks:
    dataset_root: Path
    members: tuple[MaterializedHarborMember, ...]


async def verify_definition(definition: HarborTaskDefinition, files_client: AsyncFilesClient) -> NativeTask:
    """Verify archive bytes and return native metadata without retaining extracted files.

    Algorithm:
        - Download the archive through the authenticated client with byte and digest limits.
        - Safely extract and validate its native Harbor metadata in private staging.
        - Remove all downloaded and extracted files before returning.

    This function does not compare the extracted identity with ``definition``; callers that need
    that guarantee must perform the comparison.

    Args:
        definition: Stored definition identifying the archive and expected digest.
        files_client: Authenticated asynchronous Files client.

    Returns:
        Native metadata read from the verified archive.

    Raises:
        HarborArtifactError: The Files request fails; 401, 403, and 404 are preserved, otherwise 503.
        ValueError: The archive is unreadable, unsafe, corrupt, or fails validation.
    """
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
    """Download a checksum-verified archive and extract its validated task into the owned directory."""
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
    root, native = await run_blocking_archive_operation(extract_task, archive_path, contents)
    archive_path.unlink()
    return root, native


@contextmanager
def _staging(members: Sequence[StoredHarborTask], destination: Path) -> Iterator[Path]:
    """Allocate a dataset staging directory and remove all owned files if materialization fails."""
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
    """Check the native task identity and move its tree into staging, rejecting folder-name collisions."""
    if source.definition.native_task_id != native.task_id:
        raise ValueError("Harbor native_task_id does not match the verified archive")
    target = owned / "staging" / root.name
    if any(path.name.casefold() == root.name.casefold() for path in target.parent.iterdir()):
        raise ValueError("Duplicate physical task folder")
    root.rename(target)
    return MaterializedHarborMember(source, native.task_id, target, native)


def _publish(owned: Path, members: list[MaterializedHarborMember]) -> MaterializedHarborTasks:
    """Reject duplicate native task IDs, then rename the complete staging tree to its final dataset path."""
    ids = [member.task_id.casefold() for member in members]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate Harbor task IDs")
    final = owned / "dataset"
    (owned / "staging").rename(final)
    return MaterializedHarborTasks(
        final,
        tuple(
            MaterializedHarborMember(member.source, member.task_id, final / member.task_dir.name, member.native)
            for member in members
        ),
    )


async def materialize_harbor_tasks(
    members: Sequence[StoredHarborTask], *, files_client: AsyncFilesClient, destination_root: Path
) -> MaterializedHarborTasks:
    """Build a complete local dataset using the authenticated async Files client.

    Algorithm:
        - Download, hash-check, and safely extract every member into isolated staging.
        - Match native IDs to stored definitions and reject ID or folder-name collisions.
        - Atomically rename the complete staging tree to ``dataset`` only after all members pass.

    Args:
        members: Ordered stored task descriptors to materialize.
        files_client: Authenticated asynchronous Files client.
        destination_root: Parent under which this call creates one owned directory.

    Returns:
        Dataset root and ordered members pointing at their final task directories.

    Raises:
        ValueError: Membership, archive, native identity, or destination validation fails.

    Failure removes the owned child; success transfers its cleanup to the caller. The parent and
    unrelated contents are never removed.
    """
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
    """Build a complete local dataset synchronously through the supplied Files client.

    Algorithm:
        - Download, hash-check, and safely extract every member into isolated staging.
        - Match native IDs to stored definitions and reject ID or folder-name collisions.
        - Atomically rename the complete staging tree to ``dataset`` only after all members pass.

    Args:
        members: Ordered stored task descriptors to materialize.
        files_client: Authenticated synchronous Files client.
        destination_root: Parent under which this call creates one owned directory.

    Returns:
        Dataset root and ordered members pointing at their final task directories.

    Raises:
        ValueError: Membership, archive, native identity, or destination validation fails.

    Failure removes the owned child; success transfers its cleanup to the caller. The parent and
    unrelated contents are never removed.
    """
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
