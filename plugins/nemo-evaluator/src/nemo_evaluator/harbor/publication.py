# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a self-contained task archive and verify its stored bytes before returning."""

from importlib.metadata import version
from pathlib import Path

from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskDefinition, HarborTaskHash
from nemo_evaluator.harbor.archive import MAX_ARCHIVE_BYTES, extract_task, pack_task, run_blocking_archive_operation
from nemo_evaluator.harbor.archive_io import download_verified, download_verified_async
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_archive import (
    CHUNK_BYTES,
    capture_task,
    capture_validated_task,
    normalize_harbor_instruction,
    private_directory,
)
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_tasks import HarborAgentEvalTask
from nemo_helix_plugin.client.errors import NemoTransportError
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest


def _capture_for_publication(root: Path, contents: Path, verify_against: HarborAgentEvalTask | None) -> Path:
    """Verify discovered fields against the exact upload snapshot before remote writes."""
    if verify_against is None:
        return capture_task(root, contents)
    snapshot, native = capture_validated_task(root, contents)
    inputs = {"instruction": normalize_harbor_instruction(native.instruction, task_id=native.task_id)}
    if (
        verify_against.id != native.task_id
        or verify_against.intent != native.task_id
        or verify_against.inputs != inputs
        or verify_against.reference
    ):
        raise ValueError("Harbor task fields differ from the package; rediscover before publication")
    return snapshot


def publish_harbor_task_archive(
    root: Path,
    *,
    files_client: FilesClient,
    fileset_ref: str,
    path_prefix: str = "",
    verify_against: HarborAgentEvalTask | None = None,
) -> HarborTaskDefinition:
    """Capture, upload, and verify one Harbor task without creating task entities.

    Algorithm:
        - Capture a stable task snapshot and package it into deterministic archive bytes.
        - Upload under a digest-addressed Fileset path, tolerating only an ambiguous transport result.
        - Download, hash-check, extract, and validate the stored bytes before returning their definition.

    Args:
        root: Local Harbor task directory to publish.
        files_client: Authenticated synchronous Files client.
        fileset_ref: Destination in ``workspace/fileset`` form.
        path_prefix: Optional directory prefix within the Fileset.
        verify_against: Discovered task whose id, intent, and inputs must match the captured package;
            ``None`` skips that check.

    Returns:
        Harbor definition pinned to the verified archive and native task metadata.

    Raises:
        ValueError: The task, destination, uploaded bytes, or extracted archive is invalid.
    """
    from harbor.publisher.packager import Packager

    prefix = f"{path_prefix}/" if path_prefix else ""
    # Validate destination before capture or remote writes.
    HarborArchiveSource(fileset_ref=f"{fileset_ref}#{prefix}{root.name}/{'0' * 64}/task_archive", files_hash="0" * 64)
    workspace, name = fileset_ref.split("/", 1)
    with private_directory() as owned:
        contents = owned / "contents"
        contents.mkdir(mode=0o700)
        snapshot = _capture_for_publication(root, contents, verify_against)
        fingerprint, _ = Packager.compute_content_hash(snapshot)
        archive_path = owned / "task_archive"
        digest = pack_task(snapshot, archive_path)
        path = f"{prefix}{root.name}/{digest}/task_archive"
        source = HarborArchiveSource(fileset_ref=f"{fileset_ref}#{path}", files_hash=digest)
        files_client.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=name), exist_ok=True).data()
        with archive_path.open("rb") as stream:
            try:
                # TODO: Enable If-None-Match: * and verified reuse in a follow-up PR
                # after https://github.com/NVIDIA-NeMo/nemo-helix/pull/2109 merges. Until then,
                # ordinary PUT may replace existing bytes; readback remains mandatory.
                files_client.with_headers({"Content-Length": str(archive_path.stat().st_size)}).upload_file(
                    workspace=workspace, name=name, path=path, content=iter(lambda: stream.read(CHUNK_BYTES), b"")
                ).data()
            except NemoTransportError:
                # A lost write response is successful only if mandatory readback verifies it.
                pass
        with private_directory() as check:
            downloaded = check / "task_archive"
            with downloaded.open("xb") as output:
                download_verified(
                    files_client, source.fileset_ref, output, limit=MAX_ARCHIVE_BYTES, expected_digest=digest
                )
            extracted = check / "contents"
            extracted.mkdir(mode=0o700)
            _, native = extract_task(downloaded, extracted)
        return HarborTaskDefinition(
            kind="harbor",
            native_task_id=native.task_id,
            source=source,
            harbor_hash=HarborTaskHash(digest=fingerprint, harbor_version=version("harbor")),
            instruction=native.instruction,
            config=native.config,
        )


async def publish_harbor_task_archive_async(
    root: Path,
    *,
    files_client: AsyncFilesClient,
    fileset_ref: str,
    path_prefix: str = "",
    verify_against: HarborAgentEvalTask | None = None,
) -> HarborTaskDefinition:
    """Asynchronously capture, upload, and verify one Harbor task archive.

    Algorithm:
        - Run blocking capture, hashing, and packaging under the shared worker limit.
        - Upload with native async Files I/O, tolerating only an ambiguous transport result.
        - Read back, hash-check, extract, and validate the stored bytes before returning.

    Local archive work is drained before cancellation can remove its staging files.

    Args:
        root: Local Harbor task directory to publish.
        files_client: Authenticated asynchronous Files client.
        fileset_ref: Destination in ``workspace/fileset`` form.
        path_prefix: Optional directory prefix within the Fileset.
        verify_against: Discovered task whose id, intent, and inputs must match the captured package;
            ``None`` skips that check. It verifies that the task the caller holds in memory is the
            same task whose bytes get uploaded.

    Returns:
        Harbor definition pinned to the verified archive and native task metadata.

    Raises:
        ValueError: The task, destination, uploaded bytes, or extracted archive is invalid.
    """
    from harbor.publisher.packager import Packager

    prefix = f"{path_prefix}/" if path_prefix else ""
    HarborArchiveSource(fileset_ref=f"{fileset_ref}#{prefix}{root.name}/{'0' * 64}/task_archive", files_hash="0" * 64)
    workspace, name = fileset_ref.split("/", 1)
    with private_directory() as owned:
        contents = owned / "contents"
        contents.mkdir(mode=0o700)
        snapshot = await run_blocking_archive_operation(_capture_for_publication, root, contents, verify_against)
        fingerprint, _ = await run_blocking_archive_operation(Packager.compute_content_hash, snapshot)
        archive_path = owned / "task_archive"
        digest = await run_blocking_archive_operation(pack_task, snapshot, archive_path)
        path = f"{prefix}{root.name}/{digest}/task_archive"
        source = HarborArchiveSource(fileset_ref=f"{fileset_ref}#{path}", files_hash=digest)
        (
            await files_client.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=name), exist_ok=True)
        ).data()
        with archive_path.open("rb") as stream:

            async def chunks():
                while chunk := await run_blocking_archive_operation(stream.read, CHUNK_BYTES):
                    yield chunk

            try:
                (
                    await files_client.with_headers({"Content-Length": str(archive_path.stat().st_size)}).upload_file(
                        workspace=workspace, name=name, path=path, content=chunks()
                    )
                ).data()
            except NemoTransportError:
                pass  # Same verified readback recovery as the synchronous publisher.
        with private_directory() as check:
            downloaded = check / "task_archive"
            with downloaded.open("xb") as output:
                await download_verified_async(
                    files_client, source.fileset_ref, output, limit=MAX_ARCHIVE_BYTES, expected_digest=digest
                )
            extracted = check / "contents"
            extracted.mkdir(mode=0o700)
            _, native = await run_blocking_archive_operation(extract_task, downloaded, extracted)
        return HarborTaskDefinition(
            kind="harbor",
            native_task_id=native.task_id,
            source=source,
            harbor_hash=HarborTaskHash(digest=fingerprint, harbor_version=version("harbor")),
            instruction=native.instruction,
            config=native.config,
        )
