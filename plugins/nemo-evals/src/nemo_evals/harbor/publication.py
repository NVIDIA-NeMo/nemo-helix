# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a self-contained task archive and verify its stored bytes before returning."""

from importlib.metadata import version
from pathlib import Path

from nemo_evals.api.task_definitions.harbor import HarborArchiveSource, HarborTaskDefinition, HarborTaskHash
from nemo_evals.harbor.archive import MAX_ARCHIVE_BYTES, extract_task, pack_task, run_blocking_archive_operation
from nemo_evals.harbor.archive_io import download_verified, download_verified_async
from nemo_helix_plugin.client.errors import NemoTransportError
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nhx_evals_sdk.agent_eval.runtimes.harbor.archive import (
    CHUNK_BYTES,
    capture_task,
    capture_validated_task,
    normalize_harbor_instruction,
    private_directory,
)
from nhx_evals_sdk.agent_eval.runtimes.harbor.tasks import HarborAgentEvalTask


def _capture_for_publication(root: Path, contents: Path, verify_against: HarborAgentEvalTask | None) -> Path:
    """Capture the upload snapshot and verify discovered fields against it before remote writes.

    Only checks that the in-memory task still matches the files being uploaded; scoring is validated
    by the caller.

    Algorithm:
        - Without ``verify_against``, capture the task as is; no Harbor validation runs.
        - Otherwise capture a validated copy and compare the discovered task with the native task
          Harbor reads from that copy:

          ==========================  =======  ===================================================
          Field                       Checked  Rule
          ==========================  =======  ===================================================
          ``id``, ``intent``          yes      equal to the snapshot's ``task_id``
          ``inputs``                  yes      equal to ``{"instruction": ...}`` normalized from
                                               the snapshot
          ``reference``               yes      empty; Harbor tasks keep their tests in the archive
          ``metrics``, ``views``,     no       validated and attached by the caller, never part
          ``metadata``                         of the archive
          ==========================  =======  ===================================================

    Args:
        root: Local Harbor task directory to capture.
        contents: Existing private directory that will own the captured copy.
        verify_against: Discovered task whose id, intent, inputs and reference must match the
            snapshot; ``None`` skips validation and comparison.

    Returns:
        Path of the captured task directory beneath ``contents``.

    Raises:
        ValueError: The source is unsafe or changes during capture, is not a valid Harbor task, or
            its fields differ from ``verify_against``.
    """
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
            return _definition(downloaded, check, source, fingerprint)


def describe_harbor_task_archive(
    root: Path,
    *,
    fileset_ref: str,
    path_prefix: str = "",
    verify_against: HarborAgentEvalTask | None = None,
) -> HarborTaskDefinition:
    """The definition :func:`publish_harbor_task_archive` would return for ``root``, with no remote call.

    The archive is packed deterministically and its Fileset path is addressed by its digest, so the same task
    directory always yields the same definition: a caller compares it with the stored task to skip a publication
    that would change nothing.

    Raises:
        ValueError: The task or destination is invalid.
    """
    from harbor.publisher.packager import Packager

    prefix = f"{path_prefix}/" if path_prefix else ""
    HarborArchiveSource(fileset_ref=f"{fileset_ref}#{prefix}{root.name}/{'0' * 64}/task_archive", files_hash="0" * 64)
    with private_directory() as owned:
        contents = owned / "contents"
        contents.mkdir(mode=0o700)
        snapshot = _capture_for_publication(root, contents, verify_against)
        fingerprint, _ = Packager.compute_content_hash(snapshot)
        archive_path = owned / "task_archive"
        digest = pack_task(snapshot, archive_path)
        source = HarborArchiveSource(
            fileset_ref=f"{fileset_ref}#{prefix}{root.name}/{digest}/task_archive", files_hash=digest
        )
        return _definition(archive_path, owned, source, fingerprint)


def _definition(archive: Path, scratch: Path, source: HarborArchiveSource, fingerprint: str) -> HarborTaskDefinition:
    """The Harbor definition of ``archive``, read from its extracted native task."""
    extracted = scratch / "extracted"
    extracted.mkdir(mode=0o700)
    _, native = extract_task(archive, extracted)
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
