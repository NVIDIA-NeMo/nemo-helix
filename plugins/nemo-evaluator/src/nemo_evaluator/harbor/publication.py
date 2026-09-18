# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a self-contained task archive and verify its stored bytes before returning."""

from importlib.metadata import version
from pathlib import Path

from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskDefinition, HarborTaskHash
from nemo_evaluator.harbor.archive import (
    CHUNK_BYTES,
    MAX_ARCHIVE_BYTES,
    capture_task,
    extract_task,
    pack_task,
    private_directory,
)
from nemo_evaluator.harbor.archive_io import download_verified
from nemo_platform_plugin.client.errors import NemoTransportError
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest


def publish_harbor_task_archive(
    root: Path, *, files_client: FilesClient, fileset_ref: str, path_prefix: str = ""
) -> HarborTaskDefinition:
    """Upload and verify one exact archive. No DB writes or deletion of shared objects."""
    from harbor.publisher.packager import Packager

    prefix = f"{path_prefix}/" if path_prefix else ""
    # Validate destination before capture or remote writes.
    HarborArchiveSource(fileset_ref=f"{fileset_ref}#{prefix}{root.name}/{'0' * 64}/task_archive", files_hash="0" * 64)
    workspace, name = fileset_ref.split("/", 1)
    with private_directory() as owned:
        contents = owned / "contents"
        contents.mkdir(mode=0o700)
        snapshot = capture_task(root, contents)
        fingerprint, _ = Packager.compute_content_hash(snapshot)
        archive_path = owned / "task_archive"
        digest = pack_task(snapshot, archive_path)
        path = f"{prefix}{root.name}/{digest}/task_archive"
        source = HarborArchiveSource(fileset_ref=f"{fileset_ref}#{path}", files_hash=digest)
        files_client.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=name), exist_ok=True).data()
        with archive_path.open("rb") as stream:
            try:
                # TODO: Enable If-None-Match: * and verified reuse in a follow-up PR
                # after https://github.com/NVIDIA-NeMo/nemo-platform/pull/2109 merges. Until then,
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
            source=source,
            harbor_hash=HarborTaskHash(digest=fingerprint, harbor_version=version("harbor")),
            instruction=native.instruction,
            config=native.config,
        )
