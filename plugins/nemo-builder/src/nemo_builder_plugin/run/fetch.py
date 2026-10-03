# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build fetch``: the first step. It downloads each build context onto the work volume, and unpacks archives."""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from nemo_builder_plugin.run.utils import (
    job_identity,
    read_step_config,
    split_fileset_ref,
    work_mount,
)
from nemo_builder_plugin.steps import ContextSource, FetchStepConfig, WorkLayout
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import ListFilesQueryParams
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

#: What one archive may unpack to. It unpacks onto the work volume, which every build on the node shares.
MAX_UNPACKED_BYTES = 4 * 1024**3
MAX_UNPACKED_ENTRIES = 100_000

_CHUNK = 1024 * 1024


class ArchiveRefused(Exception):
    """An archive this step won't unpack: one that isn't a tar, writes outside its directory, or is too large."""


def _safe_destination(root: Path, relative_path: str) -> Path:
    """Join a tenant-controlled fileset path to ``root``, refusing one that escapes it."""
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"fileset path {relative_path!r} escapes the context directory")
    return candidate


def _download_fileset(
    client: FilesClient,
    *,
    workspace: str,
    name: str,
    context_path: str | None,
    destination: Path,
) -> int:
    query: ListFilesQueryParams | None = {"path": context_path} if context_path else None
    listing = client.list_files(workspace=workspace, name=name, query_params=query).data()

    count = 0
    for entry in listing.data:
        target = _safe_destination(destination, entry.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        response = client.download_file(workspace=workspace, name=name, path=entry.path)
        target.write_bytes(response.read())
        count += 1
    return count


def unpack(archive: Path, destination: Path) -> int:
    """Unpack the tar ``archive``, compressed or not, into ``destination``; return how many entries it held.

    Python's ``data`` filter puts an absolute name under ``destination``, refuses a ``..`` out of it, a link out of
    it and a device, and drops set-ID bits. The caps are checked before each entry is written.
    """
    entries = total = 0
    try:
        with tarfile.open(archive, "r:*") as tar:
            for member in tar:
                entries += 1
                total += max(member.size, 0)
                if entries > MAX_UNPACKED_ENTRIES:
                    raise ArchiveRefused(f"it holds more than {MAX_UNPACKED_ENTRIES} entries")
                if total > MAX_UNPACKED_BYTES:
                    raise ArchiveRefused(f"it unpacks to more than {MAX_UNPACKED_BYTES} bytes")
                tar.extract(member, destination, filter="data")
    except tarfile.FilterError as exc:
        raise ArchiveRefused(str(exc)) from exc
    except (tarfile.TarError, EOFError) as exc:
        raise ArchiveRefused(f"it isn't a tar archive this step can read: {exc}") from exc
    return entries


def _fetch_archive(
    client: FilesClient, layout: WorkLayout, source: ContextSource, archive: str, *, workspace: str
) -> int:
    source_workspace, name = split_fileset_ref(source.fileset, workspace)
    destination = Path(layout.archive(source.fileset, archive))
    downloads = Path(layout.downloads)
    destination.mkdir(parents=True)
    downloads.mkdir(parents=True, exist_ok=True)

    logger.info(
        "fetching archive %s from fileset %s/%s -> %s",
        sanitize_for_log(archive),
        sanitize_for_log(source_workspace),
        sanitize_for_log(name),
        sanitize_for_log(destination),
    )
    # On the work volume, not the pod's own disk, which an archive may outgrow.
    with tempfile.NamedTemporaryFile(dir=downloads) as download:
        response = client.download_file(workspace=source_workspace, name=name, path=archive)
        with response.stream(_CHUNK) as chunks:
            for chunk in chunks:
                download.write(chunk)
        download.flush()
        try:
            entries = unpack(Path(download.name), destination)
        except ArchiveRefused as exc:
            raise ArchiveRefused(f"{source.fileset}/{archive}: {exc}") from exc
    logger.info("unpacked %d entries", entries)
    return entries


def fetch_source(client: FilesClient, layout: WorkLayout, source: ContextSource, *, workspace: str) -> int:
    if source.archive is not None:
        return _fetch_archive(client, layout, source, source.archive, workspace=workspace)

    source_workspace, name = split_fileset_ref(source.fileset, workspace)
    fileset_dir = Path(layout.fileset(source.fileset))
    context_dir = Path(layout.context(source))
    fileset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "fetching fileset %s/%s%s -> %s",
        sanitize_for_log(source_workspace),
        sanitize_for_log(name),
        sanitize_for_log(f" ({source.context_path})" if source.context_path else ""),
        sanitize_for_log(context_dir),
    )
    # Files reports paths from the fileset's root even when the listing is narrowed to the context
    # path, so they are written under the fileset's directory, not the context's.
    count = _download_fileset(
        client,
        workspace=source_workspace,
        name=name,
        context_path=source.context_path,
        destination=fileset_dir,
    )
    context_dir.mkdir(parents=True, exist_ok=True)
    logger.info("fetched %d file(s)", count)
    return count


def clear_earlier_attempts(layout: WorkLayout) -> None:
    """Remove whatever an earlier run of this job left in its slice of the work volume.

    A job submitted again under the same name gets the same slice.
    """
    for directory in (layout.root / "context", layout.unpacked, layout.downloads, layout.outputs):
        path = Path(directory)
        if path.exists():
            logger.info("removing %s left by an earlier run of this job", sanitize_for_log(path))
            shutil.rmtree(path)


def main() -> int:
    config = FetchStepConfig.model_validate(read_step_config())
    workspace, _ = job_identity()

    # As `service:builder`, acting for the submitter. Files refuses a fileset in a workspace where the
    # submitter has no role; nothing checks that the submitter's role there may read filesets.
    client = client_from_platform(get_task_nemo_client("builder"), FilesClient)
    layout = WorkLayout(PurePosixPath(work_mount()))
    clear_earlier_attempts(layout)

    for source in config.sources:
        try:
            fetch_source(client, layout, source, workspace=workspace)
        except ArchiveRefused as exc:
            # Its entry names are the archive's, so the message is sanitized here rather than left to a traceback.
            logger.error("refusing to unpack %s", sanitize_for_log(str(exc)))
            return 1

    return 0
