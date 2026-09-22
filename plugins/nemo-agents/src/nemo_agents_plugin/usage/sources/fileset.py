# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a NeMo Platform fileset locally for parsing.

The caller passes in the SDK client and a default workspace; the parser
then operates on the staged tempdir, which is cleaned up when the
context exits.

The workspace-prefix convention follows :class:`nemo_platform_plugin.refs.FilesetRef`
— ``"name"`` uses the default workspace, ``"workspace/name"`` overrides it.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path

from nemo_platform_plugin.client.adapter import SyncPlatformClient, client_from_platform
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.refs import FilesetRef

logger = logging.getLogger(__name__)


class FilesetRefError(ValueError):
    """Raised when a fileset reference is malformed (e.g., multi-segment)."""


class FilesetDownloadError(RuntimeError):
    """Raised when the SDK fails to download a fileset (network, auth, missing)."""


@contextlib.contextmanager
def fileset_path(
    ref: FilesetRef,
    *,
    sdk: SyncPlatformClient,
    workspace: str,
) -> Iterator[Path]:
    """Download *ref* to a tempdir and yield the path.

    *sdk* is a platform handle (``NeMoPlatform`` or
    :class:`~nemo_platform_plugin.client.client.NemoClient`).  Cleanup happens
    when the context exits.

    The accepted shapes are ``name`` (uses *workspace* as the workspace) or
    ``workspace/name``.  Multi-segment refs (``a/b/c``) are rejected here
    rather than letting the slash leak into the tempdir prefix where
    :func:`tempfile.mkdtemp` raises a confusing ``FileNotFoundError``.

    SDK errors (network failures, auth, missing filesets) are wrapped as
    :class:`FilesetDownloadError` so the CLI's catch tuple stays a precise
    list of error categories instead of growing toward bare ``Exception``.
    """
    raw = str(ref)
    if "/" in raw:
        ws, name = raw.split("/", 1)
    else:
        ws, name = workspace, raw

    if "/" in name:
        raise FilesetRefError(
            f"invalid fileset reference {raw!r}: name segment must not contain '/' "
            f"(use 'workspace/name' for a workspace-qualified reference)"
        )

    if name in {"", ".", ".."}:
        raise FilesetRefError(
            f"invalid fileset reference {raw!r}: name segment must be a real fileset name "
            f"(empty / '.' / '..' are rejected; a trailing '/' produces an empty name)"
        )

    if not ws:
        raise FilesetRefError(
            f"invalid fileset reference {raw!r}: workspace must be non-empty "
            f"(pass --workspace or use 'workspace/name' form)"
        )

    with tempfile.TemporaryDirectory(prefix=f".usage-{name}-") as tmp:
        tmp_path = Path(tmp)
        logger.debug("downloading fileset %s/%s to %s", ws, name, tmp_path)
        try:
            files = client_from_platform(sdk, FilesClient)
            listing = files.list_files(name=name, workspace=ws).data().data
            for entry in listing:
                dest = tmp_path / entry.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(files.download_file(name=name, workspace=ws, path=entry.path).read())
        except (FilesetRefError, FilesetDownloadError):
            raise
        except Exception as exc:
            raise FilesetDownloadError(f"failed to download fileset {ws}/{name}: {exc}") from exc
        yield tmp_path
