# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fileset staging helpers for agents jobs.

Platform jobs run as host subprocesses that stage nothing into their work
dir, so a config file (and its sibling data) must either already exist on the
host (absolute-path mode, used by the co-located CLI) or be delivered through
a NeMo Platform fileset that the job downloads at runtime.  These helpers
implement the fileset path so a remote client (e.g. Studio) can submit a job
without touching the platform host's filesystem.

:class:`~nemo_agents_plugin.jobs.evaluate_agent.EvaluateAgentJob` predates this
module and still carries its own private copies of the same logic; new jobs
should use these functions instead, and evaluate can migrate here later.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path
from typing import Iterator

from filesets import FilesetFileSystem
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.file_manager import FilesetFileManager
from nemo_platform_plugin.refs import (
    FilesetRef,
    LocalDir,
    OutputTarget,
    classify_output_target,
    parse_entity_ref,
)
from nemo_platform_plugin.run_dependencies import LocalRunError

logger = logging.getLogger(__name__)


def split_fileset_ref(ref: str, default_workspace: str) -> tuple[str, str]:
    """Split a ``workspace/name`` (or bare ``name``) fileset ref into ``(ws, name)``."""
    parsed = parse_entity_ref(ref, default_workspace=default_workspace)
    return parsed.workspace, parsed.name


def _fileset_manager(
    files_client: FilesClient,
    *,
    workspace: str,
    fileset: str,
    ensure_fileset_exists: bool,
) -> FilesetFileManager:
    return FilesetFileManager(
        workspace=workspace,
        fileset_name=fileset,
        filesystem=FilesetFileSystem(client=files_client),
        ensure_fileset_exists=ensure_fileset_exists,
    )


@contextlib.contextmanager
def resolve_staged_config(
    config_rel_path: str,
    fileset_ref: str | None,
    *,
    workspace: str,
    ctx: JobContext,
    sdk: NeMoPlatform | None,
    kind: str,
) -> Iterator[Path]:
    """Yield a local path to a config file, staging it from a fileset if requested.

    When ``fileset_ref`` is set, download the fileset into a tempdir under
    ``ctx.storage.ephemeral`` and resolve ``config_rel_path`` inside it (with a
    path-escape guard).  Otherwise treat ``config_rel_path`` as a real local
    path and pass it through verbatim.  ``kind`` is a short slug used in the
    tempdir prefix and log lines (e.g. ``"optimize-config"``).
    """
    if fileset_ref is None:
        yield Path(config_rel_path)
        return

    ref = FilesetRef(fileset_ref)
    ws, name = split_fileset_ref(ref, workspace)

    if sdk is None:
        raise LocalRunError(
            f"Staging {kind} from a fileset requires a 'sdk: NeMoPlatform', but no "
            "platform SDK was available.  Set NMP_BASE_URL or pass sdk via "
            "NemoJobScheduler.run_local(sdk=...)."
        )

    with tempfile.TemporaryDirectory(prefix=f".{kind}-{name}-", dir=str(ctx.storage.ephemeral)) as tmp:
        tmp_path = Path(tmp)
        logger.info("Downloading fileset %s/%s into %s for %s.", ws, name, tmp_path, kind)
        files_client = client_from_platform(sdk, FilesClient)
        manager = _fileset_manager(files_client, workspace=ws, fileset=name, ensure_fileset_exists=False)
        manager.download_from_url(f"{ws}/{name}", local_dir=tmp_path)
        # ``config_rel_path`` is caller-controlled — confirm it stays inside the
        # downloaded fileset so an absolute path or ``..`` segment can't make the
        # subprocess read arbitrary files from the task host.
        root = tmp_path.resolve()
        local_config = (tmp_path / config_rel_path).resolve()
        if not local_config.is_relative_to(root):
            raise ValueError(f"{kind} {config_rel_path!r} resolves outside the downloaded fileset")
        if not local_config.is_file():
            raise FileNotFoundError(
                f"{kind} '{config_rel_path}' was not found in fileset '{ws}/{name}' after download.  "
                f"Available files: {sorted(p.name for p in tmp_path.iterdir())}"
            )
        yield local_config


@contextlib.contextmanager
def resolve_output(
    output: OutputTarget | None,
    *,
    workspace: str,
    ctx: JobContext,
    sdk: NeMoPlatform | None,
    kind: str,
) -> Iterator[Path]:
    """Yield a local base directory for job outputs, uploading to a fileset on success.

    Branches on the union shape:

    - ``None`` → ``ctx.storage.persistent / "results"`` (no upload).
    - :class:`LocalDir` → that directory, ``mkdir -p``-ed (no upload).
    - :class:`FilesetRef` → a fresh tempdir under ``ctx.storage.ephemeral``;
      on a clean exit the tempdir contents are uploaded to the named fileset
      (auto-created if missing).  A body that raises skips the upload so
      partial/broken runs don't pollute the fileset.
    """
    if output is None:
        local = ctx.storage.persistent / "results"
        local.mkdir(parents=True, exist_ok=True)
        logger.info("Writing %s outputs to platform-persistent dir %s", kind, local)
        yield local
        return

    if classify_output_target(output) is LocalDir:
        local = Path(str(output)).expanduser().resolve()
        local.mkdir(parents=True, exist_ok=True)
        logger.info("Writing %s outputs to local dir %s", kind, local)
        yield local
        return

    ref = FilesetRef(output)
    ws, name = split_fileset_ref(ref, workspace)

    if sdk is None:
        raise LocalRunError(
            f"Uploading {kind} results to a fileset requires a 'sdk: NeMoPlatform', but no "
            "platform SDK was available.  Set NMP_BASE_URL, pass sdk via "
            "NemoJobScheduler.run_local(sdk=...), or use a local output directory instead."
        )

    with tempfile.TemporaryDirectory(prefix=f".{kind}-output-{name}-", dir=str(ctx.storage.ephemeral)) as tmp:
        tmp_path = Path(tmp)
        logger.info("Staging %s outputs in %s; will upload to fileset %s/%s on success.", kind, tmp_path, ws, name)
        should_upload = True
        try:
            yield tmp_path
        except BaseException:
            should_upload = False
            raise
        finally:
            if should_upload:
                upload_to_fileset(tmp_path, fileset=name, workspace=ws, sdk=sdk)


def upload_to_fileset(local_dir: Path, *, fileset: str, workspace: str, sdk: NeMoPlatform) -> None:
    """Upload *local_dir*'s contents recursively to the named fileset (auto-created)."""
    files_client = client_from_platform(sdk, FilesClient)
    manager = _fileset_manager(files_client, workspace=workspace, fileset=fileset, ensure_fileset_exists=True)
    manager.validate_storage()
    manager.upload(local_path=local_dir, remote_path="")
    logger.info("Uploaded outputs from %s to fileset %s/%s.", local_dir, workspace, fileset)
