# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-facing handle for publishing job results.

Single sync API — there is no async twin. ``NemoJob.run`` runs in the
task container where there is no event loop and most work calls into
sync library protocols, so ``ctx.results.save(...)`` is sync.

Concrete impls living in this codebase:

- :class:`LocalJobResults` — copies the artefact under a local directory
  rooted at ``<persistent>/results/``. Used for laptop ``run_local`` and
  any context where no NeMo Platform Files / Jobs SDK is configured.
- :class:`PlatformJobResults` — thin adapter over
  :class:`nemo_platform_plugin.jobs.result_manager.ResultManager` that registers
  results as platform artefacts.

The API is deliberately narrow: a single ``save(name, local_path, *,
ignore_patterns=None) -> ResultRef``. Read paths (``get`` / ``list``)
are not on the Protocol — task code that needs to enumerate results
goes through the typed Jobs client.
"""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.jobs.client import JobsClient
from nemo_platform_plugin.jobs.result_manager import result_manager_factory
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ResultRef(BaseModel):
    """Pointer to a saved result, returned by every ``save`` call.

    Attributes:
        name: Logical result name within the job.
        artifact_url: Where the bytes live. ``"file://..."`` locally,
            ``"fileset://..."`` on the platform.
    """

    name: str
    artifact_url: str


class JobResults(ABC):
    """Task-facing API for publishing results."""

    @abstractmethod
    def save(
        self,
        name: str,
        local_path: str | Path,
        *,
        ignore_patterns: list[str] | str | None = None,
    ) -> ResultRef: ...


class LocalJobResults(JobResults):
    """Sync :class:`JobResults` that copies artefacts under a local directory.

    Layout::

        <root>/
            <result-name-1>       # file or directory
            <result-name-2>
            ...

    ``save`` accepts ``local_path == <root>/<name>`` as a no-op so a
    job that already wrote into ``ctx.storage.persistent / "results"``
    can register the path without copying.

    Directories are copied recursively; ``ignore_patterns`` is forwarded
    to :func:`shutil.copytree` so callers can skip cache files (matches
    the evaluator's ``ignore_patterns=["cache.db", "cache/"]`` usage).
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def save(
        self,
        name: str,
        local_path: str | Path,
        *,
        ignore_patterns: list[str] | str | None = None,
    ) -> ResultRef:
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Cannot save result {name!r}: {src} does not exist.")
        self._root.mkdir(parents=True, exist_ok=True)
        dst = self._root / name
        if dst.resolve() != src.resolve():
            _copy(src, dst, ignore_patterns)
        artifact_url = f"file://{dst.resolve()}"
        logger.info("Saved result %r to %s", name, artifact_url)
        return ResultRef(name=name, artifact_url=artifact_url)


def _copy(src: Path, dst: Path, ignore_patterns: list[str] | str | None) -> None:
    """Blocking file / directory copy used by :class:`LocalJobResults`.

    Module-level so tests can exercise it directly when useful. Any
    existing ``dst`` is cleared first — regardless of whether it's a
    file or a directory — so the documented overwrite-by-name contract
    holds when the artifact kind changes between :meth:`save` calls.
    """
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        ignore = _make_ignore(ignore_patterns)
        shutil.copytree(src, dst, ignore=ignore)
    else:
        shutil.copyfile(src, dst)


def _make_ignore(patterns: list[str] | str | None):
    """Build a ``shutil.copytree`` ignore callable from ``ignore_patterns``.

    Mirrors the shape the file manager accepts (``list[str] | str | None``)
    so the local and platform impls take identical kwargs at call sites.
    Trailing ``/`` and ``\\`` are stripped so directory-style patterns
    like ``"cache/"`` match a directory named ``cache`` —
    :func:`shutil.ignore_patterns` matches via :mod:`fnmatch` over bare
    child names, where a trailing slash is a literal character that
    would otherwise never match.
    """
    if patterns is None:
        return None
    if isinstance(patterns, str):
        patterns = [patterns]
    normalized = [pattern.rstrip("/\\") for pattern in patterns]
    return shutil.ignore_patterns(*normalized)


class PlatformJobResults(JobResults):
    """:class:`JobResults` backed by the platform Files + Jobs services.

    Wraps :class:`~nemo_platform_plugin.jobs.result_manager.ResultManager` — every
    :meth:`save` call forwards to ``ResultManager.create_result``, which
    uploads into the job's fileset under
    ``results/<attempt_id>/<result_name>`` and registers the result via
    ``jobs.results.create`` (conflict-idempotent).

    Args:
        job_name: Platform job name this sink publishes results for.
        workspace: Workspace the job lives in.
        client: Typed root client used for both file uploads and the jobs-results
            registration.
        attempt_id: Optional override for the job attempt id; when
            omitted, looked up lazily via the typed Jobs client.
    """

    def __init__(
        self,
        *,
        job_name: str,
        workspace: str,
        client: NemoClient,
        attempt_id: str | None = None,
    ) -> None:
        self._configure_manager(
            job_name=job_name,
            workspace=workspace,
            client=client,
            attempt_id=attempt_id,
        )

    def _configure_manager(
        self,
        *,
        job_name: str,
        workspace: str,
        client: NemoClient,
        attempt_id: str | None,
    ) -> None:
        self._manager = result_manager_factory(
            job_name=job_name,
            workspace=workspace,
            attempt_id=attempt_id,
            files_client=FilesClient.from_client(client),
            jobs_client=JobsClient.from_client(client),
        )
        self._job_name = job_name
        self._workspace = workspace

    def save(
        self,
        name: str,
        local_path: str | Path,
        *,
        ignore_patterns: list[str] | str | None = None,
    ) -> ResultRef:
        record = self._manager.create_result(
            result_name=name,
            artifact_local_path=local_path,
            ignore_patterns=ignore_patterns,
        )
        logger.info(
            "Saved result %r to %s for job %r in workspace %r",
            name,
            record.artifact_url,
            self._job_name,
            self._workspace,
        )
        return ResultRef(name=record.name, artifact_url=record.artifact_url)


__all__ = [
    "JobResults",
    "LocalJobResults",
    "PlatformJobResults",
    "ResultRef",
]
