# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-facing handle for publishing job results.

Single sync API — there is no async twin. ``NemoJob.run`` runs in the
task container where there is no event loop and most work calls into
sync library protocols, so ``ctx.results.save(...)`` is sync.

Concrete impls living in this codebase:

- :class:`LocalJobResults` — copies the artefact under a local directory
  rooted at ``<persistent>/results/``. Used by programmatic callers and
  any context where no NeMo Helix Files / Jobs SDK is configured.
- :class:`HelixJobResults` — thin adapter over
  :class:`nemo_helix_plugin.jobs.result_manager.ResultManager` that registers
  results as platform artefacts.

The API is deliberately narrow: a single ``save(name, local_path, *,
ignore_patterns=None) -> ResultRef``. Read paths (``get`` / ``list``)
are not on the Protocol — task code that needs to enumerate results
goes through the typed Jobs client.
"""

from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Protocol

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.constants import NEMO_JOB_ARTIFACT_BASE_PATH_ENVVAR, NEMO_JOB_FILESET_ENVVAR
from nemo_helix_plugin.jobs.ignore_patterns import _IgnoreRules, compile_ignore_rules
from nemo_helix_plugin.jobs.schemas import HelixJobResultResponse
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


class _ResultManager(Protocol):
    def create_result(
        self,
        result_name: str,
        artifact_local_path: str | Path,
        ignore_patterns: list[str] | str | None = None,
    ) -> HelixJobResultResponse: ...


def result_manager_factory(
    job_name: str,
    *,
    attempt_id: str,
    files_client: FilesClient,
    fileset_name: str,
    workspace: str | None = None,
    jobs_client: JobsClient | None = None,
    artifact_base_path: str | None = None,
) -> _ResultManager:
    from nemo_helix_plugin.jobs.result_manager import result_manager_factory as create_result_manager

    return create_result_manager(
        job_name=job_name,
        attempt_id=attempt_id,
        workspace=workspace,
        files_client=files_client,
        jobs_client=jobs_client,
        fileset_name=fileset_name,
        artifact_base_path=artifact_base_path,
    )


def job_storage_context_from_env() -> tuple[str, str | None]:
    """Fileset and artifact base path for the job this task serves.

    Task entrypoints that build their own context read the same platform env as
    ``build_ctx_from_env``. The result sink cannot read the job record for this,
    so the values have to come from here.
    """
    fileset_name = os.environ.get(NEMO_JOB_FILESET_ENVVAR, "").strip()
    if not fileset_name:
        raise RuntimeError(f"{NEMO_JOB_FILESET_ENVVAR} not set; running outside the platform?")
    return fileset_name, os.environ.get(NEMO_JOB_ARTIFACT_BASE_PATH_ENVVAR, "").strip() or None


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

    Directories are copied recursively; ``ignore_patterns`` uses Git ignore
    syntax, as hosted uploads do, so callers can skip cache files (matches
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
        # Validate patterns before _copy clears the previous result.
        rules = compile_ignore_rules(ignore_patterns)
        self._root.mkdir(parents=True, exist_ok=True)
        dst = self._root / name
        if dst.resolve() != src.resolve():
            _copy(src, dst, rules)
        artifact_url = f"file://{dst.resolve()}"
        logger.info("Saved result %r to %s", name, artifact_url)
        return ResultRef(name=name, artifact_url=artifact_url)


def _copy(src: Path, dst: Path, rules: _IgnoreRules | None) -> None:
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
        shutil.copytree(src, dst, ignore=_make_ignore(rules, src))
    else:
        shutil.copyfile(src, dst)


def _make_ignore(rules: _IgnoreRules | None, root: Path) -> Callable[[str, list[str]], set[str]] | None:
    """Build a copytree callback using the hosted uploader's root-relative matching rules."""
    if rules is None:
        return None
    matcher = rules  # The closure sees the narrowed type.

    def ignore(directory: str, names: list[str]) -> set[str]:
        """Prune matching directories and exclude matching files at their bundle-relative paths."""
        return {
            name
            for name in names
            if matcher.excludes(
                PurePosixPath(Path(directory, name).relative_to(root).as_posix()),
                is_directory=Path(directory, name).is_dir(),
            )
        }

    return ignore


class HelixJobResults(JobResults):
    """:class:`JobResults` backed by the platform Files + Jobs services.

    Wraps :class:`~nemo_helix_plugin.jobs.result_manager.ResultManager` — every
    :meth:`save` call forwards to ``ResultManager.create_result``, which
    uploads into the job's fileset under
    ``results/<attempt_id>/<result_name>`` and registers the result via
    ``JobsClient.create_job_result`` (conflict-idempotent).

    Directory uploads attempt every nonignored file after ordinary file-upload
    failures, retaining successful uploads. Failures are raised after all files
    are attempted (as an ``ExceptionGroup`` for multiple failures), so a partial
    upload is not registered as a complete result. Cancellation stops immediately.

    Args:
        job_name: Platform job name this sink publishes results for.
        workspace: Workspace the job lives in.
        client: Typed root client used for both file uploads and the jobs-results
            registration.
        attempt_id: Concrete platform attempt that produces these results.
        fileset_name: Fileset holding the job's artifacts.
        artifact_base_path: Folder inside ``fileset_name`` the job's artifacts nest under;
            ``None`` means the job owns the whole fileset.
    """

    def __init__(
        self,
        *,
        job_name: str,
        workspace: str,
        client: NemoClient,
        attempt_id: str,
        fileset_name: str,
        artifact_base_path: str | None = None,
    ) -> None:
        self._manager: _ResultManager
        self._configure_manager(
            job_name=job_name,
            workspace=workspace,
            client=client,
            attempt_id=attempt_id,
            fileset_name=fileset_name,
            artifact_base_path=artifact_base_path,
        )

    def _configure_manager(
        self,
        *,
        job_name: str,
        workspace: str,
        client: NemoClient,
        attempt_id: str,
        fileset_name: str,
        artifact_base_path: str | None,
    ) -> None:
        self._manager = result_manager_factory(
            job_name=job_name,
            workspace=workspace,
            attempt_id=attempt_id,
            files_client=FilesClient.from_client(client),
            jobs_client=JobsClient.from_client(client),
            fileset_name=fileset_name,
            artifact_base_path=artifact_base_path,
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
    "HelixJobResults",
    "ResultRef",
    "job_storage_context_from_env",
]
