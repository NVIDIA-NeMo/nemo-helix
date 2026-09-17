# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime context handed to ``NemoJob.run()``.

Carries the non-client bits a job needs to execute: workspace, job id
(when one exists), filesystem paths, and a results sink. Each field
maps onto an existing ``NEMO_JOB_*`` environment variable that the
in-container runtime sets:

- ``workspace``         ← ``NEMO_JOB_WORKSPACE``
- ``job_id``            ← ``NEMO_JOB_ID`` (``None`` for local runs)
- ``storage.ephemeral``  ← ``NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH``
- ``storage.persistent`` ← ``NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH``

Single sync class — there is no async twin. ``NemoJob.run`` runs in the
task container where there is no event loop and most work calls into
sync library protocols; ``ctx.results.save(...)`` is therefore sync as
well. This matches the shared resources/jobs/functions design rationale.

Clients (files, models, ...) reach typed jobs via the runtime contract they
inherit, not through the context. Logging is not on the context either — use
``logging.getLogger(__name__)`` in each task module.
Progress reporting is a virtual method on
:class:`~nemo_platform_plugin.job.NemoJob` so each job can ship its own payload
shape.

Example::

    def run(self, config: dict, *, ctx: JobContext, client: NemoClient) -> dict:
        spec = MySpec.model_validate(config)
        out_path = ctx.storage.ephemeral / "rows.jsonl"
        ...
        ref = ctx.results.save("rows.jsonl", out_path)
        self.report_progress(ctx, status="done")
        return {"status": "completed", "result": ref.model_dump()}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nemo_platform_plugin.job_results import JobResults


class StoragePaths:
    """Filesystem locations a job can read and write during execution.

    Attributes:
        ephemeral: Scratch directory for working files and intermediate
            artifacts. No guarantees across steps or retries. Maps to
            ``NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH``.
        persistent: PVC-backed directory shared across steps within the
            same job. Only available when the job step declares
            ``NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH`` in its compile()
            environment. Raises ``RuntimeError`` if accessed without
            being provisioned. Maps to
            ``NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH``.
    """

    def __init__(self, ephemeral: Path, persistent: Path | None = None) -> None:
        self.ephemeral = ephemeral
        self._persistent = persistent

    @property
    def persistent(self) -> Path:
        """Return the persistent storage path, or raise if not provisioned."""
        if self._persistent is None:
            raise RuntimeError(
                "This job did not request persistent storage. "
                "Add NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH to the step's "
                "environment list in compile() to enable it, or use "
                "ctx.storage.ephemeral for scratch data."
            )
        return self._persistent


@dataclass(kw_only=True)
class JobContext:
    """Runtime context for :class:`~nemo_platform_plugin.job.NemoJob.run`.

    Attributes:
        workspace: Workspace scope the job runs in.
        storage: Scratch and persistent filesystem paths.
        results: Sink for publishing results (local directory for
            laptop runs, NeMo Platform fileset on the platform).
        job_id: Platform job UUID, or ``None`` for a local run.
    """

    workspace: str
    storage: StoragePaths
    results: JobResults
    job_id: str | None = None


__all__ = [
    "JobContext",
    "StoragePaths",
]
