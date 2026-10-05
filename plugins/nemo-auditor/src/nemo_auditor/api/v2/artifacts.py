# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate artifact download for audit jobs.

Registered before the generic ``/{name}/download`` catch-all so FastAPI
matches this specific path first. Fetches the individual garak report
results already stored by the job and streams them back as a single
``artifacts.tar.gz`` — no additional storage.
"""

from __future__ import annotations

import logging
import tarfile
import tempfile
from pathlib import Path

import anyio.to_thread
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from nemo_auditor.authz import scope
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.result_manager import async_result_manager_factory

logger = logging.getLogger(__name__)

router = APIRouter()

_AUDIT_SCOPE = scope.child("audit")
_AUDIT_READ_PERMISSION = _AUDIT_SCOPE.permission(
    "read",
    description="Read auditor.audit jobs, including status, logs, and results",
)

# The individual result names published by AuditJob, in preferred archive order.
_REPORT_RESULT_NAMES = ("report-jsonl", "report-html", "report-hitlog-jsonl")


@router.get(
    "/jobs/audit/{job}/results/artifacts/download",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Aggregate gzip archive of all garak report artifacts.",
            "content": {"application/gzip": {"schema": {"type": "string", "format": "binary"}}},
        },
        404: {"description": "No report artifacts found for the job."},
    },
)
@_AUDIT_SCOPE.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[_AUDIT_READ_PERMISSION])
async def download_audit_artifacts(
    workspace: str,
    job: str,
    background_tasks: BackgroundTasks,
    client: AsyncNemoClient = Depends(get_nemo_client),
) -> FileResponse:
    """Stream an aggregate tar.gz of all garak report artifacts for an audit job."""
    jobs_client = AsyncJobsClient.from_client(client)
    files_client = AsyncFilesClient.from_client(client)
    result_manager = async_result_manager_factory(
        job_name=job,
        workspace=workspace,
        files_client=files_client,
        jobs_client=jobs_client,
    )

    tmp_dir = tempfile.TemporaryDirectory()
    artifact_tmps = []
    try:
        tmp = Path(tmp_dir.name)

        for result_name in _REPORT_RESULT_NAMES:
            try:
                result_info = (await jobs_client.get_job_result(name=result_name, job=job, workspace=workspace)).data()
            except NotFoundError:
                continue

            artifact_tmps.append(await result_manager.download_artifact(artifact_url=result_info.artifact_url))

        if not artifact_tmps:
            raise HTTPException(
                status_code=404,
                detail=f"No report artifacts found for audit job '{job}' in workspace '{workspace}'.",
            )

        tar_path = tmp / "artifacts.tar.gz"

        def _create_archive() -> None:
            with tarfile.open(tar_path, "w:gz") as tar:
                for artifact_tmp in artifact_tmps:
                    tar.add(artifact_tmp.path, arcname=artifact_tmp.path.name)

        await anyio.to_thread.run_sync(_create_archive)

        def _cleanup():
            for t in artifact_tmps:
                t.cleanup_tmp_dir()
            tmp_dir.cleanup()

        background_tasks.add_task(_cleanup)
        return FileResponse(path=str(tar_path), filename="artifacts.tar.gz", media_type="application/gzip")
    except Exception:
        for t in artifact_tmps:
            t.cleanup_tmp_dir()
        tmp_dir.cleanup()
        raise
