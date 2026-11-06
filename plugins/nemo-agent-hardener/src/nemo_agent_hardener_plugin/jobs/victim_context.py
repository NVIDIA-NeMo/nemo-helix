# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``agent-hardener.victim-context`` job — stage a saved manifest's victim build context for an outside build.

A job that runs in a container (Helm) has no Docker to build the victim, and OpenShell only creates a sandbox
from an image the cluster can pull. So the operator builds it: this job runs ``agent-hardener stage`` on the
frozen manifest and saves the context as a ``.tar.gz`` result, which ``docker build -t <ref> - < archive``
consumes as-is. The pushed ``<ref>`` then goes on the manifest as ``victim_image``.
"""

from __future__ import annotations

import logging
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.jobs import _common
from nemo_agent_hardener_plugin.jobs.errors import (
    AGENT_HARDENER_ERROR_FILE_ENVVAR,
    CATEGORY_MANIFEST,
    AgentHardenerRunError,
    classify_exception,
    classify_subprocess,
    read_run_error,
)
from nemo_agent_hardener_plugin.jobs.execution import _build_native_subprocess_env
from nemo_agent_hardener_plugin.jobs.manifest import _materialize_manifest
from nemo_helix_plugin.agent_hardener.types import VictimContextSpec
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import HelixJobSpec, HelixJobStep
from pydantic import BaseModel

logger = logging.getLogger(__name__)

#: Job result name of the staged context archive.
CONTEXT_RESULT = "victim-context"
_LOG_TAIL = 4000


class AgentHardenerVictimContextJob(NemoJob):
    """Stage a saved manifest's victim build context as a ``docker build``-ready archive."""

    name = "victim-context"  # keeps the hand-written `nemo agent-hardener stage` command unshadowed
    job_collection_path = "/victim-context/jobs"
    description = "Stage a saved manifest's victim build context for building the victim image outside the job."
    container = "cpu-tasks"
    spec_schema: ClassVar[type[BaseModel] | None] = VictimContextSpec

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,  # VictimContextSpec
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> HelixJobSpec:
        """A single step running the stage task, placed like the war-game's (and needing no victim image)."""
        del workspace, entity_client, job_name, options
        stage = VictimContextSpec.model_validate(spec)
        executor, environment = await _common.task_step_executor(
            "nemo_agent_hardener_plugin.tasks.victim_context", step="stage", profile=profile, async_sdk=async_sdk
        )
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="stage", executor=executor, config=stage.model_dump(mode="json"), environment=environment
                )
            ],
        )

    def run(self, config: dict, *, ctx: JobContext, sdk: Any = None) -> dict:
        """Stage the context, classifying any failure into an operator-facing error result."""
        spec = VictimContextSpec.model_validate(config)
        return _staged_result(lambda: self._execute(spec, ctx=ctx, sdk=sdk))

    def _execute(self, spec: VictimContextSpec, *, ctx: JobContext, sdk: Any) -> Path:
        plugin_config = AgentHardenerConfig.get()
        _common.require_provisioned(plugin_config)
        manifest = _materialize_manifest(sdk, spec.manifest_id, ctx)
        return _stage_archive(plugin_config, manifest, spec.manifest_id, ctx)


def stage_local_manifest(manifest: Path, ctx: JobContext) -> dict:
    """Stage a local ``agent-hardener.yaml`` the way the job stages a saved manifest, returning the same result."""

    def execute() -> Path:
        plugin_config = AgentHardenerConfig.get()
        _common.require_provisioned(plugin_config)
        return _stage_archive(plugin_config, str(manifest.resolve()), manifest.stem, ctx)

    return _staged_result(execute)


def _staged_result(execute: Callable[[], Path]) -> dict:
    try:
        archive = execute()
    except Exception as exc:
        failure = classify_exception(exc)
        logger.exception("agent-hardener stage failed [%s]: %s", failure.category, failure.message)
        return {
            "status": "failed",
            "returncode": 1,
            "error": {"category": failure.category, "message": failure.message, "remediation": failure.remediation},
        }
    return {"status": "completed", "returncode": 0, "context_archive": str(archive)}


def _stage_archive(plugin_config: AgentHardenerConfig, manifest: str, name: str, ctx: JobContext) -> Path:
    """Stage *manifest*'s victim build context and save it as the ``<name>-victim-context.tar.gz`` result."""
    context_dir = _stage(plugin_config, manifest, ctx)
    archive = ctx.storage.persistent / f"{name}-{CONTEXT_RESULT}.tar.gz"
    _archive_context(context_dir, archive)
    ctx.results.save(CONTEXT_RESULT, archive)
    return archive


def _stage(plugin_config: AgentHardenerConfig, manifest: str, ctx: JobContext) -> Path:
    """Run ``agent-hardener stage`` on *manifest* and return the context directory it printed.

    Captured rather than run through :func:`_common.execute`, which hands a terminal to the child: the
    directory is read from stdout, and staging never prompts.
    """
    workdir = ctx.storage.persistent / "stage"
    workdir.mkdir(parents=True, exist_ok=True)
    error_file = ctx.storage.persistent / "stage-error.json"
    error_file.unlink(missing_ok=True)
    env = {**_build_native_subprocess_env(plugin_config), AGENT_HARDENER_ERROR_FILE_ENVVAR: str(error_file)}
    completed = subprocess.run(
        [str(plugin_config.agent_hardener_bin), "stage", "--config", manifest],
        cwd=workdir,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path = ctx.storage.persistent / "stage.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    ctx.results.save("stage-log", log_path)
    if completed.returncode != 0:
        raise classify_subprocess(completed.returncode, completed.stderr[-_LOG_TAIL:], read_run_error(error_file))
    printed = completed.stdout.strip().splitlines()
    context_dir = Path(printed[-1]) if printed else None
    if context_dir is None or not (context_dir / "Dockerfile").is_file():
        raise AgentHardenerRunError(
            CATEGORY_MANIFEST,
            f"agent-hardener stage did not print a build context with a Dockerfile (got {context_dir}).",
        )
    return context_dir


def _archive_context(context_dir: Path, archive: Path) -> None:
    """Write *context_dir*'s contents to a gzipped tar rooted at the context, keeping file modes and symlinks.

    Everything is kept: the image has to match the staged context ``run`` re-derives and checks it against.
    """
    with tarfile.open(archive, "w:gz") as tar:
        for entry in sorted(context_dir.iterdir()):
            tar.add(entry, arcname=entry.name)
