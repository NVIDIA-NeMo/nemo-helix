# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``eval-author.first-eval`` -- author a platform agent's first evaluation suite.

One Fabric invocation: a Deep Agents author carrying the vendored NeMo Eval Author skills is
handed the target agent's stored config (and its Ethos fileset, when one exists) as the
agent's repository and asked to run the ``eval-author-first-eval`` skill non-interactively.
Everything it writes under ``.eval-author/`` plus ``ETHOS.md`` becomes the job's
``eval_author`` result and is uploaded to the output fileset.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, ClassVar

import yaml
from filesets import FilesetFileSystem
from nemo_agents_plugin.entities import ETHOS_FILENAME, ethos_fileset_name
from nemo_agents_plugin.jobs.fileset_io import resolve_staged_config, split_fileset_ref, upload_to_fileset
from nemo_eval_author_plugin.runner import build_author_agent, build_task_input, run_author, stage_skills
from nemo_eval_author_plugin.schemas.first_eval import FirstEvalSpec
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import (
    InternalServerError,
    NemoResponseValidationError,
    NemoTransportError,
    NotFoundError,
)
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    ExecutorSpec,
    HelixJobSpec,
    HelixJobStep,
    SubprocessExecutionProviderSpec,
)
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError, HelixJobDependencyUnavailableError
from nemo_helix_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from nemo_helix_plugin.jobs.file_manager import FilesetFileManager
from nemo_helix_plugin.jobs.image import get_qualified_image
from nemo_helix_plugin.refs import parse_entity_ref
from pydantic import BaseModel

logger = logging.getLogger(__name__)

JOB_NAME = "first-eval"

#: Task entry point for the step, in both executor flavours.  The subprocess backend takes one
#: flat command; the cpu backend splits it into a container entrypoint + command.
TASK_MODULE = "nemo_eval_author_plugin.tasks.first_eval"
TASK_ENTRYPOINT = ["python", "-m"]
TASK_COMMAND = [TASK_MODULE]
TASK_IMAGE = "nhx-tasks"

RESULT_NAME = "eval_author"
SUMMARY_FILENAME = "eval-author-result.json"
EVAL_AUTHOR_DIR = ".eval-author"
AGENT_CONFIG_FILENAME = "agent.yaml"


class FirstEvalJob(NemoJob):
    """Author a platform agent's first evaluation suite with the bundled NeMo Eval Author skill."""

    name: ClassVar[str] = JOB_NAME
    description: ClassVar[str] = "Author a platform agent's first evaluation suite with NeMo Eval Author."
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = FirstEvalSpec

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]  (narrows the spec types)
        cls,
        *,
        workspace: str,
        spec: FirstEvalSpec,
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncNemoClient,
        profile: str | None = None,
        options: dict | None = None,
    ) -> HelixJobSpec:
        del entity_client, job_name, options
        config = spec.model_dump(mode="json")
        config["workspace"] = workspace
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name=JOB_NAME,
                    executor=await _resolve_executor(profile=profile or "default", async_sdk=async_sdk),
                    config=config,
                ),
            ],
        )

    def run(self, config: dict, *, ctx: JobContext, sdk: NemoClient) -> dict[str, Any]:
        spec = FirstEvalSpec.model_validate(config)
        author = build_author_agent(_author_overrides(spec, ctx=ctx, sdk=sdk), workspace=spec.workspace)

        agent_ref = parse_entity_ref(spec.agent, default_workspace=spec.workspace)
        agent_label = f"{agent_ref.workspace}/{agent_ref.name}"
        source = fetch_agent_config(sdk, workspace=agent_ref.workspace, name=agent_ref.name)

        base_dir = ctx.storage.ephemeral / "eval-author" / "fabric"
        workspace_dir = base_dir / author.environment.workspace
        shutil.rmtree(workspace_dir, ignore_errors=True)
        stage_skills(workspace_dir)
        agent_fileset = _agent_fileset(spec, sdk, workspace=agent_ref.workspace, agent_name=agent_ref.name)
        if agent_fileset is not None:
            _download_fileset(sdk, *agent_fileset, into=workspace_dir)
        # The stored config is authoritative; it overrides whatever agent.yaml the fileset carried.
        (workspace_dir / AGENT_CONFIG_FILENAME).write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

        logger.info(
            "Authoring first evals for agent %s with author model %s", agent_label, author.models["default"].model
        )
        response = run_author(author, input=build_task_input(agent_label), base_dir=base_dir)
        logger.info("Author response: %s", response)

        output_ws, output_name = (
            split_fileset_ref(spec.output, spec.workspace)
            if spec.output is not None
            else (agent_ref.workspace, f"{agent_ref.name}-evals")
        )
        results_dir = ctx.storage.ephemeral / "eval-author" / "results"
        files = collect_results(workspace_dir, results_dir)
        summary = {
            "agent": agent_label,
            "author_model": author.models["default"].model,
            "output_fileset": f"{output_ws}/{output_name}",
            "files": files,
            "response": response,
        }
        (results_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        result_ref = ctx.results.save(RESULT_NAME, results_dir)
        upload_to_fileset(results_dir, fileset=output_name, workspace=output_ws, sdk=sdk)
        return {
            "status": "completed",
            "agent": agent_label,
            "output_fileset": summary["output_fileset"],
            "files": files,
            "result": result_ref.model_dump(mode="json"),
        }


def _author_overrides(spec: FirstEvalSpec, *, ctx: JobContext, sdk: NemoClient) -> dict[str, Any] | None:
    if spec.author_config is None:
        return None
    with resolve_staged_config(
        spec.author_config,
        spec.author_config_fileset,
        workspace=spec.workspace,
        ctx=ctx,
        sdk=sdk,
        kind="eval-author-config",
    ) as config_path:
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def fetch_agent_config(sdk: NemoClient, *, workspace: str, name: str) -> dict[str, Any]:
    try:
        return client_from_platform(sdk, AgentsClient).get_agent(name=name, workspace=workspace).data().config
    except NotFoundError as exc:
        raise LocalRunError(
            f"Agent '{workspace}/{name}' does not exist; there is nothing to author evals for."
        ) from exc


def _agent_fileset(spec: FirstEvalSpec, sdk: NemoClient, *, workspace: str, agent_name: str) -> tuple[str, str] | None:
    """``(workspace, name)`` of the fileset staged as the agent's repository, or ``None`` for none."""
    if spec.agent_fileset is not None:
        return split_fileset_ref(spec.agent_fileset, spec.workspace)
    name = ethos_fileset_name(agent_name)
    try:
        client_from_platform(sdk, FilesClient).get_fileset(name=name, workspace=workspace)
    except NotFoundError:
        logger.info("Agent %s/%s has no %s fileset; authoring from its config alone.", workspace, agent_name, name)
        return None
    return workspace, name


def _download_fileset(sdk: NemoClient, workspace: str, name: str, *, into: Path) -> None:
    manager = FilesetFileManager(
        workspace=workspace,
        fileset_name=name,
        filesystem=FilesetFileSystem(client=client_from_platform(sdk, FilesClient)),
        ensure_fileset_exists=False,
    )
    logger.info("Staging fileset %s/%s into %s", workspace, name, into)
    manager.download_from_url(f"{workspace}/{name}", local_dir=into)


def collect_results(workspace_dir: Path, results_dir: Path) -> list[str]:
    """Copy ``.eval-author/**`` and ``ETHOS.md`` from *workspace_dir* into a fresh *results_dir*."""
    produced = workspace_dir / EVAL_AUTHOR_DIR
    if not produced.is_dir():
        raise LocalRunError(
            f"The author agent produced no {EVAL_AUTHOR_DIR}/ directory; nothing to save. "
            "Check the job log for the agent's final response."
        )
    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.copytree(produced, results_dir / EVAL_AUTHOR_DIR)
    ethos = workspace_dir / ETHOS_FILENAME
    if ethos.is_file():
        shutil.copy2(ethos, results_dir / ETHOS_FILENAME)
    return sorted(str(path.relative_to(results_dir)) for path in results_dir.rglob("*") if path.is_file())


async def _resolve_executor(*, profile: str, async_sdk: AsyncNemoClient) -> ExecutorSpec:
    """Pick the executor for *profile* from the backends the platform actually registered.

    ``subprocess`` is preferred: the one-shot Fabric run needs a venv carrying the Deep Agents
    harness adapter, which the platform host's own environment has.  Deployments that register
    no subprocess backend get the ``cpu`` provider, which maps to their container backend.
    """
    try:
        profiles = (await client_from_platform(async_sdk, AsyncJobsClient).get_execution_profiles()).data()
    except (NemoTransportError, NemoResponseValidationError, InternalServerError) as exc:
        raise HelixJobDependencyUnavailableError(
            f"Unable to resolve execution profile '{profile}': the Jobs service is temporarily "
            "unavailable.  Retry the submission."
        ) from exc

    if any(
        isinstance(candidate, SubprocessJobExecutionProfile) and candidate.profile == profile for candidate in profiles
    ):
        return SubprocessExecutionProviderSpec(
            provider="subprocess",
            profile=profile,
            command=[*TASK_ENTRYPOINT, *TASK_COMMAND],
        )

    if any(candidate.provider == "cpu" and candidate.profile == profile for candidate in profiles):
        return CPUExecutionProviderSpec(
            provider="cpu",
            profile=profile,
            container=ContainerSpec(
                image=get_qualified_image(TASK_IMAGE),
                entrypoint=TASK_ENTRYPOINT,
                command=TASK_COMMAND,
            ),
        )

    available = sorted({f"{candidate.provider}/{candidate.profile}" for candidate in profiles})
    raise HelixJobCompilationError(
        f"No 'subprocess' or 'cpu' execution profile named {profile!r} is registered, so the "
        f"first-eval step has nowhere to run.  Available profiles: {available or ['<none>']}."
    )
