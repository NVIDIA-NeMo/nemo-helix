# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract every agent optimization strategy implements.

A strategy is a job.  This base owns everything normalized — staging the
bundle, resolving the agent, registering the optimized agent — so a strategy
plugin implements exactly one method, :meth:`AgentOptimizeJob.optimize`.

Keep this module a leaf: ``discovery`` imports plugin job classes, which
import this, so importing ``discovery`` (or the router) from here would cycle.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, cast

import yaml
from nemo_agent_optimization_plugin.registration import register_optimized_agent
from nemo_agent_optimization_plugin.schemas.optimize import AgentOptimizeSpec
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import InternalServerError, NemoResponseValidationError, NemoTransportError
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    ExecutorSpec,
    PlatformJobSpec,
    PlatformJobStep,
    SubprocessExecutionProviderSpec,
)
from nemo_platform_plugin.jobs.client import AsyncJobsClient
from nemo_platform_plugin.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_platform_plugin.jobs.exceptions import (
    PlatformJobCompilationError,
    PlatformJobDependencyUnavailableError,
)
from nemo_platform_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from nemo_platform_plugin.jobs.image import get_qualified_image
from nemo_platform_plugin.run_dependencies import LocalRunError
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentOptimizeJob(NemoJob):
    """Base for every agent optimization strategy job."""

    # ``NemoJob``'s ``run`` is the only method the platform's ``_NamedPlugin``
    # metaclass tracks via ``abc.abstractmethod``; overriding it here makes this
    # class register as "fully concrete" the moment the class body executes,
    # which trips ``_NamedPlugin.__init_subclass__``'s "must define name"
    # check. ``optimize`` is deliberately *not* ``@abstractmethod`` (see its
    # docstring below), so it can't carry that abstractness forward instead.
    # A placeholder satisfies the check on this base; every concrete strategy
    # subclass sets its own ``name`` for real job/CLI/entry-point identity.
    name: ClassVar[str] = "agent_optimize_base"
    strategy: ClassVar[str]
    container: ClassVar[str] = "cpu-tasks"
    job_collection_path: ClassVar[str | None] = None
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = AgentOptimizeSpec

    #: Module run as ``python -m <task_module>`` inside the job's container or subprocess.
    #: Each strategy plugin ships one, because the task module names its own job class.
    task_module: ClassVar[str] = ""
    #: Image for the cpu (docker / kubernetes_job) fallback.
    task_image: ClassVar[str] = "nmp-cpu-tasks"

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]
        cls,
        *,
        workspace: str,
        spec: AgentOptimizeSpec,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        del entity_client, job_name, options
        if not cls.task_module:
            raise PlatformJobCompilationError(
                f"{cls.__name__} sets no 'task_module', so there is nothing to run remotely. "
                "Every AgentOptimizeJob must name the module its plugin ships as the task "
                "entry point, e.g. 'my_plugin.tasks.agent_optimize'."
            )
        spec_dict = spec.model_dump(mode="json")
        spec_dict["workspace"] = workspace
        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="agent-optimize",
                    executor=await _resolve_executor(
                        profile=profile or "default",
                        async_sdk=async_sdk,
                        task_module=cls.task_module,
                        task_image=cls.task_image,
                    ),
                    config=spec_dict,
                    environment=[
                        EnvironmentVariable(
                            name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
                            value=DEFAULT_JOB_STORAGE_PATH,
                        ),
                    ],
                ),
            ],
        )

    def run(self, config: dict, *, ctx: JobContext, sdk: NeMoPlatform | None = None) -> dict:
        spec = AgentOptimizeSpec.model_validate(config)
        if sdk is None:
            raise LocalRunError(
                "An agent optimization requires a platform SDK to fetch the agent it optimizes "
                "and to create the optimized one. Set NMP_BASE_URL or pass sdk via "
                "NemoJobScheduler.run_local(sdk=...)."
            )

        with _staged_bundle(spec, ctx=ctx, sdk=sdk) as (config_path, bundle_root):
            optimize_config = _load_yaml(config_path)
            source_agent_config = fetch_agent_config(spec.agent, workspace=spec.workspace, sdk=sdk)
            with _bundle_workdir(bundle_root):
                logger.info("Running agent optimization strategy %s", self.strategy)
                optimized = self.optimize(
                    source_agent_config=source_agent_config,
                    config=optimize_config,
                    ctx=ctx,
                    workspace=spec.workspace,
                    sdk=sdk,
                )

        return register_optimized_agent(
            optimized,
            name=spec.output_agent,
            source_agent_config=source_agent_config,
            workspace=spec.workspace,
            sdk=sdk,
        )

    def optimize(
        self,
        *,
        source_agent_config: dict[str, Any],
        config: dict[str, Any],
        ctx: JobContext,
        workspace: str,
        sdk: NeMoPlatform,
    ) -> dict[str, Any]:
        """Return an optimized ``nemo-agents-spec-v1`` config dict.

        ``source_agent_config`` is the stored config of the agent under test.
        Treat it as an input: copy it before mutating.  ``config`` is this
        strategy's own configuration, loaded from the staged bundle, and the
        process working directory is the bundle root while this runs.
        """
        raise NotImplementedError


def _profiles_unavailable(profile: str) -> PlatformJobDependencyUnavailableError:
    """A retryable failure while resolving the backend for *profile*."""
    return PlatformJobDependencyUnavailableError(
        f"Unable to resolve execution profile '{profile}': the Jobs service is temporarily "
        "unavailable.  Retry the submission."
    )


async def _fetch_execution_profiles(async_sdk: object) -> list:
    """The execution profiles the platform has registered.

    Split out from :func:`_resolve_executor` purely so tests can replace the network call
    without faking the whole SDK.  Kept module-level (not a nested closure) so
    ``monkeypatch.setattr(job_base, "_fetch_execution_profiles", ...)`` can intercept it.
    """
    return (
        await client_from_platform(cast(AsyncNeMoPlatform, async_sdk), AsyncJobsClient).get_execution_profiles()
    ).data()


async def _resolve_executor(*, profile: str, async_sdk: object, task_module: str, task_image: str) -> ExecutorSpec:
    """Pick the executor for *profile* from the backends the platform actually registered.

    Optimization prefers ``subprocess``: a study drives trials that may need the host's
    Docker daemon and a venv carrying the agent's harness adapters.  Deployments that do
    not register a subprocess backend (Helm / Minikube) get the ``cpu`` provider instead,
    which the platform maps to whichever backend it registered for that profile.
    """
    if async_sdk is None:
        raise _profiles_unavailable(profile)

    entrypoint = ["python", "-m"]
    command = [task_module]

    try:
        profiles = await _fetch_execution_profiles(async_sdk)
    except (NemoTransportError, NemoResponseValidationError, InternalServerError) as exc:
        raise _profiles_unavailable(profile) from exc

    if any(
        isinstance(candidate, SubprocessJobExecutionProfile) and candidate.profile == profile for candidate in profiles
    ):
        return SubprocessExecutionProviderSpec(provider="subprocess", profile=profile, command=[*entrypoint, *command])

    # Jobs keys execution profiles by (provider, profile); "cpu" is whatever container
    # backend the deployment registered under that name.
    if any(candidate.provider == "cpu" and candidate.profile == profile for candidate in profiles):
        return CPUExecutionProviderSpec(
            provider="cpu",
            profile=profile,
            container=ContainerSpec(image=get_qualified_image(task_image), entrypoint=entrypoint, command=command),
        )

    available = sorted({f"{candidate.provider}/{candidate.profile}" for candidate in profiles})
    raise PlatformJobCompilationError(
        f"No 'subprocess' or 'cpu' execution profile named {profile!r} is registered, so the "
        f"agent-optimize step has nowhere to run.  Available profiles: {available or ['<none>']}."
    )


def fetch_agent_config(agent: str, *, workspace: str, sdk: NeMoPlatform) -> dict[str, Any]:
    """Fetch a platform agent's stored ``nemo-agents-spec-v1`` config."""
    ws, _, name = agent.rpartition("/")
    ws = ws or workspace
    stored = sdk.agents.get(name, workspace=ws)
    config = stored["config"] if isinstance(stored, dict) else getattr(stored, "config", {})
    if not isinstance(config, dict) or not config:
        raise LocalRunError(f"Agent '{ws}/{name}' has an empty or invalid stored config.")
    logger.info("Resolved agent %r to %s/%s", agent, ws, name)
    return config


@contextlib.contextmanager
def _staged_bundle(spec: AgentOptimizeSpec, *, ctx: JobContext, sdk: NeMoPlatform) -> Iterator[tuple[Path, Path]]:
    """Yield ``(config path, bundle root)`` for the run."""
    # Soft dependency, mirroring registration.py's lazy imports.
    from nemo_agents_plugin.jobs.fileset_io import resolve_staged_config

    with resolve_staged_config(
        spec.config,
        spec.config_fileset,
        workspace=spec.workspace,
        ctx=ctx,
        sdk=sdk,
        kind="optimize-config",
    ) as config_path:
        yield config_path, _bundle_root_of(config_path, spec.config)


def _bundle_root_of(config_path: Path, config_rel_path: str) -> Path:
    """The download dir *config_rel_path* was resolved inside."""
    root = config_path
    for _ in PurePosixPath(config_rel_path).parts:
        root = root.parent
    return root


@contextlib.contextmanager
def _bundle_workdir(bundle_root: Path) -> Iterator[None]:
    """Run the strategy with *bundle_root* as the working directory.

    Relative paths in a strategy config are fileset-root-relative and are
    consumed in many places (datasets, Fabric ``base_dir``, hook paths, MCP
    config paths).  Moving the process resolves them all at once instead of
    chasing every schema that can hold a path.  The task subprocess runs
    exactly one job, so the process-global chdir is contained.
    """
    previous = Path.cwd()
    os.chdir(bundle_root)
    logger.info("Resolving config paths against staged bundle root %s", bundle_root)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a strategy config, expanding ``${VAR}`` against the task environment."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"optimization config must be a mapping: {path}")
    return _expand_env(raw)


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value
