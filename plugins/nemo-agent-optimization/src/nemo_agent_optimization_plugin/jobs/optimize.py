# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OptimizeJob — strategy-dispatching Agents optimization (``nemo agents optimize``).

Every installed ``nemo.optimization-strategy`` plugin runs through the same path: this job stages
the bundle, loads the config, resolves the agent, and hands all of it to the named strategy.  The
job itself knows nothing strategy-specific — preflight checks, dataset staging and study execution
all belong to the strategy that needs them.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, cast

import yaml
from nemo_agent_optimization_plugin.agents import resolve_agent_config
from nemo_agent_optimization_plugin.schemas.optimize import FILESET_REQUIRED, OptimizeSpec, OptimizeSubmitSpec
from nemo_agent_optimization_plugin.strategies import PRIMARY_ARTIFACT_KEY, discover_optimization_strategies
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import InternalServerError, NemoResponseValidationError, NemoTransportError
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    ExecutorSpec,
    PlatformJobSpec,
    SubprocessExecutionProviderSpec,
)
from nemo_platform_plugin.jobs.client import AsyncJobsClient
from nemo_platform_plugin.jobs.exceptions import (
    PlatformJobCompilationError,
    PlatformJobDependencyUnavailableError,
)
from nemo_platform_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from nemo_platform_plugin.jobs.image import get_qualified_image
from nemo_platform_plugin.refs import (
    FilesetRef,
    LocalDir,
    classify_output_target,
)
from nemo_platform_plugin.run_dependencies import LocalRunError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

#: Task entry point for the optimize step, in both executor flavours.  The subprocess backend
#: takes one flat command; the cpu backend splits it into a container entrypoint + command.
OPTIMIZE_TASK_MODULE = "nemo_agent_optimization_plugin.tasks.optimize"
OPTIMIZE_ENTRYPOINT = ["python", "-m"]
OPTIMIZE_COMMAND = [OPTIMIZE_TASK_MODULE]

#: Image for the cpu (docker / kubernetes_job) fallback.  ``nemo-agent-optimization-plugin`` is part
#: of the ``cpu-tasks`` dependency group so ``OPTIMIZE_TASK_MODULE`` imports there.
OPTIMIZE_TASK_IMAGE = "nmp-cpu-tasks"


class OptimizeJob(NemoJob):
    """Run the selected nemo.optimization-strategy plugin."""

    name: ClassVar[str] = "optimize"
    description: ClassVar[str] = "Optimize a platform agent with an installed optimization strategy."
    container: ClassVar[str] = "cpu-tasks"
    job_collection_path: ClassVar[str | None] = None
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = OptimizeSpec
    input_spec_schema: ClassVar[type[BaseModel]] = OptimizeSubmitSpec

    @classmethod
    async def to_spec(  # ty: ignore[invalid-method-override]
        cls,
        input_spec: OptimizeSubmitSpec,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncNeMoPlatform,
        is_local: bool,
    ) -> OptimizeSpec:
        del entity_client, async_sdk
        payload = input_spec.model_dump(mode="json")
        payload["workspace"] = workspace
        return OptimizeSpec.model_validate(payload, context={"is_local": is_local})

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]
        cls,
        *,
        workspace: str,
        spec: OptimizeSpec,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        from nemo_platform_plugin.jobs.api_factory import (
            EnvironmentVariable,
            PlatformJobStep,
        )
        from nemo_platform_plugin.jobs.constants import (
            DEFAULT_JOB_STORAGE_PATH,
            PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
        )

        # ``compile`` is the remote submission path only — ``NemoJobScheduler.run_local`` goes
        # straight to ``run`` — so requiring the fileset here keeps platform execution remote-safe.
        if spec.optimize_config_fileset is None:
            raise PlatformJobCompilationError(FILESET_REQUIRED)

        spec_dict = spec.model_dump(mode="json")
        spec_dict["workspace"] = workspace

        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="optimize",
                    executor=await _resolve_executor(profile=profile or "default", async_sdk=async_sdk),
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
        spec = OptimizeSpec.model_validate(config)
        # Resolve the strategy before anything expensive runs: ``--strategy`` is a required flag
        # with no client-side validation, so a typo is the likeliest user error and it should not
        # cost a full fileset download and an agent fetch first.  ``validate_config`` and ``run``
        # still happen below, where the staged config exists.
        strategies = discover_optimization_strategies()
        strategy = strategies.get(spec.strategy)
        if strategy is None:
            raise LocalRunError(
                f"Optimization strategy {spec.strategy!r} is not installed. Available strategies: {sorted(strategies)}"
            )
        with _staged_bundle(spec, ctx=ctx, sdk=sdk) as (config_path, bundle_root):
            optimize_config = _load_yaml(config_path)
            source_agent_config = _load_local_source_agent_config(spec.agent)
            agent_config = resolve_agent_config(spec.agent, workspace=spec.workspace, sdk=sdk)
            with _bundle_workdir(bundle_root):
                strategy.validate_config(optimize_config, agent=spec.agent)
                logger.info("Dispatching agents optimize strategy %s", spec.strategy)
                result = strategy.run(
                    agent_config=agent_config,
                    source_agent_config=source_agent_config,
                    config=optimize_config,
                    ctx=ctx,
                    workspace=spec.workspace,
                    sdk=sdk,
                )

        primary_artifact_value = result.pop(PRIMARY_ARTIFACT_KEY, None)
        primary_artifact = Path(primary_artifact_value) if isinstance(primary_artifact_value, str) else None
        published = _publish_results(
            spec.output,
            workspace=spec.workspace,
            ctx=ctx,
            sdk=sdk,
            primary_artifact=primary_artifact,
        )
        return result if published is None else {**result, "output": published}


def _load_local_source_agent_config(agent: str | None) -> dict[str, Any] | None:
    """Load the original platform agent mapping so local output preserves its schema."""
    if agent is None:
        return None
    agent_path = Path(agent).expanduser()
    if not agent_path.is_file():
        return None
    # Soft dependency: nemo-agent-optimization-plugin cannot declare nemo-agents-plugin (it would
    # cycle), so the package only ever arrives transitively.  Mirrors the guard in
    # ``nemo_agent_optimization_plugin.agents``.
    try:
        from nemo_agents_plugin.agent_config import load_agent_config
    except ImportError as exc:  # pragma: no cover - agents plugin always present for CLI path
        raise LocalRunError("Loading a local agent.yaml requires nemo-agents-plugin.") from exc

    return load_agent_config(agent_path).model_dump(mode="json", exclude_none=True)


def _profiles_unavailable(profile: str) -> PlatformJobDependencyUnavailableError:
    """A retryable failure while resolving the backend for *profile*."""
    return PlatformJobDependencyUnavailableError(
        f"Unable to resolve execution profile '{profile}': the Jobs service is temporarily "
        "unavailable.  Retry the submission."
    )


async def _resolve_executor(*, profile: str, async_sdk: object) -> ExecutorSpec:
    """Pick the executor for *profile* from the backends the platform actually registered.

    Optimize prefers ``subprocess``: a study drives Fabric trials that may need the host's
    Docker daemon and a venv carrying the agent's harness adapters.  Deployments that do not
    register a subprocess backend (Helm / Minikube) get the ``cpu`` provider instead, which the
    platform maps to whichever backend it registered for that profile (docker or kubernetes_job).
    """
    if async_sdk is None:
        raise _profiles_unavailable(profile)

    try:
        profiles = (
            await client_from_platform(cast(AsyncNeMoPlatform, async_sdk), AsyncJobsClient).get_execution_profiles()
        ).data()
    except (NemoTransportError, NemoResponseValidationError, InternalServerError) as exc:
        raise _profiles_unavailable(profile) from exc

    if any(
        isinstance(candidate, SubprocessJobExecutionProfile) and candidate.profile == profile for candidate in profiles
    ):
        return SubprocessExecutionProviderSpec(
            provider="subprocess",
            profile=profile,
            command=[*OPTIMIZE_ENTRYPOINT, *OPTIMIZE_COMMAND],
        )

    # Jobs keys execution profiles by (provider, profile); "cpu" is whatever container backend
    # the deployment registered under that name.
    if any(candidate.provider == "cpu" and candidate.profile == profile for candidate in profiles):
        return CPUExecutionProviderSpec(
            provider="cpu",
            profile=profile,
            container=ContainerSpec(
                image=get_qualified_image(OPTIMIZE_TASK_IMAGE),
                entrypoint=OPTIMIZE_ENTRYPOINT,
                command=OPTIMIZE_COMMAND,
            ),
        )

    available = sorted({f"{candidate.provider}/{candidate.profile}" for candidate in profiles})
    raise PlatformJobCompilationError(
        f"No 'subprocess' or 'cpu' execution profile named {profile!r} is registered, so the "
        f"optimize step has nowhere to run.  Available profiles: {available or ['<none>']}."
    )


@contextlib.contextmanager
def _staged_bundle(
    spec: OptimizeSpec,
    *,
    ctx: JobContext,
    sdk: NeMoPlatform | None,
) -> Iterator[tuple[Path, Path | None]]:
    """Yield ``(optimize config path, bundle root)`` for the run.

    In fileset mode the whole bundle is downloaded and the root is the download dir, so the
    config's relative references (dataset, ``eval.fabric.base_dir``, hook and MCP configs) can be
    resolved against it.  In absolute-path mode there is no bundle: the config is read in place and
    the root is ``None``, leaving the CLI's working directory alone.
    """
    if spec.optimize_config_fileset is None:
        yield Path(spec.optimize_config), None
        return

    # Soft dependency, mirroring nemo_agent_optimization_plugin.agents' lazy imports.
    from nemo_agents_plugin.jobs.fileset_io import resolve_staged_config

    with resolve_staged_config(
        spec.optimize_config,
        spec.optimize_config_fileset,
        workspace=spec.workspace,
        ctx=ctx,
        sdk=sdk,
        kind="optimize-config",
    ) as config_path:
        yield config_path, _bundle_root_of(config_path, spec.optimize_config)


def _bundle_root_of(config_path: Path, config_rel_path: str) -> Path:
    """The download dir ``config_rel_path`` was resolved inside.

    ``resolve_staged_config`` yields ``<download dir>/<config_rel_path>`` and keeps the download
    dir private, so walk back up one level per relative segment to recover it.
    """
    root = config_path
    for _ in PurePosixPath(config_rel_path).parts:
        root = root.parent
    return root


@contextlib.contextmanager
def _bundle_workdir(bundle_root: Path | None) -> Iterator[None]:
    """Run the study with *bundle_root* as the working directory (no-op when ``None``).

    Relative paths in the optimize config are documented as fileset-root-relative, and they are
    consumed in many places — the dataset loader, Fabric's ``base_dir``, ``run_hook.path``,
    author-supplied MCP ``config_paths``.  Rewriting each key would mean chasing every schema that
    can hold a path; moving the process instead makes them all resolve correctly at once.  The task
    subprocess runs exactly one job, so the process-global chdir is contained.
    """
    if bundle_root is None:
        yield
        return
    previous = Path.cwd()
    os.chdir(bundle_root)
    logger.info("Resolving optimize config paths against staged bundle root %s", bundle_root)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read an optimize config, expanding ``${VAR}`` against the task's environment."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"optimize config must be a mapping: {path}")
    return _expand_env(raw)


def _publish_results(
    output: str | None,
    *,
    workspace: str,
    ctx: JobContext,
    sdk: NeMoPlatform | None,
    primary_artifact: Path | None = None,
) -> dict[str, str] | None:
    """Copy the run's artifacts to *output*, returning a pointer for the job result.

    Strategies write everything under ``ctx.storage.persistent / "results"``
    and register it via ``ctx.results.save``, which on the platform lands in the
    job's own fileset under ``results/<attempt_id>/``.  That is addressable only
    through the typed Jobs results client, so a remote client that wants to read the
    optimized config back — or hand it to a follow-up job — needs a stable
    location it names up front.  Publishing the whole ``results`` tree keeps
    this strategy-agnostic: no ``RESULT_NAME`` coupling.

    A strategy that names a single primary artifact (``PRIMARY_ARTIFACT_KEY`` in its result)
    can instead be published straight to a YAML file the caller named.

    Returns ``None`` when no target was requested.
    """
    if output is None:
        return None

    if primary_artifact is not None and Path(output).suffix.lower() in {".yaml", ".yml"}:
        if not primary_artifact.is_file():
            raise FileNotFoundError(f"Optimization strategy did not write its primary artifact: {primary_artifact}")
        local_file = Path(output).expanduser().resolve()
        local_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(primary_artifact, local_file)
        logger.info("Published optimized config from %s to local file %s", primary_artifact, local_file)
        return {"type": "local_file", "path": str(local_file)}

    # Soft dependency, mirroring nemo_agent_optimization_plugin.agents' lazy imports.
    from nemo_agents_plugin.jobs.fileset_io import split_fileset_ref, upload_to_fileset

    try:
        artifacts = ctx.storage.persistent / "results"
    except RuntimeError as exc:
        raise LocalRunError(
            "Publishing optimize results requires persistent storage, which this job did not "
            "request.  This is a platform-run-only feature; drop 'output' for local runs."
        ) from exc

    if not artifacts.is_dir() or not any(path.is_file() for path in artifacts.rglob("*")):
        raise FileNotFoundError(
            f"Optimize study reported success but wrote no artifacts to {artifacts}; nothing to publish."
        )

    if classify_output_target(output) is LocalDir:
        local = Path(output).expanduser().resolve()
        local.mkdir(parents=True, exist_ok=True)
        shutil.copytree(artifacts, local, dirs_exist_ok=True)
        logger.info("Published optimize results from %s to local dir %s", artifacts, local)
        return {"type": "local_dir", "path": str(local)}

    ws, name = split_fileset_ref(FilesetRef(output), workspace)
    if sdk is None:
        raise LocalRunError(
            f"Publishing optimize results to fileset '{ws}/{name}' requires a 'sdk: NeMoPlatform', "
            "but no platform SDK was available.  Set NMP_BASE_URL, pass sdk via "
            "NemoJobScheduler.run_local(sdk=...), or use a local output directory instead."
        )
    upload_to_fileset(artifacts, fileset=name, workspace=ws, sdk=sdk)
    logger.info("Published optimize results from %s to fileset %s/%s", artifacts, ws, name)
    return {"type": "fileset", "fileset": f"{ws}/{name}"}


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value
