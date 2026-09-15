# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute-agent job."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import stat
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, cast

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.agent_config_formats import resolve_agent_config_for_deployment
from nemo_agents_plugin.config import AgentsConfig
from nemo_agents_plugin.entities import (
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    Agent,
    AgentEnvironmentInline,
    AgentInline,
    ComputeSpecInline,
)
from nemo_agents_plugin.environment_resolution import (
    EnvironmentResolutionError,
    merge_environment_spec_into_agent_config,
    resolve_environment,
)
from nemo_agents_plugin.fabric.invocation import (
    AgentConfigInvocationRequest,
    FabricDirectories,
    invoke_agent_config_request_once,
)
from nemo_agents_plugin.fabric.runtime import FabricRuntimeTimeoutError
from nemo_agents_plugin.jobs.execute_extensions import (
    NOOP_EXECUTE_AGENT_EXTENSION_KIND,
    ExecuteAgentAfterInvokeContext,
    run_execute_agent_after_invoke_extension,
    validate_execute_agent_extension_config,
)
from nemo_agents_plugin.tasks.execute.workdir import (
    AgentWorkdir,
    materialize_agent_workdir,
    validate_agent_workdir,
)
from nemo_agents_plugin.telemetry.intake_export import (
    configure_intake_atif_export,
    supports_intake_atif_export,
)
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.constants import (
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    is_workload_identity_token_file_set,
)
from nemo_platform_plugin.client.oidc_factory import resolve_workload_exchange_provider
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.job_results import ResultRef
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    EnvironmentVariableFromSecret,
    PlatformJobSpec,
    PlatformJobStep,
    ResourcesLimitsSpec,
    ResourcesSpec,
)
from nemo_platform_plugin.jobs.constants import (
    CONFIG_TASK_STORAGE_PATH_ENVVAR,
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ATTEMPT_ID_ENVVAR,
    NEMO_JOB_FILESET_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_SECRETS_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_STEP_ENVVAR,
    NEMO_JOB_TASK_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
    TASK_CONFIG_ENVVAR,
)
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nemo_platform_plugin.jobs.image import get_qualified_image
from nemo_platform_plugin.refs import ENTITY_REF_PATTERN, parse_entity_ref
from nemo_platform_plugin.sdk_provider import get_forwarding_headers
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

FABRIC_AGENT_CONFIG_FORMAT = NEMO_AGENTS_SPEC_CONFIG_FORMAT
FABRIC_BASE_DIR_NAME = "fabric"
INPUT_WORKDIR_RESULT_NAME = "input_workdir"
OUTPUT_WORKDIR_RESULT_NAME = "output_workdir"
OUTPUT_ARTIFACTS_RESULT_NAME = "output_artifacts"
NMP_BASE_URL_ENVVAR = "NMP_BASE_URL"
HEADER_ENVVAR_PREFIX = "NMP_AGENT_TELEMETRY_HEADER_"
FABRIC_RUN_RESULT_NAME = "fabric_run_result"
FABRIC_ERROR_RESULT_NAME = "fabric_error"
FABRIC_RUN_RESULT_FILENAME = "fabric_run_result.json"
FABRIC_ERROR_FILENAME = "fabric_error.json"
SUCCESSFUL_FABRIC_STATUSES = {"succeeded"}
DEFAULT_AGENT_EXECUTION_TIMEOUT_SECONDS = 60 * 60
DEFAULT_AGENT_EXECUTION_IMAGE_NAME = "nmp-api"

# Name Fabric gives the agent process's captured stderr. It lives under a
# runtime/invocation directory Fabric names itself, so it is found by search
# rather than opened by path.
_AGENT_STDERR_FILENAME = "stderr.txt"
# Bounds on what a failing agent can make this task read into memory. The
# artifacts directory is agent-writable, so neither the size nor the number of
# these files is ours to trust, and a runaway agent could otherwise exhaust the
# task's memory while it reports the failure - destroying the diagnostics this
# exists to produce. Sized never to fire on real output (a genuine failed run's
# stderr measured ~2 KiB), so it bounds the pathological case without
# truncating the ordinary one.
_MAX_LOGGED_STDERR_BYTES = 4 * 1024 * 1024
_MAX_LOGGED_STDERR_FILES = 5

# k8s resource key carrying the GPU count. The agents ``ComputeResources`` maps
# express GPUs the Kubernetes way (a ``nvidia.com/gpu`` entry in ``limits``);
# the jobs executor's ``ResourcesSpec`` instead carries a top-level integer
# ``num_gpus``. We translate that one key and pass ``cpu``/``memory`` through.
_GPU_RESOURCE_KEY = "nvidia.com/gpu"
# Resource keys we know how to translate onto the jobs ``ResourcesSpec``. Any
# other k8s resource key would be silently dropped (it has no home on the
# executor spec), so we reject it up front instead.
_SUPPORTED_RESOURCE_KEYS = frozenset({"cpu", "memory", _GPU_RESOURCE_KEY})

# Env var names a secret-backed env var must never shadow. Splitting a secret's
# resolved value over one of these would clobber platform-injected job state (the
# jobs substrate sets the ``NEMO_JOB_*``/``NMP_TASK_CONFIG`` family on every
# step) or the agent-container env the execute task relies on to reach the
# platform SDK (mirrors the deployment container's reserved set). Reject the
# collision at compile time so it can never reach the running step.
_RESERVED_ENV_VAR_NAMES = frozenset(
    {
        # Jobs substrate (nemo_platform_plugin.jobs.constants).
        EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
        PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
        CONFIG_TASK_STORAGE_PATH_ENVVAR,
        TASK_CONFIG_ENVVAR,
        NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
        NEMO_JOB_ID_ENVVAR,
        NEMO_JOB_ATTEMPT_ID_ENVVAR,
        NEMO_JOB_STEP_ENVVAR,
        NEMO_JOB_TASK_ENVVAR,
        NEMO_JOB_WORKSPACE_ENVVAR,
        NEMO_JOB_FILESET_ENVVAR,
        NEMO_JOB_SECRETS_ENVVAR,
        # Agent execution env (mirrors the deployment container's reserved set).
        "NMP_WORKSPACE",
        "NMP_AGENT_NAME",
        "NMP_BASE_URL",
        "PYTHONPATH",
        "AGENT_CONFIG_PATH",
        "NAT_CONFIG_PATH",
    }
)


class ExecuteAgentExtensionConfig(BaseModel):
    kind: str = Field(description="Trusted extension kind registered by an installed NeMo plugin.")
    config: dict[str, Any] = Field(default_factory=dict, description="Extension-owned configuration.")


def _make_noop_extension_config() -> ExecuteAgentExtensionConfig:
    return ExecuteAgentExtensionConfig(kind=NOOP_EXECUTE_AGENT_EXTENSION_KIND)


class ExecuteAgentJobConfig(BaseModel):
    model_config = {"json_schema_mode_override": "validation"}

    agent: str | AgentInline = Field(
        description=(
            "Agent to execute: an Agent entity name or workspace/name ref, or an inline "
            "agent definition for a config composed at request time."
        ),
    )
    input: str = Field(description="Prompt to pass to the agent.")
    environment: str | AgentEnvironmentInline | None = Field(
        default=None,
        description=(
            'AgentEnvironment to run under: a "workspace/name" ref to a stored '
            "AgentEnvironment, an inline environment, or None. Its EnvironmentSpec "
            "is merged into the agent config and its ComputeSpec/secret refs are "
            "snapshotted onto the job step at creation time."
        ),
    )
    workdir: AgentWorkdir | None = Field(
        default=None,
        description="Optional working directory configuration for the execution.",
    )
    timeout_seconds: float = Field(
        default=DEFAULT_AGENT_EXECUTION_TIMEOUT_SECONDS,
        gt=0,
        description="Maximum time to wait for Fabric to return an execution result.",
    )
    auto_telemetry: bool = Field(
        default=True,
        description=(
            "Let the server fill in the agent's telemetry export -- an Intake destination for a "
            "config that asks for one and does not say where. False submits the agent config as "
            "written, which still exports if the config says to; an agent config is the place to "
            "say a run should not be traced."
        ),
    )
    extension: ExecuteAgentExtensionConfig | None = Field(
        default=None,
        description="Optional trusted plugin extension to run during the execute-agent lifecycle.",
    )

    @field_validator("agent")
    @classmethod
    def _validate_agent(cls, value: str | AgentInline) -> str | AgentInline:
        """Apply this job's own requirements to whichever arm the caller used.

        ``AgentInline`` is deliberately as permissive as the ``Agent`` entity it
        backs, so the narrowing lives here rather than on the shared model: an
        entity can legitimately hold a NAT workflow config, but this job cannot
        run one.
        """
        if isinstance(value, str):
            # The ref pattern moved off the field so the union's other arm can
            # carry a config; keep enforcing it for the string arm.
            if not re.fullmatch(ENTITY_REF_PATTERN, value):
                raise ValueError(f"Agent ref {value!r} must be an entity name or a 'workspace/name' ref.")
            return value

        _validate_agent_config_format(value.config_format)
        if not value.config:
            raise ValueError("Inline agent definitions require a non-empty config.")
        return value


class ResolvedAgentConfig(BaseModel):
    name: str
    workspace: str
    config: dict[str, Any]
    config_format: str


class ExecuteAgentStepConfig(BaseModel):
    request: ExecuteAgentJobConfig
    agent: ResolvedAgentConfig
    workdir: AgentWorkdir | None = None
    # Immutable snapshot of the resolved AgentEnvironment, mirroring
    # AgentDeployment. ``agent.config`` already holds the merged config;
    # ``compute`` and ``secrets`` are snapshotted for the executor. The raw
    # request environment (a ref or inline) is retained for provenance on
    # ``request.environment`` — no separate field is needed here. Once created,
    # the job is not kept in sync with the underlying environment entities.
    compute: ComputeSpecInline | None = None
    secrets: dict[str, str] = Field(default_factory=dict)
    extension: ExecuteAgentExtensionConfig = Field(default_factory=_make_noop_extension_config)


class ExecuteAgentJob(NemoJob):
    name: ClassVar[str] = "execute"
    description: ClassVar[str] = "Execute an agent to completion as a scheduled platform job."
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False
    input_spec_schema: ClassVar[type[BaseModel]] = ExecuteAgentJobConfig
    spec_schema: ClassVar[type[BaseModel]] = ExecuteAgentStepConfig

    @staticmethod
    def _execution_image() -> str:
        return AgentsConfig.get().deployments.default_image or get_qualified_image(DEFAULT_AGENT_EXECUTION_IMAGE_NAME)

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: object,
        is_local: bool,
    ) -> BaseModel:
        """Validates an ``ExecuteAgentJobConfig`` (external-facing) and resolves
        it to an ``ExecuteAgentStepConfig`` (internal). Validating entity references
        in this step ensures errors surface to the user on their create request; storing
        resolved values on the step config obviates the need for a second fetch from the Job,
        and ensures we have an immutable snapshot of referenced entities as they existed
        at job creation time (stored on the Job record). The exceptions are File references
        in the AgentWorkdir, which are validated here, but not materialized; the Job snapshots
        them as a named result before doing any work.
        """
        del is_local
        request = cast(ExecuteAgentJobConfig, input_spec)
        typed_entity_client = cast(Any, entity_client)
        source = await _resolve_agent_source(request.agent, workspace=workspace, entity_client=typed_entity_client)

        _validate_agent_config_format(source.config_format)
        resolved_agent_config = resolve_agent_config_for_deployment(
            source.config_format,
            source.config,
            workspace=source.workspace,
            agent_name=source.name,
        )

        # Resolve and merge the referenced AgentEnvironment. The EnvironmentSpec is
        # merged into the resolved config (EnvironmentSpec-wins precedence); the
        # ComputeSpec and secret-env references are snapshotted onto the step for
        # the executor. This mirrors ``create_deployment`` so environment errors
        # surface on the create request and the referenced entities are captured
        # as an immutable snapshot. ``EnvironmentResolutionError`` is surfaced as a
        # ``ValueError`` (matching the "Agent not found" pattern) so the jobs
        # create path reports it to the caller.
        try:
            resolved_env = await resolve_environment(
                request.environment, workspace=workspace, entity_client=typed_entity_client
            )
            merged = merge_environment_spec_into_agent_config(resolved_agent_config, resolved_env.environment_spec)
        except EnvironmentResolutionError as exc:
            raise ValueError(str(exc)) from exc

        # Validate the merged config: an EnvironmentSpec can override the harness
        # provider, so validation must run after the merge to reject e.g. a spec
        # that selects a non-local Fabric environment.
        _validate_agent_config(merged.config)

        workdir = None
        if request.workdir is not None:
            async_sdk_handle = cast(AsyncNeMoPlatform, async_sdk)
            files_client = client_from_platform(async_sdk_handle, AsyncFilesClient)
            workdir = await validate_agent_workdir(request.workdir, files_client, default_workspace=workspace)

        extension = request.extension or _make_noop_extension_config()
        validate_execute_agent_extension_config(extension.kind, extension.config)

        return ExecuteAgentStepConfig(
            request=request,
            agent=ResolvedAgentConfig(
                name=source.name,
                workspace=source.workspace,
                config=merged.config,
                config_format=source.config_format,
            ),
            workdir=workdir,
            compute=resolved_env.compute_spec,
            secrets=merged.secrets,
            extension=extension,
        )

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        del workspace, entity_client, job_name, async_sdk, options
        step_config = ExecuteAgentStepConfig.model_validate(spec)

        executor = CPUExecutionProviderSpec(
            profile=profile or "default",
            provider="cpu",
            container=ContainerSpec(
                image=cls._execution_image(),
                entrypoint=["python", "-m"],
                command=["nemo_agents_plugin.tasks.execute"],
            ),
        )
        # Snapshotted compute -> executor resources, and secret-env refs ->
        # secret-backed step env vars. Both validate the snapshot and raise
        # ``ValueError`` on a bad shape (an unsupported resource key, or a secret
        # env name colliding with a reserved job env var). Surface those as a
        # ``PlatformJobCompilationError`` so the jobs create route maps them to a
        # descriptive 422 rather than an opaque 500 (the route's compile wrapper
        # only translates ``PlatformJobCompilationError``, not bare ``ValueError``).
        try:
            # Snapshotted compute -> executor resources. Injected only when the
            # environment supplied a compute spec, so the default CPU sizing is
            # left to the jobs backend otherwise.
            resources = _compute_to_resources(step_config.compute)
            # Snapshotted secret-env refs -> secret-backed step env vars. The jobs
            # substrate materializes each value into the process env under
            # ENV_NAME; Fabric and its MCP servers read it by name (env-var-name
            # indirection - see environment_resolution).
            environment = _secret_environment(step_config.secrets)
        except ValueError as exc:
            raise PlatformJobCompilationError(str(exc)) from exc

        if resources is not None:
            executor["resources"] = resources

        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="execute-agent",
                    executor=executor,
                    config=step_config.model_dump(mode="json"),
                    environment=environment,
                )
            ],
        )

    def run(self, config: dict, *, ctx: JobContext, sdk: NeMoPlatform | None = None) -> dict:
        step_config = ExecuteAgentStepConfig.model_validate(config)
        agent_ref = f"{step_config.agent.workspace}/{step_config.agent.name}"
        # Logged before validation so a config that fails to parse still names the agent it belonged to.
        logger.info("Executing agent %s (timeout %gs).", agent_ref, step_config.request.timeout_seconds)

        _validate_agent_config_format(step_config.agent.config_format)
        agent_config = _validate_agent_config(step_config.agent.config)

        fabric_dirs = FabricDirectories.create(agent_config, ctx.storage.ephemeral)

        if step_config.request.auto_telemetry and supports_intake_atif_export(
            step_config.agent.config, base_dir=fabric_dirs.base
        ):
            _configure_intake_telemetry(step_config.agent.config, workspace=ctx.workspace, sdk=sdk)
            # Wiring mutates the config mapping, not the model validated above,
            # so re-validate to carry it into what Fabric is handed.
            agent_config = _validate_agent_config(step_config.agent.config)

        if step_config.workdir is not None and _has_workdir_inputs(step_config.workdir):
            if sdk is None:
                raise RuntimeError("sdk is required to stage workdir inputs.")
            logger.info("Staging workdir inputs for agent %s.", agent_ref)
            files_client = client_from_platform(sdk, FilesClient)
            materialize_agent_workdir(step_config.workdir, files_client, fabric_dirs.workspace)

        input_workdir_ref = ctx.results.save(INPUT_WORKDIR_RESULT_NAME, fabric_dirs.workspace)

        logger.info("Invoking agent %s.", agent_ref)
        started_at = time.monotonic()
        try:
            result = asyncio.run(
                invoke_agent_config_request_once(
                    AgentConfigInvocationRequest(
                        agent_config=agent_config,
                        input=step_config.request.input,
                        base_dir=fabric_dirs.base,
                        request_id=ctx.job_id,
                        caller_context={
                            "job_id": ctx.job_id,
                            "job_workspace": ctx.workspace,
                            "agent": agent_ref,
                        },
                        timeout_seconds=step_config.request.timeout_seconds,
                    )
                )
            )
        except Exception as error:
            # Fabric never produced a result, so the agent's own stderr is the
            # only account of how far it got, if anywhere.
            logger.exception(
                "Fabric invocation failed for agent %s after %.1fs.", agent_ref, time.monotonic() - started_at
            )
            _log_agent_stderr(fabric_dirs.artifacts)
            self._save_fabric_error_results(
                ctx, workspace_dir=fabric_dirs.workspace, artifacts_dir=fabric_dirs.artifacts, error=error
            )
            raise

        logger.info("Agent %s returned status=%s after %.1fs.", agent_ref, result.status, time.monotonic() - started_at)

        fabric_run_result_ref = _save_json_result(
            ctx,
            FABRIC_RUN_RESULT_NAME,
            ctx.storage.ephemeral / FABRIC_RUN_RESULT_FILENAME,
            asdict(result),
        )
        output_workdir_ref = ctx.results.save(OUTPUT_WORKDIR_RESULT_NAME, fabric_dirs.workspace)
        output_artifacts_ref = ctx.results.save(OUTPUT_ARTIFACTS_RESULT_NAME, fabric_dirs.artifacts)
        status = "completed" if result.status in SUCCESSFUL_FABRIC_STATUSES else "failed"
        if status == "completed":
            try:
                run_execute_agent_after_invoke_extension(
                    step_config.extension.kind,
                    ExecuteAgentAfterInvokeContext(
                        ctx=ctx,
                        config=step_config.extension.config,
                        agent_name=step_config.agent.name,
                        fabric_result=result,
                    ),
                )
            except Exception:
                logger.exception("Execute-agent extension failed.")
                raise
        else:
            # A Fabric result that *reports* failure is not an exception here,
            # so this is the only place the reason is stated. Without it the
            # step exits non-zero having logged nothing at all, and the cause
            # is only reachable by downloading the saved artifacts.
            logger.error(
                "Agent %s failed: fabric_status=%s error=%s (runtime_id=%s invocation_id=%s). "
                "Fabric run result: %s. Agent stdout/stderr: %s",
                agent_ref,
                result.status,
                result.error,
                result.runtime_id,
                result.invocation_id,
                fabric_run_result_ref.artifact_url,
                output_artifacts_ref.artifact_url,
            )
            _log_agent_stderr(fabric_dirs.artifacts)

        output = {
            "status": status,
            "agent": agent_ref,
            "fabric_status": result.status,
            "runtime_id": result.runtime_id,
            "invocation_id": result.invocation_id,
            "request_id": result.request_id,
            "input_workdir": input_workdir_ref.model_dump(),
            "output_workdir": output_workdir_ref.model_dump(),
            "output_artifacts": output_artifacts_ref.model_dump(),
            "fabric_run_result": fabric_run_result_ref.model_dump(),
        }
        return output

    def _save_fabric_error_results(
        self,
        ctx: JobContext,
        *,
        workspace_dir: Path,
        artifacts_dir: Path,
        error: Exception,
    ) -> None:
        cause = error.__cause__
        payload = {
            "type": type(error).__name__,
            "message": str(error),
            "fabric_status": "timeout" if isinstance(error, (TimeoutError, FabricRuntimeTimeoutError)) else "error",
            "cause_type": type(cause).__name__ if cause is not None else None,
            "cause_message": str(cause) if cause is not None else None,
        }
        for name, path in (
            (OUTPUT_WORKDIR_RESULT_NAME, workspace_dir),
            (OUTPUT_ARTIFACTS_RESULT_NAME, artifacts_dir),
        ):
            try:
                if path.exists():
                    ctx.results.save(name, path)
            except Exception:
                logger.warning("Failed to save %s after Fabric invocation error.", name, exc_info=True)
        try:
            _save_json_result(ctx, FABRIC_ERROR_RESULT_NAME, ctx.storage.ephemeral / FABRIC_ERROR_FILENAME, payload)
        except Exception:
            logger.warning("Failed to save Fabric error result.", exc_info=True)


def _log_agent_stderr(artifacts_dir: Path) -> None:
    """Log the agent process's captured stderr after a run that failed.

    Fabric runs the agent as its own process and redirects its streams to files
    under the artifacts directory, so nothing it logs reaches the task
    container's stdout. Only stderr is worth reading: the stdout file is the
    adapter's protocol channel - a single JSON response, already saved as
    ``fabric_run_result`` - and the lifecycle harness redirects the agent's own
    stdout into stderr anyway, so nothing is lost by ignoring it.

    Read on failure only. A healthy run's execution record belongs in telemetry,
    which carries it live and structured; this exists for what telemetry cannot
    see - a failure before the agent started and produced any spans, an export
    that never landed, a process that died.

    Best-effort: it runs on an already-failing path and must never replace the
    error that got us here.
    """
    try:
        # Safe to run over an agent-writable tree: pathlib's ``**`` does not
        # follow symlinks, so a symlink loop cannot make this hang.
        paths = sorted(artifacts_dir.rglob(_AGENT_STDERR_FILENAME))
    except OSError as error:
        logger.warning("Could not search %s for agent stderr: %s", artifacts_dir, error)
        return
    if len(paths) > _MAX_LOGGED_STDERR_FILES:
        logger.warning(
            "Found %d agent stderr files under %s; logging the first %d.",
            len(paths),
            artifacts_dir,
            _MAX_LOGGED_STDERR_FILES,
        )
        paths = paths[:_MAX_LOGGED_STDERR_FILES]
    for path in paths:
        try:
            tail = _read_stderr_tail(path)
        except OSError as error:
            logger.warning("Could not read agent stderr at %s: %s", path, error)
            continue
        if not tail.text.strip():
            logger.warning("Agent stderr at %s was empty.", path)
            continue
        if tail.omitted_bytes:
            logger.error(
                "Agent stderr (%s, last %d bytes; %d earlier bytes omitted - the saved artifact has "
                "the whole stream):\n%s",
                path,
                tail.read_bytes,
                tail.omitted_bytes,
                tail.text,
            )
        else:
            logger.error("Agent stderr (%s):\n%s", path, tail.text)


class _StderrTail(NamedTuple):
    """What was read from an agent stderr file, in bytes rather than characters."""

    text: str
    read_bytes: int
    omitted_bytes: int


def _read_stderr_tail(path: Path) -> _StderrTail:
    """Return *path*'s contents and how many leading bytes were dropped.

    Takes the tail rather than the head when a stream exceeds the cap: a file
    that large means the agent ran a long while before failing, so the failure
    is at the end. A run that fails early writes little and is never truncated,
    which is where the beginning would have mattered.

    The file type is not taken on trust, because the artifacts directory is
    agent-writable and a size check alone is not a bound: a ``stderr.txt``
    symlinked to ``/dev/zero`` reports ``st_size`` 0, passes any cap, and then
    reads forever, and a FIFO of that name would block this task indefinitely
    while it is trying to report a failure. ``O_NOFOLLOW`` refuses a symlinked
    final component, ``O_NONBLOCK`` keeps a FIFO from blocking the open, and
    the ``fstat`` below rejects anything that is not a regular file - checked
    on the open descriptor so the answer cannot change underneath us.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise OSError(f"{path} is not a regular file")
        size = info.st_size
        if size > _MAX_LOGGED_STDERR_BYTES:
            handle.seek(size - _MAX_LOGGED_STDERR_BYTES)
        # Bounded by the cap regardless of what ``st_size`` claimed, so a file
        # that lied about its size or grew mid-read still cannot run away.
        chunk = handle.read(_MAX_LOGGED_STDERR_BYTES)
    return _StderrTail(
        text=chunk.decode("utf-8", errors="replace"),
        read_bytes=len(chunk),
        omitted_bytes=max(0, size - len(chunk)),
    )


class _AgentSource(BaseModel):
    """The agent definition to execute, however the request supplied it."""

    name: str
    workspace: str
    config: dict[str, Any]
    config_format: str


async def _resolve_agent_source(
    agent: str | AgentInline,
    *,
    workspace: str,
    entity_client: Any,
) -> _AgentSource:
    """Resolve the request's agent arm into one shape the rest of ``to_spec`` uses.

    An inline config is not persisted anywhere: it is validated here so a bad
    definition fails on the create request, then snapshotted onto the step config
    exactly as an entity-backed config is. Its name comes from the config itself
    and it is attributed to the job's workspace.

    NOTE: an inline config is currently gated only by the job's own
    ``agents.execute.create`` permission, whereas reaching the same outcome
    through an entity additionally requires ``agents.agents.create``. Closing
    that gap needs a body-conditional permission, which the shared jobs route
    factory has no hook for today.
    """
    if isinstance(agent, AgentInline):
        # Format and non-emptiness are already enforced by
        # ``ExecuteAgentJobConfig._validate_agent``; this is the spec check.
        try:
            inline_config = AgentConfig.model_validate(agent.config)
        except Exception as exc:
            raise ValueError(f"Inline agent config is invalid: {exc}") from exc
        return _AgentSource(
            name=inline_config.name,
            workspace=workspace,
            config=agent.config,
            config_format=agent.config_format,
        )

    agent_ref = parse_entity_ref(agent, default_workspace=workspace)
    try:
        entity = await entity_client.get(Agent, name=agent_ref.name, workspace=agent_ref.workspace)
    except NemoEntityNotFoundError as exc:
        raise ValueError(f"Agent '{agent}' not found.") from exc
    return _AgentSource(
        name=entity.name,
        workspace=entity.workspace,
        config=entity.config,
        config_format=entity.config_format,
    )


def _has_workdir_inputs(workdir: AgentWorkdir) -> bool:
    return workdir.base_workdir is not None or bool(workdir.artifact_mounts)


def _secret_environment(secrets: dict[str, str]) -> list[EnvironmentVariable]:
    """Compile snapshotted secret-env refs into secret-backed step env vars.

    Each ``ENV_NAME -> "workspace/secret"`` entry becomes an
    ``EnvironmentVariable`` whose value is populated from the referenced Secret.
    A secret env name must never shadow a platform-injected env var (the jobs
    substrate or agent-execution env - see ``_RESERVED_ENV_VAR_NAMES``): the
    resolved value would clobber platform state, so reject the collision here.
    """
    if not secrets:
        return []

    reserved = sorted(name for name in secrets if name in _RESERVED_ENV_VAR_NAMES)
    if reserved:
        raise ValueError(
            f"Secret env var(s) {', '.join(reserved)} collide with reserved job env var name(s). "
            f"Reserved names: {', '.join(sorted(_RESERVED_ENV_VAR_NAMES))}."
        )

    return [
        EnvironmentVariable(name=env_name, from_secret=EnvironmentVariableFromSecret(name=ref))
        for env_name, ref in secrets.items()
    ]


def _compute_to_resources(compute: ComputeSpecInline | None) -> ResourcesSpec | None:
    """Map a snapshotted agents ComputeSpec onto the jobs executor ResourcesSpec.

    Agents ``ComputeResources`` express requests/limits the Kubernetes way -
    ``cpu``/``memory`` scalars plus a ``nvidia.com/gpu`` GPU count. The jobs
    executor's ``ResourcesSpec`` instead carries ``cpu``/``memory`` under
    ``limits``/``requests`` and a top-level integer ``num_gpus``. We translate
    ``cpu``/``memory`` through and lift ``nvidia.com/gpu`` into ``num_gpus``,
    preferring the limits count and falling back to requests. Any other k8s
    resource key has no home on ``ResourcesSpec`` and would be silently dropped,
    so it is rejected instead.
    """
    if compute is None:
        return None

    resources = compute.resources
    _reject_unsupported_resource_keys(resources.limits, "limits")
    _reject_unsupported_resource_keys(resources.requests, "requests")

    spec: dict[str, Any] = {}
    limits = _cpu_memory_spec(resources.limits)
    if limits is not None:
        spec["limits"] = limits
    requests = _cpu_memory_spec(resources.requests)
    if requests is not None:
        spec["requests"] = requests

    num_gpus = _gpu_count(resources.limits, resources.requests)
    if num_gpus is not None:
        spec["num_gpus"] = num_gpus

    return ResourcesSpec(**spec) if spec else None


def _reject_unsupported_resource_keys(resource_map: dict[str, str], where: str) -> None:
    unsupported = sorted(key for key in resource_map if key not in _SUPPORTED_RESOURCE_KEYS)
    if unsupported:
        raise ValueError(
            f"Unsupported compute resource key(s) in {where}: {', '.join(unsupported)}. "
            f"agents.execute jobs support only {', '.join(sorted(_SUPPORTED_RESOURCE_KEYS))}."
        )


def _cpu_memory_spec(resource_map: dict[str, str]) -> ResourcesLimitsSpec | None:
    # ``ResourcesLimitsSpec`` and ``ResourcesRequestsSpec`` are the same
    # plugin-owned compute resource spec; one builder covers both sides.
    spec: dict[str, str] = {}
    if "cpu" in resource_map:
        spec["cpu"] = resource_map["cpu"]
    if "memory" in resource_map:
        spec["memory"] = resource_map["memory"]
    return ResourcesLimitsSpec(**spec) if spec else None


def _gpu_count(limits: dict[str, str], requests: dict[str, str]) -> int | None:
    raw = limits.get(_GPU_RESOURCE_KEY, requests.get(_GPU_RESOURCE_KEY))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {_GPU_RESOURCE_KEY!r} value {raw!r}; expected an integer GPU count.") from exc


def _save_json_result(ctx: JobContext, name: str, path: Path, payload: dict[str, Any]) -> ResultRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ctx.results.save(name, path)


def _validate_agent_config_format(config_format: str) -> None:
    if config_format != FABRIC_AGENT_CONFIG_FORMAT:
        raise ValueError(
            f"Config format {config_format!r} is not supported; "
            f"agents.execute jobs only support {FABRIC_AGENT_CONFIG_FORMAT!r}."
        )


def _configure_intake_telemetry(
    agent_config: dict[str, Any],
    *,
    workspace: str,
    sdk: NeMoPlatform | None,
) -> None:
    """Wire the agent's trajectory export to Intake for this job.

    Runs here rather than at create time because only the task knows both
    halves: ``NMP_BASE_URL`` is the platform URL reachable from *this* pod (the
    Jobs service rewrites it per runtime), and the task's own SDK carries the
    identity the platform gave this job -- the same ``service:agents`` principal
    and on-behalf-of delegation a deployment gets from its auth-proxy sidecar.

    Credentials go in the process environment and the config names them.
    Fabric writes the resolved agent config into the run's artifacts, and those
    are uploaded as a job result, so an inline header would be a downloadable
    one.
    """
    base_url = os.environ.get(NMP_BASE_URL_ENVVAR)
    if not base_url:
        logger.warning("%s is not set; the agent will run untraced.", NMP_BASE_URL_ENVVAR)
        return

    headers = get_forwarding_headers(sdk) if sdk is not None else {}
    headers.update(_workload_identity_headers(base_url))
    for name, value in headers.items():
        os.environ[_header_envvar(name)] = value

    configure_intake_atif_export(
        agent_config,
        workspace=workspace,
        base_url=base_url,
        header_env={name: _header_envvar(name) for name in headers},
    )


def _workload_identity_headers(base_url: str) -> dict[str, str]:
    """Bearer credentials for Relay when the job runs under workload identity.

    ``get_forwarding_headers`` returns only what the SDK was *constructed* with.
    Under workload identity that is the internal marker alone -- the bearer is
    exchanged per request by the SDK's own auth layer, which Relay's raw POST to
    Intake does not go through. Without this the export would be unauthenticated
    on exactly the deployments that enforce auth.

    The token is resolved once and read from the environment at export time, so
    a run outliving its token exports with an expired one. Relay resolves
    ``header_env`` statically, so refreshing needs a dynamic-credential hook it
    does not offer today.
    """
    if not is_workload_identity_token_file_set():
        return {}
    token_file = os.environ[WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR]
    try:
        provider = resolve_workload_exchange_provider(base_url=base_url, subject_token_file=Path(token_file))
        return {"Authorization": f"Bearer {provider.get_access_token()}"}
    except Exception:
        logger.warning(
            "Could not exchange the workload identity token for telemetry export; "
            "the trajectory will be posted without credentials.",
            exc_info=True,
        )
        return {}


def _header_envvar(header_name: str) -> str:
    """Environment variable the exporter reads one outbound header value from."""
    return f"{HEADER_ENVVAR_PREFIX}{header_name.upper().replace('-', '_')}"


def _validate_agent_config(config: dict) -> AgentConfig:
    agent_config = AgentConfig.model_validate(config)
    if agent_config.environment.provider != "local":
        raise ValueError("agents.execute jobs only support local Fabric environments.")

    return agent_config
