# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-backed agent-evaluation job for the evaluator plugin.

Runs :class:`AgentEvaluator` over a set of tasks against a Model/Agent endpoint
or an agent runner (e.g. a Fabric harness), producing an ``AgentEvalResult`` (trials +
per-trial scores + summary). The row-based counterpart is
:class:`~nemo_evaluator.jobs.evaluate.EvaluateJob`.

Per-task metrics may be given inline or as references to stored metrics;
references are resolved into inline metrics during ``to_spec`` via the shared
:mod:`nemo_evaluator.jobs.metric_resolution` helper. The full result bundle (trials +
scores + summary) is persisted as job artifacts, and a concise, queryable result entity
is written via :func:`~nemo_evaluator.jobs.result_persistence.persist_agent_eval_result`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import nemo_evaluator.agent_seeds  # noqa: F401 - registers the platform 'fileset' workspace-seed handler
from filesets import FilesetPathError, parse_fileset_ref
from nemo_evaluator.api.schemas import MetricInline
from nemo_evaluator.config import get_config
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_compiler import compile_agent_eval_job
from nemo_evaluator.jobs.agent_spec import (
    AgentEvalInputSpec,
    AgentEvalSpec,
    AgentEvalTaskSpec,
    AgentTarget,
    FabricRunnerTarget,
    GymRunnerTarget,
    HarborRunnerTarget,
    ModelTarget,
    Target,
)
from nemo_evaluator.jobs.gym_environment_package import (
    ENVIRONMENT_MANIFEST_FILENAME,
    GymEnvironmentPackageError,
    parse_environment_manifest,
    validate_environment_manifest_against_listing,
)
from nemo_evaluator.jobs.gym_sandbox import (
    SandboxUnavailableError,
    SessionBackedGymRunner,
    require_fileset_environment_sandboxed,
    require_fileset_sandbox_storage_identity,
    sandbox_plan_from_environment,
)
from nemo_evaluator.jobs.metric_resolution import resolve_metrics_to_inline, to_runtime_bundle
from nemo_evaluator.jobs.publication import publish_agent_eval_result
from nemo_evaluator.jobs.result_persistence import persist_agent_eval_result
from nemo_evaluator.jobs.utils import as_async_nemo_client, as_nemo_client
from nemo_evaluator.shared.metric_bundles.bundles import unbundle_metric
from nemo_evaluator.task_refs import resolve_agent_eval_tasks
from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner, HarborRuntimeConfig
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTarget
from nemo_evaluator_sdk.metrics.protocol import Metric
from nemo_evaluator_sdk.values import RunConfigOnline, RunConfigOnlineModel
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.errors import (
    InternalServerError,
    NemoResponseValidationError,
    NemoTransportError,
    NotFoundError,
    PermissionDeniedError,
)
from nemo_platform_plugin.entities import EntityClient
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.files.types import FilesetPurpose
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec, SubprocessExecutionProviderSpec
from nemo_platform_plugin.jobs.client import AsyncJobsClient
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError, PlatformJobDependencyUnavailableError
from nemo_platform_plugin.jobs.execution_profiles import (
    KubernetesJobExecutionProfile,
    SubprocessJobExecutionProfile,
    VolcanoJobExecutionProfile,
)
from nemo_platform_plugin.jobs.spec import BaseExecutionProfile
from nemo_platform_plugin.sdk import AsyncNeMoPlatform, NeMoPlatform
from pydantic import BaseModel

logger = logging.getLogger(__name__)

#: Job-result artifact names + the on-disk bundle directory.
DEFAULT_RESULT_NAME = "agent-eval-results"
SUMMARY_RESULT_NAME = "summary"
AGENT_BUNDLE_DIR = "agent-eval"
SUMMARY_FILE_NAME = "summary.json"

# Shared tail for every Harbor backend-compatibility rejection
_HARBOR_BACKEND_REQUIREMENT = (
    "Harbor targets currently require local execution or the subprocess backend with access to the host Docker daemon."
)
_SUBPROCESS_PROVIDER: Literal["subprocess"] = "subprocess"


def _harbor_backend_error(reason: str) -> PlatformJobCompilationError:
    """A Harbor backend rejection: the specific cause followed by the shared requirement."""
    return PlatformJobCompilationError(f"{reason} {_HARBOR_BACKEND_REQUIREMENT}")


def _profile_dependency_unavailable(profile: str) -> PlatformJobDependencyUnavailableError:
    """A retryable failure while resolving an execution profile."""
    return PlatformJobDependencyUnavailableError(
        f"Unable to resolve execution profile '{profile}': the Jobs service is temporarily unavailable. "
        "Retry the submission."
    )


async def _resolve_gym_environment(
    target: Target | None,
    *,
    workspace: str,
    async_sdk: AsyncNeMoPlatform | None,
) -> Target | None:
    """Validate and qualify a Gym environment FileSet through the Files service."""
    if not isinstance(target, GymRunnerTarget) or target.environment is None:
        return target
    if async_sdk is None:
        raise ValueError("Gym environment FileSet references require a platform connection")

    # Qualify ``workspace/name`` now so later steps do not re-parse a relative or fragmented ref.
    try:
        environment_workspace, environment_name, file_path = parse_fileset_ref(
            target.environment.root,
            workspace_fallback=workspace,
        )
    except FilesetPathError as exc:
        raise ValueError(f"invalid Gym environment FileSet reference: {target.environment.root!r}") from exc
    if file_path:
        raise ValueError("Gym environment FileSet references must not include a file fragment")

    files = client_from_platform(async_sdk, AsyncFilesClient)
    try:
        environment = (
            await files.get_fileset(
                workspace=environment_workspace,
                name=environment_name,
            )
        ).data()
    except NotFoundError as exc:
        raise ValueError(f"Gym environment FileSet {environment_workspace}/{environment_name} does not exist") from exc
    except PermissionDeniedError as exc:
        raise PermissionError(
            f"access denied to Gym environment FileSet {environment_workspace}/{environment_name}"
        ) from exc

    # ``purpose=environment`` is what keeps this FileSet off the dataset and model catalogs.
    if environment.purpose != FilesetPurpose.ENVIRONMENT:
        raise ValueError(
            f"Gym environment FileSet {environment_workspace}/{environment_name} has purpose "
            f"{environment.purpose.value!r}; expected {FilesetPurpose.ENVIRONMENT.value!r}"
        )

    try:
        listing = (
            await files.list_files(
                workspace=environment_workspace,
                name=environment_name,
            )
        ).data()
    except PermissionDeniedError as exc:
        raise PermissionError(
            f"access denied to Gym environment FileSet {environment_workspace}/{environment_name}"
        ) from exc
    except NotFoundError as exc:
        raise ValueError(f"Gym environment FileSet {environment_workspace}/{environment_name} does not exist") from exc

    paths = {item.path for item in listing.data}
    if ENVIRONMENT_MANIFEST_FILENAME not in paths:
        raise ValueError(
            f"Gym environment FileSet {environment_workspace}/{environment_name} has no "
            f"{ENVIRONMENT_MANIFEST_FILENAME} at its root"
        )

    try:
        manifest_response = await files.download_file(
            workspace=environment_workspace,
            name=environment_name,
            path=ENVIRONMENT_MANIFEST_FILENAME,
        )
        raw_manifest = await manifest_response.read()
    except PermissionDeniedError as exc:
        raise PermissionError(
            f"access denied to Gym environment FileSet {environment_workspace}/{environment_name}"
        ) from exc
    except NotFoundError as exc:
        raise ValueError(
            f"Gym environment FileSet {environment_workspace}/{environment_name} has no "
            f"{ENVIRONMENT_MANIFEST_FILENAME} at its root"
        ) from exc

    try:
        # Listing-only checks: no customer code is imported.
        manifest = parse_environment_manifest(raw_manifest)
        validate_environment_manifest_against_listing(manifest, paths)
    except GymEnvironmentPackageError as exc:
        raise ValueError(
            f"Gym environment FileSet {environment_workspace}/{environment_name} is not a valid package: {exc}"
        ) from exc

    return target.model_copy(update={"environment": FilesetRef(root=f"{environment_workspace}/{environment_name}")})


#: Identity headers forwarded from the job's platform SDK to online inference so a platform-routed
#: target authenticates as the job's principal (``get_task_nemo_client`` emits these). An explicit allowlist
#: — not an ``X-NMP-*`` prefix match — so trace/metadata headers the SDK may add later never leak to
#: a third-party model/agent endpoint. ``X-NMP-Principal-Id`` is the header the PDP authorizes on
#: (verified against an auth-enabled platform); the rest carry the delegated on-behalf-of identity.
_FORWARDED_IDENTITY_HEADERS = frozenset(
    {
        "X-NMP-Principal-Id",
        "X-NMP-Principal-Email",
        "X-NMP-Principal-Groups",
        "X-NMP-Principal-On-Behalf-Of",
        "X-NMP-Principal-On-Behalf-Of-Email",
        "X-NMP-Principal-On-Behalf-Of-Groups",
        "X-NMP-Internal",
    }
)


@dataclass(frozen=True)
class AgentEvalResultFiles:
    """Filesystem layout for a persisted agent-eval bundle."""

    bundle_dir: Path
    summary: Path


def _runtime_metric(metric: MetricInline) -> Metric:
    """Reconstruct a runtime ``Metric`` from its inline bundle DTO."""
    return unbundle_metric(to_runtime_bundle(metric))


def _to_runtime_task(task: AgentEvalTaskSpec) -> AgentEvalTask:
    """Reconstruct a runtime ``AgentEvalTask`` (live metrics) from its canonical DTO."""
    return AgentEvalTask(
        id=task.id,
        intent=task.intent,
        # The runtime task carries plain dicts; the typed DTOs collapse to them, with arbitrary
        # task-specific inputs preserved and key/value metadata pairs folded into a mapping.
        inputs=task.inputs.model_dump(exclude_none=True),
        reference=task.reference,
        metrics=[_runtime_metric(metric) for metric in task.metrics],
        views=task.views,
        metadata={item.key: item.value for item in task.metadata},
    )


class AgentEvalJob(NemoJob):
    """Run agent evaluation (``AgentEvaluator``) over tasks against a Model/Agent endpoint or runner."""

    name: ClassVar[str] = "agent-evaluate"
    description: ClassVar[str] = "Run agent evaluation over tasks against a model, agent, or runner."
    container: ClassVar[str] = "cpu-tasks"
    input_spec_schema: ClassVar[type[BaseModel] | None] = AgentEvalInputSpec
    spec_schema: ClassVar[type[BaseModel] | None] = AgentEvalSpec
    job_collection_path: ClassVar[str | None] = "/agent-evaluate/jobs"

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncNeMoPlatform | None,
        is_local: bool,
    ) -> BaseModel:
        """Resolve each task's metric references into inline metrics for the canonical spec."""
        del is_local
        submit_spec = (
            input_spec.model_copy(deep=True)
            if isinstance(input_spec, AgentEvalInputSpec)
            else AgentEvalInputSpec.model_validate_json(input_spec.model_dump_json())
        )
        entity_client = cast(EntityClient | None, entity_client)
        # A `tasks` taskset reference is loaded and expanded into inline task DTOs first, so the
        # metric-ref resolution below is identical whether the tasks were submitted inline or via a
        # stored taskset.
        task_inputs = await resolve_agent_eval_tasks(
            submit_spec.tasks, workspace=workspace, entity_client=entity_client
        )
        resolved_tasks: list[AgentEvalTaskSpec] = []
        for task in task_inputs:
            metrics = await resolve_metrics_to_inline(
                task.metrics,
                workspace=workspace,
                entity_client=entity_client,
                async_sdk=async_sdk,
            )
            resolved_tasks.append(
                AgentEvalTaskSpec(
                    id=task.id,
                    intent=task.intent,
                    inputs=task.inputs,
                    reference=task.reference,
                    metrics=metrics,
                    views=task.views,
                    metadata=task.metadata,
                )
            )
        resolved_target = await _resolve_gym_environment(
            submit_spec.target,
            workspace=workspace,
            async_sdk=async_sdk,
        )
        return AgentEvalSpec(
            tasks=resolved_tasks,
            target=resolved_target,
            trials=submit_spec.trials,
            max_concurrent_tasks=submit_spec.max_concurrent_tasks,
            fail_fast=submit_spec.fail_fast,
            labels=submit_spec.labels,
            publication=submit_spec.publication,
        )

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncNeMoPlatform | None,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        """Compile the canonical spec into a plugin-native agent-evaluation job."""
        del entity_client, job_name, options
        canonical_spec = spec if isinstance(spec, AgentEvalSpec) else AgentEvalSpec.model_validate(spec.model_dump())
        execution_profile: BaseExecutionProfile | None = None
        if isinstance(canonical_spec.target, GymRunnerTarget):
            evaluator_config = get_config() if canonical_spec.target.environment is not None else None
            require_pvc_storage = (
                evaluator_config is not None and evaluator_config.sandbox_host_provider == "opensandbox"
            )
            execution_profile = await cls._execution_profile(
                async_sdk=async_sdk,
                profile=profile or "default",
                require_pvc_storage=require_pvc_storage,
            )
            if canonical_spec.target.environment is not None:
                assert evaluator_config is not None
                try:
                    require_fileset_environment_sandboxed(canonical_spec.target, evaluator_config)
                    require_fileset_sandbox_storage_identity(
                        canonical_spec.target,
                        evaluator_config,
                        execution_profile=execution_profile,
                    )
                except SandboxUnavailableError as exc:
                    raise PlatformJobCompilationError(str(exc)) from exc
        del workspace
        platform_spec = compile_agent_eval_job(
            canonical_spec,
            profile=profile,
            use_subprocess=isinstance(execution_profile, SubprocessJobExecutionProfile),
        )
        if isinstance(canonical_spec.target, HarborRunnerTarget):
            step = next(iter(platform_spec["steps"]))
            executor = cast(dict[str, Any], step["executor"])
            step["executor"] = await cls._resolve_harbor_subprocess_executor(
                executor=executor,
                async_sdk=async_sdk,
            )
        return platform_spec

    @staticmethod
    async def _execution_profile(
        *,
        async_sdk: AsyncNeMoPlatform | None,
        profile: str,
        require_pvc_storage: bool = False,
    ) -> BaseExecutionProfile | None:
        """Resolve the profile that Jobs will use for this submission."""
        if async_sdk is None:
            return None
        try:
            profiles = (await client_from_platform(async_sdk, AsyncJobsClient).get_execution_profiles()).data()
        except (NemoTransportError, NemoResponseValidationError, InternalServerError) as exc:
            raise _profile_dependency_unavailable(profile) from exc
        if require_pvc_storage:
            for item in profiles:
                if item.profile == profile and isinstance(
                    item, KubernetesJobExecutionProfile | VolcanoJobExecutionProfile
                ):
                    return item
            raise PlatformJobCompilationError(
                f"the FileSet-backed OpenSandbox Gym execution profile '{profile}' must use a Kubernetes "
                "or Volcano backend with shared PVC storage."
            )

        # Profiles are keyed by (provider, profile), so cpu/default and subprocess/default may both
        # be advertised. The Jobs API rewrites a CPU step to subprocess whenever a same-named
        # subprocess profile is configured. Prefer that profile here as well, so Gym emits the
        # host-portable `python -m ...` command that the rewritten step will need.
        matching_profile: BaseExecutionProfile | None = None
        for item in profiles:
            if item.profile != profile:
                continue
            if isinstance(item, SubprocessJobExecutionProfile):
                return item
            matching_profile = item
        return matching_profile

    @staticmethod
    async def _resolve_harbor_subprocess_executor(
        *, executor: dict[str, Any], async_sdk: AsyncNeMoPlatform | None
    ) -> SubprocessExecutionProviderSpec:
        """Resolve Harbor's selected profile to an explicit host subprocess executor."""
        profile = cast(str, executor["profile"])
        provider = cast(str, executor["provider"])
        if async_sdk is None:
            raise _profile_dependency_unavailable(profile)

        try:
            profiles = (await client_from_platform(async_sdk, AsyncJobsClient).get_execution_profiles()).data()
        except (NemoTransportError, NemoResponseValidationError, InternalServerError) as exc:
            raise _profile_dependency_unavailable(profile) from exc

        # The concrete profile type fixes the backend to "subprocess".
        if any(
            isinstance(execution_profile, SubprocessJobExecutionProfile) and execution_profile.profile == profile
            for execution_profile in profiles
        ):
            container = cast(dict[str, Any], executor["container"])
            command = [*(container.get("entrypoint") or []), *(container.get("command") or [])]
            if not command:
                raise _harbor_backend_error(
                    f"Unable to compile execution profile '{profile}' for subprocess execution: the step command is empty."
                )
            return SubprocessExecutionProviderSpec(provider=_SUBPROCESS_PROVIDER, profile=profile, command=command)

        # Jobs keys execution profiles by (provider, profile), so at most one backend can match.
        resolved_backend = next(
            (
                execution_profile.backend
                for execution_profile in profiles
                if execution_profile.profile == profile and execution_profile.provider == provider
            ),
            None,
        )
        raise _harbor_backend_error(
            f"Execution profile '{profile}' resolves to backend '{resolved_backend}'."
            if resolved_backend is not None
            else f"Execution profile '{profile}' does not resolve to a subprocess backend."
        )

    @staticmethod
    def _endpoint_url(target: Target | None) -> str | None:
        """The HTTP endpoint a Model/Agent target generates against; ``None`` for a runner/offline."""
        if isinstance(target, ModelTarget):
            return str(target.model.url)
        if isinstance(target, AgentTarget):
            return str(target.agent.url)
        return None

    @staticmethod
    def _build_evaluator(client: NemoClient | AsyncNemoClient | None, target: Target | None) -> AgentEvaluator:
        """Construct the evaluator, forwarding the job's platform identity to online inference.

        Online generation against a *platform-routed* Model/Agent target must act as the job's
        principal, so the task SDK's identity headers (:data:`_FORWARDED_IDENTITY_HEADERS`, e.g. the
        service principal id and on-behalf-of) are forwarded to the evaluator's inference client.

        Forwarding is gated on the target sharing the client platform origin: a
        third-party endpoint (or a runner with no HTTP endpoint) gets *no* identity headers, so the
        delegated identity — which includes the user's email and group PII — never leaves the platform.
        External providers authenticate via their own api key and don't need it anyway. Isolated so
        tests can inject a fake inference seam.

        ``client`` is the SDK handle injected into ``run`` — a typed platform client in a submitted
        job (built by ``get_task_nemo_client``, threading ``NMP_PRINCIPAL`` as on-behalf-of). It is ``None``
        only for a platformless local run (e.g. offline ``run_local``), which has no identity to
        forward.

        NOTE: bearer-token auth for platform routes in an auth-enabled deployment is not yet
        forwarded (the local/internal path relies on the ``X-NMP-*`` identity headers); see
        AALGO-297 follow-ups.
        """
        identity_headers: dict[str, str] = {}
        url = AgentEvalJob._endpoint_url(target)
        if client is not None and url is not None and client.is_platform_url(url):
            headers = client.default_headers
            identity_headers = {
                key: value
                for key, value in headers.items()
                if key in _FORWARDED_IDENTITY_HEADERS and isinstance(value, str)
            }
        return AgentEvaluator(default_headers=identity_headers or None)

    @staticmethod
    def _resolve_target(
        target: Target | None, ctx: JobContext
    ) -> tuple[AgentEvalTarget | None, str | dict[str, Any] | None, RunConfigOnline | RunConfigOnlineModel | None]:
        """Resolve a target spec to ``(runtime target, prompt_template, params)`` for the SDK run config.

        Endpoint targets carry their own request config; a runner is instantiated to its runtime and
        shapes its own request, so it contributes neither a prompt template nor inference params.
        """
        if isinstance(target, ModelTarget):
            return target.model, target.prompt_template, target.params or RunConfigOnlineModel()
        if isinstance(target, AgentTarget):
            return target.agent, None, target.params or RunConfigOnline()
        if isinstance(target, FabricRunnerTarget):
            fabric_runtime = FabricAgentRuntime(
                config=target.config,
                model=target.model,
                timeout_s=target.timeout_s,
                capture_trajectory=target.capture_trajectory,
                work_root=ctx.storage.persistent / "fabric",
            )
            return fabric_runtime, None, None
        if isinstance(target, GymRunnerTarget):
            # Sandboxing is a deployment decision, not a job field: the same target runs colocated
            # on a trusted box and sandboxed on a shared cluster. The decision, and the settings it
            # needs, were resolved and validated by the compiler in the evaluator service -- the
            # only place the operator's configuration exists. This container reads the result; it
            # cannot re-derive it, because none of those variables are set here.
            plan = sandbox_plan_from_environment()
            if plan is not None:
                return (
                    SessionBackedGymRunner(
                        target=target,
                        plan=plan,
                        job_id=ctx.job_id,
                        workspace=ctx.workspace,
                        persistent_storage_path=ctx.storage.persistent,
                    ),
                    None,
                    None,
                )
            if target.environment is not None:
                # Colocated GymAgentTaskRunner ignores a staged FileSet. A missing plan here means
                # the compiler did not sandbox the run, so refuse rather than evaluate the image.
                raise SandboxUnavailableError(
                    "Gym environment FileSets require sandboxed execution. Enable `sandboxed_gym_default`, "
                    "or omit `target.environment` so colocated GymAgentTaskRunner cannot ignore the staged package."
                )
            if target.agent_config is None:
                raise ValueError(
                    "The agent_config field is required for colocated Gym execution; package-supplied agents "
                    "are supported only by the sandboxed Gym host"
                )
            gym_runtime = GymAgentTaskRunner(
                config=GymRuntimeConfig(
                    agent=target.agent,
                    agent_config=target.agent_config,
                    resources_server=target.resources_server,
                    model_type=target.model_type,
                    bind_resources_server=target.bind_resources_server,
                    hydra_params=target.hydra_params,
                    env_vars=target.env_vars,
                    num_repeats=target.num_repeats,
                    concurrency=target.concurrency,
                    startup_timeout_s=target.startup_timeout_s,
                    collection_timeout_s=target.collection_timeout_s,
                    shutdown_grace_s=target.shutdown_grace_s,
                    reward_key=target.reward_key,
                )
            )
            return gym_runtime, None, None
        if isinstance(target, HarborRunnerTarget):
            harbor_runtime = HarborAgentTaskRunner(
                config=HarborRuntimeConfig(
                    jobs_dir=ctx.storage.persistent / "harbor",
                    agent_name=target.agent_name,
                    agent_import_path=target.agent_import_path,
                    agent_model_name=target.agent_model_name,
                    agent_kwargs=target.agent_kwargs,
                    n_attempts=target.n_attempts,
                    n_concurrent_trials=target.n_concurrent_trials,
                    max_retries=target.max_retries,
                    artifacts=target.artifacts,
                    trace_dir=target.trace_dir,
                    reward_key=target.reward_key,
                )
            )
            return harbor_runtime, None, None
        return None, None, None

    @staticmethod
    def _write_result_files(result: AgentEvalResult, persistent_dir: Path) -> AgentEvalResultFiles:
        """Persist the run bundle (trials/scores/tasks/summary) under the job's storage."""
        bundle_dir = persistent_dir / AGENT_BUNDLE_DIR
        # No HTML dashboard for job runs: the artifact is consumed programmatically, and the job
        # config asked for no dashboard before persistence became an explicit call.
        result.persist(bundle_dir, write_dashboard=False)
        return AgentEvalResultFiles(bundle_dir=bundle_dir, summary=bundle_dir / SUMMARY_FILE_NAME)

    def run(
        self,
        config: dict,
        *,
        ctx: JobContext,
        sdk: NemoClient | NeMoPlatform | None = None,
        async_sdk: AsyncNemoClient | AsyncNeMoPlatform | None = None,
    ) -> dict:
        """Run the agent evaluation locally and persist its result bundle as artifacts."""
        client = as_nemo_client(sdk)
        async_client = as_async_nemo_client(async_sdk)
        spec = AgentEvalSpec.model_validate(config)
        tasks = [_to_runtime_task(task) for task in spec.tasks]
        target, prompt_template, params = self._resolve_target(spec.target, ctx)
        run_config = AgentEvalRunConfig(
            params=params,
            prompt_template=prompt_template,
            parallelism=spec.max_concurrent_tasks,
            labels=spec.labels,
            fail_fast=spec.fail_fast,
        )
        # Forward whichever identity is present, preferring async when both are — the same
        # precedence the SDK-backed dataset resolver uses.
        evaluator = self._build_evaluator(async_client or client, spec.target)
        result = evaluator.run_sync(tasks=tasks, trials=spec.trials, target=target, config=run_config)

        files = self._write_result_files(result, ctx.storage.persistent)
        artifact = ctx.results.save(DEFAULT_RESULT_NAME, files.bundle_dir)
        ctx.results.save(SUMMARY_RESULT_NAME, files.summary)

        # Persist the queryable result record (aggregates + coverage); the full bundle (trials) lives
        # in the fileset referenced by `artifact`. Best-effort: the authoritative output (bundle +
        # summary artifacts) is already saved above, so a persistence failure must not fail an
        # otherwise-successful eval — log and continue.
        try:
            persist_agent_eval_result(
                result, target=spec.target, ctx=ctx, bundle_ref=artifact.artifact_url, async_sdk=async_client
            )
        except Exception:
            logger.warning(
                "Failed to persist agent-eval result record; the result bundle artifact is unaffected",
                exc_info=True,
            )

        output = {"status": "completed", "artifact": artifact.model_dump()}

        # Publication runs last, after the bundle and the queryable record are both durable, so a
        # failed publish costs a re-publish rather than a re-run. It is also the only step here that
        # can fail the job (when `required`), which is why nothing depends on its result.
        publication = spec.publication.intake if spec.publication is not None else None
        if publication is not None:
            intake = AsyncIntakeClient.from_client(async_client) if async_client is not None else None
            outcome = publish_agent_eval_result(
                result,
                spec=publication,
                target=spec.target,
                workspace=ctx.workspace,
                intake=intake,
            )
            output["publication"] = outcome.model_dump(exclude_none=True)

        return output
