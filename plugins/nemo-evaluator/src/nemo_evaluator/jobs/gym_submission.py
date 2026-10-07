# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a Gym runner target needs at submission and at run time, apart from the job's orchestration.

At submission: the rows are checked against Gym's input contract and the environment FileSet, if
any, is validated and qualified through the Files service. At run time: the target becomes the
runtime that collects rollouts, sandboxed when the compiler's plan says so, colocated otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence

from filesets import FilesetPathError, parse_fileset_ref
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_spec import GymRunnerTarget, RegisteredAgentSource, ResolvedTask
from nemo_evaluator.jobs.gym_environment_package import (
    ENVIRONMENT_MANIFEST_FILENAME,
    GymEnvironmentPackageError,
    parse_environment_manifest,
    validate_environment_manifest_against_listing,
)
from nemo_evaluator.jobs.gym_sandbox import (
    SandboxUnavailableError,
    SessionBackedGymRunner,
    sandbox_plan_from_environment,
)
from nemo_evaluator.jobs.kinds.types import SubmitContext
from nemo_evaluator.jobs.secret_env import JobEnvSecretSource
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig, validate_gym_task_row
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import FilesetPurpose
from nemo_helix_plugin.job_context import JobContext


async def resolve_gym_environment(
    target: GymRunnerTarget,
    *,
    workspace: str,
    async_client: AsyncHelixClient | None,
) -> GymRunnerTarget:
    """Validate and qualify a Gym environment FileSet through the Files service."""
    if target.environment is None:
        return target

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

    files = client_from_platform(async_client, AsyncFilesClient)
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


async def prepare_gym_submission(
    resolved_tasks: Sequence[ResolvedTask], target: GymRunnerTarget, ctx: SubmitContext
) -> GymRunnerTarget:
    """Validate Gym rows and resolve its environment after task compatibility checks."""
    for task in resolved_tasks:
        if task.spec.kind != "evaluator":
            raise ValueError("Gym requires evaluator tasks")
        validate_gym_task_row(
            task_id=task.id,
            inputs=task.spec.inputs.model_dump(exclude_none=True),
            metadata={item.key: item.value for item in task.metadata},
        )
    return await resolve_gym_environment(target, workspace=ctx.workspace, async_client=ctx.async_client)


def gym_runtime_target(target: GymRunnerTarget, ctx: JobContext) -> SessionBackedGymRunner | GymAgentTaskRunner:
    """The runtime a Gym target runs as inside the job: sandboxed when the compiler planned it, else colocated."""
    # Sandboxing is a deployment decision, not a job field: the same target runs colocated
    # on a trusted box and sandboxed on a shared cluster. The decision, and the settings it
    # needs, were resolved and validated by the compiler in the evaluator service -- the
    # only place the operator's configuration exists. This container reads the result; it
    # cannot re-derive it, because none of those variables are set here.
    plan = sandbox_plan_from_environment()
    if plan is not None:
        return SessionBackedGymRunner(
            target=target,
            plan=plan,
            job_id=ctx.job_id,
            workspace=ctx.workspace,
            persistent_storage_path=ctx.storage.persistent,
        )
    if target.environment is not None:
        # Colocated GymAgentTaskRunner ignores a staged FileSet. A missing plan here means
        # the compiler did not sandbox the run, so refuse rather than evaluate the image.
        raise SandboxUnavailableError(
            "Gym environment FileSets require sandboxed execution. Enable `sandboxed_gym_default`, "
            "or omit `target.environment` so colocated GymAgentTaskRunner cannot ignore the staged package."
        )
    if isinstance(target.source, RegisteredAgentSource):
        raise SandboxUnavailableError(
            "A registered agent runs from the environment package the staging step assembles, which only "
            "the sandboxed Gym host mounts. Enable `sandboxed_gym_default`, or select a Gym agent by `component`."
        )
    if target.agent_ref_name is not None:
        raise SandboxUnavailableError(
            "The agent_ref_name field requires sandboxed execution; colocated GymAgentTaskRunner "
            "resolves its agent from Gym config and would route rollouts to "
            f"{target.agent!r} instead of {target.agent_ref_name!r}. Enable `sandboxed_gym_default`, "
            "or omit it."
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
            env_secrets=target.env_secrets,
            num_repeats=target.num_repeats,
            concurrency=target.concurrency,
            startup_timeout_s=target.startup_timeout_s,
            collection_timeout_s=target.collection_timeout_s,
            shutdown_grace_s=target.shutdown_grace_s,
            reward_key=target.reward_key,
        ),
        secret_resolver=JobEnvSecretSource(workspace=ctx.workspace),
    )
    return gym_runtime
