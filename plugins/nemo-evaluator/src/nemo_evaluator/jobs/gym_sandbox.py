# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a submitted ``GymRunnerTarget`` into a sandboxed Gym host, service-side.

Whether a Gym evaluation is sandboxed is a property of the deployment, not of the job. The same
target runs colocated on a trusted dev box and sandboxed on a shared cluster, so nothing here is
authorable by a submitter: the rollout URL and token do not exist until a session is provisioned,
and the PVC claims, images and egress policy are cluster facts. Customizer settled this the same
way for GRPO, which keeps one submit contract across both.

What the job *does* supply is the environment selection, and that maps onto Gym's own configuration
mechanism without needing to understand its schema. Gym resolves ``config_paths`` itself, after
merging whatever initial config it is handed, so the selection travels as a list of YAML paths and
Gym loads them inside the sandbox where the environment is mounted.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from nemo_evaluator.config import EvaluatorConfig
from nemo_evaluator.jobs.agent_spec import GymRunnerTarget
from nemo_platform_plugin.jobs.execution_profiles import (
    KubernetesJobExecutionProfile,
    VolcanoJobExecutionProfile,
)
from nemo_platform_plugin.jobs.image import get_qualified_image
from nemo_platform_plugin.jobs.spec import BaseExecutionProfile
from pydantic import BaseModel, ConfigDict, Field

#: Env-var names that look like a credential. Used to refuse a sandboxed run that would hand one to
#: user-supplied environment code through `env_vars`; `env_secrets` is the supported route.
_CREDENTIAL_PATTERN = re.compile(r"(API_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_?KEY)", re.IGNORECASE)
#: Removed by the Gym host before handing the config to NeMo Gym.
ENVIRONMENT_COMPONENT_SELECTION_CONFIG_KEY = "_nmp_environment_component_selection"
GYM_HOST_IMAGE = "nmp-gym-host"


class SandboxUnavailableError(RuntimeError):
    """The deployment cannot run this evaluation sandboxed, and will not fall back silently."""


def _asset_config_path(parent: str, value: str) -> str:
    """Resolve a named Gym asset to its config path, as the ``gym`` CLI's selectors do.

    ``--resources-server mcqa`` and ``--model-type openai_model`` are sugar: each appends
    ``<parent>/<name>/configs/<flavor>.yaml`` to ``config_paths``, with the flavor defaulting to the
    name and a ``name/flavor`` value selecting a different one. Reproduced here so the sandboxed
    path selects the same configs the CLI path would.

    The path stays relative. Gym resolves it against ``NEMO_GYM_EXTRA_ROOTS``, then the working
    directory, then its install root -- which is how a mounted environment is found inside the
    sandbox, and why resolving it here would be wrong.
    """
    name, _, flavor = value.partition("/")
    return f"{parent}/{name}/configs/{flavor or name}.yaml"


def gym_global_config(target: GymRunnerTarget) -> dict[str, Any]:
    """Build the Gym global config for a target, as nested data rather than Hydra strings.

    The CLI path flattens `hydra_params` into `+a.b.c=value` overrides because that is what a
    command line accepts. Here the config *is* a dict, so the nested form is passed through as-is --
    fewer conversions, and no quoting grammar to get wrong.
    """
    model_config = _asset_config_path("responses_api_models", target.model_type)
    resources_server_config = _asset_config_path("resources_servers", target.resources_server)
    config_paths = [model_config, resources_server_config]
    if target.agent_config is not None:
        config_paths.insert(0, target.agent_config)
    config: dict[str, Any] = {"config_paths": config_paths}

    if target.bind_resources_server:
        # The CLI's `+{agent}.responses_api_agents.{agent}.resources_server.name={server}`, as data.
        agent_instance = (target.agent_ref_name or target.agent) if target.environment is not None else target.agent
        config[agent_instance] = {
            "responses_api_agents": {
                target.agent: {"resources_server": {"name": target.resources_server}},
            }
        }

    # Merged last so an explicit override wins over the derived binding, matching the CLI ordering.
    for key, value in target.hydra_params.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value

    if target.environment is not None:
        # Gym does not read this key. The host uses it to rebuild config_paths.
        config[ENVIRONMENT_COMPONENT_SELECTION_CONFIG_KEY] = {
            "agent_instance": target.agent_ref_name or target.agent,
            "agent_config": target.agent_config,
            "resources_server_instance": target.resources_server,
            "resources_server_config": resources_server_config,
            "model_config": model_config,
        }
    return config


def credential_shaped_env_vars(target: GymRunnerTarget) -> list[str]:
    """Names in ``env_vars`` that look like credentials.

    ``env_vars`` reaches the Gym host's bootstrap environment, and in sandboxed mode that host runs
    user-supplied environment code. A key that arrives this way is readable by it.
    """
    return sorted(name for name in target.env_vars if _CREDENTIAL_PATTERN.search(name))


def require_sandbox_available(config: EvaluatorConfig) -> None:
    """Raise unless this deployment can actually provision a sandboxed Gym host.

    Checked before anything is provisioned, so a misconfigured deployment fails at compile time with
    a message naming the setting rather than after the cost of a partial run.
    """
    if not config.sandbox_cluster_capable:
        raise SandboxUnavailableError(
            "sandboxed Gym execution is enabled but this cluster is not marked sandbox-capable. "
            "Set `sandbox_cluster_capable` once OpenSandbox is available, or disable "
            "`sandboxed_gym_default` to run Gym in the job container."
        )
    missing = [
        name for name, value in (("sandbox_job_storage_pvc_claim", config.sandbox_job_storage_pvc_claim),) if not value
    ]
    if missing:
        raise SandboxUnavailableError(
            f"sandboxed Gym execution needs {', '.join(missing)} configured; a host cannot be provisioned without them."
        )
    if not config.sandbox_policy_base_urls and not config.sandbox_egress_allow:
        # A sandboxed host denies all egress except the broker. With no model endpoint allowed it
        # comes up healthy and fails at the first rollout, far from the setting that caused it --
        # so refuse here, where the message can name it.
        raise SandboxUnavailableError(
            "sandboxed Gym execution needs at least one of `sandbox_policy_base_urls` or "
            "`sandbox_egress_allow`; a sandboxed host denies all egress except the episode broker, "
            "so an environment would have no route to a model."
        )


def require_no_plaintext_credentials(target: GymRunnerTarget) -> None:
    """Raise if a sandboxed run would hand a plaintext credential to user environment code."""
    exposed = credential_shaped_env_vars(target)
    if not exposed:
        return
    raise SandboxUnavailableError(
        f"these `env_vars` look like credentials and would be readable by the sandboxed environment's "
        f"own code: {', '.join(exposed)}. Move them to `env_secrets` as secret references, which the "
        "service resolves into the job environment, or disable sandboxed execution for this deployment."
    )


#: Step-environment variable carrying the resolved :class:`SandboxPlan` to the job. Its presence
#: *is* the decision to sandbox: the job container has no evaluator configuration to consult.
GYM_SANDBOX_PLAN_ENVVAR = "NEMO_EVALUATOR_GYM_SANDBOX_PLAN"


class SandboxPlan(BaseModel):
    """The deployment's sandbox settings, resolved and validated once, service-side.

    Every field here comes from :class:`EvaluatorConfig` -- that is, from the environment of the
    *evaluator service*, which is where the operator configures a deployment. The job runs somewhere
    else entirely (the CPU tasks container for sandboxed Gym), with no reason to carry those
    variables, so reading them there would silently yield model defaults and quietly run an
    evaluation colocated that the operator asked to be sandboxed. Resolving at compile time also
    means a deployment that cannot sandbox is refused at submit, which is what
    :func:`require_sandbox_available` already claims.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    host_provider: str
    runtime_image: str
    job_storage_pvc_claim: str
    environment_sub_path: str
    workspace_sub_path: str
    resources: dict[str, str] | None = None
    host_provider_options: dict[str, Any] = Field(default_factory=dict)
    egress_allow: tuple[str, ...] = ()
    policy_base_urls: tuple[str, ...] = ()
    episode_backend: str
    allow_insecure_memory_backend: bool = False
    approved_images: tuple[str, ...] = ()


def resolve_sandbox_plan(
    config: EvaluatorConfig,
    target: GymRunnerTarget,
    *,
    sandbox_server_protocol: str | None = None,
) -> SandboxPlan | None:
    """The sandbox settings for this target, or ``None`` when the deployment runs Gym colocated.

    Raises rather than falling back: a cluster that cannot sandbox refuses the run instead of
    quietly running user environment code beside this job's credentials. A FileSet environment
    cannot run colocated at all -- ``GymAgentTaskRunner`` would ignore the staged package.
    """
    if target.environment is not None:
        require_fileset_environment_sandboxed(target, config)
    if not config.sandboxed_gym_default:
        return None
    require_sandbox_available(config)
    require_no_plaintext_credentials(target)
    runtime_image = config.sandbox_runtime_image or get_qualified_image(GYM_HOST_IMAGE)
    # `require_sandbox_available` has just established that this is not empty.
    assert config.sandbox_job_storage_pvc_claim is not None
    host_provider_options = dict(config.sandbox_host_provider_options)
    # In-cluster OpenSandbox speaks http. If we leave this unset, the host probes the
    # sandbox over https and the readiness wait times out.
    if config.sandbox_host_provider == "opensandbox" and sandbox_server_protocol:
        connection = dict(host_provider_options.get("connection") or {})
        connection["protocol"] = sandbox_server_protocol
        host_provider_options["connection"] = connection
    return SandboxPlan(
        host_provider=config.sandbox_host_provider,
        runtime_image=runtime_image,
        job_storage_pvc_claim=config.sandbox_job_storage_pvc_claim,
        environment_sub_path=config.sandbox_environment_sub_path,
        workspace_sub_path=config.sandbox_workspace_sub_path,
        resources=config.sandbox_resources,
        host_provider_options=host_provider_options,
        egress_allow=tuple(config.sandbox_egress_allow),
        policy_base_urls=tuple(config.sandbox_policy_base_urls),
        episode_backend=config.sandbox_episode_backend,
        allow_insecure_memory_backend=config.sandbox_allow_insecure_memory_backend,
        approved_images=tuple(config.sandbox_approved_images),
    )


def require_fileset_environment_sandboxed(target: GymRunnerTarget, config: EvaluatorConfig) -> None:
    """Refuse a custom environment that colocated execution would silently ignore."""
    if target.environment is None:
        return
    if not config.sandboxed_gym_default:
        raise SandboxUnavailableError(
            "Gym environment FileSets require sandboxed execution. Enable `sandboxed_gym_default`, "
            "or omit `target.environment` so colocated GymAgentTaskRunner cannot ignore the staged package."
        )
    require_sandbox_available(config)
    require_no_plaintext_credentials(target)


def job_storage_pvc_name(profile: BaseExecutionProfile) -> str | None:
    """PVC claim used by Kubernetes/Volcano execution-profile job storage, if any."""
    if not isinstance(profile, KubernetesJobExecutionProfile | VolcanoJobExecutionProfile):
        return None
    pvc_name = profile.config.storage.pvc_name
    return pvc_name or None


def require_fileset_sandbox_storage_identity(
    target: GymRunnerTarget,
    config: EvaluatorConfig,
    *,
    execution_profile: BaseExecutionProfile | None,
) -> None:
    """Fail when staging would write PVC A and OpenSandbox would mount PVC B."""
    if target.environment is None or config.sandbox_host_provider == "docker":
        return
    job_claim = job_storage_pvc_name(execution_profile) if execution_profile is not None else None
    sandbox_claim = config.sandbox_job_storage_pvc_claim
    if job_claim is None or sandbox_claim is None:
        return
    if job_claim != sandbox_claim:
        raise SandboxUnavailableError(
            f"FileSet-backed Gym execution stages onto the Jobs execution-profile storage PVC "
            f"{job_claim!r} but OpenSandbox mounts `sandbox_job_storage_pvc_claim` {sandbox_claim!r}. "
            f"Set `sandbox_job_storage_pvc_claim` to {job_claim!r} so the staged environment is the "
            "one the Gym host mounts."
        )


def _egress_rules(plan: SandboxPlan) -> list[dict[str, Any]]:
    """Parse `host:port` egress entries into the host spec's rule shape.

    A malformed entry raises rather than being skipped: an allowlist that silently drops what it
    cannot parse grants less than the operator asked for, and the run then fails at a network call
    with nothing pointing back at the typo.
    """
    rules: list[dict[str, Any]] = []
    for entry in plan.egress_allow:
        host, separator, port = entry.rpartition(":")
        if not separator or not host or not port.isdigit():
            raise SandboxUnavailableError(f"`sandbox_egress_allow` entries must be `host:port`; got {entry!r}")
        rules.append({"host": host, "port": int(port)})
    return rules


def host_env(target: GymRunnerTarget) -> dict[str, str]:
    """The target's own environment variables, for the Gym host container.

    Two sources, both belonging to the job rather than the deployment: ``env_vars`` travels on the
    spec, and ``env_secrets`` was resolved by the service into *this* container's environment, which
    is the only place its value exists. Without this the host runs with neither, and an environment
    that needs a model API key fails inside the sandbox with no indication that the credential it
    was given never arrived.

    Credential-shaped ``env_vars`` never reach here: ``require_no_plaintext_credentials`` refuses
    the run instead, since the sandbox executes the environment's own code.
    """
    env = dict(target.env_vars)
    for name in target.env_secrets:
        value = os.environ.get(name)
        if value is None:
            # The service resolves every `env_secrets` entry into the job container, so a missing
            # one means the resolution failed. Failing here names the variable; letting it through
            # fails inside the sandbox as whatever the environment does without its credential.
            raise SandboxUnavailableError(
                f"`env_secrets` entry {name!r} was not resolved into this job's environment, so the "
                "sandboxed Gym host cannot be given it."
            )
        env[name] = value
    return env


def serve_config(
    target: GymRunnerTarget,
    plan: SandboxPlan,
    *,
    job_id: str,
    workspace: str = "default",
    persistent_storage_path: Path | None = None,
) -> dict[str, Any]:
    """Assemble the ``SandboxedGymServeConfig`` payload for one evaluation.

    Job-derived and deployment-derived settings meet here and nowhere else: the target supplies the
    environment selection and its own environment variables, the resolved plan supplies the cluster
    facts, and the broker token and rollout URL are minted by the session itself.
    """
    environment_pvc_claim = plan.job_storage_pvc_claim
    host_provider_options = dict(plan.host_provider_options)
    fileset_environment = target.environment is not None

    if fileset_environment:
        # Each FileSet is staged onto this job's persistent directory. A shared environment mount
        # would miss that tree (and let concurrent jobs clobber each other).
        environment_sub_path = f"jobs/{workspace}/{job_id}/{plan.environment_sub_path}"
        workspace_sub_path = f"jobs/{workspace}/{job_id}/{plan.workspace_sub_path}"
    else:
        environment_sub_path = plan.environment_sub_path
        # Job-scoped, unlike the shared read-only environment mount: the configured workspace
        # sub-path is deployment-wide, so two concurrent evaluations would otherwise scribble over
        # each other's state in one directory.
        workspace_sub_path = f"{plan.workspace_sub_path.rstrip('/')}/{job_id}"

    # The subprocess backend gives both evaluator steps one real host directory. A FileSet-backed
    # Docker host must bind that directory rather than reconstructing a Kubernetes PVC layout under
    # its configured root. Keep this trusted runtime path out of the submitted target.
    if plan.host_provider == "docker" and fileset_environment:
        if persistent_storage_path is None:
            raise SandboxUnavailableError(
                "FileSet-backed Docker Gym execution requires the job's persistent storage path"
            )
        persistent_storage_path = persistent_storage_path.resolve()
        host_provider_options["root_dir"] = str(persistent_storage_path.parent)
        environment_pvc_claim = persistent_storage_path.name
        environment_sub_path = plan.environment_sub_path
        workspace_sub_path = plan.workspace_sub_path

    return {
        "job_id": job_id,
        "host_provider": plan.host_provider,
        "environment_path": "/job/environment" if fileset_environment else None,
        "sandbox": {
            "image": plan.runtime_image,
            # One claim, two sub-paths. The environment mount is read-only and the workspace is not,
            # so they must not resolve to the same directory.
            "environment_pvc_claim": environment_pvc_claim,
            "environment_sub_path": environment_sub_path,
            "workspace_pvc_claim": environment_pvc_claim,
            "workspace_sub_path": workspace_sub_path,
            "resources": plan.resources,
            "host_provider_options": host_provider_options,
            "network_policy": {"egress_allow": _egress_rules(plan)},
        },
        "episode_broker": {
            "job_id": job_id,
            "backend": plan.episode_backend,
            "allow_insecure_memory_backend": plan.allow_insecure_memory_backend,
            "approved_images": plan.approved_images,
        },
        # The host reaches a model through these. The broker's own address is added by the
        # orchestrator, so it is deliberately absent here.
        "policy_base_urls": plan.policy_base_urls,
        "gym_global_config": gym_global_config(target),
        "host_env": host_env(target),
    }


class SessionBackedGymRunner:
    """Provisions a sandboxed Gym host for the duration of one ``run_tasks`` and collects from it.

    The session's lifetime is the run's, which is why provisioning happens here rather than at
    target-resolution time: a host outliving its run is a leaked pod holding a PVC, and one created
    before the run is charged for whatever setup happens in between. Torn down in ``finally``, so a
    failed collection reclaims it too.
    """

    #: Job id used when the run has none -- an in-process run outside the platform. The broker requires a
    #: non-empty id because it scopes episode ownership and orphan reconciliation by it.
    LOCAL_JOB_ID = "agent-eval-local"

    def __init__(
        self,
        *,
        target: GymRunnerTarget,
        plan: SandboxPlan,
        job_id: str | None,
        workspace: str = "default",
        persistent_storage_path: Path | None = None,
    ) -> None:
        """Keep the resolved plan and the job's storage identity for ``run_tasks``.

        ``workspace`` and ``persistent_storage_path`` exist so a FileSet-backed host mounts the
        tree the stage step wrote, rather than a shared deployment-wide environment cache.
        """
        self._target = target
        self._plan = plan
        self._job_id = job_id or self.LOCAL_JOB_ID
        self._workspace = workspace
        self._persistent_storage_path = persistent_storage_path
        self._delegate: Any | None = None

    def runner_info(self) -> Any:
        """Identify the run before a session exists, so provenance does not depend on provisioning."""
        from nemo_evaluator_sdk.agent_eval.trials import RunnerInfo

        if self._delegate is not None:
            return self._delegate.runner_info()
        return RunnerInfo(
            name="gym",
            kind="runner",
            config={
                "mode": "sandboxed",
                "resources_server": self._target.resources_server,
                "agent": self._target.agent,
                "reward_key": self._target.reward_key,
            },
        )

    def run_aggregate_scores(self) -> Any:
        if self._delegate is None:
            return []
        return self._delegate.run_aggregate_scores()

    async def run_tasks(self, tasks: Any, config: Any = None) -> Any:
        import asyncio

        from nemo_evaluator_sdk.agent_eval.runtimes.gym.sandboxed import (
            SandboxedGymAgentTaskRunner,
            SandboxedGymRuntimeConfig,
        )
        from sandboxed_gym import SandboxedGymOrchestrator, SandboxedGymServeConfig

        payload = serve_config(
            self._target,
            self._plan,
            job_id=self._job_id,
            workspace=self._workspace,
            persistent_storage_path=self._persistent_storage_path,
        )
        orchestrator = SandboxedGymOrchestrator()
        # `start` provisions a host and blocks on its readiness probe, so it runs off the event loop.
        session = await asyncio.to_thread(orchestrator.start, SandboxedGymServeConfig.model_validate(payload))
        try:
            descriptor = session.descriptor()
            self._delegate = SandboxedGymAgentTaskRunner(
                config=SandboxedGymRuntimeConfig(
                    rollout_url=descriptor.rollout_url,
                    auth_token=descriptor.rollout_auth_token,
                    headers=dict(descriptor.headers),
                    agent_ref_name=self._target.agent_ref_name or self._target.agent,
                    reward_key=self._target.reward_key,
                )
            )
            return await self._delegate.run_tasks(tasks, config)
        finally:
            await asyncio.to_thread(session.shutdown)


def sandbox_plan_from_environment() -> SandboxPlan | None:
    """The plan the compiler resolved for this job, or ``None`` when Gym runs colocated."""
    raw = os.environ.get(GYM_SANDBOX_PLAN_ENVVAR)
    if not raw:
        return None
    return SandboxPlan.model_validate_json(raw)
