# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell job backend: run job steps as OpenShell sandboxes.

The sandbox runs the jobs-launcher as its canonical main process
(``SandboxSpec.command``). The launcher fetches the step config from the jobs
API (``--fetch-step-config``), injects secrets, and execs the task command, so
the platform never has to reach into the sandbox after creation. The sandbox's
main-process exit code is the job's exit code: ``Completed`` maps to
``COMPLETED``, a nonzero ``exit_code`` to ``ERROR``.

The channel/policy/launch code is a separate copy from the deployments plugin's
OpenShell backend, so the jobs service does not depend on that plugin.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from nemo_helix_plugin.auth import AuthContext as PluginAuthContext
from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.capabilities import CapabilityUnavailableError
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.jobs.constants import OPENSHELL_JOB_TASK_IMAGE
from nemo_helix_plugin.jobs.execution_profiles import (
    OpenShellJobEgressConfig,
)
from nemo_helix_plugin.jobs.execution_profiles import (
    OpenShellJobExecutionProfileConfig as PluginOpenShellJobExecutionProfileConfig,
)
from nemo_helix_plugin.jobs.image import get_qualified_image
from nemo_helix_plugin.jobs.types import HelixJobStepWithContext
from nemo_helix_plugin.openshell.types import GatewayWorkload, ProviderAccessRevoked
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.common.auth import AuthContext
from nhx.common.config import get_platform_config, nhx_user_data_dir
from nhx.common.jobs.constants import (
    CONFIG_TASK_STORAGE_PATH_ENVVAR,
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ATTEMPT_ID_ENVVAR,
    NEMO_JOB_FILESET_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_SECRETS_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_NAME,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_STEP_ENVVAR,
    NEMO_JOB_TASK_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
)
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.constants import (
    JOB_ATTEMPT_ID_LABEL,
    JOB_CONTROLLER_INSTANCE_ID_LABEL,
    JOB_EXECUTION_BACKEND_LABEL,
    JOB_EXECUTION_PROFILE_LABEL,
    JOB_ID_LABEL,
    JOB_MANAGED_BY_JOBS_CONTROLLER,
    JOB_MANAGED_BY_LABEL,
    JOB_STEP_NAME_LABEL,
    JOB_TASK_ID_LABEL,
    JOB_WORKSPACE_ID_LABEL,
)
from nhx.core.jobs.controllers.backends.base import (
    NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR,
    JobBackend,
    JobUpdate,
    get_job_runtime_shared_envvars,
    get_logs_endpoint_from_fileset,
    step_ttl_seconds_active,
)
from nhx.core.jobs.controllers.backends.openshell.policy import (
    DEFAULT_EGRESS_BINARIES,
    HelixEgress,
    SandboxFilesystem,
    build_sandbox_policy,
    generate_policy_dict,
    inject_platform_egress,
    layer_policy_dict,
    load_policy_dict,
)

_OPENSHELL_IMPORT_ERROR: ImportError | None = None
try:
    import grpc
    from nemo_helix_plugin.openshell.isolation import restricted_platform_policy, validate_provider_environment
    from nhx.core.jobs.controllers.backends.openshell.sandbox_client import JobsSandboxClient
    from openshell import SandboxError, TlsConfig
    from openshell._proto import openshell_pb2 as pb

    from .provisioning import JobProviderProvisioner
except ImportError as exc:
    _OPENSHELL_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

_OPENSHELL_INSTALL_HINT = (
    "The 'openshell' package is required for the OpenShell job backend. "
    'Install it with: uv pip install "openshell>=0.1.6" "grpcio>=1.78.0" "protobuf>=6.31.1"'
)

# OpenShell rejects a longer sandbox name with INVALID_ARGUMENT (63-char DNS label).
_MAX_ROUTABLE_NAME_LEN = 19

# Default entrypoint for job steps that do not declare one. The supervisor ignores the
# image ENTRYPOINT, so the sandbox command must name the binary explicitly. It resolves
# on the image's own PATH: the supervisor applies the image ENV and WORKDIR to the
# workload, so the backend sets neither PATH, VIRTUAL_ENV, HOME, nor a working directory.
_DEFAULT_ENTRYPOINT = ("nemo-helix",)

# Job scratch storage, inside the sandbox's private /tmp so a job image needs no
# pre-created scratch tree. The task storage is /tmp itself, which exists and is
# world-writable in every image. The launcher creates the step-config directory
# when it writes the fetched config.
_TASK_STORAGE_PATH = "/tmp"
_CONFIG_STORAGE_PATH = "/tmp/nhx-job-config"
_STEP_CONFIG_FILE_PATH = f"{_CONFIG_STORAGE_PATH}/{NEMO_JOB_STEP_CONFIG_FILE_NAME}"

# Platform URLs the backend injects into the sandbox environment. The sandbox
# can only reach the platform at the egress host (the policy allows exactly that
# host:port), so all of these are rewritten to it.
_PLATFORM_URL_ENVVARS = (
    "NHX_BASE_URL",
    "NHX_AUTH_URL",
    "NHX_JOBS_URL",
    "NHX_FILES_URL",
    "NHX_MODELS_URL",
    "NHX_SECRETS_URL",
    NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR,
)


def _rewrite_platform_urls(env: dict[str, str], egress: OpenShellJobEgressConfig | None) -> None:
    """Point injected platform URLs at the egress host:port.

    The controller's own base URL is not resolvable from inside a sandbox: it is
    a loopback address on the docker driver and a short cluster service name on
    k8s. The egress host is the operator-declared address the sandbox uses to
    reach the platform, and the sandbox policy allows exactly that host:port.
    """
    if egress is None:
        return
    for key in _PLATFORM_URL_ENVVARS:
        url = env.get(key)
        if not url:
            continue
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            continue
        port = egress.port or parsed.port
        netloc = egress.host if port is None else f"{egress.host}:{port}"
        env[key] = urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _sandbox_name(workspace: str, job: str, attempt_id: str, step: str) -> str:
    """Sandbox name for a job step, within ``_MAX_ROUTABLE_NAME_LEN``.

    Deterministic across schedule/sync so sync can recompute it; the attempt id
    is included so a rescheduled attempt gets a fresh sandbox. The human-readable
    identity travels in labels.
    """
    digest = hashlib.sha256(f"{workspace}/{job}/{attempt_id}/{step}".encode()).hexdigest()[:14]
    return f"nhx-{digest}"


def _resolve_jobs_controller_instance_id() -> str:
    configured = os.getenv("NHX_JOBS_DOCKER_OWNER_ID")
    if configured:
        return configured
    owner_source = f"nhx-data-dir:{nhx_user_data_dir().expanduser().resolve()}"
    return hashlib.sha256(owner_source.encode("utf-8")).hexdigest()[:32]


def _ensure_openshell() -> None:
    """Report missing optional dependencies when this backend is selected."""
    if _OPENSHELL_IMPORT_ERROR is not None:
        raise CapabilityUnavailableError(_OPENSHELL_INSTALL_HINT) from _OPENSHELL_IMPORT_ERROR


# Sandbox phase -> job step status. The canonical main process result is the
# workload result: ``Completed`` is a successful run (exit 0), ``Error`` a failed
# run (nonzero exit). The lifecycle phases in between are scheduling states.
# Keyed by enum name so the table also loads without the optional SDK.
_PHASE_TO_STATUS: dict[str, HelixJobStatus] = {
    "SANDBOX_PHASE_UNSPECIFIED": HelixJobStatus.PENDING,
    "SANDBOX_PHASE_PROVISIONING": HelixJobStatus.PENDING,
    "SANDBOX_PHASE_READY": HelixJobStatus.ACTIVE,
    "SANDBOX_PHASE_ERROR": HelixJobStatus.ERROR,
    "SANDBOX_PHASE_DELETING": HelixJobStatus.ACTIVE,
    "SANDBOX_PHASE_UNKNOWN": HelixJobStatus.PENDING,
    "SANDBOX_PHASE_STARTING": HelixJobStatus.PENDING,
    "SANDBOX_PHASE_STOPPING": HelixJobStatus.PENDING,
    "SANDBOX_PHASE_STOPPED": HelixJobStatus.ERROR,
    "SANDBOX_PHASE_COMPLETED": HelixJobStatus.COMPLETED,
}


def _phase_name(phase: int) -> str:
    """Enum name for a sandbox phase; a value newer than this client is UNKNOWN."""
    try:
        return pb.SandboxPhase.Name(phase)
    except ValueError:
        return "SANDBOX_PHASE_UNKNOWN"


def _condition_message(status: Any) -> str:
    for condition in reversed(status.conditions):
        if condition.message:
            return condition.message
    return ""


class OpenShellJobBackend(JobBackend[Any, PluginOpenShellJobExecutionProfileConfig]):
    """Schedule job steps as OpenShell sandboxes via the gateway gRPC API."""

    BACKEND_NAME = "openshell"

    def init(self) -> None:
        _ensure_openshell()
        self._executor_config = PluginOpenShellJobExecutionProfileConfig.model_validate(self._execution_profile_config)
        self._workspace = self._executor_config.workspace
        self._client = self._create_client()
        self._policy = self._build_executor_policy()
        self._jobs_controller_instance_id = _resolve_jobs_controller_instance_id()

    # --- policy ---

    def _build_executor_policy(self) -> Any:
        """The SandboxPolicy applied to created sandboxes.

        A generated default-deny policy, with the hand-written YAML (``policy_path``)
        layered over it when given (see ``layer_policy_dict``). The platform egress rule
        is always injected as mandatory so a job can never lose its path back to the
        platform. When ``platform_egress`` is null the sandbox gets no direct egress at
        all, which breaks the launcher's config and secrets fetch and OTLP log export;
        the profile default sets it.
        """
        cfg = self._executor_config.platform_egress
        egress = (
            HelixEgress(
                host=cfg.host,
                port=cfg.port,
                protocol=cfg.protocol,
                tls=cfg.tls,
                access=cfg.access,
                # The launcher fetches the step config and secrets, so it always needs egress.
                binaries=tuple(
                    dict.fromkeys(
                        (self._executor_config.launcher_tool_path, *(cfg.binaries or DEFAULT_EGRESS_BINARIES))
                    )
                ),
            )
            if cfg is not None
            else None
        )
        policy_dict = generate_policy_dict(filesystem=SandboxFilesystem(), egress=egress)
        path = self._executor_config.policy_path
        if path:
            policy_dict = layer_policy_dict(policy_dict, load_policy_dict(path))
        if egress is not None:
            inject_platform_egress(policy_dict, egress)
        return build_sandbox_policy(policy_dict)

    # --- client ---

    def _create_client(self) -> JobsSandboxClient:
        """Plaintext for an http endpoint; otherwise TLS, mTLS when client material is set.

        With no CA configured the channel trusts the system roots.
        """
        config = self._executor_config
        tls = None
        if not config.use_insecure():
            material = config.tls
            tls = TlsConfig(
                ca_path=_optional_path(material.ca_cert_path if material else None),
                cert_path=_optional_path(material.client_cert_path if material else None),
                key_path=_optional_path(material.client_key_path if material else None),
            )
        return JobsSandboxClient(config.grpc_target(), tls=tls, timeout=config.request_timeout_seconds)

    def shutdown(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()

    # --- JobBackend contract ---

    @staticmethod
    def _require_step_spec(step: HelixJobStepWithContext) -> Any:
        if step.step_spec is None:
            raise ValueError("OpenShell job step requires a step_spec")
        return step.step_spec

    def schedule(
        self,
        executor_config: Any,
        step: HelixJobStepWithContext,
    ) -> JobUpdate:
        step_spec = self._require_step_spec(step)
        if any(item.from_secret is not None for item in step_spec.environment or []):
            return JobUpdate(
                status=HelixJobStatus.ERROR,
                status_details={
                    "message": "OpenShell secrets require gateway provider bindings; process-env injection is not supported."
                },
            )
        platform_config = get_platform_config()

        env = dict(self._execution_profile_config.env)
        task_id = f"task-{uuid.uuid4().hex}"
        # The sandbox reaches the platform at the egress host (the supervisor's
        # network), not at the API server's localhost; rewrite loopback URLs to it.
        egress = self._executor_config.platform_egress
        loopback_address = egress.host if egress is not None else None
        env.update(
            {
                NEMO_JOB_ID_ENVVAR: step.job,
                NEMO_JOB_ATTEMPT_ID_ENVVAR: step.attempt_id,
                NEMO_JOB_STEP_ENVVAR: step.name,
                NEMO_JOB_TASK_ENVVAR: task_id,
                NEMO_JOB_WORKSPACE_ENVVAR: step.workspace,
                NEMO_JOB_FILESET_ENVVAR: step.fileset,
                EPHEMERAL_TASK_STORAGE_PATH_ENVVAR: _TASK_STORAGE_PATH,
                CONFIG_TASK_STORAGE_PATH_ENVVAR: _CONFIG_STORAGE_PATH,
                NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR: _STEP_CONFIG_FILE_PATH,
                # The launcher fetches the step config from the jobs API instead of a
                # pre-written file, so it must be able to reach the platform.
                NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR: get_logs_endpoint_from_fileset(
                    platform_config,
                    step.workspace,
                    step.fileset,
                    loopback_address=loopback_address,
                ),
                NEMO_JOB_SECRETS_ENVVAR: self.get_secrets_environment_variable_for_injection(step),
            }
        )

        # Set auth context env vars so the launcher and the workload can make
        # authenticated API calls on behalf of the job creator.
        if step.auth_context:
            auth_context = AuthContext.model_validate(step.auth_context.model_dump(mode="python", exclude_none=True))
            env.update(auth_context.to_principal().get_env_var())

        # Step env vars (non-secret values).
        for envvar in step_spec.environment or []:
            if envvar.value is not None:
                env[envvar.name] = envvar.value

        # Thread through shared platform envvars (NHX_*_URL etc.).
        env.update(get_job_runtime_shared_envvars(platform_config, loopback_address=loopback_address))

        # The sandbox reaches the platform only at the egress host: the policy
        # allows exactly that host:port, and the controller's own base URL is
        # not resolvable from inside a sandbox (loopback on the docker driver,
        # a short cluster service name on k8s). Point every injected platform
        # URL at the egress host.
        _rewrite_platform_urls(env, egress)

        # The docker driver gives the sandbox no network of its own: all egress goes
        # through the supervisor's HTTP proxy, which enforces the sandbox policy. The
        # k8s driver fences with NetworkPolicies instead, so the proxy is injected only
        # when the profile configures one.
        if self._execution_profile_config.egress_proxy:
            env.setdefault("HTTP_PROXY", self._execution_profile_config.egress_proxy)
            env.setdefault("HTTPS_PROXY", self._execution_profile_config.egress_proxy)
            env.setdefault("NO_PROXY", "127.0.0.1,localhost")

        container = getattr(executor_config, "container", None)
        image = (
            (container.image if container is not None else None)
            or self._execution_profile_config.image
            or get_qualified_image(OPENSHELL_JOB_TASK_IMAGE)
        )
        entrypoint = list(container.entrypoint or []) if container is not None else []
        if not entrypoint:
            entrypoint = list(self._execution_profile_config.default_entrypoint or _DEFAULT_ENTRYPOINT)
        command = list(container.command or []) if container is not None else []

        # The launcher is the canonical main process: it fetches the step config,
        # injects secrets, then execs the task command.
        launcher = self._executor_config.launcher_tool_path
        sandbox_command = [launcher, "run", "--fetch-step-config", "--"] + entrypoint + command

        sandbox_name = _sandbox_name(step.workspace, step.job, step.attempt_id, step.name)
        labels = {
            **self._base_controller_labels(),
            JOB_WORKSPACE_ID_LABEL: step.workspace,
            JOB_ID_LABEL: step.job,
            JOB_ATTEMPT_ID_LABEL: step.attempt_id,
            JOB_STEP_NAME_LABEL: step.name,
            JOB_TASK_ID_LABEL: task_id,
        }

        template = pb.SandboxTemplate(image=image, environment=env)
        spec = pb.SandboxSpec(template=template, environment=env, policy=self._policy, command=sandbox_command)

        if step_spec.gateway_providers:
            try:
                if (
                    self._executor_config.gateway_credentials is None
                    or egress is None
                    or (platform_auth_enabled() and step.auth_context is None)
                ):
                    raise ValueError(
                        "Gateway providers require gateway_credentials, platform egress and creator identity."
                    )
                secret_env = {
                    item.name: item.from_secret.name
                    for item in step_spec.environment or []
                    if item.from_secret is not None
                }
                validate_provider_environment(
                    step_spec.gateway_providers,
                    workspace=step.workspace,
                    secret_env=secret_env,
                    env_names=set(env),
                )
                policy = restricted_platform_policy(
                    self._policy,
                    host=egress.host,
                    port=egress.port,
                    workspace=step.workspace,
                    binaries=list(dict.fromkeys((launcher, *(egress.binaries or DEFAULT_EGRESS_BINARIES)))),
                    job=step.job,
                    attempt=step.attempt_id,
                    step=step.name,
                    fileset=step.fileset,
                    secret_env=secret_env,
                    read_filesets=step_spec.gateway_read_filesets,
                )
                spec.policy.CopyFrom(policy)
                spec.providers.extend(asyncio.run(self._prepare_providers(step, sandbox_name)))
            except Exception:
                logger.warning("Gateway provider provisioning failed for job step %s", step.id)
                return JobUpdate(
                    status=HelixJobStatus.ERROR, status_details={"message": "Gateway provider provisioning failed."}
                )

        try:
            self._client.create(workspace=self._workspace, spec=spec, name=sandbox_name, labels=labels)
        except grpc.RpcError as exc:
            # The name is unique per attempt, so an existing sandbox is this attempt's own
            # (created by an earlier schedule whose PENDING update was not persisted).
            if _rpc_code(exc) == grpc.StatusCode.ALREADY_EXISTS:
                return JobUpdate(
                    status=HelixJobStatus.PENDING,
                    status_details={"message": f"Sandbox {sandbox_name} already exists; awaiting READY"},
                )
            return JobUpdate(
                status=HelixJobStatus.ERROR,
                status_details={"message": f"CreateSandbox failed: {_rpc_detail(exc)}"},
            )
        except SandboxError as exc:
            return JobUpdate(status=HelixJobStatus.ERROR, status_details={"message": f"CreateSandbox failed: {exc}"})

        return JobUpdate(
            status=HelixJobStatus.PENDING,
            status_details={"message": f"Sandbox {sandbox_name} created; awaiting READY"},
        )

    def sync(self, step: HelixJobStepWithContext) -> JobUpdate:
        sandbox_name = _sandbox_name(step.workspace, step.job, step.attempt_id, step.name)
        step_status = HelixJobStatus(getattr(step.status, "value", step.status))
        try:
            sandbox = self._client.get_sandbox(sandbox_name, workspace=self._workspace)
        except grpc.RpcError as exc:
            if _rpc_code(exc) == grpc.StatusCode.NOT_FOUND:
                if step_status == HelixJobStatus.CANCELLING:
                    if not self._cleanup_step_providers(step, sandbox_name):
                        return JobUpdate(
                            status=HelixJobStatus.CANCELLING,
                            status_details={"message": "Provider cleanup will be retried."},
                        )
                    return JobUpdate(status=HelixJobStatus.CANCELLED, status_details={"message": "Sandbox deleted"})
                return JobUpdate(
                    status=HelixJobStatus.ERROR,
                    status_details={"message": f"Sandbox {sandbox_name} not found"},
                )
            # Transient gateway error: keep the current status so the next sync retries.
            return JobUpdate(
                status=step_status,
                status_details={"message": f"GetSandbox error: {_rpc_detail(exc)}"},
            )

        phase = sandbox.status.phase
        terminal = phase in (pb.SANDBOX_PHASE_COMPLETED, pb.SANDBOX_PHASE_ERROR)

        # A cancel request stops the sandbox so the main process dies. A sandbox that
        # already finished is cancelled as-is: CANCELLING cannot move to COMPLETED.
        if step_status == HelixJobStatus.CANCELLING:
            if terminal and not (step.step_spec and step.step_spec.gateway_providers):
                return JobUpdate(
                    status=HelixJobStatus.CANCELLED,
                    status_details={"message": f"Sandbox already finished ({_phase_name(phase)})"},
                )
            try:
                self._client.delete(sandbox_name, workspace=self._workspace, allow_missing=True)
            except grpc.RpcError as exc:
                logger.warning(
                    "Failed to delete sandbox %s during cancel",
                    sandbox_name,
                    extra={"job": step.job, "step": step.name},
                )
                return JobUpdate(
                    status=HelixJobStatus.CANCELLING,
                    status_details={"message": f"Sandbox delete failed, retrying: {_rpc_detail(exc)}"},
                )
            if not self._cleanup_step_providers(step, sandbox_name):
                return JobUpdate(
                    status=HelixJobStatus.CANCELLING, status_details={"message": "Provider cleanup will be retried."}
                )
            return JobUpdate(status=HelixJobStatus.CANCELLED, status_details={"message": "Sandbox deleted"})

        if not terminal and (timeout := self._sync_ttl_exceeded(step, step_status, phase)) is not None:
            self._delete_sandbox_best_effort(sandbox_name)
            message = f"Job timed out after reaching max TTL of {timeout} seconds"
            return JobUpdate(
                status=HelixJobStatus.ERROR,
                status_details={"message": message},
                error_details={"message": message},
            )

        # exit_code is a proto3 optional: unset reads as 0, so presence must be checked.
        exit_code = sandbox.status.exit_code if sandbox.status.HasField("exit_code") else None

        if phase == pb.SANDBOX_PHASE_COMPLETED:
            if exit_code in (None, 0):
                return JobUpdate(status=HelixJobStatus.COMPLETED, status_details={"message": "Job completed"})
            return JobUpdate(
                status=HelixJobStatus.ERROR,
                status_details={"message": f"Job exited with code {exit_code}"},
                error_details={"exit_code": exit_code},
            )

        if phase == pb.SANDBOX_PHASE_ERROR:
            error_details: dict[str, Any] = {"sandbox_phase": pb.SandboxPhase.Name(phase)}
            if exit_code is not None:
                message = f"Job exited with code {exit_code}"
                error_details["exit_code"] = exit_code
            else:
                message = _condition_message(sandbox.status) or f"Sandbox phase {pb.SandboxPhase.Name(phase)}"
            return JobUpdate(
                status=HelixJobStatus.ERROR,
                status_details={"message": message},
                error_details=error_details,
            )

        phase_name = _phase_name(phase)
        status = _PHASE_TO_STATUS.get(phase_name, step_status)
        # An ACTIVE step never moves back to PENDING (e.g. a STOPPING sandbox).
        if status == HelixJobStatus.PENDING and step_status == HelixJobStatus.ACTIVE:
            status = HelixJobStatus.ACTIVE
        credentials = self._executor_config.gateway_credentials
        if (
            credentials is not None
            and credentials.renewal_mode == "helix"
            and step.step_spec is not None
            and step.step_spec.gateway_providers
            and step_status in (HelixJobStatus.PENDING, HelixJobStatus.ACTIVE)
            and phase in (pb.SANDBOX_PHASE_PROVISIONING, pb.SANDBOX_PHASE_READY)
        ):
            try:
                asyncio.run(self._prepare_providers(step, sandbox_name, renew=True))
            except ProviderAccessRevoked:
                return JobUpdate(
                    status=status,
                    status_details={
                        "message": "Gateway provider access was removed. Start a new workload to restore access."
                    },
                )
            except Exception:
                # RPC errors can contain registration tickets. Keep public status
                # and logs generic; the next reconcile retries without extending expiry.
                logger.warning("Gateway credential renewal failed for job step %s; retrying.", step.id)
                return JobUpdate(
                    status=status,
                    status_details={"message": "Gateway credential renewal failed; retrying. Credentials may expire."},
                )
        return JobUpdate(status=status, status_details={"message": f"Sandbox phase {phase_name}"})

    def _sync_ttl_exceeded(self, step: HelixJobStepWithContext, step_status: HelixJobStatus, phase: int) -> int | None:
        """The exceeded TTL in seconds for a non-terminal sandbox, or None."""
        config = self._execution_profile_config
        if step_status == HelixJobStatus.ACTIVE:
            ttl_seconds = step_ttl_seconds_active(step, config.ttl_seconds_active)
            return ttl_seconds if self.check_step_ttl(step, ttl_seconds) else None
        not_started = _PHASE_TO_STATUS.get(_phase_name(phase)) == HelixJobStatus.PENDING
        if (
            step_status == HelixJobStatus.PENDING
            and not_started
            and self.should_enforce_before_active_ttl(step)
            and self.check_step_ttl_before_active(step, config.ttl_seconds_before_active)
        ):
            return config.ttl_seconds_before_active
        return None

    def cleanup_steps(self) -> None:
        if self._executor_config.gateway_credentials is not None:
            try:
                asyncio.run(self._cleanup_unused_providers())
            except Exception:
                logger.warning("Gateway provider cleanup will retry on the next sweep.")
        try:
            sandboxes = self._client.list_all(workspace=self._workspace, label_selector=self._cleanup_label_selector())
        except (grpc.RpcError, SandboxError):
            # SandboxError: the SDK pager rejects a repeated or over-budget page token.
            logger.warning("Failed to list sandboxes for cleanup", exc_info=True)
            return

        for sandbox in sandboxes:
            labels = dict(sandbox.labels)
            if not self._sandbox_owned_by_this_controller(labels):
                continue
            workspace = labels.get(JOB_WORKSPACE_ID_LABEL)
            job = labels.get(JOB_ID_LABEL)
            attempt_id = labels.get(JOB_ATTEMPT_ID_LABEL)
            step_name = labels.get(JOB_STEP_NAME_LABEL)
            if not workspace or not job or not step_name:
                continue

            # Verify the step is terminal before cleaning up, so a resource that was
            # marked active is never reclaimed early and then reported as error.
            step = self.get_step_safe(
                job=job,
                step_name=step_name,
                workspace=workspace,
                attempt_id=attempt_id,
            )
            if step is not None and step.status not in ("cancelled", "error", "completed"):
                continue

            # A sandbox still running under a terminal step (e.g. a timed-out step whose
            # delete failed) or whose step entity is gone has nothing left to inspect.
            if step is None or sandbox.phase not in (pb.SANDBOX_PHASE_COMPLETED, pb.SANDBOX_PHASE_ERROR):
                self._delete_sandbox_best_effort(sandbox.name)
                continue

            if (
                sandbox.phase == pb.SANDBOX_PHASE_COMPLETED
                and self._execution_profile_config.cleanup_completed_jobs_immediately
            ):
                self._delete_sandbox_best_effort(sandbox.name)
                continue

            # The TTL runs from the sandbox's terminal transition, which only the full
            # proto carries and only some compute drivers stamp; otherwise from the
            # step's last update, i.e. when it went terminal.
            try:
                finished_at = self._terminal_transition_time(
                    self._client.get_sandbox(sandbox.name, workspace=self._workspace)
                )
            except grpc.RpcError:
                logger.warning("Failed to read sandbox %s for TTL cleanup", sandbox.name, exc_info=True)
                continue
            if finished_at is None:
                finished_at = _aware(step.updated_at)
            if finished_at is None or finished_at + datetime.timedelta(
                seconds=self._execution_profile_config.ttl_seconds_after_finished
            ) < datetime.datetime.now(datetime.timezone.utc):
                self._delete_sandbox_best_effort(sandbox.name)

    # --- helpers ---

    def _cleanup_step_providers(self, step: HelixJobStepWithContext, sandbox_name: str) -> bool:
        if step.step_spec is None or not step.step_spec.gateway_providers:
            return True
        try:
            asyncio.run(self._cleanup_providers(step.workspace, sandbox_name))
        except Exception:
            logger.warning("Provider cleanup failed for job step %s; retrying.", step.id)
            return False
        return True

    async def _cleanup_providers(self, workspace: str, sandbox_name: str) -> None:
        async with self._provider_provisioner() as provisioner:
            await provisioner.cleanup(workload_workspace=workspace, workload_id=sandbox_name)

    async def _cleanup_unused_providers(self) -> None:
        async with self._provider_provisioner() as provisioner:
            await provisioner.cleanup_unused()

    @asynccontextmanager
    async def _provider_provisioner(self) -> AsyncGenerator[JobProviderProvisioner]:
        config = self._executor_config.gateway_credentials
        if config is None:
            raise ValueError("Gateway credentials are not configured.")
        target = self._executor_config.grpc_target()
        if self._executor_config.use_insecure():
            channel = grpc.aio.insecure_channel(target)
        else:
            tls = self._executor_config.tls
            credentials = grpc.ssl_channel_credentials(
                root_certificates=Path(tls.ca_cert_path).read_bytes() if tls and tls.ca_cert_path else None,
                private_key=Path(tls.client_key_path).read_bytes() if tls and tls.client_key_path else None,
                certificate_chain=Path(tls.client_cert_path).read_bytes() if tls and tls.client_cert_path else None,
            )
            channel = grpc.aio.secure_channel(target, credentials)
        # Jobs invokes provider operations with asyncio.run; never retain a
        # channel across those event loops or create it on the controller thread.
        async with channel:
            yield JobProviderProvisioner(
                config, self._workspace, channel, timeout=self._executor_config.request_timeout_seconds
            )

    async def _prepare_providers(
        self, step: HelixJobStepWithContext, sandbox_name: str, *, renew: bool = False
    ) -> list[str]:
        if (
            (platform_auth_enabled() and step.auth_context is None)
            or step.step_spec is None
            or self._executor_config.platform_egress is None
        ):
            raise ValueError("Incomplete provider workload context.")
        context = PluginAuthContext.model_validate(step.auth_context.model_dump()) if step.auth_context else None
        async with (
            get_async_nemo_client(
                as_service="jobs",
                internal=True,
                on_behalf_of=context.to_principal() if platform_auth_enabled() and context else None,
            ) as client,
            self._provider_provisioner() as provisioner,
        ):
            return await provisioner.prepare(
                step.step_spec.gateway_providers,
                workload_workspace=step.workspace,
                workload_id=sandbox_name,
                workload=GatewayWorkload(
                    kind="job",
                    workspace=step.workspace,
                    name=step.job,
                    id=step.id,
                    attempt_id=step.attempt_id,
                    step_name=step.name,
                ),
                auth_context=context,
                auth_enabled=platform_auth_enabled(),
                secrets_client=AsyncSecretsClient.from_client(client),
                platform_host=self._executor_config.platform_egress.host,
                renew=renew,
            )

    def _base_controller_labels(self) -> dict[str, str]:
        return {
            JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
            JOB_CONTROLLER_INSTANCE_ID_LABEL: self._jobs_controller_instance_id,
            JOB_EXECUTION_BACKEND_LABEL: self.BACKEND_NAME,
            JOB_EXECUTION_PROFILE_LABEL: self._profile_name,
        }

    def _cleanup_label_selector(self) -> str:
        return ",".join(
            [
                f"{JOB_MANAGED_BY_LABEL}={JOB_MANAGED_BY_JOBS_CONTROLLER}",
                f"{JOB_CONTROLLER_INSTANCE_ID_LABEL}={self._jobs_controller_instance_id}",
                f"{JOB_EXECUTION_BACKEND_LABEL}={self.BACKEND_NAME}",
                f"{JOB_EXECUTION_PROFILE_LABEL}={self._profile_name}",
            ]
        )

    def _sandbox_owned_by_this_controller(self, labels: dict[str, str]) -> bool:
        return (
            labels.get(JOB_MANAGED_BY_LABEL) == JOB_MANAGED_BY_JOBS_CONTROLLER
            and labels.get(JOB_CONTROLLER_INSTANCE_ID_LABEL) == self._jobs_controller_instance_id
            and labels.get(JOB_EXECUTION_BACKEND_LABEL) == self.BACKEND_NAME
            and labels.get(JOB_EXECUTION_PROFILE_LABEL) == self._profile_name
        )

    def _terminal_transition_time(self, sandbox: Any) -> datetime.datetime | None:
        """The last condition transition time for a terminal sandbox, or None."""
        for condition in reversed(sandbox.status.conditions):
            # A message field is always truthy; an unset Timestamp would read as the epoch.
            if condition.HasField("transition_time"):
                # protobuf ToDatetime() returns naive UTC; make it aware so TTL
                # comparisons against datetime.now(timezone.utc) do not raise.
                return condition.transition_time.ToDatetime(tzinfo=datetime.timezone.utc)
        return None

    def _delete_sandbox_best_effort(self, sandbox_name: str) -> None:
        try:
            self._client.delete(sandbox_name, workspace=self._workspace, allow_missing=True)
        except Exception:
            logger.warning("Failed to delete sandbox %s during cleanup", sandbox_name, exc_info=True)


def _rpc_code(exc: Any) -> Any:
    getter = getattr(exc, "code", None)
    if not callable(getter):
        return None
    return getter()


def _rpc_detail(exc: Any) -> str:
    code = _rpc_code(exc)
    details_getter = getattr(exc, "details", None)
    details = details_getter() if callable(details_getter) else str(exc)
    return f"{code.name if code is not None else 'UNKNOWN'}: {details}"


def _aware(value: datetime.datetime | None) -> datetime.datetime | None:
    """Treat a naive datetime as UTC."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=datetime.timezone.utc)


def _optional_path(path: str | None) -> Path | None:
    return Path(path) if path else None
