# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the agent-hardener jobs.

Both the war-game and (future) synth stages shell out to agent-hardener's CLI inside its dedicated venv and
need the same env wiring (garak venv + agent-hardener's own inference key + the victim's secrets) and the same
TTY-aware subprocess execution. This module holds that shared logic so the jobs don't duplicate it.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from subprocess import run
from typing import Any, NamedTuple, cast

from nemo_agent_hardener_plugin.config import (
    GARAK_PYTHON_ENVVAR,
    INFERENCE_API_KEY_ENVVAR,
    AgentHardenerConfig,
    declared_secrets,
    missing_secrets,
    read_env_file,
    read_manifest_agent,
    unprovided_secrets,
    write_env_file,
)
from nemo_agent_hardener_plugin.entities import AgentHardenerManifest
from nemo_agent_hardener_plugin.jobs.errors import (
    CATEGORY_MISSING_CREDENTIAL,
    CATEGORY_PROVISIONING,
    AgentHardenerRunError,
)
from nemo_agent_hardener_plugin.jobs.openshell_gateway import register_openshell_gateway
from nemo_agent_hardener_plugin.model_config import ModelChoice, WarGameModels
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.entities.base import parse_qualified_name
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import EnvironmentVariable, ExecutorSpec, SubprocessExecutionProviderSpec
from nemo_helix_plugin.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.jobs.executor import resolve_task_executor
from nemo_helix_plugin.secrets.client import AsyncSecretsClient, SecretsClient

logger = logging.getLogger(__name__)

TASK_IMAGE = "nhx-agent-hardener-tasks"
WAR_GAME_TASK = "nemo_agent_hardener_plugin.tasks.war_game"
# The subprocess executor forwards only PATH/VIRTUAL_ENV, but on the host `setup` provisioned, openshell reads
# its gateway registration from $HOME/.config/openshell (or $XDG_CONFIG_HOME) and reaches Docker via $DOCKER_HOST.
_HOST_ENVVARS = ("HOME", "DOCKER_HOST", "XDG_CONFIG_HOME")
# Task pods get no platform config file, so the job reads these from NEMO_AGENT_HARDENER_* env instead.
_CONFIG_ENV_PREFIX = "NEMO_AGENT_HARDENER_"
_TASK_POD_CONFIG_FIELDS = (
    "inference_secret_name",
    "openshell_gateway_endpoint",
    "openshell_ca_secret",
    "openshell_client_cert_secret",
    "openshell_client_key_secret",
)


async def task_step_executor(
    task_module: str, *, step: str, profile: str | None, async_sdk: object
) -> tuple[ExecutorSpec, list[EnvironmentVariable]]:
    """The executor and env for a job step: the provisioned host's venv locally, :data:`TASK_IMAGE` on Helm."""
    executor = await resolve_task_executor(
        profile=profile or "default",
        async_sdk=cast(AsyncNemoClient, async_sdk),
        image=TASK_IMAGE,
        entrypoint=["python", "-m"],
        command=[task_module],
        step=step,
    )
    environment = [EnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=DEFAULT_JOB_STORAGE_PATH)]
    if isinstance(executor, SubprocessExecutionProviderSpec):
        environment += [
            EnvironmentVariable(name=name, value=value) for name in _HOST_ENVVARS if (value := os.environ.get(name))
        ]
        return executor, environment

    plugin_config = AgentHardenerConfig.get()
    if not plugin_config.container_jobs:
        raise HelixJobCompilationError(
            f"The agent-hardener {step} step cannot run on this deployment: it registers no subprocess profile "
            "and container jobs are off. Set `agent_hardener.container_jobs: true` with an OpenShell gateway "
            "endpoint, or run it from a host provisioned with `nemo agent-hardener setup`."
        )
    environment += [
        EnvironmentVariable(name=f"{_CONFIG_ENV_PREFIX}{field.upper()}", value=str(value))
        for field in _TASK_POD_CONFIG_FIELDS
        if (value := getattr(plugin_config, field))
    ]
    return executor, environment


async def victim_image_required(*, async_sdk: object, profile: str | None = None) -> bool:
    """Whether a war-game submitted here would run in the task image, and so need each manifest's victim image.

    Asks the same question ``compile`` does. A deployment that cannot run war-games at all reports ``False``:
    there is no image to ask for, and submitting explains what is missing.
    """
    try:
        executor, _ = await task_step_executor(WAR_GAME_TASK, step="war-game", profile=profile, async_sdk=async_sdk)
    except HelixJobCompilationError:
        return False
    return not isinstance(executor, SubprocessExecutionProviderSpec)


async def load_manifest(
    entity_client: NemoEntitiesClient, *, workspace: str, manifest_id: str
) -> AgentHardenerManifest:
    """The saved manifest a job targets; a missing one is rejected at submit rather than when the job runs."""
    try:
        return await entity_client.get(AgentHardenerManifest, name=manifest_id, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HelixJobCompilationError(
            f"Manifest {manifest_id!r} not found in workspace {workspace!r}. List the saved manifests with "
            "`nemo agent-hardener manifest list`, or create one with `nemo agent-hardener init`."
        ) from exc


def require_victim_image(executor: ExecutorSpec, *, step: str, manifest: AgentHardenerManifest | None) -> None:
    """Fail at submit, not at sandbox creation, when a container job's manifest has no pre-built victim image.

    A container job has no Docker to build the victim, and OpenShell only creates a sandbox from an image the
    cluster can pull. A host (subprocess) job still builds it locally.
    """
    if isinstance(executor, SubprocessExecutionProviderSpec) or manifest is None:
        return
    if not manifest.victim_image:
        raise HelixJobCompilationError(
            f"The agent-hardener {step} step runs in a container here, so manifest {manifest.name!r} needs a victim "
            "image: run `agent-hardener stage` on it, build and push the printed context to a registry the "
            "OpenShell gateway can pull from, then set the image on the manifest."
        )


def require_provisioned(plugin_config: AgentHardenerConfig) -> None:
    """Raise if agent-hardener or the garak venv isn't provisioned on this host."""
    if not plugin_config.agent_hardener_bin.exists():
        raise AgentHardenerRunError(
            CATEGORY_PROVISIONING,
            f"agent-hardener is not provisioned at {plugin_config.agent_hardener_bin}. "
            "Run `nemo agent-hardener setup` on the host that executes this job.",
        )
    if not plugin_config.garak_python.exists():
        raise AgentHardenerRunError(
            CATEGORY_PROVISIONING,
            f"garak venv is not provisioned at {plugin_config.garak_venv_path}. "
            "Run `nemo agent-hardener setup` on the host that executes this job.",
        )


def build_subprocess_env(plugin_config: AgentHardenerConfig, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Subprocess env: garak venv for the agent_breaker + agent-hardener's own key from the operator dotenv.

    Explicit shell env still wins over the operator dotenv (``setdefault``). ``extra_env`` (the user's
    per-run model selection, see :func:`build_model_env`) is applied last so a chosen model / endpoint /
    key overrides both the process env and the operator dotenv default.
    """
    env = {**os.environ, GARAK_PYTHON_ENVVAR: str(plugin_config.garak_python)}
    for key, value in read_env_file(plugin_config.operator_env_file).items():
        env.setdefault(key, value)
    if extra_env:
        env.update(extra_env)
    return env


def warn_shadowed_operator_env(plugin_config: AgentHardenerConfig) -> list[str]:
    """Warn about operator-dotenv names the platform's own environment overrides with another value.

    The shell that started the platform wins (see :func:`build_subprocess_env`), so a stale export such as a
    different ``INFERENCE_API_KEY`` silently replaces the dotenv key and surfaces only as a 401 mid-run.
    Returns the names; values are never logged.
    """
    shadowed = sorted(
        key
        for key, value in read_env_file(plugin_config.operator_env_file).items()
        if key in os.environ and os.environ[key] != value
    )
    if shadowed:
        logger.warning(
            "the platform environment overrides %s from %s; unset them where the platform was started "
            "to use the file's values",
            ", ".join(shadowed),
            plugin_config.operator_env_file,
        )
    return shadowed


# Map each model group to agent-hardener's native env knobs. attack → garak's red-team + detector (name/uri
# + NIM_API_KEY); analysis → the shared llm factory default (AGENT_HARDENER_MODEL/BASE_URL + INFERENCE_API_KEY).
# safety → the in-victim guardrail judge's endpoint + key (agent-hardener hands the key to the sandbox). Its
# model is not an env knob: it travels in the manifest, as the guardrails defender entry's `config`.
_ATTACK_MODEL_ENVVARS = ("GARAK_RED_TEAM_MODEL_NAME", "GARAK_DETECTOR_MODEL_NAME")
_ATTACK_BASE_URL_ENVVARS = ("GARAK_RED_TEAM_MODEL_URI", "GARAK_DETECTOR_MODEL_URI")
_ATTACK_KEY_ENVVAR = "NIM_API_KEY"  # pragma: allowlist secret
_ANALYSIS_MODEL_ENVVAR = "AGENT_HARDENER_MODEL"
_ANALYSIS_BASE_URL_ENVVAR = "AGENT_HARDENER_BASE_URL"
_ANALYSIS_KEY_ENVVAR = "INFERENCE_API_KEY"  # pragma: allowlist secret
_SAFETY_BASE_URL_ENVVAR = "AGENT_HARDENER_SAFETY_BASE_URL"
_SAFETY_KEY_ENVVAR = "AGENT_HARDENER_SAFETY_API_KEY"  # pragma: allowlist secret


def _resolve_secret(sdk: Any, name: str, workspace: str) -> str | None:
    """Fetch a Secret's plaintext value via the platform SDK; None if unavailable (caller warns/fails)."""
    if sdk is None:
        return None
    secret = client_from_platform(sdk, SecretsClient).access_secret(name=name, workspace=workspace).data()
    value = getattr(secret, "value", None)
    return str(value) if value else None


def resolve_model_key(sdk: Any, api_key_secret: str | None, *, workspace: str) -> str | None:
    """The key a model choice will actually use: its named Secret, else the provisioned agent-hardener key.

    One rule, shared by the run's preflight and the ``model-config/validate`` endpoint, so a choice is
    never validated against a different credential than the run will use. Mirrors the order
    :func:`~nemo_agent_hardener_plugin.cli.credentials.resolve_inference_key` establishes — the Secrets store
    is authoritative, the operator dotenv is the offline fallback — rather than inventing a second policy.

    Secrets are tried before the dotenv so the rule also holds on a deployed platform, where the server
    has no operator dotenv and that branch simply yields ``None``.
    """
    if api_key_secret:
        # Explicitly chosen: a failure here is the user's to see, so it is not swallowed.
        return _resolve_secret(sdk, api_key_secret, workspace)
    config = AgentHardenerConfig.get()
    provisioned = _optional_secret(sdk, config.inference_secret_name, workspace)
    return provisioned or build_subprocess_env(config).get(INFERENCE_API_KEY_ENVVAR)


def _optional_secret(sdk: Any, name: str, workspace: str) -> str | None:
    """A Secret's value, or ``None`` only when it does not exist; an unreachable or refusing store raises."""
    try:
        return _resolve_secret(sdk, name, workspace)
    except NotFoundError:
        return None


_MTLS_PURPOSE = "the OpenShell mTLS bundle"


def job_runtime_env(
    plugin_config: AgentHardenerConfig, *, sdk: Any, workspace: str, openshell_home: Path
) -> dict[str, str]:
    """Env a job adds for what `nemo agent-hardener setup` would otherwise have left on the host.

    agent-hardener's own inference key is the one :func:`resolve_model_key` picks, so the run uses the
    credential its preflight validated; a configured OpenShell gateway is registered under *openshell_home*.
    """
    env: dict[str, str] = {}
    key = resolve_model_key(sdk, None, workspace=workspace)
    if key:
        env[INFERENCE_API_KEY_ENVVAR] = key
    env.update(
        register_openshell_gateway(
            plugin_config,
            resolve_secret=lambda name: resolve_secret_refs(sdk, {_MTLS_PURPOSE: name}, workspace)[_MTLS_PURPOSE],
            config_home=openshell_home,
        )
    )
    return env


def build_model_env(models: WarGameModels | None, *, sdk: Any, workspace: str) -> dict[str, str]:
    """Translate the user's model selection into agent-hardener subprocess env vars.

    Only set knobs the user actually chose (``None`` leaves agent-hardener's built-in default in force). A
    group's ``api_key_secret`` is resolved to its plaintext value and bound to that group's key env var,
    so a custom provider's credential reaches garak (NIM_API_KEY) / the llm factory (INFERENCE_API_KEY) /
    the guardrail judge (AGENT_HARDENER_SAFETY_API_KEY).
    """
    env: dict[str, str] = {}
    for group in _model_groups(models):
        _apply_group(env, group.choice, group.model_envvars, group.base_url_envvars, group.key_envvar, sdk, workspace)
    return env


def model_key_secrets(models: WarGameModels | None) -> dict[str, str]:
    """The ``key env var -> Secret name`` bindings :func:`build_model_env` makes for *models*."""
    return {
        group.key_envvar: group.choice.api_key_secret
        for group in _model_groups(models)
        if group.choice and group.choice.api_key_secret
    }


def model_key_refs(models: WarGameModels | None) -> dict[str, str]:
    """Each group's chosen key Secret, labelled for messages (``"attack model key" -> secret``)."""
    return {
        f"{group.label} model key": group.choice.api_key_secret
        for group in _model_groups(models)
        if group.choice and group.choice.api_key_secret
    }


class _ModelGroup(NamedTuple):
    """One model group and the env vars its model, endpoint and key are bound to."""

    label: str
    choice: ModelChoice | None
    model_envvars: tuple[str, ...]
    base_url_envvars: tuple[str, ...]
    key_envvar: str


def _model_groups(models: WarGameModels | None) -> tuple[_ModelGroup, ...]:
    """Each model group of *models*; empty when nothing is selected."""
    if models is None:
        return ()
    return (
        _ModelGroup("attack", models.attack, _ATTACK_MODEL_ENVVARS, _ATTACK_BASE_URL_ENVVARS, _ATTACK_KEY_ENVVAR),
        _ModelGroup(
            "analysis", models.analysis, (_ANALYSIS_MODEL_ENVVAR,), (_ANALYSIS_BASE_URL_ENVVAR,), _ANALYSIS_KEY_ENVVAR
        ),
        _ModelGroup("safety", models.safety, (), (_SAFETY_BASE_URL_ENVVAR,), _SAFETY_KEY_ENVVAR),
    )


def _apply_group(
    env: dict[str, str],
    choice: ModelChoice | None,
    model_envvars: tuple[str, ...],
    base_url_envvars: tuple[str, ...],
    key_envvar: str,
    sdk: Any,
    workspace: str,
) -> None:
    """Set a group's model/base_url/key env vars from a :class:`ModelChoice` (skipping unset fields)."""
    if choice is None:
        return
    if choice.model:
        for name in model_envvars:
            env[name] = choice.model
    if choice.base_url:
        for name in base_url_envvars:
            env[name] = choice.base_url
    if choice.api_key_secret:
        value = _resolve_secret(sdk, choice.api_key_secret, workspace)
        if not value:
            raise AgentHardenerRunError(
                CATEGORY_MISSING_CREDENTIAL, f"secret {workspace}/{choice.api_key_secret} is missing or empty."
            )
        env[key_envvar] = value


def materialize_victim_env_file(manifest: str, env: dict[str, str], dest_dir: Path) -> str | None:
    """Write the manifest's declared victim secrets (sourced from *env*) to a dotenv; return its path.

    Studio submits with no ``--env-file``, but agent-hardener reads the victim's provider credentials from a
    project dotenv. We source the manifest's declared secrets from the subprocess env (which carries
    agent-hardener's operator key, see :func:`build_subprocess_env`) and write them so the war-game has creds.
    Returns ``None`` when the manifest declares no secrets or none are present in *env*.
    """
    present = {name: env[name] for name in declared_secrets(read_manifest_agent(Path(manifest))) if name in env}
    if not present:
        return None
    dest = dest_dir / ".env"
    write_env_file(dest, present)  # holds provider creds (INFERENCE_API_KEY et al.) — 0600 from creation
    return str(dest)


def resolve_secret_refs(sdk: Any, refs: dict[str, str], workspace: str) -> dict[str, str]:
    """Read each ``ENV_VAR -> [workspace/]secret`` ref from the Secrets store; fail naming the ref, never the value."""
    values: dict[str, str] = {}
    for env_name, ref in refs.items():
        secret_workspace, secret_name = parse_qualified_name(ref, default_workspace=workspace)
        try:
            value = _resolve_secret(sdk, secret_name, secret_workspace)
        except Exception as exc:
            raise AgentHardenerRunError(
                CATEGORY_MISSING_CREDENTIAL,
                f"could not read secret {secret_workspace}/{secret_name} for {env_name}: {type(exc).__name__}",
            ) from exc
        if not value:
            raise AgentHardenerRunError(
                CATEGORY_MISSING_CREDENTIAL, f"secret {secret_workspace}/{secret_name} for {env_name} is empty."
            )
        values[env_name] = value
    return values


async def missing_secret_refs(async_client: AsyncNemoClient, refs: dict[str, str], workspace: str) -> list[str]:
    """The ``purpose -> [workspace/]secret`` refs whose Secret does not exist, as ``"workspace/name (purpose)"``.

    Checks existence only (metadata, never the value). Errors other than not-found raise, so an unreachable
    store is not mistaken for a missing secret.
    """
    secrets_client = AsyncSecretsClient.from_client(async_client)
    missing: list[str] = []
    for purpose, ref in refs.items():
        secret_workspace, secret_name = parse_qualified_name(ref, default_workspace=workspace)
        try:
            await secrets_client.get_secret(name=secret_name, workspace=secret_workspace)
        except NotFoundError:
            missing.append(f"{secret_workspace}/{secret_name} ({purpose})")
    return missing


def missing_secret_refs_message(missing: list[str]) -> str:
    """Operator-facing message for :func:`missing_secret_refs` results, naming each ref and the fix."""
    return (
        f"Secrets not found in the Secrets store: {', '.join(missing)}. Create each with `nemo secrets create`, "
        "or point the reference at an existing secret."
    )


def prepare_victim_env(
    manifest: str, env: dict[str, str], user_env_file: str | None, secret_values: dict[str, str] | None = None
) -> tuple[dict[str, str], str | None]:
    """Write the victim ``.env`` from *env*, *secret_values* and the user's ``--env-file``; return ``(env, env_file)``.

    The manifest's ``agent.secrets_file`` points at ``<manifest_dir>/.env``, so it must be written even
    when the user supplies their own file. Precedence, lowest first: the operator env, values read from
    the Secrets store, then the user's file, which is still what is passed on as ``--env-file``.
    """
    merged = {**env, **(secret_values or {})}
    if user_env_file:
        merged |= read_env_file(Path(user_env_file))
    materialized = materialize_victim_env_file(manifest, merged, Path(manifest).parent)
    return merged, user_env_file or materialized


def check_victim_secrets(manifest: str, env: dict[str, str], env_file: str | None) -> None:
    """Fail fast if the manifest declares victim secrets no available source provides."""
    extra_env_files = [Path(env_file)] if env_file else []
    missing = missing_secrets(Path(manifest), env_files=extra_env_files, environ=env)
    if missing:
        raise missing_victim_secrets_error(missing)


def missing_victim_secrets_error(missing: list[str]) -> AgentHardenerRunError:
    """The classified error for victim secrets no source provides, with the exact flags that fix it."""
    # Secrets-store names are lowercase kebab-case, so the store entry cannot share the env var's name.
    refs = " ".join(f"--secrets {name}=<workspace>/{name.lower().replace('_', '-')}" for name in missing)
    return AgentHardenerRunError(
        CATEGORY_MISSING_CREDENTIAL,
        f"missing required secrets for the victim agent: {', '.join(missing)}.",
        remediation=f"Store each value in the Secrets store and map it on the manifest ({refs}), "
        "or supply it with --env-file or the platform environment.",
    )


def unprovided_host_secrets(
    manifest: AgentHardenerManifest,
    plugin_config: AgentHardenerConfig,
    *,
    env_file: str | None,
    models: WarGameModels | None,
) -> list[str]:
    """The manifest's declared victim secrets a host (subprocess) job on this machine would not find.

    Mirrors the sources the job assembles (see :func:`prepare_victim_env`): Secrets-store refs (checked for
    existence separately), the keys the job derives from the provisioned inference secret and the model
    choice, this host's environment and operator dotenv, and the run's ``--env-file``. The job's own
    environment is a subset of this process's, so this never rejects what the job would accept.
    """
    provided = {
        *manifest.deployed_secret_refs,
        *manifest.secret_refs,
        *model_key_secrets(models),
        # Set by job_runtime_env from the provisioned inference secret, when one exists.
        INFERENCE_API_KEY_ENVVAR,
    }
    return unprovided_secrets(
        [name for name in manifest.secrets if name not in provided],
        env_files=[Path(env_file)] if env_file else [],
        environ=build_subprocess_env(plugin_config),
    )


def execute(
    cmd: list[str], env: dict[str, str], log_path: Path, ctx: JobContext, *, artifact_name: str
) -> tuple[subprocess.CompletedProcess, str, Any]:
    """Run *cmd*, TTY-aware.

    With a terminal attached (a shell invocation) we inherit it so agent-hardener's interactive prompts + rich
    UI work; agent-hardener writes its own logs, so we capture nothing. Headless (deployed job / no tty) we
    capture stdout to *log_path* and save it as the *artifact_name* result. Returns
    ``(completed, log_text, log_ref)``.
    """
    if sys.stdin.isatty():
        completed = run(cmd, text=True, check=False, env=env)  # inherits the terminal
        return completed, "", None
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    log_ref = ctx.results.save(artifact_name, log_path)
    return completed, log_text, log_ref
