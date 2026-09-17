# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the agent-hardener jobs.

Both the war-game and (future) synth stages shell out to agent-hardener's CLI inside its dedicated venv and
need the same env wiring (garak venv + agent-hardener's own inference key + the victim's secrets) and the same
TTY-aware subprocess execution. This module holds that shared logic so the jobs don't duplicate it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from nemo_agent_hardener_plugin.config import (
    GARAK_PYTHON_ENVVAR,
    INFERENCE_API_KEY_ENVVAR,
    AgentHardenerConfig,
    missing_secrets,
    read_env_file,
    write_env_file,
)
from nemo_agent_hardener_plugin.jobs.errors import (
    CATEGORY_MISSING_CREDENTIAL,
    CATEGORY_PROVISIONING,
    AgentHardenerRunError,
)
from nemo_agent_hardener_plugin.model_config import ModelChoice, WarGameModels
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.secrets.client import SecretsClient


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


# Map each model group to agent-hardener's native env knobs. attack → garak's red-team + detector (name/uri
# + NIM_API_KEY); analysis → the shared llm factory default (AGENT_HARDENER_MODEL/BASE_URL + INFERENCE_API_KEY).
# The safety (guardrail) model is not an env knob — it travels in the manifest, as the guardrails
# defender entry's `config`, because a defender consumes it rather than the agent-hardener process.
_ATTACK_MODEL_ENVVARS = ("GARAK_RED_TEAM_MODEL_NAME", "GARAK_DETECTOR_MODEL_NAME")
_ATTACK_BASE_URL_ENVVARS = ("GARAK_RED_TEAM_MODEL_URI", "GARAK_DETECTOR_MODEL_URI")
_ATTACK_KEY_ENVVAR = "NIM_API_KEY"  # pragma: allowlist secret
_ANALYSIS_MODEL_ENVVAR = "AGENT_HARDENER_MODEL"
_ANALYSIS_BASE_URL_ENVVAR = "AGENT_HARDENER_BASE_URL"
_ANALYSIS_KEY_ENVVAR = "INFERENCE_API_KEY"  # pragma: allowlist secret


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
    try:
        provisioned = _resolve_secret(sdk, config.inference_secret_name, workspace)
    except Exception:  # absent secret / unreachable store — expected, the dotenv is the fallback
        provisioned = None
    return provisioned or build_subprocess_env(config).get(INFERENCE_API_KEY_ENVVAR)


def build_model_env(models: WarGameModels | None, *, sdk: Any, workspace: str) -> dict[str, str]:
    """Translate the user's model selection into agent-hardener subprocess env vars.

    Only set knobs the user actually chose (``None`` leaves agent-hardener's built-in default in force). A
    group's ``api_key_secret`` is resolved to its plaintext value and bound to that group's key env var,
    so a custom provider's credential reaches garak (NIM_API_KEY) / the llm factory (INFERENCE_API_KEY).
    """
    if models is None:
        return {}
    env: dict[str, str] = {}
    _apply_group(
        env, models.attack, _ATTACK_MODEL_ENVVARS, _ATTACK_BASE_URL_ENVVARS, _ATTACK_KEY_ENVVAR, sdk, workspace
    )
    _apply_group(
        env,
        models.analysis,
        (_ANALYSIS_MODEL_ENVVAR,),
        (_ANALYSIS_BASE_URL_ENVVAR,),
        _ANALYSIS_KEY_ENVVAR,
        sdk,
        workspace,
    )
    return env


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
        if value:
            env[key_envvar] = value


def materialize_victim_env_file(manifest: str, env: dict[str, str], dest_dir: Path) -> str | None:
    """Write the manifest's declared victim secrets (sourced from *env*) to a dotenv; return its path.

    Studio submits with no ``--env-file``, but agent-hardener reads the victim's provider credentials from a
    project dotenv. We source the manifest's declared secrets from the subprocess env (which carries
    agent-hardener's operator key, see :func:`build_subprocess_env`) and write them so the war-game has creds.
    Returns ``None`` when the manifest declares no secrets or none are present in *env*.
    """
    try:
        data = yaml.safe_load(Path(manifest).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    agent = data.get("agent", {}) if isinstance(data, dict) else {}
    declared = [name for name in (agent.get("secrets") or []) if isinstance(name, str)]
    present = {name: env[name] for name in declared if name in env}
    if not present:
        return None
    dest = dest_dir / ".env"
    write_env_file(dest, present)  # holds provider creds (INFERENCE_API_KEY et al.) — 0600 from creation
    return str(dest)


def check_victim_secrets(manifest: str, env: dict[str, str], env_file: str | None) -> None:
    """Fail fast if the manifest declares victim secrets no available source provides."""
    extra_env_files = [Path(env_file)] if env_file else []
    missing = missing_secrets(Path(manifest), env_files=extra_env_files, environ=env)
    if missing:
        raise AgentHardenerRunError(
            CATEGORY_MISSING_CREDENTIAL,
            f"missing required secrets for the victim agent: {', '.join(missing)}. "
            "Provide them via --env-file or the environment.",
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
        completed = subprocess.run(cmd, text=True, check=False, env=env)  # inherits the terminal
        return completed, "", None
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
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
