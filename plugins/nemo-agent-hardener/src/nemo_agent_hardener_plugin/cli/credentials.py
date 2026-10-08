# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision agent-hardener's own ``INFERENCE_API_KEY``.

agent-hardener's orchestrator reads it straight from the process env (public NVIDIA endpoint, not the
platform gateway), so ``setup`` resolves it (Secrets → env → prompt) into the operator dotenv.
"""

from __future__ import annotations

import os
import sys

import typer
from nemo_agent_hardener_plugin.cli.client import base_url, make_sdk
from nemo_agent_hardener_plugin.config import (
    INFERENCE_API_KEY_ENVVAR,
    AgentHardenerConfig,
    read_env_file,
    write_env_file,
)
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.secrets.client import SecretsClient


def resolve_inference_key(config: AgentHardenerConfig) -> tuple[str | None, str]:
    """Resolve agent-hardener's own inference key: NeMo Secrets -> env -> interactive prompt.

    The platform Secrets store is authoritative (house standard); env is the offline fallback. An
    explicit ``INFERENCE_API_KEY`` still wins at run time, where the job injects via ``setdefault``.
    """
    try:
        secrets = client_from_platform(make_sdk(base_url()), SecretsClient)
        secret = secrets.access_secret(name=config.inference_secret_name, workspace=config.default_workspace).data()
        if secret and secret.value:
            return secret.value, f"secret '{config.inference_secret_name}'"
    except Exception:  # Secrets store unreachable/absent → fall back to env
        pass

    env_value = os.environ.get(INFERENCE_API_KEY_ENVVAR)
    if env_value:
        return env_value, "environment"

    if sys.stdin.isatty():
        value = typer.prompt(f"Enter {INFERENCE_API_KEY_ENVVAR}", hide_input=True, default="")
        if value:
            return value, "prompt"

    return None, "unresolved"


def write_operator_env(config: AgentHardenerConfig, value: str) -> None:
    """Persist INFERENCE_API_KEY into the operator dotenv, preserving other keys, mode 0600."""
    path = config.operator_env_file
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    values = read_env_file(path)
    values[INFERENCE_API_KEY_ENVVAR] = value
    write_env_file(path, values)


def provision_operator_env(config: AgentHardenerConfig, *, force: bool) -> None:
    """Ensure agent-hardener's own inference credential is provisioned in the operator dotenv."""
    if not force and read_env_file(config.operator_env_file).get(INFERENCE_API_KEY_ENVVAR):
        typer.echo(f"Inference credential already present in {config.operator_env_file}.")
        return

    value, source = resolve_inference_key(config)
    if value is None:
        typer.secho(
            f"Could not resolve {INFERENCE_API_KEY_ENVVAR} (no env var, no "
            f"'{config.inference_secret_name}' secret, no tty to prompt). Create it with "
            f"`nemo secrets create {config.inference_secret_name}` and re-run setup, or export "
            f"{INFERENCE_API_KEY_ENVVAR} yourself.",
            fg="red",
        )
        raise typer.Exit(code=1)

    write_operator_env(config, value)
    typer.secho(f"Inference credential provisioned from {source} -> {config.operator_env_file}.", fg="green")
