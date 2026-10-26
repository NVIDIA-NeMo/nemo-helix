# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch-time credential materialization.

Evaluations store provider -> platform secret references. The evaluation Job
spec injects each referenced secret into the Job process environment, and the
dispatcher maps those values to the env vars runtime backends expect.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from scaled_evals.api.settings import settings

_PROVIDER_ENV_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    # Recipe 12 maps the same inference key to OPENAI_API_KEY for OpenAI-wire agents.
    "nvidia": ("NGC_INFERENCE_API_KEY", "POLICY_API_KEY", "OPENAI_API_KEY"),
}
# Values taken out of os.environ, so backend subprocesses that inherit the
# environment do not also receive every injected secret.
_injected: dict[str, str] = {}


def credential_env_var(provider: str) -> str:
    """Name the Job environment variable that carries ``provider``'s secret value."""
    return f"SCALED_EVALS_CREDENTIAL_{provider.upper()}"


def _env_file_value(value: str) -> str:
    if not value:
        raise ValueError("credential environment values must be non-empty strings")
    if any(ch.isspace() for ch in value) or any(ch in value for ch in "#='\""):
        escaped = value.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def write_env_file(path: Path, env: Mapping[str, str]) -> None:
    """Write an env file from already-materialized values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("".join(f"{key}={_env_file_value(value)}\n" for key, value in env.items()))


def credential_env(providers: list[str], values: Mapping[str, str]) -> dict[str, str]:
    """Map provider -> secret value into the env vars runtime backends read."""
    env: dict[str, str] = {}
    for provider in providers:
        value = values.get(provider)
        if not value:
            raise ValueError(f"credential secret for {provider} was not injected into the evaluation Job")
        if provider in _PROVIDER_ENV_KEYS:
            for key in _PROVIDER_ENV_KEYS[provider]:
                env[key] = value
            if provider == "anthropic":
                env.setdefault("ANTHROPIC_BASE_URL", settings.nvidia_anthropic_base_url)
            elif provider == "nvidia":
                env.setdefault("POLICY_BASE_URL", settings.nvidia_inference_base_url)
        elif provider == "nhx":
            env["NHX_INTAKE_CREDENTIAL_YAML"] = value
        elif provider == "openshift":
            # A user's OpenShift bearer token. NOT a runtime env var for the
            # agent — sandbox_k8s extracts it to write a per-eval kubeconfig and
            # strips it before the agent env-file, so it never reaches the pod.
            env["SANDBOX_OC_TOKEN"] = value
        else:
            raise ValueError(f"unsupported credential provider: {provider}")
    return env


def injected_credential_env(credentials: Mapping[str, str]) -> dict[str, str]:
    """Build runtime env vars from the secrets the Job spec injected for ``credentials``."""
    for provider in credentials:
        name = credential_env_var(provider)
        if name in os.environ:
            _injected[name] = os.environ.pop(name)
    values = {provider: _injected.get(credential_env_var(provider), "") for provider in credentials}
    return credential_env(sorted(credentials), values)


def merged_env_file(
    *,
    source_env_file: Path,
    output_env_file: Path,
    credential_env: Mapping[str, str],
) -> Path:
    """Create an evaluation-scoped env file with credential overrides."""
    if not credential_env:
        return source_env_file
    merged = source_env_file.read_text()
    if merged and not merged.endswith("\n"):
        merged += "\n"
    output_env_file.parent.mkdir(parents=True, exist_ok=True)
    output_env_file.write_text(merged)
    with output_env_file.open("a") as handle:
        for key, value in credential_env.items():
            handle.write(f"{key}={_env_file_value(value)}\n")
    output_env_file.chmod(0o600)
    return output_env_file
