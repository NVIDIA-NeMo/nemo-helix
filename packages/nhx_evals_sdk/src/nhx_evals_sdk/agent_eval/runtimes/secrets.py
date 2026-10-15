# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve ``env_secrets`` to values for harnesses that are handed values, such as Gym."""

from __future__ import annotations

import os
from collections.abc import Mapping

from nhx_evals_sdk.resolver_protocols import EnvSecretSource, MissingSecretError
from nhx_evals_sdk.values.common import SecretRef


def env_secret_vars(env_secrets: Mapping[str, SecretRef], source: EnvSecretSource) -> dict[str, str]:
    """Map each ``env_secrets`` key to the environment variable holding its secret.

    Example:
        With ``MY_WORKSPACE_PROBE_API_KEY`` set in the process environment::

            env_secrets = {"LLM_API_KEY": SecretRef("my-workspace/probe-api-key")}
            source = LocalSecretResolver()
            env_secret_vars(env_secrets, source)
            # Returns: {"LLM_API_KEY": "MY_WORKSPACE_PROBE_API_KEY"}

        Output keys retain the destination names; values name the source variables.
    """
    sources: dict[str, str] = {}
    for env_name, secret_ref in env_secrets.items():
        try:
            sources[env_name] = source.env_var_for(secret_ref, env_name)
        except MissingSecretError as error:
            raise MissingSecretError(f"env_secrets[{env_name!r}] -> {error}") from error
    return sources


def env_secret_values(env_secrets: Mapping[str, SecretRef], source: EnvSecretSource) -> dict[str, str]:
    """Read each secret from the environment variable named by ``source``.

    Example:
        With ``MY_WORKSPACE_PROBE_API_KEY="example-secret-value"`` in the process environment::

            env_secrets = {"LLM_API_KEY": SecretRef("my-workspace/probe-api-key")}
            source = LocalSecretResolver()
            env_secret_values(env_secrets, source)
            # Returns: {"LLM_API_KEY": "example-secret-value"}

        Output keys retain the destination names; values contain the resolved secrets.
    """
    return {name: os.environ[var] for name, var in env_secret_vars(env_secrets, source).items()}
