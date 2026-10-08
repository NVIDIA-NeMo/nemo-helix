# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolver protocols for evals SDK references."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nhx_evals_sdk.values.common import SecretRef
from nhx_evals_sdk.values.models import Model, ModelRef


@runtime_checkable
class SecretResolver(Protocol):
    """Resolve evaluator secret references to secret values."""

    async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
        """Return the secret value for ``secret_ref`` when available."""
        ...


class MissingSecretError(ValueError):
    """A secret is unavailable in this execution context; the message explains how to supply it."""


@runtime_checkable
class EnvSecretSource(Protocol):
    """Name the environment variable that already holds a secret, without reading its value.

    Lets a harness that templates env vars from ``os.environ`` (Harbor's ``${NAME}``) receive a secret
    without the value being copied or written to ``os.environ``.

    Implemented by :class:`~nhx_evals_sdk.resolvers.LocalSecretResolver` (standalone) and the
    evals plugin's ``JobEnvSecretSource`` (platform jobs). Used by:

    * ``HarborAgentTaskRunner(secret_resolver=...)``: the resolver's type, checked at construction
      (hence ``runtime_checkable``), so a resolver that can't name env vars fails early.
    * ``env_secret_vars`` / ``harbor_env_templates``: name and template each secret source variable.
    * ``GymAgentTaskRunner`` / ``env_secret_values``: read each named variable for Gym's subprocess.
    """

    def env_var_for(self, secret_ref: SecretRef, env_name: str) -> str:
        """Name of the non-empty env var holding ``secret_ref`` for ``env_name``.

        Raises:
            MissingSecretError: With a context-specific hint for supplying the secret.
        """
        ...


@runtime_checkable
class ModelResolver(Protocol):
    """Resolve evaluator model references to concrete SDK model bindings."""

    async def resolve_model(self, model_ref: ModelRef) -> Model:
        """Return the concrete model binding for ``model_ref``."""
        ...
