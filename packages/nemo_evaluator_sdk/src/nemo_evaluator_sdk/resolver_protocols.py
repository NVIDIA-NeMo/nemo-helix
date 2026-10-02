# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolver protocols for evaluator SDK references."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nemo_evaluator_sdk.values.common import SecretRef
from nemo_evaluator_sdk.values.models import Model, ModelRef


@runtime_checkable
class SecretResolver(Protocol):
    """Resolve evaluator secret references to secret values."""

    async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
        """Return the secret value for ``secret_ref`` when available."""
        ...


@runtime_checkable
class EnvSecretSource(Protocol):
    """Name the environment variable that already holds a secret, without reading its value.

    Lets a harness that templates env vars from ``os.environ`` (Harbor's ``${NAME}``) receive a secret
    without the value being copied or written to ``os.environ``.

    Implemented by :class:`~nemo_evaluator_sdk.resolvers.LocalSecretResolver` (standalone) and the
    evaluator plugin's ``JobEnvSecretResolver`` (platform jobs). Used by:

    * ``HarborAgentTaskRunner(secret_resolver=...)``: the resolver's type, checked at construction
      (hence ``runtime_checkable``), so a resolver that can't name env vars fails early.
    * ``harbor_env_templates``: finds each ``env_secrets`` source variable and templates it.
    * ``resolve_env_secrets`` (Gym's value path): uses ``missing_secret_message`` when a resolver
      provides it.
    """

    def find_env_name(self, secret_ref: SecretRef, env_name: str) -> str | None:
        """Name of the non-empty env var holding ``secret_ref`` for the variable ``env_name``, or ``None``."""
        ...

    def missing_secret_message(self, secret_ref: SecretRef, env_name: str) -> str:
        """Explain, for this execution context, how to supply ``secret_ref`` when it is missing."""
        ...


@runtime_checkable
class ModelResolver(Protocol):
    """Resolve evaluator model references to concrete SDK model bindings."""

    async def resolve_model(self, model_ref: ModelRef) -> Model:
        """Return the concrete model binding for ``model_ref``."""
        ...
