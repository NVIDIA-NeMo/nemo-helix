# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local resolver implementations for evaluator SDK refs."""

from __future__ import annotations

import logging
import os

from nemo_evaluator_sdk.resolver_protocols import MissingSecretError
from nemo_evaluator_sdk.values.common import SecretRef
from nemo_evaluator_sdk.values.models import Model, ModelRef

logger = logging.getLogger(__name__)


def _candidate_env_names(secret_name: str) -> list[str]:
    """Generate environment variable names that may contain one secret."""
    names = [secret_name, secret_name.upper()]
    normalized = secret_name.replace("-", "_").replace("/", "_")
    names.extend([normalized, normalized.upper()])
    if normalized and normalized[0].isdigit():
        prefixed = f"_{normalized}"
        names.extend([prefixed, prefixed.upper()])
    return list(dict.fromkeys(names))


def _shell_env_name(secret_name: str) -> str:
    """The uppercase candidate a user can ``export``: what a missing-secret message tells them to set."""
    normalized = secret_name.replace("-", "_").replace("/", "_").upper()
    return f"_{normalized}" if normalized and normalized[0].isdigit() else normalized


class LocalSecretResolver:
    """Resolve secrets from local environment variables.

    Structurally implements ``SecretResolver`` via ``resolve_secret`` (values for metrics) and
    ``EnvSecretSource`` via ``env_var_for`` (source variable names for harnesses).

    A workspace-qualified ref (``my-workspace/openai-api-key``) is looked up under its prefixed names
    first (``MY_WORKSPACE_OPENAI_API_KEY``) and then, with ``bare_fallback``, under the names of its
    bare secret name (``OPENAI_API_KEY``). A bare ref is only ever looked up under its own names.

    Args:
        bare_fallback: Let a workspace-qualified ref fall back to its bare names. Standalone runs keep
            it on; job entrypoints turn it off, because there the service injects each secret under
            its own env name and a bare name may hold a different consumer's secret.
    """

    def __init__(self, *, bare_fallback: bool = True) -> None:
        self._bare_fallback = bare_fallback

    def _candidates(self, secret_ref: SecretRef) -> tuple[list[str], list[str]]:
        """``(prefixed, bare)`` candidate env names, searched in that order."""
        _, sep, name = secret_ref.root.rpartition("/")
        if not sep:
            return [], _candidate_env_names(name)
        return _candidate_env_names(secret_ref.root), _candidate_env_names(name) if self._bare_fallback else []

    def _find_env_name(self, secret_ref: SecretRef) -> str | None:
        """Name of the first non-empty candidate environment variable, or ``None``."""
        prefixed, bare = self._candidates(secret_ref)
        bare_set = next((name for name in bare if os.getenv(name)), None)
        for name in prefixed:
            if os.getenv(name):
                if bare_set is not None:
                    logger.info(
                        "Both %s and %s are set for secret %r; using the more specific, workspace-prefixed %s.",
                        name,
                        bare_set,
                        secret_ref.root,
                        name,
                    )
                return name
        return bare_set

    def _missing_message(self, secret_ref: SecretRef) -> str:
        """Name the env var(s) to export before a local run."""
        _, sep, name = secret_ref.root.rpartition("/")
        env_vars = _shell_env_name(secret_ref.root)
        if sep and self._bare_fallback:
            env_vars = f"{env_vars} (or {_shell_env_name(name)})"
        return (
            f"secret {secret_ref.root!r} is not found. Make sure to set the {env_vars} env var "
            "before launching evaluation locally."
        )

    def env_var_for(self, secret_ref: SecretRef, env_name: str = "") -> str:
        """Name of the non-empty env var holding ``secret_ref``; ``env_name`` is unused locally."""
        found = self._find_env_name(secret_ref)
        if found is None:
            raise MissingSecretError(self._missing_message(secret_ref))
        return found

    async def resolve_secret(self, secret_ref: SecretRef) -> str | None:
        """Resolve one secret value from environment variables."""
        name = self._find_env_name(secret_ref)
        return os.getenv(name) if name is not None else None


class LocalModelResolver:
    """Resolve model references from an in-process registry."""

    def __init__(self) -> None:
        """Create a resolver with an empty local model registry."""
        self._models: dict[str, Model] = {}

    def register_model(self, model_ref: ModelRef, model: Model, *, replace: bool = False) -> None:
        """Register a local model binding for a model reference."""
        if not replace and model_ref.root in self._models:
            raise ValueError(f"Model reference '{model_ref.root}' is already registered.")
        self._models[model_ref.root] = model

    def get_model(self, model_ref: ModelRef) -> Model:
        """Return the registered model binding for a model reference."""
        try:
            return self._models[model_ref.root]
        except KeyError as exc:
            raise ValueError(
                f"Model reference '{model_ref.root}' is not registered. "
                "Register it with LocalBackend.model_resolver.register_model() before local execution."
            ) from exc

    def unregister_model(self, model_ref: ModelRef) -> Model:
        """Remove and return a registered local model binding."""
        try:
            return self._models.pop(model_ref.root)
        except KeyError as exc:
            raise ValueError(f"Model reference '{model_ref.root}' is not registered.") from exc

    async def resolve_model(self, model_ref: ModelRef) -> Model:
        """Resolve one model reference from the local registry."""
        return self.get_model(model_ref)
