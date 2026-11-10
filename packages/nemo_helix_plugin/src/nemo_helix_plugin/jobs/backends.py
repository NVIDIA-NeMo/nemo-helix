# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-only job backend contributions; implementations load in the controller."""

from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

from nemo_helix_plugin import discovery
from nemo_helix_plugin.jobs.spec import BaseExecutionProfile
from pydantic import Field

BUILTIN_JOB_BACKENDS = frozenset({"docker", "kubernetes_job", "volcano_job", "subprocess", "e2e", "openshell"})


@dataclass(frozen=True)
class JobBackendContribution:
    """Register an instance under ``nemo.job_backends``.

    ``profile_model`` must declare a matching backend Literal and a config field.
    ``implementations`` maps providers to ``module:ClassName`` references to
    concrete JobBackend subclasses. Keep runtime imports out of this module:
    API processes and SDK clients also discover these schema contributions.
    """

    backend: str
    profile_model: type[BaseExecutionProfile]
    implementations: dict[str, str]


def _valid_implementation(provider: str, reference: str) -> bool:
    if not isinstance(provider, str) or not provider or provider.strip() != provider:
        return False
    if not isinstance(reference, str) or reference.count(":") != 1:
        return False
    module, symbol = reference.split(":")
    return all(part.isidentifier() for part in module.split(".")) and symbol.isidentifier()


def discover_job_backends() -> dict[str, JobBackendContribution]:
    """Discover allowlisted schemas, rejecting reserved or inconsistent declarations.

    Entry-point import failures follow generic discovery's warning-and-skip
    policy. Malformed contributions that loaded successfully raise ValueError.
    """
    result: dict[str, JobBackendContribution] = {}
    for name, item in discovery.discover("nemo.job_backends").items():
        if not isinstance(item, JobBackendContribution):
            raise ValueError(f"Invalid job backend contribution: {name}")
        if not isinstance(item.backend, str) or not item.backend or item.backend.strip() != item.backend:
            raise ValueError(f"Invalid job backend name for entry point {name}: {item.backend!r}")
        if item.backend in BUILTIN_JOB_BACKENDS:
            raise ValueError(f"Job backend entry point {name} collides with built-in backend {item.backend!r}")
        if item.backend in result:
            raise ValueError(f"Duplicate job backend {item.backend!r} from entry point {name}")
        if not isinstance(item.profile_model, type) or not issubclass(item.profile_model, BaseExecutionProfile):
            raise ValueError(f"Invalid job backend profile model: {name}")
        field = item.profile_model.model_fields.get("backend")
        if (
            field is None
            or get_origin(field.annotation) is not Literal
            or get_args(field.annotation) != (item.backend,)
        ):
            raise ValueError(f"Job backend profile must declare Literal[{item.backend!r}]: {name}")
        if "config" not in item.profile_model.model_fields:
            raise ValueError(f"Job backend profile must declare config: {name}")
        if (
            not isinstance(item.implementations, dict)
            or not item.implementations
            or not all(_valid_implementation(provider, ref) for provider, ref in item.implementations.items())
        ):
            raise ValueError(f"Invalid job backend implementation references: {name}")
        result[item.backend] = item
    return result


class GenericExecutionProfile(BaseExecutionProfile):
    """Readback shape for clients without the backend-specific schema installed."""

    backend: str
    config: dict[str, Any] = Field(default_factory=dict)
