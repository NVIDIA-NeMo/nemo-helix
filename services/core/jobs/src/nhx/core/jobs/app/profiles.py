# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

from nemo_helix_plugin.jobs.backends import GenericExecutionProfile, discover_job_backends
from nemo_helix_plugin.jobs.spec import BaseExecutionProfile
from nhx.core.jobs.controllers.backends.docker import DockerJobExecutionProfile
from nhx.core.jobs.controllers.backends.kubernetes import (
    KubernetesJobExecutionProfile,
    VolcanoJobExecutionProfile,
)
from nhx.core.jobs.controllers.backends.openshell import OpenShellJobExecutionProfile
from nhx.core.jobs.controllers.backends.subprocess import SubprocessJobExecutionProfile
from nhx.core.jobs.controllers.backends.test import E2EJobExecutionProfile
from pydantic import SerializeAsAny, TypeAdapter, WrapValidator

_BuiltinExecutionProfile = (
    DockerJobExecutionProfile
    | KubernetesJobExecutionProfile
    | VolcanoJobExecutionProfile
    | SubprocessJobExecutionProfile
    | E2EJobExecutionProfile
    | OpenShellJobExecutionProfile
)
_BUILTIN_PROFILE_MODELS = {
    model.model_fields["backend"].default: model
    for model in (
        DockerJobExecutionProfile,
        KubernetesJobExecutionProfile,
        VolcanoJobExecutionProfile,
        SubprocessJobExecutionProfile,
        E2EJobExecutionProfile,
        OpenShellJobExecutionProfile,
    )
}
_builtin_profile_adapter = TypeAdapter(_BuiltinExecutionProfile)


def parse_execution_profile(value: Any) -> BaseExecutionProfile:
    """Validate configured profiles using installed schemas, never the SDK fallback."""
    if isinstance(value, BaseExecutionProfile):
        value = value.model_dump()
    if not isinstance(value, dict):
        raise ValueError("Execution profile must be an object")
    if "backend" not in value:
        return _builtin_profile_adapter.validate_python(value)
    backend = value["backend"]
    if not isinstance(backend, str):
        raise ValueError("Execution profile backend must be a string")
    model = _BUILTIN_PROFILE_MODELS.get(backend)
    if model is not None:
        return model.model_validate(value)
    contribution = discover_job_backends().get(backend)
    if contribution is None:
        raise ValueError(f"No installed, allowlisted job backend: {backend}")
    profile = contribution.profile_model.model_validate(value)
    if profile.provider not in contribution.implementations:
        raise ValueError(f"No implementation for backend {backend!r} and provider {profile.provider!r}")
    return profile


# The stable schema includes built-ins plus an extensibility shape. Validation
# dispatches explicitly so the generic client fallback never admits server config.
ExecutionProfileT = Annotated[
    SerializeAsAny[_BuiltinExecutionProfile | GenericExecutionProfile],
    WrapValidator(lambda value, _handler: parse_execution_profile(value)),
]
