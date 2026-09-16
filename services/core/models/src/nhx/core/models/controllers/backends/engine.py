# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-agnostic engine dispatch + readiness-probe helpers.

The ``engine`` discriminant on a ``ModelDeploymentConfig`` selects the compiler
path (nim / vllm / generic). These constants and helpers are shared by every
service backend (docker container labels, k8s object labels) so engine selection
and readiness-probe resolution behave identically regardless of where the
deployment runs.
"""

from typing import Any

from nhx.core.models.controllers.backends.common import DeploymentConfigView

ENGINE_NIM = "nim"
ENGINE_VLLM = "vllm"
ENGINE_GENERIC = "generic"

# Label recording the engine, read back at status time to pick the health probe.
# Used as a docker container label and a k8s object/pod label.
ENGINE_LABEL = "nhx.nvidia.com/engine"

# Label recording the resolved readiness-probe path, read back at status time.
# Stamped at create so status polling doesn't need the deployment config.
HEALTH_PATH_LABEL = "nhx.nvidia.com/health-path"

# Per-engine readiness probe paths (relative to the container/pod host URL).
ENGINE_HEALTH_PATHS: dict[str, str] = {
    ENGINE_NIM: "/v1/health/ready",
    ENGINE_VLLM: "/health",
}


def config_engine(config: Any) -> str:
    """Return the engine discriminant as a lowercase string (defaults to nim)."""
    engine = getattr(config, "engine", None)
    if engine is None:
        return ENGINE_NIM
    # engine may be an enum or a plain string depending on the SDK model.
    return str(getattr(engine, "value", engine)).lower()


def resolve_health_path(engine: str, view: DeploymentConfigView) -> str:
    """Resolve the readiness-probe path for a deployment.

    Precedence: an explicit ``executor_config.health_check_path`` wins; otherwise
    fall back to the engine's standard endpoint. ``generic`` containers have no
    engine default, so they fall back to the NIM path unless they set their own.
    """
    explicit_path = getattr(view, "health_check_path", None)
    if explicit_path:
        return explicit_path
    return ENGINE_HEALTH_PATHS.get(engine, ENGINE_HEALTH_PATHS[ENGINE_NIM])
