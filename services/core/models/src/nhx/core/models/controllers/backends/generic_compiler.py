# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-agnostic compiler for the ``generic`` engine.

The ``generic`` engine runs an arbitrary container as-is: there is no
inference-engine compiler synthesizing args or env. The deployment declares a
container image + tag, a readiness-probe path, and (optionally) raw args and
environment variables; the platform runs the image and probes it.

Like :mod:`vllm_compiler`, these functions take a :class:`DeploymentConfigView`
and return plain data (image tuple, arg vector, env dict). They are NOT specific
to any service backend: the docker backend renders the result into a
``docker run`` container; the k8s backend renders it into a native Kubernetes
Deployment. Keep this module free of backend-specific imports so both can reuse
it.

Unlike the vLLM/NIM engines there is no platform-default image for a generic
container, so :func:`resolve_generic_image` raises when ``image_name`` is unset.
The create/update API layer validates this up front; the compiler enforces it
again as a defensive backstop.
"""

from nhx.core.models.controllers.backends.common import DeploymentConfigView


def resolve_generic_image(view: DeploymentConfigView) -> tuple[str, str]:
    """Resolve the generic container image name and tag.

    There is no platform default for a generic image (it is an arbitrary
    user-supplied container), so ``image_name`` is required. ``image_tag``
    defaults to ``latest`` when unset, mirroring Docker/Kubernetes conventions.

    Both values are trimmed: the API layer already rejects whitespace-padded
    inputs for generic configs, but this stays robust to any other call path.
    """
    if not (view.image_name and view.image_name.strip()):
        raise ValueError("The 'generic' engine requires executor_config.image_name to be set (no platform default).")
    image_name = view.image_name.strip()
    image_tag = (view.image_tag or "").strip() or "latest"
    return image_name, image_tag


def compile_generic_args(view: DeploymentConfigView) -> list[str]:
    """Return the container arg vector for a generic container.

    The platform synthesizes nothing for the generic engine: the user's
    ``additional_args`` become the container arg vector. On Kubernetes this
    replaces the image CMD (ENTRYPOINT is unchanged); on Docker it replaces the
    run command. When the image has no ENTRYPOINT, include the executable as
    argv[0] (e.g. ``["python3", "-m", "http.server", "8000"]`` for
    ``python:alpine``). Returns an empty list when none are supplied, in which
    case the image's default command/args are used unchanged.
    """
    return list(view.additional_args or [])


def compile_generic_env_vars(view: DeploymentConfigView) -> dict[str, str]:
    """Return the environment variables for a generic container.

    Only the user's ``additional_envs`` are applied; the platform injects no
    engine-specific environment for a generic container.
    """
    return {str(k): str(v) for k, v in (view.additional_envs or {}).items()}
