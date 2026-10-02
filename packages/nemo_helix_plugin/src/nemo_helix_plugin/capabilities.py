# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared backend capability probes for NeMo Helix.

Runtime (``platform.runtime``) describes *where the platform process runs*.
Capability probes answer *which backends can run jobs/deployments right now*.

This module owns the Docker probe used by jobs, deployments, setup, and
customization. GPU/Kubernetes probes are deferred to AIRCORE-972.

Caching
-------
Results are memoized per Docker endpoint for long-lived server processes
(registry boot). CLI / preflight paths that need a fresh verdict should pass
``use_cache=False``. Tests that construct Docker backends should call
:func:`reset_capability_cache` so a prior miss does not poison later fixtures.
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_DOCKER_PROBE_TIMEOUT_SECONDS = 5


class CapabilityUnavailableError(RuntimeError):
    """A required backend capability is unavailable.

    Raised when an optional packaging extra is missing or a runtime substrate
    (e.g. Docker daemon/socket) is unreachable. Registries may soft-skip
    optional backends that raise this, while still fail-closing when a
    configured default still names the unavailable backend.
    """


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a capability probe."""

    available: bool
    detail: str | None = None


# Cache keyed by docker host (None → default DOCKER_HOST / from_env).
_docker_probe_cache: dict[str | None, ProbeResult] = {}


def reset_capability_cache() -> None:
    """Clear memoized probe results.

    Call from test fixtures so a prior unavailable verdict does not pin the
    process. CLI helpers that need a fresh probe should prefer
    ``probe_docker(use_cache=False)`` instead of resetting the whole cache.
    """
    _docker_probe_cache.clear()


def probe_docker(
    *,
    docker_host: str | None = None,
    use_cache: bool = True,
) -> ProbeResult:
    """Probe Docker daemon reachability with a short timeout.

    Args:
        docker_host: Optional Docker API URL override (same meaning as
            ``DOCKER_HOST`` / deployments ``docker_host``). ``None`` uses the
            environment default via ``docker.from_env``. Passed by setting
            ``DOCKER_HOST`` in the env dict — docker-py 7.x rejects
            ``base_url=`` on ``from_env``.
        use_cache: When True (default), reuse a prior result for this host key.
            Pass False for CLI retry UX after the user starts Docker.
    """
    cache_key = docker_host
    if use_cache and cache_key in _docker_probe_cache:
        return _docker_probe_cache[cache_key]

    result = _probe_docker_uncached(docker_host=docker_host)
    if use_cache:
        _docker_probe_cache[cache_key] = result
    return result


def docker_from_env_kwargs(*, timeout: float, docker_host: str | None = None) -> dict[str, object]:
    """Build kwargs for ``docker.from_env`` that docker-py 7.x accepts.

    When ``docker_host`` is set, override ``DOCKER_HOST`` via an ``environment``
    copy. Leave the process environment untouched when no host override is
    configured (default socket / existing ``DOCKER_HOST`` still apply).
    """
    kwargs: dict[str, object] = {"timeout": timeout}
    if docker_host:
        kwargs["environment"] = {**os.environ, "DOCKER_HOST": docker_host}
    return kwargs


def _probe_docker_uncached(*, docker_host: str | None) -> ProbeResult:
    try:
        from docker.errors import DockerException
        from requests.exceptions import ConnectionError as RequestsConnectionError
        from requests.exceptions import Timeout as RequestsTimeout

        import docker
    except ImportError as exc:
        detail = f"Docker Python package is not installed ({exc})"
        logger.debug(detail)
        return ProbeResult(available=False, detail=detail)

    client = None
    try:
        client = docker.from_env(
            **docker_from_env_kwargs(timeout=_DOCKER_PROBE_TIMEOUT_SECONDS, docker_host=docker_host)
        )
        client.ping()
        return ProbeResult(available=True, detail=None)
    except (DockerException, RequestsConnectionError, RequestsTimeout, OSError) as exc:
        detail = f"Docker daemon unreachable ({exc})"
        logger.debug(detail)
        return ProbeResult(available=False, detail=detail)
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()


def require_docker(*, docker_host: str | None = None, use_cache: bool = True) -> None:
    """Raise :class:`CapabilityUnavailableError` when Docker is not available."""
    result = probe_docker(docker_host=docker_host, use_cache=use_cache)
    if not result.available:
        raise CapabilityUnavailableError(result.detail or "Docker daemon is unavailable")
