# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common health check router factory."""

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

import httpx
from nhx.common.config import HelixConfig
from nhx.common.observability import MARK_INTERNAL_REQUEST_HEADERS
from nhx.common.platform_endpoint import resolve_service_endpoint

logger = logging.getLogger(__name__)


def _status_names(values: object) -> set[str]:
    if not isinstance(values, list):
        return set()

    names: set[str] = set()
    for value in values:
        if isinstance(value, Mapping):
            name = value.get("name")
        else:
            name = getattr(value, "name", value)
        if isinstance(name, str):
            names.add(name)
    return names


def service_ready_state_from_status(data: object, service_name: str) -> bool | None:
    """Return service readiness from a platform /status payload.

    ``True`` means the service is ready or absent from this platform deployment.
    ``False`` means the service is explicitly present but not ready.
    ``None`` means the payload shape is unusable and should be retried.
    """
    if not isinstance(data, Mapping):
        return None

    services = data.get("services") or {}
    if not isinstance(services, Mapping):
        return None

    ready = _status_names(services.get("ready") or [])
    if service_name in ready:
        return True

    not_ready = _status_names(services.get("not_ready") or [])
    if service_name in not_ready:
        return False

    # Service is absent from this deployment — treat as ready so callers don't block.
    return True


async def async_wait_for_service_ready(
    platform_config: HelixConfig,
    service_name: str,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
    http_client: httpx.AsyncClient | None = None,
) -> bool:
    """Wait for a specific Helix service to be ready by polling its /status endpoint.

    Uses platform_config.get_service_url(service_name) so each service can have its own URL
    (e.g. from service_discovery). Returns True when the named service appears in
    the response's services.ready list.

    Args:
        platform_config: Platform config; the status URL is derived via get_service_url(service_name).
        service_name: Name of the service to wait for (e.g. "entities", "auth", "files").
        timeout: Maximum time to wait in seconds.
        poll_interval: Time between polling attempts in seconds.
        http_client: Optional client (e.g. for test injection). If None, a temporary client is used.

    Returns:
        True if the service became ready, False if timeout.
    """
    import asyncio

    endpoint = resolve_service_endpoint(service_name, platform_config)
    status_url = f"{endpoint.connect_base_url.rstrip('/')}/status"
    own_client = http_client is None
    if http_client is None:
        http_client = endpoint.async_http_client(timeout=2.0)

    logger.debug("Waiting for service to be ready", extra={"service": service_name, "url": status_url})

    try:
        start = time.monotonic()
        while (time.monotonic() - start) < timeout:
            try:
                response = await http_client.get(
                    status_url,
                    headers=MARK_INTERNAL_REQUEST_HEADERS,
                )
                if response.status_code == 200:
                    data: Any = response.json()
                    ready = service_ready_state_from_status(data, service_name)
                    if ready is True:
                        logger.info("Service is ready", extra={"service": service_name})
                        return True
            except (httpx.RequestError, ValueError) as e:
                logger.debug("Status check failed, will retry", extra={"service": service_name, "error": str(e)})
            await asyncio.sleep(poll_interval)

        logger.warning(
            "Timeout waiting for service to be ready",
            extra={"service": service_name, "url": status_url, "timeout": timeout},
        )
        return False
    finally:
        if own_client:
            await http_client.aclose()


async def async_wait_for_dependencies(
    platform_config: HelixConfig,
    dependency_names: list[str],
    timeout_per_service: float = 120.0,
    poll_interval: float = 0.5,
    http_client: httpx.AsyncClient | None = None,
) -> bool:
    """Wait for all named Helix services to be ready (same pattern as Service._wait_for_dependencies).

    Uses get_service_url(service_name) for each dependency so service APIs may live at different URLs.
    Waits for each dependency in order; returns False if any timeout.

    Args:
        platform_config: Platform config (from Configuration.get_platform_config()).
        dependency_names: Service names to wait for (e.g. ["entities", "auth", "files"]).
        timeout_per_service: Maximum time to wait per service, in seconds.
        poll_interval: Time between polling attempts in seconds.
        http_client: Optional client (e.g. for test injection).

    Returns:
        True if all dependencies became ready, False if any timed out.
    """
    for dep in dependency_names:
        if not await async_wait_for_service_ready(
            platform_config,
            dep,
            timeout=timeout_per_service,
            poll_interval=poll_interval,
            http_client=http_client,
        ):
            return False
    return True


def wait_for_service_ready(
    platform_config: HelixConfig,
    service_name: str,
    stop_signal: threading.Event,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> bool:
    """Wait for a specific Helix service to be ready by polling /status.

    Polls the platform's /status endpoint (which always returns 200 with
    per-service status). Returns True when the named service appears in
    services.ready, so e.g. the jobs controller can start once the jobs
    service is ready without waiting for models or other services.

    Args:
        platform_config: Platform config; the status URL is derived via get_service_url(service_name).
        service_name: Name of the service to wait for (e.g. "models", "jobs", "entities").
        stop_signal: Event to check for early termination.
        timeout: Maximum time to wait in seconds.
        poll_interval: Time between polling attempts in seconds.

    Returns:
        True if the service became ready, False if timeout or stop signal.
    """
    start_time = time.time()
    endpoint = resolve_service_endpoint(service_name, platform_config)
    status_url = f"{endpoint.connect_base_url.rstrip('/')}/status"
    http_client = endpoint.sync_http_client(timeout=2.0)

    logger.info(
        "Waiting for service to be ready",
        extra={"service": service_name, "url": status_url},
    )

    try:
        while not stop_signal.is_set() and (time.time() - start_time) < timeout:
            try:
                response = http_client.get(status_url, headers=MARK_INTERNAL_REQUEST_HEADERS)
                if response.status_code == 200:
                    data: Any = response.json()
                    ready = service_ready_state_from_status(data, service_name)
                    if ready is True:
                        logger.debug(
                            "Service is ready",
                            extra={"service": service_name, "url": status_url},
                        )
                        return True
            except (httpx.RequestError, ValueError) as e:
                logger.debug(
                    "Status check failed, will retry",
                    extra={"service": service_name, "url": status_url, "error": str(e)},
                )
            time.sleep(poll_interval)
    finally:
        http_client.close()

    if stop_signal.is_set():
        logger.debug("Stop signal received while waiting for service")
    else:
        logger.warning(
            "Timeout waiting for service to be ready; check that the platform URL is reachable and the service has started",
            extra={"service": service_name, "url": status_url, "timeout": timeout},
        )
    return False
