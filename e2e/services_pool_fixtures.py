# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable pytest fixtures backed by the E2E services pool."""

import os
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import pytest
from _pytest.reports import TestReport
from nemo_helix import DefaultHttpxClient, NeMoHelix

from e2e.services_pool import E2EServicesPool, RunningServices, admin_headers

_services_pool_manager_key = pytest.StashKey[E2EServicesPool]()
_services_log_key = pytest.StashKey[dict[str, Path] | Path]()

_TAIL_LINES_ON_FAILURE = 100


def _read_services_log_tail(log_path: Path) -> list[str]:
    with log_path.open() as log_file:
        return list(deque(log_file, maxlen=_TAIL_LINES_ON_FAILURE))


def configure_services_pool(config: pytest.Config, *, configure_mock_provider: bool = True) -> None:
    """Initialize the shared service-pool manager for a pytest session."""
    if configure_mock_provider:
        os.environ.setdefault("NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX", "igw-mock-")

        from nemo_helix_plugin.config import Configuration

        Configuration.clear_cache()
    if config.stash.get(_services_pool_manager_key, None) is None:
        config.stash[_services_pool_manager_key] = E2EServicesPool()


def register_services_pool_items(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Register collected modules that may use the service pool."""
    config.stash[_services_pool_manager_key].register_collected_items(items)


def append_services_pool_report_sections(
    item: pytest.Item,
    report: TestReport,
    *,
    metadata_section_name: str,
) -> None:
    """Append service-pool metadata and log tails to failed test reports."""
    if not report.failed:
        return

    metadata: dict[str, str] | None = None
    module = item.getparent(pytest.Module)
    if module is not None:
        manager = item.config.stash[_services_pool_manager_key]
        active_metadata = manager.describe_active_module_binding(module.nodeid)
        if active_metadata:
            metadata = {key: str(value) for key, value in active_metadata.items() if value is not None}

    log_path: Path | None = None
    if metadata and metadata.get("service_log_path"):
        log_path = Path(metadata["service_log_path"])
    if log_path is None and module is not None:
        log_paths_by_module = item.session.stash.get(_services_log_key, None)
        if isinstance(log_paths_by_module, dict):
            log_path = log_paths_by_module.get(module.nodeid)
    if log_path and log_path.exists():
        tail = _read_services_log_tail(log_path)
        if tail:
            header = f"--- services log (last {len(tail)} lines) [{log_path}] ---"
            report.sections.append(("Services Log", f"{header}\n{''.join(tail)}"))
    if metadata:
        report.sections.append(
            (
                metadata_section_name,
                "\n".join(f"{key}: {value}" for key, value in sorted(metadata.items())),
            )
        )


@pytest.fixture(scope="session")
def _services_pool_manager(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[E2EServicesPool]:
    manager = request.config.stash[_services_pool_manager_key]
    manager.bind_tmp_path_factory(tmp_path_factory)
    yield manager
    manager.shutdown_all()


@pytest.fixture(scope="module")
def _services_instance(
    request: pytest.FixtureRequest,
    _services_pool_manager: E2EServicesPool,
) -> Iterator[RunningServices]:
    module = request.node.getparent(pytest.Module)
    if module is None:
        raise RuntimeError("Expected module-scoped service-pool fixture to have a pytest module parent")
    services = _services_pool_manager.acquire_for_module(module)
    if services.log_path is not None:
        log_paths_by_module = request.session.stash.get(_services_log_key, None)
        if not isinstance(log_paths_by_module, dict):
            log_paths_by_module = {}
            request.session.stash[_services_log_key] = log_paths_by_module
        log_paths_by_module[module.nodeid] = services.log_path
    try:
        yield services
    finally:
        _services_pool_manager.release_for_module(module)


@pytest.fixture(scope="module")
def _services(_services_instance: RunningServices) -> Iterator[str]:
    yield _services_instance.url


@pytest.fixture(scope="module", name="services_pool_sdk")
def services_pool_sdk(_services: str, _services_instance: RunningServices) -> NeMoHelix:
    access_token = os.environ.get("NHX_ACCESS_TOKEN")
    context_name = os.environ.get("NHX_CONTEXT_NAME")
    headers = admin_headers() if _services_instance.auth_enabled else {}
    http_client = DefaultHttpxClient(base_url=_services, verify=True) if _services_instance.proc is not None else None
    return NeMoHelix(
        base_url=_services,
        access_token=access_token,
        context_name=context_name,
        http_client=http_client,
        max_retries=2,
        default_headers=headers,
    )
