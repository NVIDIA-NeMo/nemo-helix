# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level auth integration fixtures backed by the E2E services pool."""

import pytest

from e2e.services_pool_fixtures import (  # noqa: F401
    _services,
    _services_instance,
    _services_pool_manager,
    append_services_pool_report_sections,
    configure_services_pool,
    register_services_pool_items,
    services_pool_client,
)


def pytest_configure(config: pytest.Config) -> None:
    configure_services_pool(config, configure_mock_provider=False)


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    pool_items = [item for item in items if item.get_closest_marker("e2e_config") is not None]
    register_services_pool_items(config, pool_items)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):  # noqa: ARG001
    outcome = yield
    report = outcome.get_result()
    append_services_pool_report_sections(item, report, metadata_section_name="Auth Integration Services Binding")
