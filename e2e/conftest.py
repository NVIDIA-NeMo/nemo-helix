# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E test fixtures that run against a real ``nemo services`` process.

Usage::

    # Start services, run e2e tests, stop services
    make test-e2e

    # Or manually
    uv run --frozen pytest e2e -v --run-e2e

    # If you already have services running
    NHX_BASE_URL=http://localhost:9090 uv run --frozen pytest e2e -v --run-e2e

When ``NHX_BASE_URL`` is set the harness skips service startup/shutdown and
connects to the given URL.  Otherwise it spawns ``nemo services run`` as a
child process on a free port, polls ``/status`` until ready, and
terminates the process after the session.

Config selection::

    # Default local platform config
    pytestmark = [pytest.mark.e2e_config()]

    # Single repo-root-relative config file
    pytestmark = [pytest.mark.e2e_config("e2e/configs/local-subprocess.yaml")]

    # Ordered platform config layers: files first, then inline overlays
    pytestmark = [
        pytest.mark.e2e_config(
            "e2e/configs/local-subprocess.yaml",
            {"auth": {"enabled": True}},
        )
    ]

    # Platform layers plus separate harness metadata
    pytestmark = [
        pytest.mark.e2e_config(
            "contrib/auth/authentik/config/platform-compose-authentik.yaml",
            harness={"backend": "docker_compose", ...},
        )
    ]

Why this exists:

- E2E modules should be able to declare the platform shape they need rather
  than inheriting one global config from ``conftest.py``.
- Different modules can exercise different backends or auth modes in the same
  pytest session.
- Identical effective configs are pooled and reused, so config selection does
  not imply one fresh ``nemo services`` process per module.

How pooling works:

- The harness resolves the ordered platform ``e2e_config(...)`` layers into one
  effective config dict, and keeps any ``harness=...`` metadata separate.
- The platform config plus harness config are normalized into one canonical
  pool identity and hashed.
- Modules that resolve to the same hash share one running services instance for
  the session.
- The pooled instance is shut down as soon as the last module using that hash
  finishes, so mixed-config runs do not keep every started platform alive until
  the end of the session.

The pool implementation itself lives in ``e2e.services_pool`` so this file can
stay focused on pytest hooks and fixtures.
"""

import os
import uuid
from collections.abc import Iterator

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import PlatformSecretCreateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest

from e2e.services_pool_fixtures import (  # noqa: F401
    _services,
    _services_instance,
    _services_log_key,
    _services_pool_manager,
    append_services_pool_report_sections,
    configure_services_pool,
    register_services_pool_items,
    services_pool_sdk,
)


def pytest_configure(config: pytest.Config) -> None:
    """Enable mock inference provider for e2e tests.

    Sets the env var and clears the Configuration cache so that
    InferenceGatewayConfig picks up the new value. The cache must be
    cleared because the config module evaluates ``get_service_config()``
    at import time, which may run before this hook.
    """
    configure_services_pool(config)


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Register collected E2E modules with the services pool manager."""
    register_services_pool_items(config, items)


NGC_API_KEY_ENV = "NGC_API_KEY"


@pytest.fixture
def ngc_api_key() -> str:
    """Return the NGC API key from the environment.

    Skips the test when the key is missing or set to a CI placeholder
    value (e.g. ``not-used-for-ghcr-cpu-*``).
    """
    key = os.environ.get(NGC_API_KEY_ENV, "")
    if not key or key.startswith("not-used"):
        pytest.skip(f"{NGC_API_KEY_ENV} not set or is a placeholder")
    return key


@pytest.fixture
def ngc_secret(sdk: NeMoHelix, workspace: str, ngc_api_key: str) -> Iterator[str]:
    """Create a secret containing the NGC API key, cleaned up after test."""
    secret_name = f"e2e-ngc-key-{uuid.uuid4().hex[:8]}"
    secrets = client_from_platform(sdk, SecretsClient)
    secrets.create_secret(workspace=workspace, body=PlatformSecretCreateRequest(name=secret_name, value=ngc_api_key))
    yield secret_name
    try:
        secrets.delete_secret(workspace=workspace, name=secret_name)
    except Exception:
        pass  # Best-effort cleanup; the workspace is deleted anyway


# ---- Services log tail on failure ------------------------------------------


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):  # noqa: ARG001
    """Append the services log tail to the report when a test fails.

    This hook is the pytest-sanctioned way to add extra sections to test
    reports (``report.sections``).  Fixtures cannot do this because they
    don't have access to the report object.
    """
    outcome = yield
    report = outcome.get_result()
    append_services_pool_report_sections(item, report, metadata_section_name="E2E Services Binding")


@pytest.fixture(scope="module", name="sdk")
def e2e_sdk(request: pytest.FixtureRequest) -> NeMoHelix:
    """Provide the conventional e2e SDK fixture name."""
    return request.getfixturevalue("services_pool_sdk")


@pytest.fixture(scope="module")
def files_client(sdk: NeMoHelix) -> FilesClient:
    """Provide a FilesClient derived from the SDK."""
    return client_from_platform(sdk, FilesClient)


@pytest.fixture(scope="function")
def workspace(sdk: NeMoHelix) -> Iterator[str]:
    """Create a unique workspace for each test, deleted on teardown."""
    workspaces = client_from_platform(sdk, WorkspacesClient)
    name = f"e2e-{uuid.uuid4().hex[:8]}"
    workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
    yield name
    workspaces.delete_workspace(name=name).data()
