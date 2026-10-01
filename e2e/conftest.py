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
import subprocess
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.jobs.image import image_builder
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing import NemoRun
from nhx.testing.e2e.jobs import wait_budget
from pydantic import SecretStr

from e2e.services_pool_fixtures import (  # noqa: F401
    _services,
    _services_instance,
    _services_log_key,
    _services_pool_manager,
    append_services_pool_report_sections,
    configure_services_pool,
    register_services_pool_items,
    services_pool_client,
    services_pool_sdk,
)

DEFAULT_DOCKER_INTERNAL_HOST = "nhx-quickstart:8080"
DEFAULT_KUBERNETES_INTERNAL_HOST = "nemo-helix-api:8080"
DEFAULT_PRINCIPAL_ID = "e2e-test-user@example.com"
PLATFORM_DEPLOY_E2E_PREFIXES = (
    "e2e/auditor/",
    "e2e/cli/",
    "e2e/customizer/",
    "e2e/notebooks/",
)
PLATFORM_DEPLOY_E2E_FILES = {
    "e2e/test_files.py",
    "e2e/test_hello_world.py",
    "e2e/test_jobs_kai_scheduler.py",
    "e2e/test_models.py",
    "e2e/test_safe_synthesizer.py",
    "e2e/test_secrets.py",
    "e2e/test_workspace_cleanup.py",
    "e2e/test_workspaces.py",
}
DEFAULT_QUICKSTART_CONFIG = "e2e/quickstart/default.yaml"
AUTH_QUICKSTART_CONFIG = "e2e/quickstart/auth.yaml"

_active_platform: str | None = None
_selected_features: set[str] = set()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add Platform-Deploy-compatible e2e options."""
    group = parser.getgroup("e2e", "E2E testing options")
    group.addoption(
        "--docker",
        action="store_true",
        default=False,
        help="Run e2e tests with the Docker quickstart harness.",
    )
    group.addoption(
        "--kubernetes",
        action="store_true",
        default=False,
        help="Run e2e tests against an already-deployed cluster.",
    )
    group.addoption(
        "--config",
        action="store",
        default=None,
        help="Path to the e2e platform config file used by the local harness.",
    )
    group.addoption(
        "--principal-id",
        action="store",
        default=None,
        help="Principal ID for auth-enabled e2e tests.",
    )
    group.addoption(
        "--registry",
        action="store",
        default=None,
        help="Docker registry for e2e images.",
    )
    group.addoption(
        "--tag",
        action="store",
        default=None,
        help="Docker image tag for e2e images.",
    )
    group.addoption(
        "--cluster-url",
        action="store",
        default=None,
        help="URL of an already-running e2e cluster.",
    )
    group.addoption(
        "--access-token",
        action="store",
        default=None,
        help="Bearer token for an auth-enabled e2e cluster.",
    )
    group.addoption(
        "--workspace",
        action="store",
        default=None,
        help="Use an existing workspace instead of creating one per test.",
    )
    group.addoption(
        "--run-external-network",
        action="store_true",
        default=False,
        help="Run tests marked external_network.",
    )
    group.addoption(
        "--wait-for-auth",
        action="store_true",
        default=False,
        help="Poll service readiness after creating per-test workspaces.",
    )
    group.addoption(
        "--internal-host",
        action="store",
        default=None,
        help="Internal host that job pods/containers use to reach the platform API.",
    )


def _cluster_url(config: pytest.Config) -> str | None:
    return config.getoption("--cluster-url", default=None) or os.environ.get("NHX_E2E_CLUSTER_URL")


def _features(config: pytest.Config) -> set[str]:
    return {feature.lower() for feature in config.getoption("--feature", default=[]) or []}


def _default_config_for_features(features: set[str]) -> str:
    return AUTH_QUICKSTART_CONFIG if "auth" in features else DEFAULT_QUICKSTART_CONFIG


def pytest_configure(config: pytest.Config) -> None:
    """Enable mock inference provider and Platform-Deploy-compatible e2e mode."""
    global _active_platform, _selected_features

    _selected_features = _features(config)
    use_docker = bool(config.getoption("--docker", default=False))
    use_kubernetes = bool(config.getoption("--kubernetes", default=False))
    cluster_url = _cluster_url(config)

    if use_docker and use_kubernetes:
        raise pytest.UsageError("Cannot use both --docker and --kubernetes. Choose one platform.")
    if use_kubernetes:
        _active_platform = "kubernetes"
        if not cluster_url:
            raise pytest.UsageError("When using --kubernetes, set --cluster-url or NHX_E2E_CLUSTER_URL.")
        os.environ.setdefault("NHX_BASE_URL", cluster_url)
        os.environ.setdefault("NHX_E2E_INTERNAL_HOST", DEFAULT_KUBERNETES_INTERNAL_HOST)
    elif use_docker:
        _active_platform = "docker"
        if cluster_url:
            os.environ.setdefault("NHX_BASE_URL", cluster_url)
        os.environ.setdefault("NHX_E2E_INTERNAL_HOST", DEFAULT_DOCKER_INTERNAL_HOST)
    else:
        _active_platform = "kubernetes" if cluster_url else "subprocess"
        if cluster_url:
            os.environ.setdefault("NHX_BASE_URL", cluster_url)

    if use_docker or use_kubernetes or cluster_url:
        # Root conftest skips e2e tests unless --run-e2e is set. Selecting an
        # explicit e2e backend is also an explicit request to run e2e tests.
        config.option.run_e2e = True

    access_token = config.getoption("--access-token", default=None) or os.environ.get("NHX_E2E_ACCESS_TOKEN")
    if access_token:
        os.environ.setdefault("NHX_ACCESS_TOKEN", access_token)
    registry = config.getoption("--registry", default=None) or os.environ.get("NHX_E2E_REGISTRY")
    if registry:
        os.environ.setdefault("NHX_E2E_IMAGE_REGISTRY", registry)
    tag = config.getoption("--tag", default=None) or os.environ.get("NHX_E2E_TAG")
    if tag:
        os.environ.setdefault("NHX_E2E_IMAGE_TAG", tag)
    if "gpu" in _selected_features:
        os.environ.setdefault("NHX_E2E_GPU_REQUESTED", "1")
    config_ref = _selected_config_ref(config)
    if config_ref:
        os.environ["NHX_E2E_CONFIG_REF"] = config_ref
    os.environ["NHX_E2E_HARNESS_BACKEND"] = _selected_harness()["backend"]

    config.addinivalue_line("markers", "platform(*names): restrict test to e2e platform(s).")
    config.addinivalue_line("markers", "feature(*names): optional e2e feature requirement.")
    config.addinivalue_line("markers", "external_network: test requires outbound network access.")
    config.addinivalue_line("markers", "skip_on_astra: skip this test when NHX_E2E_ON_ASTRA=1.")
    config.addinivalue_line("markers", "platform_deploy: tests owned by Platform-Deploy orchestration jobs.")

    configure_services_pool(config)


def _selected_config_ref(config: pytest.Config) -> str | None:
    explicit = config.getoption("--config", default=None)
    if explicit:
        return explicit
    if _active_platform == "docker":
        return _default_config_for_features(_selected_features)
    return None


def _selected_harness() -> dict[str, str]:
    if _active_platform == "docker" and not os.environ.get("NHX_BASE_URL"):
        return {"backend": "docker"}
    return {"backend": "subprocess"}


def _is_running_on_astra() -> bool:
    return os.environ.get("NHX_E2E_ON_ASTRA", "").lower() in {"1", "true", "yes"}


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply Platform-Deploy-compatible filters, then register services-pool modules."""
    run_external_network = config.getoption("--run-external-network", default=False)
    on_astra = _is_running_on_astra()
    config_ref = _selected_config_ref(config)
    harness = _selected_harness()

    for item in items:
        relpath = item.path.relative_to(config.rootpath).as_posix()
        if relpath in PLATFORM_DEPLOY_E2E_FILES or relpath.startswith(PLATFORM_DEPLOY_E2E_PREFIXES):
            item.add_marker(pytest.mark.platform_deploy)

        if on_astra and item.get_closest_marker("skip_on_astra"):
            item.add_marker(pytest.mark.skip(reason="Test is known to fail on Astra (skip_on_astra)."))
        if item.get_closest_marker("external_network") and not run_external_network:
            item.add_marker(pytest.mark.skip(reason="Requires --run-external-network."))

        platform_marker = item.get_closest_marker("platform")
        if platform_marker and _active_platform:
            allowed_platforms = set(platform_marker.args)
            if _active_platform not in allowed_platforms:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Test requires one of {allowed_platforms}, but platform is {_active_platform!r}"
                    )
                )

        feature_marker = item.get_closest_marker("feature")
        if feature_marker:
            required_features = {str(feature).lower() for feature in feature_marker.args}
            if _selected_features:
                if not required_features.issubset(_selected_features):
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Test requires all of {required_features}; only {_selected_features} selected"
                        )
                    )
            else:
                item.add_marker(
                    pytest.mark.skip(reason=f"Test requires feature(s) {required_features}. Run with --feature.")
                )
        elif _selected_features and (
            "sdk" in getattr(item, "fixturenames", []) or "client" in getattr(item, "fixturenames", [])
        ):
            item.add_marker(pytest.mark.skip(reason="Feature-selected runs only include tests with feature markers."))

        if config_ref and ("sdk" in getattr(item, "fixturenames", []) or "client" in getattr(item, "fixturenames", [])):
            if item.get_closest_marker("e2e_config") is None:
                item.add_marker(pytest.mark.e2e_config(config_ref, harness=harness))

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
def ngc_secret(client: NemoClient, workspace: str, ngc_api_key: str) -> Iterator[str]:
    """Create a secret containing the NGC API key, cleaned up after test."""
    secret_name = f"e2e-ngc-key-{uuid.uuid4().hex[:8]}"
    secrets = SecretsClient.from_client(client)
    secrets.create_secret(
        workspace=workspace, body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(ngc_api_key))
    )
    yield secret_name
    try:
        secrets.delete_secret(workspace=workspace, name=secret_name)
    except Exception:
        pass  # Best-effort cleanup; the workspace is deleted anyway


# ---- Job wait budget --------------------------------------------------------

# Share of a test's pytest-timeout budget that job waits may consume. The rest
# covers the call phase's other work: setup before the wait, the diagnostics a
# failed wait collects, and ``finally`` cleanup.
_JOB_WAIT_BUDGET_FRACTION = 0.8


def _effective_pytest_timeout(item: pytest.Item) -> float | None:
    """Return the pytest-timeout budget in force for *item*, in seconds."""
    marker = item.get_closest_marker("timeout")
    if marker is not None:
        value = marker.kwargs.get("timeout")
        if value is None and marker.args:
            value = marker.args[0]
        if value is not None:
            return float(value)
    try:
        configured = item.config.getini("timeout")
    except (ValueError, KeyError):  # pytest-timeout not installed
        return None
    try:
        return float(configured)
    except (TypeError, ValueError):
        return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """Cap job waits below this test's own pytest-timeout budget.

    Wraps the call phase because that is the phase pytest-timeout arms. See the
    module docstring of ``nhx/testing/e2e/jobs.py`` for why the order matters.
    """
    timeout = _effective_pytest_timeout(item)
    budget = timeout * _JOB_WAIT_BUDGET_FRACTION if timeout else None
    with wait_budget(budget):
        yield


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
    """Generated SDK handle for the Data Designer engine probes that still take one."""
    return request.getfixturevalue("services_pool_sdk")


@pytest.fixture(scope="module", name="client")
def e2e_client(request: pytest.FixtureRequest) -> NemoClient:
    """Typed platform client for the module's pooled services instance."""
    return request.getfixturevalue("services_pool_client")


@pytest.fixture(scope="module")
def files_client(client: NemoClient) -> FilesClient:
    """Provide a FilesClient sharing the typed client's transport."""
    return FilesClient.from_client(client)


@pytest.fixture(scope="session")
def principal_id(request: pytest.FixtureRequest) -> str:
    """Principal ID used by auth-enabled harnesses."""
    return request.config.getoption("--principal-id") or os.environ.get("NHX_E2E_PRINCIPAL_ID") or DEFAULT_PRINCIPAL_ID


@pytest.fixture(scope="session")
def cluster_url(request: pytest.FixtureRequest) -> str | None:
    """Return the external cluster URL, when one was provided."""
    return _cluster_url(request.config)


@pytest.fixture(scope="session")
def registry(request: pytest.FixtureRequest) -> str | None:
    """Return the selected e2e image registry."""
    return request.config.getoption("--registry") or os.environ.get("NHX_E2E_IMAGE_REGISTRY")


@pytest.fixture(scope="session")
def tag(request: pytest.FixtureRequest) -> str | None:
    """Return the selected e2e image tag."""
    return request.config.getoption("--tag") or os.environ.get("NHX_E2E_IMAGE_TAG")


@pytest.fixture(scope="session")
def image(registry: str | None, tag: str | None) -> Callable[[str], str]:
    """Return a function that builds fully-qualified e2e image references."""
    return image_builder(registry=registry, tag=tag)


@pytest.fixture(scope="session")
def internal_host(request: pytest.FixtureRequest) -> str:
    """Host that jobs use to call the platform API from inside the test backend."""
    default_host = (
        DEFAULT_KUBERNETES_INTERNAL_HOST if _active_platform == "kubernetes" else DEFAULT_DOCKER_INTERNAL_HOST
    )
    return request.config.getoption("--internal-host") or os.environ.get("NHX_E2E_INTERNAL_HOST") or default_host


@pytest.fixture(scope="function")
def workspace(request: pytest.FixtureRequest, client: NemoClient) -> Iterator[str]:
    """Create a unique workspace for each test, deleted on teardown unless fixed."""
    fixed_name = request.config.getoption("--workspace") or os.environ.get("NHX_E2E_WORKSPACE")
    if fixed_name:
        yield fixed_name
        return

    workspaces = WorkspacesClient.from_client(client)
    name = f"e2e-{uuid.uuid4().hex[:8]}"
    workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
    yield name
    workspaces.delete_workspace(name=name).data()


@pytest.fixture(scope="function")
def nemo_run(request: pytest.FixtureRequest) -> NemoRun:
    """Run the NeMo CLI from the repo root with e2e connection env vars."""
    services_url = request.getfixturevalue("_services")
    base_url = (services_url or "").rstrip("/") or None
    repo_root = Path(request.config.rootpath)

    def run(
        *args: str,
        workspace: str | None = None,
        env_extra: dict[str, str] | None = None,
        timeout: int | None = 60,
        capture_output: bool = True,
        stdin: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        if base_url:
            env["NHX_BASE_URL"] = base_url
        if workspace is not None:
            env["NHX_WORKSPACE"] = workspace
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            ["uv", "run", "--frozen", "nemo", "-f", "json", *args],
            cwd=repo_root,
            env=env,
            timeout=timeout,
            capture_output=capture_output,
            stdin=stdin,
            text=True,
        )

    return run
