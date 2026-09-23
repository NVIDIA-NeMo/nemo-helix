# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Root-level conftest.py for NeMo Helix

This file contains shared fixtures and configuration that are available to all tests
in the repository. Individual packages and services can add their own conftest.py
files for more specific fixtures.
"""

import logging
import os
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

from tests.xdist_cache import xdist_worker_xdg_cache_home

_xdist_cache_home = xdist_worker_xdg_cache_home(os.environ)
if _xdist_cache_home is not None:
    os.environ["XDG_CACHE_HOME"] = _xdist_cache_home
    os.environ["NHX_PYTEST_XDIST_CACHE_HOME_ISOLATED"] = "1"

import pytest  # noqa: E402
from nhx.testing.pytest_outcomes import pytest_skip as skip_test  # noqa: E402

from tests.auth_idp.xdist import append_xdist_group_suffix  # noqa: E402
from tests.discovery_exclusions import TEST_DISCOVERY_EXCLUSIONS  # noqa: E402

# Set test environment variables BEFORE any imports
# This must happen at module level, before pytest even starts processing
if "MODE" not in os.environ:
    os.environ["MODE"] = "development"

# Silence config warnings during tests - services don't need the config file
# when using create_test_client() which programmatically sets configuration overrides
if "NHX_CONFIG_WARNINGS_DISABLED" not in os.environ:
    os.environ["NHX_CONFIG_WARNINGS_DISABLED"] = "1"


# ============================================================================
# Suppress httpcore debug logging during cleanup
# This prevents "I/O operation on closed file" errors when huggingface_hub's
# atexit handler tries to log debug messages after pytest closes stdout/stderr
# ============================================================================
_httpcore_logger = logging.getLogger("httpcore")
_httpcore_logger.setLevel(logging.WARNING)
_httpcore_logger.propagate = False  # Prevent any handlers from receiving these messages

_httpx_logger = logging.getLogger("httpx")
_httpx_logger.setLevel(logging.WARNING)
_httpx_logger.propagate = False


# ============================================================================
# Session-level fixtures
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def cleanup_huggingface_hub_session():
    """
    Clean up huggingface_hub HTTP sessions before pytest shuts down.

    This prevents "I/O operation on closed file" logging errors that occur when
    huggingface_hub's atexit handler tries to close HTTP connections after
    pytest has already closed stdout/stderr.
    """
    yield
    # Suppress httpcore logging during cleanup to prevent errors on closed streams
    # This must be done again here in case new loggers were created during tests
    logging.getLogger("httpcore").setLevel(logging.CRITICAL)
    logging.getLogger("httpcore").propagate = False
    logging.getLogger("httpcore").disabled = True

    # Clean up huggingface_hub's HTTP client before pytest closes streams
    try:
        from huggingface_hub.utils import _http

        # Close the session if it exists
        if hasattr(_http, "get_session"):
            # Newer versions use get_session()
            try:
                session = _http.get_session()
                if session is not None:
                    session.close()
            except Exception:
                pass
        # Also try the module-level close function
        if hasattr(_http, "close_session"):
            try:
                _http.close_session()
            except Exception:
                pass
    except (ImportError, AttributeError):
        pass  # huggingface_hub not installed or structure changed


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "tests" / "data"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the path to the repository root."""
    return Path(__file__).parent


# ============================================================================
# Function-level fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for a test.

    The directory is automatically cleaned up after the test completes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    """
    Create a temporary file for a test.

    The file is automatically cleaned up after the test completes.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
        tmpfile_path = Path(tmpfile.name)

    yield tmpfile_path

    # Cleanup
    if tmpfile_path.exists():
        tmpfile_path.unlink()


@pytest.fixture
def mock_env_vars(monkeypatch) -> dict:
    """
    Provide a way to set environment variables for a test.

    Usage:
        def test_something(mock_env_vars):
            mock_env_vars['MY_VAR'] = 'value'
            # Test code that reads MY_VAR
    """
    env_vars = {}

    def set_env(key: str, value: str):
        env_vars[key] = value
        monkeypatch.setenv(key, value)

    # Return a dict-like object that sets env vars when assigned
    class EnvVarSetter(dict):
        def __setitem__(self, key, value):
            set_env(key, value)
            super().__setitem__(key, value)

    return EnvVarSetter()


# ============================================================================
# Pytest hooks
# ============================================================================


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    """Skip test trees with explicit temporary root-CI discovery exclusions."""
    try:
        relative_path = collection_path.relative_to(Path(__file__).parent)
    except ValueError:
        return False

    return any(
        relative_path == excluded_path or excluded_path in relative_path.parents
        for excluded_path in TEST_DISCOVERY_EXCLUSIONS
    )


def pytest_load_initial_conftests(early_config, parser, args):
    """
    Called before any conftest files are loaded.

    This is the earliest hook that runs, making it ideal for setting
    environment variables that are needed by module-level code in conftest files.
    """
    # Set default test environment variables BEFORE any imports happen
    # This is necessary because some modules create singleton configs at import time
    if "MODE" not in os.environ:
        os.environ["MODE"] = "development"


def pytest_configure(config):
    """
    Configure pytest with custom settings.

    This runs once at the start of the test session.
    """
    config.option.importmode = "importlib"
    _enable_crash_dumps(config)
    global _SESSION_TIMEOUT_SECONDS
    _SESSION_TIMEOUT_SECONDS = getattr(config.option, "timeout", None)


def _enable_crash_dumps(config) -> None:
    """Route fatal-signal tracebacks to a per-worker file when ``PYTEST_CRASH_DUMP_DIR`` is set.

    xdist does not forward worker streams, so a signal death otherwise reports only ``node down``.
    """
    dump_dir = os.environ.get("PYTEST_CRASH_DUMP_DIR")
    if not dump_dir:
        return

    import faulthandler

    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    directory = Path(dump_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Line buffered: a fatal signal gives no chance to flush.
    handle = (directory / f"crash-{worker}-{os.getpid()}.log").open("w", buffering=1)
    faulthandler.enable(file=handle, all_threads=True)
    # Module-level so the handle outlives this function; closing it would disable the handler.
    global _CRASH_DUMP_FILE
    _CRASH_DUMP_FILE = handle
    handle.write(f"--- worker {worker} pid {os.getpid()}\n")


#: Open crash-dump file for this process, or None when crash dumps are disabled.
_CRASH_DUMP_FILE = None

#: Session-wide per-test timeout, to judge a lost worker's elapsed time against.
_SESSION_TIMEOUT_SECONDS = None

#: How long before the session timeout the stack dump fires, so it lands before os._exit.
_TIMEOUT_DUMP_LEAD_SECONDS = 5

#: Start time of each in-flight test, by nodeid. xdist forwards `logstart` to the controller, so a
#: lost worker's test can still be timed from there.
_TEST_START_TIMES: dict[str, float] = {}


def pytest_runtest_logstart(nodeid, location):
    """Record the test being entered, so the tail of a crash dump names it."""
    del location
    _TEST_START_TIMES[nodeid] = time.monotonic()
    if _CRASH_DUMP_FILE is not None:
        _CRASH_DUMP_FILE.write(f"--- entering {nodeid}\n")


def pytest_runtest_logfinish(nodeid, location):
    """Forget a test that finished, so only genuinely in-flight tests are timed."""
    del location
    _TEST_START_TIMES.pop(nodeid, None)
    if _CRASH_DUMP_FILE is not None:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()


def _effective_timeout_seconds(item) -> float | None:
    """The pytest-timeout budget for ``item``: its closest ``timeout`` marker, else the session default."""
    marker = item.get_closest_marker("timeout")
    if marker is None:
        return _SESSION_TIMEOUT_SECONDS
    value = marker.args[0] if marker.args else marker.kwargs.get("timeout")
    return float(value) if value is not None else _SESSION_TIMEOUT_SECONDS


def _arm_timeout_stack_dump(timeout_seconds: float | None) -> None:
    """Dump every thread's stack into the crash file just before pytest-timeout kills the worker.

    pytest-timeout's thread method prints its stacks to worker stderr, which xdist discards, so a
    timed-out test otherwise leaves only its name behind.
    """
    if not timeout_seconds or timeout_seconds <= _TIMEOUT_DUMP_LEAD_SECONDS:
        return
    import faulthandler

    faulthandler.dump_traceback_later(timeout_seconds - _TIMEOUT_DUMP_LEAD_SECONDS, file=_CRASH_DUMP_FILE, exit=False)


def pytest_handlecrashitem(crashitem, report, sched):
    """Say how long a lost worker's test ran, and name a timeout when it looks like one.

    pytest-timeout's thread method ``os._exit(1)``s the worker, which the controller cannot tell
    from a crash. Elapsed time is the one number that distinguishes them.
    """
    del sched
    started = _TEST_START_TIMES.pop(crashitem, None)
    if started is None:
        return None
    elapsed = time.monotonic() - started
    if _SESSION_TIMEOUT_SECONDS and elapsed >= _SESSION_TIMEOUT_SECONDS * 0.9:
        note = (
            f"ran {elapsed:.1f}s against --timeout={_SESSION_TIMEOUT_SECONDS:g}s "
            f"— likely a TIMEOUT, not a crash (pytest-timeout os._exit()s the worker)"
        )
    else:
        note = f"ran {elapsed:.1f}s before the worker was lost"
    report.longrepr = f"{report.longrepr}\n  {note}"
    return None


def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection.

    Auto-marks tests based on their location:
    - Tests in /integration/ directories get the 'integration' marker
    - Tests without category markers get the 'unit' marker
    """
    # Category markers that determine test type
    category_markers = {
        "unit",
        "e2e",
        "auth_idp",
        "auth_idp_docker",
        "auth_idp_k8s",
        "smoke_gpu_tasks",
        "smoke_nhx_customizer_tasks",
        "smoke_nhx_automodel_training",
        "integration",
        "regression",
        "canary",
        "slow",
        "skip_in_ci",
    }

    for item in items:
        # Get current marker names
        marker_names = {marker.name for marker in item.iter_markers()}
        fspath_str = str(item.fspath)

        # Auto-mark tests in e2e directories
        if "/e2e/" in fspath_str:
            if "e2e" not in marker_names:
                item.add_marker(pytest.mark.e2e)
                marker_names.add("e2e")

        # Auto-mark integration tests (e.g., /services/core/jobs/tests/integration/)
        elif "/integration/" in fspath_str:
            if "integration" not in marker_names:
                item.add_marker(pytest.mark.integration)
                marker_names.add("integration")

        # Auto-mark async tests that might be slow
        if "async" in item.name and item.get_closest_marker("timeout") is None:
            # Give async tests a longer timeout by default
            item.add_marker(pytest.mark.timeout(600))

        # Auto-mark tests without category markers as unit tests
        if not marker_names.intersection(category_markers):
            item.add_marker(pytest.mark.unit)

        if getattr(config.option, "numprocesses", None) or getattr(config, "workerinput", None) is not None:
            group_names = set()
            for mark in item.iter_markers("xdist_group"):
                name = mark.args[0] if mark.args else mark.kwargs.get("name", "default")
                group_names.add(str(name))
            item._nodeid = append_xdist_group_suffix(item.nodeid, group_names)


# ============================================================================
# Pytest command-line options
# ============================================================================


def pytest_addoption(parser):
    """
    Add custom command-line options for pytest.
    """
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (skipped by default)",
    )
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests (skipped by default in quick test runs)",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=True,
        help="Run integration tests (enabled by default)",
    )
    parser.addoption(
        "--feature",
        action="append",
        default=[],
        metavar="NAME",
        help="Enable optional e2e feature sets (repeatable), e.g. --feature gpu",
    )


def _e2e_features_enabled(config: pytest.Config) -> set[str]:
    features = {feature.lower() for feature in config.getoption("--feature") or []}
    if os.environ.get("RUN_NSS_K8S_E2E") == "1":
        features.add("gpu")
    return features


def pytest_runtest_setup(item):
    """
    Run before each test to check if it should be skipped based on command-line options.
    """
    if _CRASH_DUMP_FILE is not None:
        _arm_timeout_stack_dump(_effective_timeout_seconds(item))
    if "slow" in [marker.name for marker in item.iter_markers()]:
        if not item.config.getoption("--run-slow"):
            skip_test("Skipping slow test (use --run-slow to run)")
    if "e2e" in [marker.name for marker in item.iter_markers()]:
        auth_idp_runtime = item.config.getoption("--auth-idp-runtime", default=None)
        auth_idp_runtime_selected = auth_idp_runtime and "auth_idp" in [marker.name for marker in item.iter_markers()]
        if not item.config.getoption("--run-e2e") and not auth_idp_runtime_selected:
            skip_test("Skipping e2e test (use --run-e2e to run)")
    if "subprocess_only" in [marker.name for marker in item.iter_markers()]:
        if os.environ.get("NHX_BASE_URL"):
            skip_test("Skipping subprocess-only test (NHX_BASE_URL is set)")
    if "container_only" in [marker.name for marker in item.iter_markers()]:
        if not os.environ.get("NHX_BASE_URL"):
            skip_test("Skipping container-only test (requires NHX_BASE_URL)")
    if "needs_agents_e2e_image" in [marker.name for marker in item.iter_markers()]:
        if not (os.environ.get("NHX_E2E_IMAGE_REGISTRY") and os.environ.get("NHX_E2E_IMAGE_TAG")):
            skip_test("Skipping agents-image test (set NHX_E2E_IMAGE_REGISTRY and NHX_E2E_IMAGE_TAG)")
    if "requires_gpu" in [marker.name for marker in item.iter_markers()]:
        if "gpu" not in _e2e_features_enabled(item.config):
            skip_test("Skipping GPU container e2e (pass --feature gpu)")


from xdist.scheduler.loadgroup import LoadGroupScheduling  # noqa: E402
from xdist.scheduler.loadscope import LoadScopeScheduling  # noqa: E402

# Temporary workaround for https://github.com/pytest-dev/pytest-xdist/issues/1189
# Remove once pytest-xdist > 3.8.0 is released with the fix from PR #1299

_original_reschedule = LoadScopeScheduling._reschedule


def _patched_reschedule(self, node):
    if node not in self.registered_collections:
        return
    _original_reschedule(self, node)


LoadScopeScheduling._reschedule = _patched_reschedule  # type: ignore[invalid-assignment]


def pytest_xdist_make_scheduler(config, log):
    dist = config.getvalue("dist")
    if dist == "loadgroup":
        return LoadGroupScheduling(config, log)
    if dist == "loadscope":
        return LoadScopeScheduling(config, log)
    return None
