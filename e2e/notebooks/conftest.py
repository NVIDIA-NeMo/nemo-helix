# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for documentation notebook e2e tests.

Provides fixtures that start the quickstart backend and expose
NHX_BASE_URL so notebooks executed via papermill can reach the API.

By default every notebook runs inside an **isolated virtualenv** that
contains only the public ``nemo-helix`` package (``packages/nemo_helix/``, installed
as editable) and ``ipykernel``.  This catches accidental imports of
internal packages and mirrors what end-users have installed.

``VIRTUAL_ENV`` and ``PATH`` are set so that ``%%bash`` cells running
``pip install`` or ``uv pip install`` target the sandbox, not the host.

Pass ``--notebook-kernel python3`` to skip sandbox creation and use
the host kernel instead (handy for fast local iteration).

Usage:
    # Run all CPU notebooks against Docker quickstart (sandboxed kernel)
    uv run pytest e2e/notebooks -v

    # Same but use the host kernel (faster, no isolation)
    uv run pytest e2e/notebooks -v --notebook-kernel python3

    # GPU-requiring notebooks
    uv run pytest e2e/notebooks -v --feature gpu

    # Execute both Python and shell cells
    uv run pytest e2e/notebooks -v --notebook-language all

    # Against an already-running quickstart or cluster
    uv run pytest e2e/notebooks -v --docker --cluster-url=http://localhost:8080
"""

import logging
import os
from collections.abc import Generator, Iterator
from pathlib import Path

import pytest
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.models.types import CreateModelProviderRequest
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nhx.testing.notebooks import cleanup_temp_venv_and_kernel, create_temp_venv_with_kernel
from pydantic import SecretStr

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SDK_PACKAGE = REPO_ROOT / "packages/nemo_helix"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("notebooks", "Documentation notebook testing options")
    group.addoption(
        "--notebook-language",
        choices=["all", "python", "shell"],
        default="python",
        help="Which notebook cells to execute: python (default), shell, or all",
    )
    group.addoption(
        "--notebook-kernel",
        default="sandbox",
        help=(
            'Jupyter kernel for notebooks. "sandbox" (default) creates an '
            "isolated venv with only the SDK installed. Any other value "
            "(e.g. python3) is used as-is."
        ),
    )


@pytest.fixture(scope="session")
def notebook_language(request: pytest.FixtureRequest) -> str:
    """Return the selected notebook language filter."""
    return request.config.getoption("--notebook-language")


@pytest.fixture(scope="session")
def notebook_kernel(request: pytest.FixtureRequest) -> Iterator[str]:
    """Provide a Jupyter kernel name for notebook execution.

    When ``--notebook-kernel sandbox`` (the default), creates an isolated
    virtualenv via :func:`nhx.testing.notebooks.create_temp_venv_with_kernel`
    containing only ``packages/nemo_helix/`` (editable with extras)
    and ``ipykernel``, then registers a temporary Jupyter kernel pointing
    at that venv.

    ``VIRTUAL_ENV`` and ``PATH`` are updated so that ``%%bash`` cells
    running ``pip install`` / ``uv pip install`` target the sandbox
    rather than the host environment.

    Pass any other value (e.g. ``--notebook-kernel python3``) to skip
    sandbox creation and use an existing kernel directly.
    """
    requested = request.config.getoption("--notebook-kernel")

    if requested != "sandbox":
        yield requested
        return

    sdk_spec = f"{SDK_PACKAGE}[nemo-safe-synthesizer-plugin,nemo-data-designer-plugin]"
    kernel_name, venv_dir, kernel_spec_dir = create_temp_venv_with_kernel(
        extra_pip_args=["pip", "-e", sdk_spec],
    )

    old_virtual_env = os.environ.get("VIRTUAL_ENV")
    old_path = os.environ.get("PATH", "")
    os.environ["VIRTUAL_ENV"] = venv_dir
    os.environ["PATH"] = f"{venv_dir}/bin:{old_path}"
    logger.info("Set VIRTUAL_ENV=%s for %%%%bash cell isolation", venv_dir)

    try:
        yield kernel_name
    finally:
        if old_virtual_env is None:
            os.environ.pop("VIRTUAL_ENV", None)
        else:
            os.environ["VIRTUAL_ENV"] = old_virtual_env
        os.environ["PATH"] = old_path
        cleanup_temp_venv_and_kernel(kernel_name, venv_dir, kernel_spec_dir)
        logger.info("Cleaned up sandbox kernel %r", kernel_name)


@pytest.fixture(scope="module", autouse=True)
def nvidia_build_provider(request: pytest.FixtureRequest, nhx_base_url: str) -> None:
    """Create the system/nvidia-build provider used by provider-backed docs notebooks.

    Quickstart does not run ``nemo setup``, so docs E2E must register the
    provider explicitly before notebooks call Data Designer or Inference
    Gateway examples that reference ``system/nvidia-build``.
    """
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_api_key:
        return

    client = request.getfixturevalue("client")
    workspace = "system"
    secret_name = "nvidia-build-api-key"
    provider_name = "nvidia-build"

    try:
        client.secrets.create_secret(
            workspace=workspace,
            body=HelixSecretCreateRequest(
                name=secret_name,
                value=SecretStr(nvidia_api_key),
                description="NVIDIA Build API key for docs notebook E2E",
            ),
        )
    except ConflictError:
        pass

    provider_names = {provider.name for provider in client.models.list_providers(workspace=workspace).items()}
    if provider_name in provider_names:
        return

    client.models.create_provider(
        workspace=workspace,
        body=CreateModelProviderRequest(
            name=provider_name,
            host_url="https://integrate.api.nvidia.com",
            api_key_secret_name=secret_name,
            description="NVIDIA Build provider for docs notebook E2E",
        ),
    )


@pytest.fixture(scope="session", autouse=True)
def guardrails_tutorial_assets() -> Generator[None, None, None]:
    """Set GUARDRAILS_TUTORIAL_ASSETS to the in-repo image directory for CI runs.

    Notebooks read this variable to locate tutorial assets (ex. images).
    When unset (the default for end-users), notebooks fall back to the current
    working directory, so users can simply place files in the same directory
    as the notebook.
    """
    if os.environ.get("GUARDRAILS_TUTORIAL_ASSETS"):
        yield
        return

    assets_path = str(REPO_ROOT / "docs/guardrails/_snippets/input")
    os.environ["GUARDRAILS_TUTORIAL_ASSETS"] = assets_path
    try:
        yield
    finally:
        os.environ.pop("GUARDRAILS_TUTORIAL_ASSETS", None)


@pytest.fixture(scope="module")
def nhx_base_url(request: pytest.FixtureRequest, cluster_url: str | None) -> Iterator[str]:
    """Derive the NHX base URL and set it as an environment variable.

    Notebooks read NHX_BASE_URL from the environment at runtime,
    so we set it here for the duration of the test module.
    """
    if cluster_url:
        url = cluster_url
    else:
        client = request.getfixturevalue("client")
        url = str(client.base_url)

    old = os.environ.get("NHX_BASE_URL")
    os.environ["NHX_BASE_URL"] = url
    yield url
    if old is None:
        os.environ.pop("NHX_BASE_URL", None)
    else:
        os.environ["NHX_BASE_URL"] = old
