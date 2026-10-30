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

import httpx
import pytest
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.models.types import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    CreateModelEntityRequest,
    Engine,
    ModelDeploymentConfigModelSpec,
    UpsertModelProviderRequest,
)
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest, HelixSecretUpdateRequest
from nemo_helix_plugin.virtual_models.client import VirtualModelsClient
from nhx.testing.notebooks import (
    cleanup_notebook_deployments,
    cleanup_temp_venv_and_kernel,
    create_temp_venv_with_kernel,
)
from nhx.testing.utils import ensure_passthrough_virtual_model, wait_for_model_entity
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


def _nvidia_base_url() -> str:
    """Use an OpenAI-compatible API base, including /v1, for provider-backed docs.

    NVIDIA_BASE_URL overrides the public Build endpoint for local/CI testing.
    The key must be authorized for that endpoint and the tutorial's model IDs.
    """
    return os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")


def _probe_nvidia_build_credentials(api_key: str) -> None:
    """Diagnose CI credentials directly, without changing notebook outcomes."""
    try:
        response = httpx.post(
            f"{_nvidia_base_url()}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"),
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
                "stream": False,
            },
            timeout=30,
        )
    except httpx.HTTPError as exc:
        # Exception messages can contain sensitive request information.
        logger.warning("NVIDIA credential probe (direct, bypasses platform): error_type=%s", type(exc).__name__)
    else:
        # Never log the response body, request headers, or credential.
        logger.info("NVIDIA credential probe (direct, bypasses platform): status=%s", response.status_code)


@pytest.fixture(scope="module", autouse=True)
def nvidia_build_provider(request: pytest.FixtureRequest, nhx_base_url: str, cluster_url: str | None) -> None:
    """Create the default/nvidia-build provider used by provider-backed docs notebooks.

    Quickstart does not run ``nemo setup``, so docs E2E must register the
    provider explicitly before notebooks call Data Designer or Inference
    Gateway examples that reference ``default/nvidia-build``.
    """
    # Existing external providers and credentials belong to their operator.
    if cluster_url:
        return
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_api_key:
        return

    client = request.getfixturevalue("client")
    workspace = "default"
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
        client.secrets.update_secret(
            workspace=workspace,
            name=secret_name,
            body=HelixSecretUpdateRequest(value=SecretStr(nvidia_api_key)),
        )

    stored_value = client.secrets.access_secret(workspace=workspace, name=secret_name).data().value
    if stored_value != nvidia_api_key:
        # Do not use an assertion: pytest introspection can expose credentials.
        raise RuntimeError("Docs NVIDIA provider secret did not round-trip correctly")

    _probe_nvidia_build_credentials(nvidia_api_key)

    # Quickstart may already have seeded this provider with a different secret.
    # Bind it to the refreshed fixture-owned key, not the stale seed reference.
    client.models.upsert_provider(
        workspace=workspace,
        name=provider_name,
        body=UpsertModelProviderRequest(
            host_url=_nvidia_base_url(),
            api_key_secret_name=secret_name,
            description="NVIDIA Build provider for docs notebook E2E",
        ),
    )


@pytest.fixture
def inference_tutorial_model(
    request: pytest.FixtureRequest, nhx_base_url: str, cluster_url: str | None
) -> Iterator[None]:
    """Provide the deployed model explicitly required by the inference tutorial."""
    if cluster_url:
        # An external cluster's operator owns its model prerequisites and lifecycle.
        yield
        return
    client = request.getfixturevalue("client")
    models = client.models
    workspace = "default"
    config_name = "docs-inference-llama-config"
    deployment_name = "llama-3-2-1b-deployment"
    model_name = "meta-llama-3-2-1b-instruct"
    models.create_deployment_config(
        workspace=workspace,
        body=CreateModelDeploymentConfigRequest(
            name=config_name,
            engine=Engine.NIM,
            # This is the served platform ID, not the vendor artifact path. NIM's
            # image selects the weights; the compiler uses this ID to name the
            # endpoint and the provider reconciler uses it as the routing key.
            model_spec=ModelDeploymentConfigModelSpec(model_namespace=workspace, model_name=model_name),
            executor_config=ContainerExecutorConfig(
                gpu=1, image_name="nvcr.io/nim/meta/llama-3.2-1b-instruct", image_tag="1.8.6"
            ),
        ),
    )
    try:
        models.create_deployment(
            workspace=workspace, body=CreateModelDeploymentRequest(name=deployment_name, config=config_name)
        )
        try:
            if not models.wait_for_deployment_status(deployment_name, "READY", workspace=workspace, timeout=1800):
                raise TimeoutError("Inference tutorial model did not become ready")
            if not models.wait_for_provider_status(deployment_name, "READY", workspace=workspace, timeout=300):
                raise TimeoutError("Inference tutorial provider did not become ready")
            # Deployment-backed discovery publishes served-model mappings but
            # intentionally does not create Model Entities. A VirtualModel alone
            # therefore advertises an OpenAI ID whose model-entity route 404s.
            model_created = False
            try:
                models.create_model(
                    workspace=workspace,
                    body=CreateModelEntityRequest(
                        name=model_name,
                        model_providers=[f"{workspace}/{deployment_name}"],
                        description="Model Entity for the docs inference deployment",
                    ),
                )
                model_created = True
            except ConflictError:
                # A pre-existing entity is not fixture-owned; never delete it.
                pass
            try:
                ensure_passthrough_virtual_model(client, workspace, model_name, timeout=120, autoprovisioned=False)
                # Check the actual model-entity proxy route, not only VM existence.
                wait_for_model_entity(client, workspace, model_name, timeout=120)
                yield
            finally:
                try:
                    VirtualModelsClient.from_client(client).delete_virtual_model(name=model_name, workspace=workspace)
                finally:
                    if model_created:
                        models.delete_model(name=model_name, workspace=workspace)
        finally:
            models.delete_deployment(name=deployment_name, workspace=workspace)
            if not models.wait_for_deployment_status(deployment_name, "DELETED", workspace=workspace, timeout=600):
                raise TimeoutError("Inference tutorial deployment was not deleted")
    finally:
        models.delete_deployment_config(name=config_name, workspace=workspace)


@pytest.fixture(scope="module")
def anonymizer_tutorial_inputs(request: pytest.FixtureRequest, nhx_base_url: str, cluster_url: str | None) -> None:
    """Seed shared fictional input only on the isolated notebook test backend."""
    if cluster_url:
        return
    client = request.getfixturevalue("client")
    client.files.create_fileset(
        workspace="default",
        body=CreateFilesetRequest(name="anonymizer-inputs", description="Fictional tutorial records"),
        exist_ok=True,
    )
    client.files.upload_file(
        workspace="default",
        name="anonymizer-inputs",
        path="anonymizer-input.csv",
        content=(Path(__file__).parent / "assets/anonymizer-input.csv").read_bytes(),
    )


@pytest.fixture(autouse=True)
def notebook_prerequisites(request: pytest.FixtureRequest) -> None:
    callspec = getattr(request.node, "callspec", None)
    path = callspec.params.get("notebook_path") if callspec else None
    if path is not None and Path(path).name == "run-inference.mdx":
        request.getfixturevalue("inference_tutorial_model")
    elif path is not None and "anonymizer" in Path(path).parts:
        request.getfixturevalue("anonymizer_tutorial_inputs")


@pytest.fixture(autouse=True)
def customizer_deployment_cleanup(
    request: pytest.FixtureRequest, nhx_base_url: str, cluster_url: str | None
) -> Iterator[None]:
    """Do not let one GPU tutorial's inference server starve subsequent training.

    External clusters may have concurrent users; only clean the isolated backend
    managed by this test run. Skips do not instantiate the fixture.
    """
    callspec = getattr(request.node, "callspec", None)
    path = callspec.params.get("notebook_path") if callspec else None
    if cluster_url or path is None or "customizer" not in Path(path).parts:
        yield
        return
    client = request.getfixturevalue("client")
    with cleanup_notebook_deployments(client):
        yield


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
