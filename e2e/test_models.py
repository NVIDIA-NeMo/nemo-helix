# ty: ignore
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for models service (ModelDeployment lifecycle).

These tests verify the full deployment lifecycle including config creation,
deployment creation, status transitions, model autodiscovery, and inference
via the inference gateway. Requires GPU (use --docker --feature gpu with a GPU config, or --kubernetes --feature gpu).
"""

from __future__ import annotations

import dataclasses
import logging
import time

import openai
import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoTransportError, NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.storage_config import HuggingfaceStorageConfig
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    CreateModelEntityRequest,
    Engine,
    ModelDeploymentConfigModelSpec,
    ModelEntity,
    ModelProvider,
    ServedModelMapping,
)
from nhx.core.models.app import normalize_model_entity_name
from nhx.core.models.app.utils import _get_k8s_safe_name
from nhx.testing import wait_for_model_entity

logger = logging.getLogger(__name__)

# Per-test timeout (seconds). Keep below CI job timeout (1h) so pytest reports timeouts
# and JUnit is written before the runner kills the job; 30 min per test allows at least
# two tests to complete with full output.
_E2E_TEST_TIMEOUT = 1800

pytestmark = [
    pytest.mark.feature("gpu"),
    pytest.mark.slow,
    pytest.mark.timeout(_E2E_TEST_TIMEOUT),
]

# Timeouts (seconds)
_CLEANUP_WAIT_DELETED_TIMEOUT = 120  # Max wait for each deployment to reach DELETED during teardown
_SERVED_MODELS_TIMEOUT = 120  # Max wait for provider served_models to be populated by autodiscovery
_WAIT_PROVIDER_READY_TIMEOUT = 40  # Max wait for provider (and gateway) to reach READY before route checks
_WAIT_MODEL_ENTITY_TIMEOUT = 40  # Max wait for model entity to appear in IGW cache
_POLL_INTERVAL = 2.0  # Seconds between polls in _wait_for_served_models and wait_for_model_entity
_MIN_MODELS_POLL_TIMEOUT = 60  # Max wait for /v1/models to report at least assert_min_models (e.g. base + LoRA)
_DEPLOYMENT_READY_TIMEOUT = 900  # Max wait for deployment to reach READY (covers image pull and load)
# Inference route verification
_CHAT_MAX_TOKENS = 10
_CHAT_VERIFICATION_MESSAGE = {"role": "user", "content": "Say hello in one word."}


def _entity_name_from_id(model_entity_id: str) -> str:
    """Extract entity name from 'workspace/name' format (or return as-is if no slash)."""
    return model_entity_id.split("/", 1)[1] if "/" in model_entity_id else model_entity_id


def print_model_deployment_debug(
    workspace: str,
    deployment_name: str,
    *,
    namespace: str = "default",
) -> None:
    """Print k8s pod describe and logs for a model deployment before it is deleted.

    Used only in e2e/test_models.py to aid debugging when a deployment times out or
    fails. Uses the Kubernetes Python client to list pods by label (app.kubernetes.io/name
    = k8s-safe deployment name), then for each pod prints a describe-like summary and
    container logs. Safe to call when not on kubernetes or when kubeconfig is missing
    (no-op with a log message).
    """
    try:
        import kubernetes.config
        from kubernetes.client import CoreV1Api
        from kubernetes.client.rest import ApiException
    except ImportError:
        logger.debug("kubernetes client not available; skipping print_model_deployment_debug")
        return
    try:
        kubernetes.config.load_kube_config()
    except Exception as e:
        logger.debug("Could not load kube config (not on k8s?): %s", e)
        return
    k8s_name = _get_k8s_safe_name(deployment_name, max_length=63, name_type="label")
    v1 = CoreV1Api()
    try:
        ret = v1.list_namespaced_pod(
            namespace,
            label_selector=f"app.kubernetes.io/name={k8s_name}",
        )
        pods = list(ret.items)
        if not pods:
            # Fallback: pods may have name prefix from the deployment/statefulset
            ret = v1.list_namespaced_pod(namespace)
            pods = [p for p in ret.items if p.metadata.name.startswith(k8s_name)]
    except ApiException as e:
        logger.warning("Failed to list pods for deployment %s: %s", deployment_name, e)
        return
    if not pods:
        logger.info(
            "No k8s pods found for deployment %s (k8s name %s)",
            deployment_name,
            k8s_name,
        )
        return
    print(f"\n===== Model deployment debug: {workspace}/{deployment_name} (k8s name={k8s_name}) =====")
    for pod in pods:
        name = pod.metadata.name
        print(f"\n--- Pod: {namespace}/{name} ---")
        print("Metadata:")
        print(f"  namespace={pod.metadata.namespace}, name={pod.metadata.name}")
        print(f"  labels={pod.metadata.labels}")
        print("Status:")
        if pod.status:
            print(f"  phase={getattr(pod.status, 'phase', None)}")
            if pod.status.conditions:
                for c in pod.status.conditions:
                    print(f"  condition: {c.type}={c.status} ({c.reason or ''}) {c.message or ''}")
            if getattr(pod.status, "container_statuses", None):
                for cs in pod.status.container_statuses:
                    print(f"  container {cs.name}: ready={cs.ready} restart_count={cs.restart_count}")
        try:
            events = v1.list_namespaced_event(
                namespace,
                field_selector=f"involvedObject.name={name}",
            )
            if events.items:
                print("Events:")
                for ev in events.items[:15]:
                    print(f"  {ev.last_timestamp} {ev.type} {ev.reason}: {ev.message}")
        except ApiException:
            pass
        print("Logs:")
        containers = [c.name for c in (pod.spec.containers or [])]
        for container in containers:
            try:
                logs = v1.read_namespaced_pod_log(name, namespace, container=container, tail_lines=200)
                print(f"  --- container: {container} (tail 200) ---")
                print(logs if isinstance(logs, str) else logs.decode("utf-8", errors="replace"))
            except ApiException as e:
                print(f"  (logs for {container}: {e.status})")
    print(f"===== End model deployment debug: {workspace}/{deployment_name} =====\n")


@dataclasses.dataclass
class ModelDeploymentTestContext:
    """Accumulates state across the IGW readiness, assertion, and route-check phases."""

    workspace: str
    deployment_name: str
    provider: ModelProvider | None = None
    model_entity: ModelEntity | None = None
    assert_min_models: int | None = None


class ModelDeploymentCleanup:
    """Tracks model deployments, configs, model entities, and filesets for scoped teardown.

    Register resources after creation; cleanup() deletes deployments (wait DELETED),
    configs, model entities, and filesets. Prevents cross-test collisions.
    """

    def __init__(self, client: NemoClient) -> None:
        self._models = ModelsClient.from_client(client)
        self._files = FilesClient.from_client(client)
        self._deployments: list[tuple[str, str]] = []  # (workspace, deployment_name)
        self._configs: list[tuple[str, str]] = []  # (workspace, config_name)
        self._model_entities: list[tuple[str, str]] = []  # (workspace, model_entity_name)
        self._filesets: list[tuple[str, str]] = []  # (workspace, fileset_name)

    def register(self, workspace: str, deployment_name: str, config_name: str) -> None:
        """Register a deployment and its config for cleanup."""
        self._deployments.append((workspace, deployment_name))
        self._configs.append((workspace, config_name))

    def register_model_entity(self, workspace: str, model_entity_name: str) -> None:
        """Register a model entity for cleanup (e.g. created for HuggingFace tests)."""
        self._model_entities.append((workspace, model_entity_name))

    def register_fileset(self, workspace: str, fileset_name: str) -> None:
        """Register a fileset for cleanup (e.g. created for HuggingFace tests)."""
        self._filesets.append((workspace, fileset_name))

    def cleanup(self) -> None:
        """Delete deployments (wait DELETED), configs, model entities, and filesets."""
        for workspace, deployment_name in self._deployments:
            try:
                print_model_deployment_debug(workspace, deployment_name)
            except Exception:
                logger.warning(
                    "Cleanup debug (print_model_deployment_debug %s/%s)",
                    workspace,
                    deployment_name,
                    exc_info=True,
                )
        for workspace, deployment_name in self._deployments:
            try:
                self._models.delete_deployment(name=deployment_name, workspace=workspace)
            except NotFoundError:
                logger.debug(
                    "Cleanup: deployment %s/%s not found (skip delete)",
                    workspace,
                    deployment_name,
                )
            except Exception:
                logger.warning(
                    "Cleanup warning (delete %s/%s)",
                    workspace,
                    deployment_name,
                    exc_info=True,
                )
        for workspace, deployment_name in self._deployments:
            try:
                self._models.wait_for_deployment_status(
                    deployment_name,
                    "DELETED",
                    workspace=workspace,
                    timeout=_CLEANUP_WAIT_DELETED_TIMEOUT,
                )
            except NotFoundError:
                logger.debug(
                    "Cleanup: deployment %s/%s not found (skip wait DELETED)",
                    workspace,
                    deployment_name,
                )
            except Exception:
                logger.warning(
                    "Cleanup warning (wait DELETED %s/%s)",
                    workspace,
                    deployment_name,
                    exc_info=True,
                )

        seen_configs: set[tuple[str, str]] = set()
        for workspace, config_name in self._configs:
            key = (workspace, config_name)
            if key in seen_configs:
                continue
            seen_configs.add(key)
            try:
                self._models.delete_deployment_config(name=config_name, workspace=workspace)
            except NotFoundError:
                logger.debug(
                    "Cleanup: config %s/%s not found (skip delete)",
                    workspace,
                    config_name,
                )
            except Exception:
                logger.warning(
                    "Cleanup warning (config %s/%s)",
                    workspace,
                    config_name,
                    exc_info=True,
                )

        seen_entities: set[tuple[str, str]] = set()
        for workspace, model_entity_name in self._model_entities:
            key = (workspace, model_entity_name)
            if key in seen_entities:
                continue
            seen_entities.add(key)
            try:
                self._models.delete_model(name=model_entity_name, workspace=workspace)
            except NotFoundError:
                logger.debug(
                    "Cleanup: model entity %s/%s not found (skip delete)",
                    workspace,
                    model_entity_name,
                )
            except Exception:
                logger.warning(
                    "Cleanup warning (model entity %s/%s)",
                    workspace,
                    model_entity_name,
                    exc_info=True,
                )

        seen_filesets: set[tuple[str, str]] = set()
        for workspace, fileset_name in self._filesets:
            key = (workspace, fileset_name)
            if key in seen_filesets:
                continue
            seen_filesets.add(key)
            try:
                self._files.delete_fileset(name=fileset_name, workspace=workspace)
            except NotFoundError:
                logger.debug(
                    "Cleanup: fileset %s/%s not found (skip delete)",
                    workspace,
                    fileset_name,
                )
            except Exception:
                logger.warning(
                    "Cleanup warning (fileset %s/%s)",
                    workspace,
                    fileset_name,
                    exc_info=True,
                )


@pytest.fixture
def model_deployment_cleanup(client: NemoClient):
    """Fixture that yields a tracker for model deployment cleanup.

    Register deployments/configs with register(); optionally register_model_entity()
    and register_fileset() for tests that create those resources. Teardown runs
    tracker.cleanup() so only this test's resources are removed.
    """
    tracker = ModelDeploymentCleanup(client)
    yield tracker
    tracker.cleanup()


def _wait_for_served_models(
    client: NemoClient,
    provider_name: str,
    workspace: str,
    timeout: int = _SERVED_MODELS_TIMEOUT,
) -> None:
    """Poll until provider's served_models is populated by autodiscovery."""
    models = ModelsClient.from_client(client)
    start = time.time()
    while time.time() - start < timeout:
        try:
            provider = models.get_provider(name=provider_name, workspace=workspace).data()
            if provider.served_models and len(provider.served_models) >= 1:
                return
        except NotFoundError:
            pass
        time.sleep(_POLL_INTERVAL)
    pytest.fail(f"Timeout: served_models not populated within {timeout}s")


def _wait_for_gateway_ready(
    client: NemoClient,
    provider_name: str,
    workspace: str,
    timeout: float,
) -> bool:
    """Poll the inference gateway until it can route to *provider_name*."""
    gateway = InferenceGatewayClient.from_client(client)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            gateway.provider_ready(name=provider_name, workspace=workspace)
            return True
        except (NotFoundError, NemoTransportError):
            time.sleep(_POLL_INTERVAL)
    return False


def _assert_chat_route(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
) -> None:
    """Create an OpenAI client at base_url, call chat completions with model/messages, assert content."""
    client = openai.OpenAI(base_url=base_url, api_key="not-needed")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=_CHAT_MAX_TOKENS,
    )
    assert response.choices[0].message.content


def _populate_igw_context(
    client: NemoClient,
    ctx: ModelDeploymentTestContext,
    *,
    base_model_entity_name: str | None = None,
) -> ModelDeploymentTestContext:
    """Wait for provider READY, served_models autodiscovery, and model entity in IGW cache.

    Populates ctx.provider and ctx.model_entity. When base_model_entity_name is set, uses that
    entity for context (finds it in served_models; order of served_models is not guaranteed).
    """
    models = ModelsClient.from_client(client)
    logger.info("Waiting for provider READY (and gateway)...")
    if not models.wait_for_provider_status(
        ctx.deployment_name,
        "READY",
        workspace=ctx.workspace,
        timeout=_WAIT_PROVIDER_READY_TIMEOUT,
    ) or not _wait_for_gateway_ready(client, ctx.deployment_name, ctx.workspace, timeout=_WAIT_PROVIDER_READY_TIMEOUT):
        pytest.fail("Provider did not reach READY or gateway within timeout")

    logger.info("Waiting for served_models autodiscovery...")
    _wait_for_served_models(client, ctx.deployment_name, ctx.workspace, timeout=_SERVED_MODELS_TIMEOUT)

    ctx.provider = models.get_provider(name=ctx.deployment_name, workspace=ctx.workspace).data()
    served_models = ctx.provider.served_models
    assert served_models is not None and len(served_models) >= 1, "Expected at least 1 served model"

    # served_models order is not guaranteed (e.g. LoRA deployments may list adapter before base).
    # When we have a preferred entity (e.g. base model), look it up by model_entity_id instead of
    # using index 0 so we wait for and set ctx.model_entity to the right one.
    if base_model_entity_name:
        full_id = f"{ctx.workspace}/{base_model_entity_name}"
        sm = next((s for s in served_models if s.model_entity_id == full_id), None)
        entity_name = _entity_name_from_id(sm.model_entity_id) if sm else base_model_entity_name
    else:
        entity_name = _entity_name_from_id(served_models[0].model_entity_id)

    logger.info("Waiting for model entity in IGW cache...")
    try:
        wait_for_model_entity(
            client,
            ctx.workspace,
            entity_name,
            timeout=_WAIT_MODEL_ENTITY_TIMEOUT,
            poll_interval=_POLL_INTERVAL,
        )
    except TimeoutError as e:
        pytest.fail(f"Model entity route not available: {e}")

    ctx.model_entity = models.get_model(name=entity_name, workspace=ctx.workspace).data()
    return ctx


def _assert_provider_and_entity(
    ctx: ModelDeploymentTestContext,
    *,
    assert_served_model_name: str | None = None,
    assert_model_entity_name: str | None = None,
    assert_entity_linked_by_autodiscovery: bool = False,
) -> None:
    """Assert provider and model entity fields are correct.

    served_models order is not guaranteed; we find the matching entry by entity id or name.
    """
    logger.info("Verifying ModelProvider...")
    assert ctx.provider is not None and ctx.model_entity is not None, "IGW context is not populated"
    served_models = ctx.provider.served_models or []
    if assert_served_model_name is not None:
        has_served = any(sm.served_model_name == assert_served_model_name for sm in served_models)
        assert has_served, (
            f"Expected a served_model with served_model_name {assert_served_model_name!r}, "
            f"got: {[sm.served_model_name for sm in served_models]}"
        )
    if assert_model_entity_name is not None:
        full_id = f"{ctx.workspace}/{assert_model_entity_name}"
        has_entity = any(sm.model_entity_id == full_id for sm in served_models)
        assert has_entity, (
            f"Expected a served_model with model_entity_id for {assert_model_entity_name!r}, "
            f"got: {[sm.model_entity_id for sm in served_models]}"
        )

    logger.info("Verifying Model Entity...")
    assert ctx.model_entity.workspace == ctx.workspace
    if assert_model_entity_name is not None:
        assert ctx.model_entity.name == assert_model_entity_name
    if assert_entity_linked_by_autodiscovery:
        assert f"{ctx.workspace}/{ctx.deployment_name}" in (ctx.model_entity.model_providers or []), (
            f"Deployment {ctx.deployment_name} not in entity model_providers"
        )
        if ctx.model_entity.description is not None:
            assert "Auto-discovered" in ctx.model_entity.description


def _find_served_mapping_for_entity(
    provider: ModelProvider,
    workspace: str,
    model_entity_name: str,
) -> ServedModelMapping:
    """Return the served_models entry for the given model entity name or LoRA adapter name.

    For deployment-backed autodiscovery, LoRA entries use model_entity_id
    ``{base_id}&adapters/{adapter_workspace}/{adapter_name}`` (see provider reconciler); we match
    ``&adapters/{workspace}/{model_entity_name}`` when the name refers to an adapter.
    """
    full_id = f"{workspace}/{model_entity_name}"
    normalized_name = normalize_model_entity_name(model_entity_name)
    full_id_normalized = f"{workspace}/{normalized_name}"

    for sm in provider.served_models or []:
        if (
            sm.model_entity_id == full_id
            or sm.model_entity_id.endswith(f"/{model_entity_name}")
            or sm.served_model_name.endswith(f"/{model_entity_name}")
            or sm.served_model_name == model_entity_name
            or sm.model_entity_id == full_id_normalized
            or sm.model_entity_id.endswith(f"-{normalized_name}")
            or sm.model_entity_id.endswith(f"&adapters/{workspace}/{model_entity_name}")
        ):
            return sm

    raise ValueError(
        f"No served_model matching {model_entity_name!r} (normalized: {normalized_name!r}) in provider "
        f"(model_entity_ids: {[sm.model_entity_id for sm in provider.served_models or []]})"
    )


def _check_igw_routes(
    client: NemoClient,
    ctx: ModelDeploymentTestContext,
    test_messages: list[dict[str, str]],
    *,
    served_model_name: str | None = None,
    model_entity: ModelEntity | None = None,
) -> None:
    """Hit /v1/models, /v1/chat/completions, and all four OpenAI proxy routes.

    When served_model_name and model_entity are not provided, uses the first
    served model (index 0) and ctx.model_entity. Pass them to verify routes
    for a specific model in the list (e.g. LoRA adapter or prompt-tuned).
    """
    assert ctx.provider is not None and ctx.provider.served_models, "IGW context is not populated"
    models = ModelsClient.from_client(client)
    gateway = InferenceGatewayClient.from_client(client)
    if served_model_name is None:
        served_model_name = ctx.provider.served_models[0].served_model_name
    if model_entity is None:
        model_entity = ctx.model_entity

    logger.info("Testing /v1/models endpoint...")
    if ctx.assert_min_models is not None:
        # Poll for min models (e.g. base + LoRA) so LoRA sidecar has time to poll and load adapters.
        num_attempts = int(_MIN_MODELS_POLL_TIMEOUT / _POLL_INTERVAL)
        num_models = 0
        models_response = None
        for attempt in range(num_attempts):
            models_response = gateway.provider_get(
                trailing_uri="v1/models",
                workspace=ctx.workspace,
                name=ctx.deployment_name,
            ).data()
            if models_response and isinstance(models_response, dict) and "data" in models_response:
                num_models = len(models_response["data"])
                if num_models >= ctx.assert_min_models:
                    elapsed = (attempt + 1) * _POLL_INTERVAL
                    logger.info("%d model(s) (after %.0fs poll)", num_models, elapsed)
                    break
            if attempt < num_attempts - 1:
                time.sleep(_POLL_INTERVAL)
        else:
            # Poll exhausted; still below threshold.
            logger.info("%d model(s) after %.0fs poll", num_models, _MIN_MODELS_POLL_TIMEOUT)
            assert num_models >= ctx.assert_min_models, (
                f"Expected at least {ctx.assert_min_models} model(s) from /v1/models (e.g. base + LoRA), got {num_models}"
            )
    else:
        models_response = gateway.provider_get(
            trailing_uri="v1/models",
            workspace=ctx.workspace,
            name=ctx.deployment_name,
        ).data()
    assert models_response is not None
    if isinstance(models_response, dict) and "data" in models_response and ctx.assert_min_models is None:
        logger.info("%d model(s)", len(models_response["data"]))

    logger.info("Testing /v1/chat/completions endpoint...")
    chat_response = gateway.provider_post(
        trailing_uri="v1/chat/completions",
        workspace=ctx.workspace,
        name=ctx.deployment_name,
        body=JsonBody(
            {
                "model": served_model_name,
                "messages": [_CHAT_VERIFICATION_MESSAGE],
                "max_tokens": _CHAT_MAX_TOKENS,
            }
        ),
    ).data()
    assert chat_response is not None

    logger.info("get_provider_route_openai_url...")
    _assert_chat_route(
        models.get_provider_route_openai_url(ctx.provider),
        served_model_name,
        test_messages,
    )
    logger.info("get_provider_route_openai_url_for_deployment...")
    deployment = models.get_deployment(name=ctx.deployment_name, workspace=ctx.workspace).data()
    _assert_chat_route(
        models.get_provider_route_openai_url_for_deployment(deployment),
        served_model_name,
        test_messages,
    )
    # IGW has 3 route types: provider (served_model_name in body), model-entity (entity in path),
    # openai (workspace/model_entity_name in body). LoRA adapters have no ModelEntity (Adapter sub-entity only),
    # so model-entity URL is skipped when model_entity is None; OpenAI route with the fully qualified
    # model_entity_id (e.g. workspace/base&adapters/ws/adapter) is asserted in _verify_igw_inference_routes
    # for LoRA entries (see also_verify_routes_for_models).
    if model_entity is not None:
        logger.info("get_model_entity_route_openai_url...")
        _assert_chat_route(
            models.get_model_entity_route_openai_url(model_entity),
            served_model_name,
            test_messages,
        )
        logger.info("get_openai_client...")
        openai_base_url = models.get_openai_route_base_url(workspace=ctx.workspace)
        model_string = f"{model_entity.workspace}/{model_entity.name}"
        _assert_chat_route(
            openai_base_url,
            model_string,
            test_messages,
        )


def _verify_igw_inference_routes(
    client: NemoClient,
    workspace: str,
    deployment_name: str,
    *,
    assert_served_model_name: str | None = None,
    assert_model_entity_name: str | None = None,
    assert_entity_linked_by_autodiscovery: bool = False,
    assert_min_models: int | None = None,
    also_verify_routes_for_models: list[str] | None = None,
) -> None:
    """Verify ModelProvider, model entity, and all IGW proxy inference routes.

    Waits for provider READY (and gateway), served_models autodiscovery, and model entity
    in IGW cache, then runs all route checks. Optionally runs the same route checks for
    additional models (e.g. LoRA adapter, prompt-tuned) by matching model_entity_id in
    the provider's served_models list.

    Args:
        assert_served_model_name: If set, assert provider's first served_model_name equals this.
        assert_model_entity_name: If set, assert model_entity_id and model_entity.name equal this.
        assert_entity_linked_by_autodiscovery: If True, assert deployment is in the entity's
            model_providers and (when description is set) description contains "Auto-discovered".
        assert_min_models: If set, assert GET /v1/models returns at least this many models
            (e.g. 2 for base + LoRA adapter).
        also_verify_routes_for_models: If set, for each model entity name (or adapter name
            for LoRA, matching model_entity_id), look up served_model_name and run the same
            route checks (chat completions and all proxy routes).
    """
    ctx = ModelDeploymentTestContext(
        workspace=workspace,
        deployment_name=deployment_name,
        assert_min_models=assert_min_models,
    )
    ctx = _populate_igw_context(client, ctx, base_model_entity_name=assert_model_entity_name)
    _assert_provider_and_entity(
        ctx,
        assert_served_model_name=assert_served_model_name,
        assert_model_entity_name=assert_model_entity_name,
        assert_entity_linked_by_autodiscovery=assert_entity_linked_by_autodiscovery,
    )
    _check_igw_routes(client, ctx, test_messages=[_CHAT_VERIFICATION_MESSAGE])

    if also_verify_routes_for_models:
        assert ctx.provider is not None
        models = ModelsClient.from_client(client)
        ctx.assert_min_models = None  # Already asserted above; skip poll in subsequent route checks
        for model_entity_name in also_verify_routes_for_models:
            logger.info("Verifying IGW routes for model entity %s...", model_entity_name)
            served_mapping = _find_served_mapping_for_entity(ctx.provider, ctx.workspace, model_entity_name)
            served = served_mapping.served_model_name
            try:
                entity = models.get_model(name=model_entity_name, workspace=ctx.workspace).data()
            except NotFoundError:
                # LoRA adapters are on ModelEntity.adapters (Adapter.name), not standalone model entities.
                # We still verify provider routes using served_model_name; model-entity route is skipped.
                entity = None
            _check_igw_routes(
                client,
                ctx,
                test_messages=[_CHAT_VERIFICATION_MESSAGE],
                served_model_name=served,
                model_entity=entity,
            )
            if entity is not None:
                assert served_mapping.model_entity_id == f"{ctx.workspace}/{entity.name}", (
                    "served_models model_entity_id must match workspace/entity_name used for OpenAI routing "
                    f"(got {served_mapping.model_entity_id!r})"
                )
            # LoRA: IGW OpenAI proxy must accept the fully qualified model_entity_id for routing
            # (workspace/base&adapters/adapter-ws/adapter-name, split on first /).
            if entity is None and "&adapters/" in served_mapping.model_entity_id:
                logger.info(
                    "Verifying OpenAI route with LoRA fully qualified model_entity_id %s",
                    served_mapping.model_entity_id,
                )
                openai_base_url = models.get_openai_route_base_url(workspace=ctx.workspace)
                _assert_chat_route(
                    openai_base_url,
                    served_mapping.model_entity_id,
                    [_CHAT_VERIFICATION_MESSAGE],
                )
            # Prompt-tuned: OpenAI route with workspace/entity_name is already exercised in _check_igw_routes
            # when model_entity is set (same as other first-class model entities).


# This test requires a larger GPU allocation than we currently have available on Astra (@tmutch)
@pytest.mark.skip_on_astra
def test_model_deployment_lifecycle(
    client: NemoClient,
    workspace: str,
    model_deployment_cleanup: ModelDeploymentCleanup,
) -> None:
    """Test ModelDeployment lifecycle with GPU backend (Docker or K8s).

    Validates: config and deployment creation, status transitions to READY,
    model autodiscovery from NIM /v1/models, inference via IGW, and
    scoped cleanup (deployment then config).
    """
    md_config_name = "llama-config"
    md_name = "llama-deployment"

    models = ModelsClient.from_client(client)
    logger.info("Creating ModelDeploymentConfig")
    config = models.create_deployment_config(
        workspace=workspace,
        body=CreateModelDeploymentConfigRequest(
            name=md_config_name,
            description="E2E test configuration for LLaMA-3.2-1B-Instruct (GPU)",
            engine=Engine.NIM,
            model_spec=ModelDeploymentConfigModelSpec(model_name="meta/llama-3.2-1b-instruct"),
            executor_config=ContainerExecutorConfig(
                gpu=1,
                image_name="nvcr.io/nim/meta/llama-3.2-1b-instruct",
                image_tag="1.8.6",
            ),
        ),
    ).data()
    assert config.name == md_config_name
    assert config.workspace == workspace

    logger.info("Creating ModelDeployment...")
    deployment = models.create_deployment(
        workspace=workspace,
        body=CreateModelDeploymentRequest(name=md_name, config=md_config_name),
    ).data()
    assert deployment.name == md_name
    assert deployment.workspace == workspace
    assert deployment.id is not None

    model_deployment_cleanup.register(workspace, md_name, md_config_name)

    logger.info("Waiting for deployment READY (with gateway check)...")
    ready = models.wait_for_deployment_status(
        md_name,
        "READY",
        workspace=workspace,
        timeout=_DEPLOYMENT_READY_TIMEOUT,
    ) and _wait_for_gateway_ready(client, md_name, workspace, timeout=_WAIT_PROVIDER_READY_TIMEOUT)
    if not ready:
        pytest.fail("Deployment did not reach READY within timeout")

    _verify_igw_inference_routes(
        client,
        workspace,
        md_name,
        assert_served_model_name="meta/llama-3.2-1b-instruct",
        assert_model_entity_name="meta-llama-3-2-1b-instruct",
        assert_entity_linked_by_autodiscovery=True,
    )
    logger.info("Model deployment lifecycle test complete (cleanup via fixture)")


def test_model_deployment_huggingface_multi_llm(
    client: NemoClient,
    workspace: str,
    model_deployment_cleanup: ModelDeploymentCleanup,
) -> None:
    """Test HuggingFace model deployment using multi-LLM (default image, no image_name/image_tag).

    Creates a fileset pointing to the HuggingFace model at hf_repo_id, registers a model
    entity, and deploys using the default multi-LLM NIM image. Validates READY, autodiscovery,
    and inference via the gateway.
    """
    hf_repo_id = "Qwen/Qwen2.5-1.5B-Instruct"
    fileset_name = "qwen-2-5-1-5b"
    model_entity_name = "qwen-2-5-1-5b"
    md_config_name = "qwen-multillm-config"
    md_name = "qwen-multillm-deployment"

    models = ModelsClient.from_client(client)
    logger.info("Creating HuggingFace fileset and model entity")
    FilesClient.from_client(client).create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(
            name=fileset_name,
            description=f"{hf_repo_id} from HuggingFace (public)",
            storage=HuggingfaceStorageConfig(repo_id=hf_repo_id, repo_type="model"),
        ),
    )
    models.create_model(
        workspace=workspace,
        body=CreateModelEntityRequest(name=model_entity_name, fileset=f"{workspace}/{fileset_name}"),
    )
    model_deployment_cleanup.register_model_entity(workspace, model_entity_name)
    model_deployment_cleanup.register_fileset(workspace, fileset_name)

    logger.info("Creating ModelDeploymentConfig (multi-LLM: no image_name/image_tag)")
    config = models.create_deployment_config(
        workspace=workspace,
        body=CreateModelDeploymentConfigRequest(
            name=md_config_name,
            description="E2E HuggingFace + multi-LLM (default NIM image)",
            engine=Engine.NIM,
            model_spec=ModelDeploymentConfigModelSpec(model_namespace=workspace, model_name=model_entity_name),
            executor_config=ContainerExecutorConfig(gpu=1),
        ),
    ).data()
    assert config.name == md_config_name

    logger.info("Creating ModelDeployment...")
    deployment = models.create_deployment(
        workspace=workspace,
        body=CreateModelDeploymentRequest(name=md_name, config=md_config_name),
    ).data()
    assert deployment.name == md_name
    assert deployment.workspace == workspace
    assert deployment.id is not None

    model_deployment_cleanup.register(workspace, md_name, md_config_name)

    logger.info("Waiting for deployment READY (model pull + load may take several minutes)...")
    ready = models.wait_for_deployment_status(
        md_name,
        "READY",
        workspace=workspace,
        timeout=_DEPLOYMENT_READY_TIMEOUT,
    ) and _wait_for_gateway_ready(client, md_name, workspace, timeout=_WAIT_PROVIDER_READY_TIMEOUT)
    if not ready:
        pytest.fail("Deployment did not reach READY within timeout")

    _verify_igw_inference_routes(client, workspace, md_name, assert_model_entity_name=model_entity_name)
    logger.info("HuggingFace multi-LLM deployment test complete (cleanup via fixture)")


@pytest.mark.skip(
    reason=(
        "Deferred during plugin migration: Automodel does not currently expose deployment_config, "
        "so the old flat customizer auto-deploy flow cannot be submitted."
    )
)
def test_model_deployment_lora_and_prompt_tuning() -> None:
    """Deferred until plugin-backed training and deployment ownership is settled."""
