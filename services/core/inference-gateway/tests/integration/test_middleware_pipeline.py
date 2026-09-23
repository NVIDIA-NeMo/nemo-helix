# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the IGW middleware pipeline.

Verifies that ``NemoInferenceMiddleware`` plugins are invoked during request
processing, can mutate the request body, mutate the response body, and
short-circuit the backend proxy via ``ImmediateResponse``.

Four test scenarios:

1. **ImmediateResponse** (no Docker) — request middleware returns a canned
   payload without proxying; response middleware adds a marker.

2. **OpenAI endpoint: request + response mutation** (Docker) — request middleware
   rewrites ``body["model"]`` so IGW can route to the live mock-NIM via the
   OpenAI-compatible endpoint; response middleware stamps the response.

3. **Model endpoint: request + response mutation** (Docker) — same as #2 but
   exercising the ``/v2/workspaces/{ws}/model/{name}/-/...`` path.

4. **Model endpoint: non-model body mutation** (Docker) — regression test for
   the double-body-read bug.  Request middleware adds a non-``model`` field to
   the body; response middleware echoes it back so we can confirm it was
   forwarded to the backend rather than dropped.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from nemo_deployments_plugin.backends.labels import container_name as plugin_container_name
from nemo_deployments_plugin.backends.labels import docker_volume_name
from nemo_helix import NeMoHelix, NotFoundError
from nemo_helix.types.inference.virtual_model import VirtualModel as SDKVirtualModel
from nemo_helix_plugin.inference_middleware import (
    ImmediateResponse,
    InferenceMiddlewareContext,
    InferenceRequest,
    InferenceResponse,
    NemoInferenceMiddleware,
)
from nhx.core.inference_gateway.api.dependencies import (
    global_middleware_registry,
    global_virtual_model_cache,
)
from nhx.core.inference_gateway.api.middleware_registry import (
    MiddlewareRegistry,
    ResolvedMiddlewareCall,
)
from nhx.core.inference_gateway.api.model_cache import ModelProviderInfo
from nhx.core.inference_gateway.api.virtual_model_cache import VirtualModelCache
from nhx.core.models.controllers.backends.deployments_plugin.naming import entity_names
from nhx.core.models.controllers.models_controller import ModelsController
from tenacity import retry, stop_after_delay, wait_fixed

DEFAULT_WORKSPACE = "default"


def _wait_for_deployment_deleted(
    controller: ModelsController,
    sdk: NeMoHelix,
    deployment_name: str,
    max_wait: float = 30,
    poll_interval: float = 0.1,
) -> None:
    """Step the controller until a deployment reaches DELETED (or is gone).

    The deployments_plugin delete path is non-blocking, so a single
    ``controller.step()`` is not enough to drive a deployment to DELETED: the
    reconciler must run several times (DELETING -> backend teardown -> DELETED).
    Deleting the ModelDeploymentConfig before its deployment is DELETED fails
    with a 409 Conflict.
    """

    @retry(stop=stop_after_delay(max_wait), wait=wait_fixed(poll_interval), reraise=True)
    def _poll() -> None:
        controller.step()
        try:
            deployment = sdk.inference.deployments.retrieve(
                deployment_name,
                workspace=DEFAULT_WORKSPACE,
            )
        except NotFoundError:
            return
        assert deployment.status == "DELETED", f"Deployment not DELETED: {deployment.status}"

    _poll()


# Sentinel stamped on the response by response middleware so tests can assert
# the pipeline ran end-to-end.
RESPONSE_MIDDLEWARE_MARKER = "middleware-response-applied"


# ---------------------------------------------------------------------------
# Test middleware implementations
# ---------------------------------------------------------------------------


class ImmediateResponseMiddleware(NemoInferenceMiddleware):
    """Request middleware that short-circuits the backend with a canned response.

    The canned data is then passed to ``process_response`` so that a separate
    response middleware plugin can still observe and mutate it.
    """

    REQUEST_MARKER = "middleware-immediate-response"

    async def on_startup(self) -> None:
        pass

    async def on_shutdown(self) -> None:
        pass

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: Any,
    ) -> InferenceRequest | ImmediateResponse:
        return ImmediateResponse(
            data={
                "id": self.REQUEST_MARKER,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "from middleware"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: Any,
    ) -> InferenceResponse:
        return response


class ResponseMarkerMiddleware(NemoInferenceMiddleware):
    """Response middleware that stamps a sentinel key onto any dict response.

    Used alongside other request middleware to verify that both the request
    and response phases of the pipeline were executed.
    """

    async def on_startup(self) -> None:
        pass

    async def on_shutdown(self) -> None:
        pass

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: Any,
    ) -> InferenceRequest | ImmediateResponse:
        # Pass-through — this plugin only acts on responses.
        return request

    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: Any,
    ) -> InferenceResponse:
        if isinstance(response.result, dict):
            return InferenceResponse(
                result={**response.result, RESPONSE_MIDDLEWARE_MARKER: True},
                headers=response.headers,
            )
        return response


class RequestBodyEchoMiddleware(NemoInferenceMiddleware):
    """Response middleware that echoes a request-body field into the response.

    Used in regression tests for the double-body-read bug: if the mutation
    stamped by a *prior* request middleware survived to ``process_request``
    output, this middleware will find it in ``request_body`` and reflect it
    in the response under ``ECHO_KEY``.  If the mutation was lost (dropped by
    re-reading raw bytes), ``request_body[SOURCE_KEY]`` will be absent and the
    echoed value will be ``None``.

    Important: ``request_body`` passed to ``process_response`` is the
    ``json_body`` local variable in the proxy handler — that is always the
    middleware-mutated dict regardless of the bug.  The echo therefore does
    **not** expose the bug directly; the bug is exposed by the unit test that
    inspects what ``mock_proxy_client`` actually received.  This class exists
    to make the integration test self-documenting and to anchor future
    assertions if the NIM ever gains body-echo support.
    """

    SOURCE_KEY = "x_custom_middleware_field"
    ECHO_KEY = "x_custom_middleware_field_echoed"

    async def on_startup(self) -> None:
        pass

    async def on_shutdown(self) -> None:
        pass

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: Any,
    ) -> InferenceRequest | ImmediateResponse:
        return request

    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: Any,
    ) -> InferenceResponse:
        if isinstance(response.result, dict):
            # ctx.proxied_request carries the middleware-mutated body that was
            # forwarded to the backend — SOURCE_KEY is present there.
            source_value = ctx.proxied_request.body.get(self.SOURCE_KEY) if ctx.proxied_request else None
            return InferenceResponse(
                result={**response.result, self.ECHO_KEY: source_value},
                headers=response.headers,
            )
        return response


class ModelRouterMiddleware(NemoInferenceMiddleware):
    """Request middleware that rewrites ``body["model"]`` to a real model-entity ID.

    Simulates a routing plugin (e.g. nemo-switchyard) that resolves an
    opaque VirtualModel alias to the underlying model entity so that IGW can
    look up a provider.  Also stamps the original alias onto the request body
    so tests can verify the mutation occurred before proxying.
    """

    REQUEST_MUTATION_KEY = "x_original_model"

    def __init__(self, target_model_entity_id: str) -> None:
        super().__init__()
        self._target = target_model_entity_id

    async def on_startup(self) -> None:
        pass

    async def on_shutdown(self) -> None:
        pass

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: Any,
    ) -> InferenceRequest | ImmediateResponse:
        # Record the original alias for assertion, then rewrite model.
        return InferenceRequest(
            body={
                **request.body,
                self.REQUEST_MUTATION_KEY: request.body.get("model"),
                "model": self._target,
            },
            headers=request.headers,
            path=request.path,
        )

    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: Any,
    ) -> InferenceResponse:
        return response


# ---------------------------------------------------------------------------
# Registry injection helpers
# ---------------------------------------------------------------------------


def _make_call(plugin_key: str) -> ResolvedMiddlewareCall:
    return ResolvedMiddlewareCall(plugin_name=plugin_key, config_type="test_config", resolved_config={})


def _inject_vm_and_plugins(
    registry: MiddlewareRegistry,
    vm_cache: VirtualModelCache,
    workspace: str,
    vm_name: str,
    *,
    request_plugins: list[tuple[str, NemoInferenceMiddleware]],
    response_plugins: list[tuple[str, NemoInferenceMiddleware]],
    default_model_entity: str | None = None,
) -> None:
    """Register plugins and wire them into the registry + cache for one VirtualModel."""
    all_keys: list[str] = []

    for key, plugin in request_plugins + response_plugins:
        registry.plugins[key] = plugin
        all_keys.append(key)

    registry.request_middleware_calls[(workspace, vm_name)] = [_make_call(k) for k, _ in request_plugins]
    registry.response_middleware_calls[(workspace, vm_name)] = [_make_call(k) for k, _ in response_plugins]
    registry.post_response_middleware_calls[(workspace, vm_name)] = []

    vm = SDKVirtualModel(
        id=f"{workspace}/{vm_name}",
        entity_id=f"{workspace}/{vm_name}",
        name=vm_name,
        workspace=workspace,
        parent=workspace,
        db_version=1,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        default_model_entity=default_model_entity,
    )
    existing = list(vm_cache.virtual_model_map.values())
    vm_cache.rebuild([*existing, vm])


def _cleanup(
    registry: MiddlewareRegistry,
    vm_cache: VirtualModelCache,
    workspace: str,
    vm_name: str,
    plugin_keys: list[str],
) -> None:
    for key in plugin_keys:
        registry.plugins.pop(key, None)
    registry.request_middleware_calls.pop((workspace, vm_name), None)
    registry.response_middleware_calls.pop((workspace, vm_name), None)
    registry.post_response_middleware_calls.pop((workspace, vm_name), None)
    remaining = [
        vm for vm in vm_cache.virtual_model_map.values() if not (vm.workspace == workspace and vm.name == vm_name)
    ]
    vm_cache.rebuild(remaining)


# ---------------------------------------------------------------------------
# Test 1: ImmediateResponse + response mutation (no Docker)
# ---------------------------------------------------------------------------


def test_middleware_immediate_response_and_response_mutation(test_clients):
    """Request middleware short-circuits the proxy; response middleware mutates the result.

    Verifies two things in one round-trip:
    - ``ImmediateResponseMiddleware.process_request`` returns a canned payload
      (no backend call is made).
    - ``ResponseMarkerMiddleware.process_response`` stamps the sentinel marker
      onto that payload before it reaches the caller.
    """
    test_uuid = uuid.uuid4().hex[:8]
    vm_name = f"test-vm-imm-{test_uuid}"
    req_key = f"req-imm-{test_uuid}"
    resp_key = f"resp-marker-{test_uuid}"

    registry = global_middleware_registry()
    vm_cache = global_virtual_model_cache()

    _inject_vm_and_plugins(
        registry,
        vm_cache,
        DEFAULT_WORKSPACE,
        vm_name,
        request_plugins=[(req_key, ImmediateResponseMiddleware())],
        response_plugins=[(resp_key, ResponseMarkerMiddleware())],
        default_model_entity=None,
    )

    try:
        response = test_clients.sdk.inference.gateway.openai.post(
            "v1/chat/completions",
            workspace=DEFAULT_WORKSPACE,
            body={
                "model": vm_name,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        # Request middleware ran: response has the ImmediateResponse canned id
        assert response.get("id") == ImmediateResponseMiddleware.REQUEST_MARKER, (
            f"Expected request middleware marker in response id: {response}"
        )
        assert "choices" in response, f"Expected 'choices' in response: {response}"

        # Response middleware ran: sentinel key was stamped onto the payload
        assert response.get(RESPONSE_MIDDLEWARE_MARKER) is True, (
            f"Expected response middleware marker in response: {response}"
        )
    finally:
        _cleanup(registry, vm_cache, DEFAULT_WORKSPACE, vm_name, [req_key, resp_key])


# ---------------------------------------------------------------------------
# Test 2: Request mutation + response mutation through live mock-NIM (Docker)
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_middleware_request_and_response_mutation_through_backend(
    controller_with_docker_and_igw,
    docker_client,
):
    """Request middleware rewrites body['model']; response middleware stamps the response.

    Flow:
    1. Deploy the mock-NIM container via Docker.
    2. VirtualModel alias has no ``default_model_entity`` — the router plugin
       must supply the real model-entity ID.
    3. ``ModelRouterMiddleware.process_request`` rewrites ``body["model"]`` from
       the VM alias to the real entity ID; without this the request would 422.
    4. The request reaches the live mock-NIM and a real response is returned.
    5. ``ResponseMarkerMiddleware.process_response`` adds the sentinel key to
       the response dict.
    6. Assertions verify both mutations: a valid NIM response AND the sentinel.
    """
    controller, model_cache, sdk, mock_nim_image, ctx, _ = controller_with_docker_and_igw
    test_uuid = uuid.uuid4().hex[:8]

    config_name = f"test-mw-{test_uuid}"
    deployment_name = f"test-mw-{test_uuid}"
    model_entity_name = f"test-mw-model-{test_uuid}"
    served_model_name = "mock-model"
    vm_name = f"test-mw-alias-{test_uuid}"
    router_key = f"test-router-{test_uuid}"
    marker_key = f"test-marker-{test_uuid}"
    names = entity_names(deployment_name)
    container_name = plugin_container_name(DEFAULT_WORKSPACE, names.server)

    ctx.register_container(container_name)
    ctx.register_volume(docker_volume_name(DEFAULT_WORKSPACE, names.volume))
    ctx.register_volume(docker_volume_name(DEFAULT_WORKSPACE, names.scratch))

    # ---- Phase 1: Deploy mock NIM ----------------------------------------
    image_name, image_tag = mock_nim_image.rsplit(":", 1)
    sdk.inference.deployment_configs.create(
        workspace=DEFAULT_WORKSPACE,
        name=config_name,
        engine="nim",
        model_spec={},
        executor_config={"gpu": 0, "image_name": image_name, "image_tag": image_tag},
    )
    sdk.inference.deployments.create(
        workspace=DEFAULT_WORKSPACE,
        name=deployment_name,
        config=config_name,
    )

    @retry(stop=stop_after_delay(30), wait=wait_fixed(0.1), reraise=True)
    def _wait_ready():
        controller.step()
        dep = sdk.inference.deployments.retrieve(deployment_name, workspace=DEFAULT_WORKSPACE)
        assert dep.status == "READY", f"Not READY: {dep.status}"

    _wait_ready()

    # ---- Phase 2: Wire up served_models and populate the model cache ------
    sdk.inference.providers.update_status(
        deployment_name,
        workspace=DEFAULT_WORKSPACE,
        served_models=[
            {
                "model_entity_id": f"{DEFAULT_WORKSPACE}/{model_entity_name}",
                "served_model_name": served_model_name,
            }
        ],
    )

    provider = sdk.inference.providers.retrieve(deployment_name, workspace=DEFAULT_WORKSPACE)
    model_cache.workspace_name_provider_map[(DEFAULT_WORKSPACE, deployment_name)] = ModelProviderInfo(
        model_provider=provider
    )
    model_cache.rebuild_model_entity_map()

    # ---- Phase 3: Register middleware and VirtualModel --------------------
    registry = global_middleware_registry()
    vm_cache = global_virtual_model_cache()

    _inject_vm_and_plugins(
        registry,
        vm_cache,
        DEFAULT_WORKSPACE,
        vm_name,
        request_plugins=[
            (router_key, ModelRouterMiddleware(f"{DEFAULT_WORKSPACE}/{model_entity_name}")),
        ],
        response_plugins=[
            (marker_key, ResponseMarkerMiddleware()),
        ],
        default_model_entity=None,  # router middleware supplies the real entity ID
    )

    try:
        # ---- Phase 4: Make inference request via VirtualModel alias --------
        # Without the request middleware this would 422 because vm_name has
        # no default_model_entity and is not itself a model entity.
        response = sdk.inference.gateway.openai.post(
            "v1/chat/completions",
            workspace=DEFAULT_WORKSPACE,
            body={
                "model": vm_name,
                "messages": [{"role": "user", "content": "hello from middleware test"}],
            },
        )

        # Request mutation: routing succeeded → mock-NIM returned a valid chat response
        assert "choices" in response or "message" in response, (
            f"Expected valid chat response from mock-NIM, got: {response}"
        )

        # Response mutation: sentinel key was added by ResponseMarkerMiddleware
        assert response.get(RESPONSE_MIDDLEWARE_MARKER) is True, (
            f"Expected response middleware marker '{RESPONSE_MIDDLEWARE_MARKER}' in response: {response}"
        )

    finally:
        _cleanup(registry, vm_cache, DEFAULT_WORKSPACE, vm_name, [router_key, marker_key])

        # ---- Phase 5: Cleanup --------------------------------------------
        sdk.inference.deployments.delete(deployment_name, workspace=DEFAULT_WORKSPACE)
        _wait_for_deployment_deleted(controller, sdk, deployment_name)
        sdk.inference.deployment_configs.delete(config_name, workspace=DEFAULT_WORKSPACE)


# ---------------------------------------------------------------------------
# Test 3: Model endpoint — request + response mutation (Docker)
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_model_endpoint_request_and_response_mutation_through_backend(
    controller_with_docker_and_igw,
    docker_client,
):
    """Same as test_middleware_request_and_response_mutation_through_backend but
    exercising the ``/v2/workspaces/{ws}/model/{name}/-/...`` route instead of
    the OpenAI-compatible route.

    Ensures the model entity proxy path executes the full middleware pipeline.
    """
    controller, model_cache, sdk, mock_nim_image, ctx, _ = controller_with_docker_and_igw
    test_uuid = uuid.uuid4().hex[:8]

    config_name = f"test-mep-{test_uuid}"
    deployment_name = f"test-mep-{test_uuid}"
    model_entity_name = f"test-mep-model-{test_uuid}"
    served_model_name = "mock-model"
    vm_name = f"test-mep-alias-{test_uuid}"
    router_key = f"test-mep-router-{test_uuid}"
    marker_key = f"test-mep-marker-{test_uuid}"
    names = entity_names(deployment_name)
    container_name = plugin_container_name(DEFAULT_WORKSPACE, names.server)

    ctx.register_container(container_name)
    ctx.register_volume(docker_volume_name(DEFAULT_WORKSPACE, names.volume))
    ctx.register_volume(docker_volume_name(DEFAULT_WORKSPACE, names.scratch))

    # ---- Phase 1: Deploy mock NIM ----------------------------------------
    image_name, image_tag = mock_nim_image.rsplit(":", 1)
    sdk.inference.deployment_configs.create(
        workspace=DEFAULT_WORKSPACE,
        name=config_name,
        engine="nim",
        model_spec={},
        executor_config={"gpu": 0, "image_name": image_name, "image_tag": image_tag},
    )
    sdk.inference.deployments.create(workspace=DEFAULT_WORKSPACE, name=deployment_name, config=config_name)

    @retry(stop=stop_after_delay(30), wait=wait_fixed(0.1), reraise=True)
    def _wait_ready():
        controller.step()
        dep = sdk.inference.deployments.retrieve(deployment_name, workspace=DEFAULT_WORKSPACE)
        assert dep.status == "READY", f"Not READY: {dep.status}"

    _wait_ready()

    # ---- Phase 2: Wire up served_models and populate the model cache ------
    sdk.inference.providers.update_status(
        deployment_name,
        workspace=DEFAULT_WORKSPACE,
        served_models=[
            {
                "model_entity_id": f"{DEFAULT_WORKSPACE}/{model_entity_name}",
                "served_model_name": served_model_name,
            }
        ],
    )
    provider = sdk.inference.providers.retrieve(deployment_name, workspace=DEFAULT_WORKSPACE)
    model_cache.workspace_name_provider_map[(DEFAULT_WORKSPACE, deployment_name)] = ModelProviderInfo(
        model_provider=provider
    )
    model_cache.rebuild_model_entity_map()

    # ---- Phase 3: Register middleware and VirtualModel --------------------
    registry = global_middleware_registry()
    vm_cache = global_virtual_model_cache()

    _inject_vm_and_plugins(
        registry,
        vm_cache,
        DEFAULT_WORKSPACE,
        vm_name,
        request_plugins=[
            (router_key, ModelRouterMiddleware(f"{DEFAULT_WORKSPACE}/{model_entity_name}")),
        ],
        response_plugins=[(marker_key, ResponseMarkerMiddleware())],
        default_model_entity=None,
    )

    try:
        # ---- Phase 4: Make inference request via model endpoint -----------
        response = sdk.inference.gateway.model.post(
            "v1/chat/completions",
            name=vm_name,
            workspace=DEFAULT_WORKSPACE,
            body={
                "model": vm_name,
                "messages": [{"role": "user", "content": "hello from model endpoint test"}],
            },
        )

        assert "choices" in response or "message" in response, (
            f"Expected valid chat response from mock-NIM, got: {response}"
        )
        assert response.get(RESPONSE_MIDDLEWARE_MARKER) is True, (
            f"Expected response middleware marker in response: {response}"
        )

    finally:
        _cleanup(registry, vm_cache, DEFAULT_WORKSPACE, vm_name, [router_key, marker_key])
        sdk.inference.deployments.delete(deployment_name, workspace=DEFAULT_WORKSPACE)
        _wait_for_deployment_deleted(controller, sdk, deployment_name)
        sdk.inference.deployment_configs.delete(config_name, workspace=DEFAULT_WORKSPACE)


# ---------------------------------------------------------------------------
# Test 4: Model endpoint — non-model body mutation regression (Docker)
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_model_endpoint_non_model_body_mutation_regression(
    controller_with_docker_and_igw,
    docker_client,
):
    """Regression test for the double-body-read bug on the model entity proxy path.

    ``_update_request_with_served_name`` previously re-read the body from raw
    request bytes, discarding mutations added by request middleware to fields
    other than ``model``.  The fix passes ``body=json_body`` directly to
    ``build_next_request``.

    Note: the mock-NIM does not echo request body fields, so we cannot assert
    the extra field reached the NIM directly.  The definitive assertion is in
    ``test_model_entity_proxy_middleware_body_mutations_reach_backend`` (unit
    test) which inspects the raw bytes sent to the mock HTTP client.  This
    integration test validates the full pipeline runs without error and the
    response middleware sees the correct ``request_body`` (which always contains
    the mutation regardless of the bug).  See the docstring on
    ``RequestBodyEchoMiddleware`` for details.
    """
    controller, model_cache, sdk, mock_nim_image, ctx, _ = controller_with_docker_and_igw
    test_uuid = uuid.uuid4().hex[:8]

    config_name = f"test-mep-reg-{test_uuid}"
    deployment_name = f"test-mep-reg-{test_uuid}"
    model_entity_name = f"test-mep-reg-model-{test_uuid}"
    served_model_name = "mock-model"
    vm_name = f"test-mep-reg-alias-{test_uuid}"
    router_key = f"test-mep-reg-router-{test_uuid}"
    echo_key = f"test-mep-reg-echo-{test_uuid}"
    names = entity_names(deployment_name)
    container_name = plugin_container_name(DEFAULT_WORKSPACE, names.server)

    ctx.register_container(container_name)
    ctx.register_volume(docker_volume_name(DEFAULT_WORKSPACE, names.volume))
    ctx.register_volume(docker_volume_name(DEFAULT_WORKSPACE, names.scratch))

    # ---- Phase 1: Deploy mock NIM ----------------------------------------
    image_name, image_tag = mock_nim_image.rsplit(":", 1)
    sdk.inference.deployment_configs.create(
        workspace=DEFAULT_WORKSPACE,
        name=config_name,
        engine="nim",
        model_spec={},
        executor_config={"gpu": 0, "image_name": image_name, "image_tag": image_tag},
    )
    sdk.inference.deployments.create(workspace=DEFAULT_WORKSPACE, name=deployment_name, config=config_name)

    @retry(stop=stop_after_delay(30), wait=wait_fixed(0.1), reraise=True)
    def _wait_ready():
        controller.step()
        dep = sdk.inference.deployments.retrieve(deployment_name, workspace=DEFAULT_WORKSPACE)
        assert dep.status == "READY", f"Not READY: {dep.status}"

    _wait_ready()

    # ---- Phase 2: Wire up served_models and populate the model cache ------
    sdk.inference.providers.update_status(
        deployment_name,
        workspace=DEFAULT_WORKSPACE,
        served_models=[
            {
                "model_entity_id": f"{DEFAULT_WORKSPACE}/{model_entity_name}",
                "served_model_name": served_model_name,
            }
        ],
    )
    provider = sdk.inference.providers.retrieve(deployment_name, workspace=DEFAULT_WORKSPACE)
    model_cache.workspace_name_provider_map[(DEFAULT_WORKSPACE, deployment_name)] = ModelProviderInfo(
        model_provider=provider
    )
    model_cache.rebuild_model_entity_map()

    # ---- Phase 3: Register middleware ------------------------------------
    # Request middleware #1: route the alias → real entity AND stamp a custom field.
    # Request middleware #2: would fail (503) if the bug caused it to receive
    # the original body instead of the mutated one — but since both plugins run
    # on the same json_body local variable the bug doesn't affect plugin chaining.
    class RouterWithExtraField(NemoInferenceMiddleware):
        async def on_startup(self) -> None:
            pass

        async def on_shutdown(self) -> None:
            pass

        async def process_request(self, ctx, request, cfg) -> InferenceRequest | ImmediateResponse:
            return InferenceRequest(
                body={
                    **request.body,
                    "model": f"{DEFAULT_WORKSPACE}/{model_entity_name}",
                    RequestBodyEchoMiddleware.SOURCE_KEY: "was-mutated",
                },
                headers=request.headers,
                path=request.path,
            )

        async def process_response(self, ctx, response, cfg) -> InferenceResponse:
            return response

    registry = global_middleware_registry()
    vm_cache = global_virtual_model_cache()

    _inject_vm_and_plugins(
        registry,
        vm_cache,
        DEFAULT_WORKSPACE,
        vm_name,
        request_plugins=[(router_key, RouterWithExtraField())],
        response_plugins=[(echo_key, RequestBodyEchoMiddleware())],
        default_model_entity=None,
    )

    try:
        response = sdk.inference.gateway.model.post(
            "v1/chat/completions",
            name=vm_name,
            workspace=DEFAULT_WORKSPACE,
            body={
                "model": vm_name,
                "messages": [{"role": "user", "content": "regression test"}],
            },
        )

        assert "choices" in response or "message" in response, f"Expected valid chat response, got: {response}"
        # ResponseMarkerMiddleware echoes SOURCE_KEY from request_body.
        # request_body in process_response is json_body (always has the mutation).
        assert response.get(RequestBodyEchoMiddleware.ECHO_KEY) == "was-mutated", (
            f"Expected echoed mutation in response: {response}"
        )

    finally:
        _cleanup(registry, vm_cache, DEFAULT_WORKSPACE, vm_name, [router_key, echo_key])
        sdk.inference.deployments.delete(deployment_name, workspace=DEFAULT_WORKSPACE)
        _wait_for_deployment_deleted(controller, sdk, deployment_name)
        sdk.inference.deployment_configs.delete(config_name, workspace=DEFAULT_WORKSPACE)
