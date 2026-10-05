#!/app/.venv/bin/python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Set up mock inference provider for the content safety eval.

This script runs after the NeMo Helix API server is healthy but before the agent starts.
It configures a mock inference provider that always returns "Yes" to any prompt,
which makes it suitable for use with guardrails self-check rails (blocking all content).

The agent is responsible for creating the guardrail configuration itself.
"""

import json
import sys
import time

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import (
    ConflictError,
    InternalServerError,
    NemoTransportError,
    NotFoundError,
    UnprocessableEntityError,
)
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import (
    CreateModelEntityRequest,
    CreateModelProviderRequest,
    ServedModelMapping,
    UpdateModelProviderStatusRequest,
    UpsertModelProviderRequest,
)

# Exceptions we treat as transient readiness errors during setup polling.
# Anything outside this set (auth errors, bad-request, schema validation
# failures) gets re-raised so misconfiguration fails fast instead of
# silently burning the whole timeout budget.
_TRANSIENT_ROUTING_ERRORS = (
    NemoTransportError,
    NotFoundError,
    UnprocessableEntityError,
    InternalServerError,
)

NHX_BASE_URL = "http://localhost:8080"
WORKSPACE = "default"
MOCK_MODEL_NAME = "mock-llm"
MOCK_PROVIDER_NAME = f"igw-mock-{MOCK_MODEL_NAME}"
MOCK_PROVIDER_HOST_URL = "http://mock.local"

# Header name used by the mock provider for model-specific fixed responses
MOCK_RESPONSE_MAP_HEADER = "X-Mock-Response-Map"

# Mock chat completion response: always answers "Yes" to any prompt.
MOCK_SELF_CHECK_RESPONSE = {
    "id": "chatcmpl-mock-safety",
    "object": "chat.completion",
    "created": 1700000000,
    "model": MOCK_MODEL_NAME,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Yes"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
}
# IGW commit 12ddafae2e (fix(igw): preserve qualified body["model"] for mock
# providers) keys the mock-response-map by the workspace-qualified entity id
# rather than the bare served-model name. Match that contract so the lookup
# in handle_mock_request resolves and we get the canned "Yes" back.
MOCK_RESPONSE_MAP = {
    f"{WORKSPACE}/{MOCK_MODEL_NAME}": [
        {
            "response_body": MOCK_SELF_CHECK_RESPONSE,
            "response_code": 200,
        }
    ]
}


def _register_served_model(client: NemoClient, workspace: str, provider_name: str, model_name: str) -> None:
    """Register provider served-model mapping for IGW model discovery."""
    ModelsClient.from_client(client).update_provider_status(
        name=provider_name,
        workspace=workspace,
        body=UpdateModelProviderStatusRequest(
            served_models=[
                ServedModelMapping(model_entity_id=f"{workspace}/{model_name}", served_model_name=model_name)
            ]
        ),
    )


def _ensure_model_entity(client: NemoClient, workspace: str, model_name: str) -> None:
    """Create model entity if missing; ignore already-exists conflicts."""
    try:
        ModelsClient.from_client(client).create_model(
            workspace=workspace,
            body=CreateModelEntityRequest(
                name=model_name,
                description="Mock model entity for guardrails agentic test",
            ),
        )
    except ConflictError:
        pass


def wait_for_model(
    client: NemoClient,
    workspace: str,
    provider_name: str,
    model_name: str,
    timeout: float = 120.0,
) -> None:
    """Poll until model entity route is visible in IGW cache."""
    start = time.time()
    attempts = 0
    full_model_name = f"{workspace}/{model_name}"
    last_error: Exception | None = None
    while time.time() - start < timeout:
        attempts += 1
        if attempts == 1 or attempts % 5 == 0:
            _register_served_model(client, workspace, provider_name, model_name)
        try:
            models_response = InferenceGatewayClient.from_client(client).list_openai_models(workspace=workspace).data()
            if any(model.id == full_model_name for model in models_response.data):
                return
        except _TRANSIENT_ROUTING_ERRORS as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(
        f"Model '{full_model_name}' not available in workspace '{workspace}' after "
        f"{timeout}s (last routing error: {last_error!r})"
    )


def wait_for_openai_routing(
    client: NemoClient,
    workspace: str,
    model_name: str,
    timeout: float = 60.0,
) -> None:
    """Poll until IGW OpenAI route can complete a chat call for the model.

    Transient readiness errors (connection/timeout, 404/422 from cache miss,
    5xx from a restarting upstream) are retried; everything else (auth,
    bad-request from a misconfigured mock provider) is raised immediately so
    test fixtures don't silently burn the whole timeout window when the mock
    contract drifts.
    """
    start = time.time()
    full_model_name = f"{workspace}/{model_name}"
    last_routing_error: Exception | None = None
    while time.time() - start < timeout:
        try:
            InferenceGatewayClient.from_client(client).openai_post(
                trailing_uri="v1/chat/completions",
                workspace=workspace,
                body=JsonBody(
                    {
                        "model": full_model_name,
                        "messages": [{"role": "user", "content": "healthcheck"}],
                        "max_tokens": 1,
                        "temperature": 0,
                    }
                ),
            ).data()
            return
        except _TRANSIENT_ROUTING_ERRORS as exc:
            last_routing_error = exc
            time.sleep(1.0)
    raise TimeoutError(
        f"Model '{full_model_name}' not routable via IGW OpenAI after "
        f"{timeout}s (last routing error: {last_routing_error!r})"
    )


def setup() -> None:
    client = NemoClient(base_url=NHX_BASE_URL)

    # Step 1: Create or update mock provider with static "Yes" response.
    # AGENT and VERIFY run in separate containers against the same DB state,
    # so on rerun the provider may already exist with the previous response
    # contract. We update default_extra_headers in that case so changes to
    # MOCK_RESPONSE_MAP (e.g. the workspace-qualified key migration) take
    # effect rather than silently routing through stale state.
    desired_headers = {MOCK_RESPONSE_MAP_HEADER: json.dumps(MOCK_RESPONSE_MAP)}
    print(f"Creating mock provider: {MOCK_PROVIDER_NAME}")
    models = ModelsClient.from_client(client)
    try:
        models.create_provider(
            workspace=WORKSPACE,
            body=CreateModelProviderRequest(
                name=MOCK_PROVIDER_NAME,
                host_url=MOCK_PROVIDER_HOST_URL,
                default_extra_headers=desired_headers,
            ),
        )
    except ConflictError:
        print(f"Provider already exists, reconciling headers: {MOCK_PROVIDER_NAME}")
        models.upsert_provider(
            name=MOCK_PROVIDER_NAME,
            workspace=WORKSPACE,
            body=UpsertModelProviderRequest(
                host_url=MOCK_PROVIDER_HOST_URL,
                default_extra_headers=desired_headers,
            ),
        )

    # Ensure model entity exists so provider served_models mapping stays stable.
    print(f"Ensuring model entity exists: {MOCK_MODEL_NAME}")
    _ensure_model_entity(client, WORKSPACE, MOCK_MODEL_NAME)

    # Step 2: Register served models via provider status update
    # This tells IGW that this provider serves the 'mock-llm' model entity
    print(f"Registering model entity: {MOCK_MODEL_NAME}")
    _register_served_model(client, WORKSPACE, MOCK_PROVIDER_NAME, MOCK_MODEL_NAME)

    # Step 3: Wait for model entity to become available in IGW's cache
    print(f"Waiting for model '{MOCK_MODEL_NAME}' to become available...")
    wait_for_model(client, WORKSPACE, MOCK_PROVIDER_NAME, MOCK_MODEL_NAME)
    print(f"Model '{WORKSPACE}/{MOCK_MODEL_NAME}' is available.")
    print(f"Waiting for model '{WORKSPACE}/{MOCK_MODEL_NAME}' to be routable...")
    wait_for_openai_routing(client, WORKSPACE, MOCK_MODEL_NAME)
    print(f"Model '{WORKSPACE}/{MOCK_MODEL_NAME}' is routable via IGW OpenAI.")
    # Allow downstream services a brief window to observe IGW model registration.
    # This avoids intermittent verify-time races where model lookup is briefly stale.
    time.sleep(20.0)


if __name__ == "__main__":
    try:
        setup()
        print("\n=== Mock Inference Setup Complete ===")
        print(f"  Mock model:       {WORKSPACE}/{MOCK_MODEL_NAME}")
        print(f"  Mock provider:    {MOCK_PROVIDER_NAME}")
        print("  Behavior: Always returns 'Yes' to any prompt")
        print("  Note: Agent must create a guardrail configuration to use this model.")
    except Exception as e:
        print(f"Error during mock setup: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
