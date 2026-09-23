#!/app/.venv/bin/python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Set up mock inference provider for the chat completions eval.

This script runs after the NeMo Helix API server is healthy but before the agent starts.
It configures a mock inference provider that returns a realistic chat completion
response, allowing the agent to exercise the full chat completions flow via CLI
without needing a real LLM backend.
"""

import json
import sys
import time

from nemo_helix import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    NeMoHelix,
    NotFoundError,
    UnprocessableEntityError,
)
from nemo_helix import (
    ConflictError as SDKConflictError,
)
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelEntityRequest

# Exceptions we treat as transient readiness errors during setup polling.
# Anything outside this set (auth errors, bad-request, schema validation
# failures) gets re-raised so misconfiguration fails fast instead of
# silently burning the whole timeout budget.
_TRANSIENT_ROUTING_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    NotFoundError,
    UnprocessableEntityError,
    InternalServerError,
)

NHX_BASE_URL = "http://localhost:8080"
WORKSPACE = "default"
MOCK_MODEL_NAME = "chat-model"
MOCK_PROVIDER_NAME = f"igw-mock-{MOCK_MODEL_NAME}"
MOCK_PROVIDER_HOST_URL = "http://mock-chat.local"

# Header name used by the mock provider to return a fixed response
MOCK_RESPONSE_HEADER = "X-Mock-Response"

# Mock chat completion response matching OpenAI format
MOCK_CHAT_RESPONSE = {
    "id": "chatcmpl-mock-harbor",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "chat-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris. It is known as the City of Light.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 15, "completion_tokens": 14, "total_tokens": 29},
}


def register_served_model(sdk: NeMoHelix, workspace: str, provider_name: str, model_name: str) -> None:
    """Register provider served-model mapping for IGW model discovery."""
    sdk.inference.providers.update_status(
        name=provider_name,
        workspace=workspace,
        served_models=[
            {
                "model_entity_id": f"{workspace}/{model_name}",
                "served_model_name": model_name,
            }
        ],
    )


def wait_for_model_with_reregistration(
    sdk: NeMoHelix,
    workspace: str,
    provider_name: str,
    model_name: str,
    timeout: float = 120.0,
) -> None:
    """Poll until model appears, periodically re-registering served_models."""
    start = time.time()
    attempts = 0
    last_error: Exception | None = None
    while time.time() - start < timeout:
        attempts += 1
        if attempts == 1 or attempts % 5 == 0:
            register_served_model(sdk, workspace, provider_name, model_name)
        try:
            sdk.inference.gateway.model.get(
                "v1/models",
                name=model_name,
                workspace=workspace,
            )
            return
        except _TRANSIENT_ROUTING_ERRORS as exc:
            last_error = exc
            print(f"Polling for model {model_name}: {exc}", file=sys.stderr)
            time.sleep(1.0)
    raise TimeoutError(
        f"Model '{model_name}' not available in workspace '{workspace}' after "
        f"{timeout}s (last routing error: {last_error!r})"
    )


def ensure_model_entity(sdk: NeMoHelix, workspace: str, model_name: str) -> None:
    """Create model entity if missing; ignore already-exists conflicts."""
    try:
        client_from_platform(sdk, ModelsClient).create_model(
            workspace=workspace,
            body=CreateModelEntityRequest(
                name=model_name,
                description="Mock model entity for chat completions agentic test",
            ),
        )
    except ConflictError:
        pass


def wait_for_openai_routing(
    sdk: NeMoHelix,
    workspace: str,
    model_name: str,
    timeout: float = 60.0,
) -> None:
    """Poll until IGW OpenAI route can complete a chat call for the model."""
    start = time.time()
    full_model_name = f"{workspace}/{model_name}"
    last_error: Exception | None = None
    while time.time() - start < timeout:
        try:
            sdk.inference.gateway.openai.post(
                "v1/chat/completions",
                workspace=workspace,
                body={
                    "model": full_model_name,
                    "messages": [{"role": "user", "content": "healthcheck"}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            return
        except _TRANSIENT_ROUTING_ERRORS as exc:
            last_error = exc
            print(f"Polling for OpenAI chat routing on {full_model_name}: {exc}", file=sys.stderr)
            time.sleep(1.0)
    raise TimeoutError(
        f"Model '{full_model_name}' not routable via IGW OpenAI after {timeout}s (last routing error: {last_error!r})"
    )


def wait_for_model_route_chat(
    sdk: NeMoHelix,
    workspace: str,
    model_name: str,
    timeout: float = 60.0,
) -> None:
    """Poll until IGW model route can complete a chat call for the model."""
    start = time.time()
    last_error: Exception | None = None
    while time.time() - start < timeout:
        try:
            sdk.inference.gateway.model.post(
                "v1/chat/completions",
                name=model_name,
                workspace=workspace,
                body={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "healthcheck"}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )
            return
        except _TRANSIENT_ROUTING_ERRORS as exc:
            last_error = exc
            print(f"Polling for model-route chat routing on {workspace}/{model_name}: {exc}", file=sys.stderr)
            time.sleep(1.0)
    raise TimeoutError(
        f"Model '{workspace}/{model_name}' not routable via IGW model route after "
        f"{timeout}s (last routing error: {last_error!r})"
    )


def setup() -> None:
    sdk = NeMoHelix(base_url=NHX_BASE_URL)

    # Step 1: Create or update mock provider with static chat completion response.
    # AGENT and VERIFY share persisted DB state, so on rerun the provider may
    # already exist from a previous setup. We update headers in that case so a
    # changed MOCK_CHAT_RESPONSE / MOCK_RESPONSE_HEADER contract takes effect
    # rather than getting masked by stale state.
    desired_headers = {MOCK_RESPONSE_HEADER: json.dumps(MOCK_CHAT_RESPONSE)}
    print(f"Creating mock provider: {MOCK_PROVIDER_NAME}")
    try:
        sdk.inference.providers.create(
            workspace=WORKSPACE,
            name=MOCK_PROVIDER_NAME,
            host_url=MOCK_PROVIDER_HOST_URL,
            default_extra_headers=desired_headers,
        )
    except SDKConflictError:
        print(f"Provider already exists, reconciling headers: {MOCK_PROVIDER_NAME}")
        sdk.inference.providers.update(
            name=MOCK_PROVIDER_NAME,
            workspace=WORKSPACE,
            host_url=MOCK_PROVIDER_HOST_URL,
            default_extra_headers=desired_headers,
        )

    # Verify provider was created
    provider = sdk.inference.providers.retrieve(name=MOCK_PROVIDER_NAME, workspace=WORKSPACE)
    print(f"Provider created: {provider.name}")

    # Step 2: Ensure model entity exists and register served model mapping
    print(f"Ensuring model entity exists: {MOCK_MODEL_NAME}")
    ensure_model_entity(sdk, WORKSPACE, MOCK_MODEL_NAME)
    print(f"Registering model entity: {MOCK_MODEL_NAME}")
    register_served_model(sdk, WORKSPACE, MOCK_PROVIDER_NAME, MOCK_MODEL_NAME)

    # Step 3: Wait for model entity to become available in IGW's cache.
    # Re-register served models while waiting to tolerate cache propagation races.
    print(f"Waiting for model '{MOCK_MODEL_NAME}' to become available...")
    wait_for_model_with_reregistration(sdk, WORKSPACE, MOCK_PROVIDER_NAME, MOCK_MODEL_NAME)
    print(f"Model '{MOCK_MODEL_NAME}' is available.")
    print(f"Waiting for model '{WORKSPACE}/{MOCK_MODEL_NAME}' to be routable...")
    wait_for_openai_routing(sdk, WORKSPACE, MOCK_MODEL_NAME)
    print(f"Model '{WORKSPACE}/{MOCK_MODEL_NAME}' is routable via IGW OpenAI.")
    print(f"Waiting for model '{WORKSPACE}/{MOCK_MODEL_NAME}' to be routable via model route...")
    wait_for_model_route_chat(sdk, WORKSPACE, MOCK_MODEL_NAME)
    print(f"Model '{WORKSPACE}/{MOCK_MODEL_NAME}' is routable via IGW model route.")


if __name__ == "__main__":
    try:
        setup()
        print("\n=== Mock Chat Provider Setup Complete ===")
        print(f"  Mock model:       {WORKSPACE}/{MOCK_MODEL_NAME}")
        print(f"  Mock provider:    {MOCK_PROVIDER_NAME}")
        print("  Behavior: Returns a fixed chat completion response to any prompt")
    except Exception as e:
        print(f"Error during mock setup: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
