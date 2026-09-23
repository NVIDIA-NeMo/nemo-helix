<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Mock Provider Mode for Inference Gateway

## Overview

Mock provider mode allows the Inference Gateway to return mock responses without connecting to real inference backends. This is useful for:

- **E2E Tests**: Test services that make inference calls without real LLM backends
- **Integration Tests**: Verify IGW routing and response handling
- **Local Development**: Test IGW flows without deploying real inference backends

**Key capabilities:**
- Returns mock JSON responses for any inference endpoint
- Supports both non-streaming and streaming (SSE) responses
- Per-model dynamic responses with sequential call tracking
- Configurable via SDK or HTTP headers

## Key Feature: Prefix-Based Matching

Unlike a global mock mode that affects ALL providers, mock provider mode uses **prefix-based matching**:

- Only providers whose name **starts with the configured prefix** return mock responses
- Other providers continue to proxy to their real backends
- This enables mixing real and mock providers in the same environment

For example, with `mock_provider_prefix=igw-mock-`:
- `igw-mock-judge` → Returns mock response
- `igw-mock-embeddings` → Returns mock response
- `real-llm-provider` → Proxies to real backend (NOT mocked)

## Using Mock Provider Mode in Tests

The recommended way to use mock provider mode in tests is via `nhx_testing`:

```python
import pytest
from typing import Generator

from nhx.testing import ClientContext, add_mock_provider, create_test_client
from nhx.core.inference_gateway.service import InferenceGatewayService
from nhx.core.models.service import ModelsService


@pytest.fixture
def mock_provider_test_clients() -> Generator[ClientContext, None, None]:
    """Create test clients with mock provider mode enabled."""
    with create_test_client(
        InferenceGatewayService,
        ModelsService,
        client_type=ClientContext,
    ) as clients:
        yield clients


def test_my_service(mock_provider_test_clients: ClientContext):
    """Test using mock provider mode."""
    # Create a mock provider with a pre-configured response
    provider = add_mock_provider(
        mock_provider_test_clients.sdk,
        workspace="default",
        name="my-judge",  # Becomes "igw-mock-my-judge"
        mock_response_body={
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        },
    )

    sdk = mock_provider_test_clients.sdk

    # === Route 1: Provider route ===
    response = sdk.inference.gateway.provider.post(
        "v1/chat/completions",
        name=provider.name,  # Use provider.name from returned ModelProvider
        workspace="default",
        body={"model": "test", "messages": []},
    )

    # === Route 2: Model Entity route ===
    # Uses default served_models mapping (entity name = "my-judge")
    response = sdk.inference.gateway.model.post(
        "v1/chat/completions",
        name="my-judge",
        workspace="default",
        body={"model": "test", "messages": []},
    )

    # === Route 3: OpenAI route ===
    response = sdk.inference.gateway.openai.post(
        "v1/chat/completions",
        workspace="default",
        body={
            "model": "default/my-judge",  # workspace/entity_name format
            "messages": [],
        },
    )
```

### The `add_mock_provider` Function

```python
def add_mock_provider(
    sdk: NeMoHelix,
    *,
    workspace: str,
    name: str,
    mock_response_body: dict | None = None,
    mock_response_body_by_model: dict[str, list[MockProviderResponse]] | None = None,
    mock_status: int | None = None,
    host_url: str = "http://mock.local",
    served_models: dict[str, str] | None = None,
    enabled_models: list[str] | None = None,
) -> ModelProvider:
```

**Parameters:**
- `sdk`: The NeMoHelix SDK client
- `workspace`: Provider workspace
- `name`: Provider name (auto-prefixed with `igw-mock-`)
- `mock_response_body`: Static JSON response to return for all requests (optional - uses smart defaults if this and mock_response_body_by_model are omitted)
- `mock_response_body_by_model`: Dict mapping `"workspace/model"` → list of `MockProviderResponse` for sequential calls (optional - uses smart defaults if this and mock_response_body are omitted)
- `mock_status`: HTTP status code (default: 200, only applies to `mock_response_body`)
- `served_models`: Dict mapping `entity_name` → `served_model_name` (optional - defaults to `{name: name}`)
- `enabled_models`: List of enabled models (optional - defaults to None i.e. all models are enabled)

**Key behaviors:**
- Provider name is auto-prefixed (e.g., `"judge"` becomes `"igw-mock-judge"`)
- Returns the created `ModelProvider` with `.name` containing the prefixed name
- Creates a default model entity mapping so all 3 route types work automatically

### Dynamic Per-Model Responses

In most cases, returning a static response via `mock_response_body` is sufficient. However, some tests may need to return different responses for different models, or configure responses for a model that gets invoked multiple times (for example, Guardrails content safety checks). In this case, use `mock_response_body_by_model`:

```python
from nhx.testing import MockProviderResponse

provider = add_mock_provider(
    sdk,
    workspace=workspace,
    name="nim-provider",
    mock_response_body_by_model={
        f"{workspace}/main-llm": [
            MockProviderResponse(
                response_code=200,
                response_body={"id": "1", "choices": [{"message": {"content": "Hello!"}}]},
            ),
        ],
        f"{workspace}/content-safety": [
            MockProviderResponse(
                response_code=200,
                response_body={"id": "2", "choices": [{"message": {"content": '{"safe": true}'}}]},
            ),  # 1st call
            MockProviderResponse(
                response_code=200,
                response_body={"id": "3", "choices": [{"message": {"content": '{"safe": false}'}}]},
            ),  # 2nd call
        ],
    },
    served_models={
        "main-llm": "main-llm",
        "content-safety": "content-safety",
    },
)
```

**How it works:**
- Each model has a list of responses that are returned sequentially
- Call counts are tracked per model (ex. the first call to the `content-safety` model returns response[0])
- If calls exceed the response list, the last response is returned
- The `/v1/models` endpoint automatically returns the model IDs derived from the keys
- Streaming is automatically supported: if the request has `stream=True`, the mock response is converted to SSE format

**Why call counts are stored in FastAPI app.state:**

In E2E tests, the NeMo Helix API runs inside a Docker container while pytest runs on the host.
These are separate processes with no shared memory, so the test code cannot pass a
Python object (the call tracker) directly to the server. Tests pass mock response *configuration* via headers,
but the server must maintain its own call tracking state.

We store this state on `app.state.mock_call_tracker` (FastAPI's built-in mechanism for
application state). Each test uses a unique workspace (e.g., `e2e-abc123`), and models are keyed by full path
(`workspace/model-name`), so call counts are isolated between individual tests.
Integration tests reset counts between test functions in `create_test_client` via `reset_call_counts(app.state)`.

### Streaming Responses

Mock provider mode supports streaming responses. When `stream=True` is in the request body, the mock provider automatically converts the response to Server-Sent Events (SSE) format:

```python
import json

provider = add_mock_provider(
    sdk,
    workspace="default",
    name="streaming-model",
    mock_response_body={
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}],
    },
)

# Request with streaming enabled
response = sdk.inference.gateway.openai.with_streaming_response.post(
    "v1/chat/completions",
    workspace="default",
    body={
        "model": "default/streaming-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    },
)

# Parse Server-Sent Events (SSE) stream
with response as stream:
    for line in stream.iter_lines():
        if line.startswith("data: "):
            line = line[len("data: "):]
        
        if not line or line == "[DONE]":
            continue
        
        try:
            chunk = json.loads(line)
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                print(content, end="")
        except json.JSONDecodeError:
            continue
```

**How it works:**
- The non-streaming response is automatically converted to SSE chunks
- Content is split word-by-word to simulate realistic streaming behavior
- Each chunk follows OpenAI's streaming format with `delta` containing incremental content
- Stream terminates with `data: [DONE]` marker

### Simulating Errors

```python
from nemo_helix import RateLimitError

provider = add_mock_provider(
    sdk,
    workspace="default",
    name="rate-limited",
    mock_response_body={"error": {"message": "Rate limit exceeded"}},
    mock_status=429,
)

with pytest.raises(RateLimitError):
    sdk.inference.gateway.provider.post(
        "v1/chat/completions",
        name=provider.name,
        workspace="default",
        body={"model": "test", "messages": []},
    )
```

### Inline Mock Response (No Provider Needed)

For one-off scenarios, pass the response via header:

```python
from nhx.core.inference_gateway.api.mock_provider import MOCK_RESPONSE_HEADER

response = sdk.inference.gateway.provider.post(
    "v1/chat/completions",
    name="any-provider",  # Doesn't need to exist
    workspace="default",
    body={"model": "test", "messages": []},
    extra_headers={MOCK_RESPONSE_HEADER: json.dumps({"id": "inline-response"})},
)
```

## Headers

Mock provider mode uses special HTTP headers to control behavior:

| Header | Required | Description |
| ------ | -------- | ----------- |
| `X-Mock-Response` | Yes* | JSON string containing the mock response body. *Required unless endpoint has a smart default or `X-Mock-Response-Map` is configured. |
| `X-Mock-Status` | No | HTTP status code to return (default: `200`). Only applies to `X-Mock-Response`. |
| `X-Mock-Response-Map` | No | JSON map of model name → list of `{response_body, response_code}` items for per-model sequential responses. |

## Smart Defaults

These endpoints return automatic responses without needing headers:

| Endpoint           | Method | Response                              |
| ------------------ | ------ | ------------------------------------- |
| `/v1/health/ready` | GET    | `{"status": "ready"}`                 |
| `/v1/health/live`  | GET    | `{"status": "live"}`                  |
| `/health/ready`    | GET    | `{"status": "ready"}`                 |
| `/health/live`     | GET    | `{"status": "live"}`                  |
| `/v1/models`       | GET    | Generic model list with `mock-model`  |

## Example Mock Responses

### Chat Completion (Non-Streaming)

```python
chat_response = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello!"},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
}
```

### Chat Completion (Streaming)

When `stream=True` is in the request, the same non-streaming response is automatically converted to SSE format:

```text
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Note:** Content is split word-by-word (last word has no trailing space) to simulate realistic streaming behavior.

### LLM Judge (Structured JSON Output)

```python
judge_response = {
    "id": "chatcmpl-judge",
    "object": "chat.completion",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": json.dumps({
                "score": 4,
                "judgment": "Good response.",
                "strengths": ["Accurate", "Well-structured"],
            }),
        },
        "finish_reason": "stop",
    }],
}
```

### Embeddings

```python
embeddings_response = {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
    "model": "text-embedding-ada-002",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}
```

### Multiple Choices (n > 1)

```python
multi_choice_response = {
    "id": "chatcmpl-multi",
    "object": "chat.completion",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Option A"}, "finish_reason": "stop"},
        {"index": 1, "message": {"role": "assistant", "content": "Option B"}, "finish_reason": "stop"},
    ],
}
```

## Enabling Mock Provider Mode (Manual Configuration)

For non-test scenarios, set the environment variable:

```bash
NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX=igw-mock-
```

Or in `config.yaml`:

```yaml
inference_gateway:
  mock_provider_prefix: igw-mock-
```

**Note:** In tests using `create_test_client`, mock provider mode is enabled by default via `igw_mock_provider_mode=True`.

## Limitations

- **No authentication validation**: Auth is bypassed in mock mode
- **Internal use only**: Not for production or customer use

## Reference

See `tests/integration/test_mock_provider_mode.py` for comprehensive examples including:
- All 3 route types (Provider, Model Entity, OpenAI)
- Error simulation (429, 500)
- LLM Judge patterns
- Smart defaults
- Header overrides
