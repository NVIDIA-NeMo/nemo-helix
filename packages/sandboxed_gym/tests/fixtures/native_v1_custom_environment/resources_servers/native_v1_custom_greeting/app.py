# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import emoji  # ty: ignore[unresolved-import] - installed from this environment fixture's requirements
from nemo_gym.base_resources_server import (  # ty: ignore[unresolved-import] - available in the Gym runtime
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class NativeV1CustomGreetingResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the customer-supplied greeting verifier."""


class NativeV1CustomGreetingVerifyRequest(BaseVerifyRequest):
    """Verify that the assistant emitted the expected greeting."""

    expected_greeting: str


class NativeV1CustomGreetingVerifyResponse(BaseVerifyResponse):
    """Greeting-verification result returned to Gym."""

    expected_greeting: str
    observed_text: str


def _assistant_text(response: Any) -> str:
    """Collect output text from the assistant response."""
    parts: list[str] = []
    for output_item in response.output:
        if getattr(output_item, "type", None) != "message":
            continue
        for content_item in getattr(output_item, "content", []):
            if getattr(content_item, "type", None) == "output_text":
                parts.append(content_item.text)
    return "".join(parts)


def _normalized_greeting(value: str) -> str:
    """Normalize emoji so the fixture exercises its package-index dependency."""
    return emoji.demojize(value, language="en").casefold()


class NativeV1CustomGreetingResourcesServer(SimpleResourcesServer):
    """Minimal custom server used to prove source delivery from a FileSet."""

    config: NativeV1CustomGreetingResourcesServerConfig

    async def verify(self, body: NativeV1CustomGreetingVerifyRequest) -> NativeV1CustomGreetingVerifyResponse:
        observed_text = _assistant_text(body.response)
        reward = float(_normalized_greeting(body.expected_greeting) in _normalized_greeting(observed_text))
        return NativeV1CustomGreetingVerifyResponse(
            **body.model_dump(),
            reward=reward,
            observed_text=observed_text,
        )


if __name__ == "__main__":
    NativeV1CustomGreetingResourcesServer.run_webserver()
