# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FileSet-delivered Gym resources server for the ASCII Tree example.

Gym launches this module inside OpenSandbox using ``ascii_tree.yaml``. It
extracts assistant text from each model response, calls the scorer installed
from the adjacent wheel requirement, and returns the reward and parsed tree.
The server is environment code; developers invoke the parent ``run.py`` rather
than running this file locally.
"""

from __future__ import annotations

from typing import Any

from nemo_gym.base_resources_server import (  # ty: ignore[unresolved-import] - supplied by the Gym runtime
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nmp_ascii_tree_evaluator.scoring import (  # ty: ignore[unresolved-import] - installed from the FileSet wheel
    ascii_tree_reward,
    extract_ascii_formatted,
)


class AsciiTreeVerifyRequest(BaseVerifyRequest):
    """Expected ASCII tree and model response supplied by Gym."""

    answer: str


class AsciiTreeVerifyResponse(BaseVerifyResponse):
    """Custom reward and parsed model output returned to Gym."""

    answer: str
    observed_text: str
    parsed_ascii: str | None


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either SDK response models or decoded test mappings."""
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def assistant_text(response: Any) -> str:
    """Join assistant output-text parts in their original response order."""
    return "".join(
        _field(content, "text", "")
        for output in _field(response, "output", [])
        if _field(output, "type") == "message"
        for content in _field(output, "content", [])
        if _field(content, "type") == "output_text"
    )


class AsciiTreeResourcesServer(SimpleResourcesServer):
    """Score model responses with FileSet-provided ASCII Tree code."""

    config: BaseResourcesServerConfig

    async def verify(
        self,
        body: AsciiTreeVerifyRequest,
    ) -> AsciiTreeVerifyResponse:
        """Score one Gym turn and return both the reward and readable evidence."""
        observed_text = assistant_text(body.response)

        return AsciiTreeVerifyResponse(
            **body.model_dump(),
            reward=ascii_tree_reward(observed_text, body.answer),
            observed_text=observed_text,
            parsed_ascii=extract_ascii_formatted(observed_text),
        )


if __name__ == "__main__":
    AsciiTreeResourcesServer.run_webserver()
