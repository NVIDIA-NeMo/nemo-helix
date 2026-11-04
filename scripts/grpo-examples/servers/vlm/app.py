# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym resources server for NeMo-RL's ``VLMEnvironment`` verifier.

The Ray worker pool, ``EnvironmentInterface.step``, and the batch metrics stay in
NeMo-RL. This server is the ``verify`` half: ``expected_answer`` on the row is the
ground truth, and ``reward_functions`` on the server config is ``VLMEnvConfig``.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, Response
from nemo_gym.base_resources_server import (  # ty: ignore[unresolved-import]
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from pydantic import BaseModel, ConfigDict
from rewards import score_response

# 16x16 pink PNG. simple_agent stores the HTTP body as a string, and the training
# postprocess treats a string that starts with ``data:image/`` as a tool image.
INSPECT_SCENE_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAI0lEQVR4nGP8z9DAQApg"
    "Ikk1w6gG4gATkergYFQDMYDkUAIAxr0Bn/6xQzgAAAAASUVORK5CYII="
)


class RewardFunctionConfig(BaseModel):
    name: str
    weight: float
    kwargs: Optional[dict[str, Any]] = None


class VlmResourcesServerConfig(BaseResourcesServerConfig):
    reward_functions: list[RewardFunctionConfig]


class VlmVerifyRequest(BaseVerifyRequest):
    expected_answer: str


class VlmVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str] = None
    format_reward: Optional[float] = None
    answer_reward: Optional[float] = None


class InspectSceneRequest(BaseModel):
    """The model calls ``inspect_scene`` with an empty object."""

    model_config = ConfigDict(extra="allow")


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for output in body.response.output:
        if getattr(output, "type", None) == "message" and getattr(output, "role", None) == "assistant":
            content = getattr(output, "content", None)
            if isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


class VlmResourcesServer(SimpleResourcesServer):
    config: VlmResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/inspect_scene")(self.inspect_scene)
        return app

    async def inspect_scene(self, body: InspectSceneRequest) -> Response:
        """Return a second view of the scene as a raw image data URL.

        A JSON string would arrive quoted (``"data:image/..."``) and the
        training postprocess would not treat it as an image.
        """
        del body
        return Response(content=INSPECT_SCENE_IMAGE, media_type="text/plain")

    async def verify(self, body: VlmVerifyRequest) -> VlmVerifyResponse:
        scored = score_response(
            _extract_last_assistant_text(body),
            body.expected_answer,
            list(self.config.reward_functions),
        )
        return VlmVerifyResponse(**body.model_dump(), **scored)


if __name__ == "__main__":
    VlmResourcesServer.run_webserver()
