# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Inference Gateway proxy routes.

Three proxy surfaces share the ``/-/{trailing_uri}`` convention, where
``trailing_uri`` is the provider-relative path such as ``v1/chat/completions``:

- ``provider``: route straight to a registered ModelProvider by name.
- ``model``: route through a model entity (``workspace/name``).
- ``openai``: the workspace's OpenAI-compatible surface, where the body's
  ``model`` field selects the VirtualModel.

``stream_openai`` and ``stream_model`` return the raw SSE byte stream so callers
can decode events (including ``event: error`` frames) themselves.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from nemo_platform_plugin.client.endpoint import delete, get, patch, post, put
from nemo_platform_plugin.client.types import BinaryContent
from nemo_platform_plugin.inference_gateway.types import JsonBody, OpenAIModel, OpenAIModelList, ProviderReadyResponse

_BASE = "/apis/inference-gateway/v2/workspaces/{workspace}"

# ---------------------------------------------------------------------------
# Provider proxy
# ---------------------------------------------------------------------------


@get(f"{_BASE}/provider/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def provider_get(*, workspace: str | None = None, name: str, trailing_uri: str) -> Any: ...


@post(f"{_BASE}/provider/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def provider_post(*, workspace: str | None = None, name: str, trailing_uri: str, body: JsonBody) -> Any: ...


@put(f"{_BASE}/provider/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def provider_put(*, workspace: str | None = None, name: str, trailing_uri: str, body: JsonBody) -> Any: ...


@patch(f"{_BASE}/provider/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def provider_patch(*, workspace: str | None = None, name: str, trailing_uri: str, body: JsonBody) -> Any: ...


@delete(f"{_BASE}/provider/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def provider_delete(*, workspace: str | None = None, name: str, trailing_uri: str) -> None: ...


@post(f"{_BASE}/provider/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def stream_provider(*, workspace: str | None = None, name: str, trailing_uri: str, body: JsonBody) -> BinaryContent: ...


@get(f"{_BASE}/provider/{{name}}/-/v1/models")
@abstractmethod
def get_provider_models_raw(*, workspace: str | None = None, name: str) -> BinaryContent: ...


@get(f"{_BASE}/provider/{{name}}/ready")
@abstractmethod
def provider_ready(*, workspace: str | None = None, name: str) -> ProviderReadyResponse: ...


# ---------------------------------------------------------------------------
# Model entity proxy
# ---------------------------------------------------------------------------


@get(f"{_BASE}/model/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def model_get(*, workspace: str | None = None, name: str, trailing_uri: str) -> Any: ...


@post(f"{_BASE}/model/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def model_post(*, workspace: str | None = None, name: str, trailing_uri: str, body: JsonBody) -> Any: ...


@put(f"{_BASE}/model/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def model_put(*, workspace: str | None = None, name: str, trailing_uri: str, body: JsonBody) -> Any: ...


@patch(f"{_BASE}/model/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def model_patch(*, workspace: str | None = None, name: str, trailing_uri: str, body: JsonBody) -> Any: ...


@delete(f"{_BASE}/model/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def model_delete(*, workspace: str | None = None, name: str, trailing_uri: str) -> None: ...


@post(f"{_BASE}/model/{{name}}/-/{{trailing_uri}}")
@abstractmethod
def stream_model(*, workspace: str | None = None, name: str, trailing_uri: str, body: JsonBody) -> BinaryContent: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible surface
# ---------------------------------------------------------------------------


@get(f"{_BASE}/openai/-/{{trailing_uri}}")
@abstractmethod
def openai_get(*, workspace: str | None = None, trailing_uri: str) -> Any: ...


@post(f"{_BASE}/openai/-/{{trailing_uri}}")
@abstractmethod
def openai_post(*, workspace: str | None = None, trailing_uri: str, body: JsonBody) -> Any: ...


@post(f"{_BASE}/openai/-/{{trailing_uri}}")
@abstractmethod
def stream_openai(*, workspace: str | None = None, trailing_uri: str, body: JsonBody) -> BinaryContent: ...


@get(f"{_BASE}/openai/-/v1/models")
@abstractmethod
def list_openai_models(*, workspace: str | None = None) -> OpenAIModelList: ...


@get(f"{_BASE}/openai/-/v1/models/{{name}}")
@abstractmethod
def get_openai_model(*, workspace: str | None = None, name: str) -> OpenAIModel: ...
