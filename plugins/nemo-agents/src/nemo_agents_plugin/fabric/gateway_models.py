# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The models a translated Fabric config calls through the Platform Inference Gateway."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard
from urllib.parse import urlsplit

PLATFORM_IGW_PATH_MARKER = "/apis/inference-gateway/"
#: Where a gateway model reads its key when its config names no ``api_key_env``.
PLATFORM_IGW_API_KEY_ENV = "NEMO_AGENTS_IGW_API_KEY"

_VIRTUAL_MODEL_ROUTE = re.compile(
    r"/apis/inference-gateway/v2/workspaces/(?P<workspace>[^/]+)/(?:model/(?P<name>[^/]+)/-|openai/-)(?:/|$)"
)


@dataclass(frozen=True, slots=True)
class GatewayModel:
    """One shared or harness model of a translated Fabric config that calls the Inference Gateway."""

    config: dict[str, Any]
    """The model's own mapping inside the config, so a caller can rewrite it in place."""
    base_url: str
    """The URL it calls: ``base_url``, else the legacy ``settings.base_url``."""
    virtual_model: str | None
    """``workspace/name`` of the VirtualModel its route names: the ``model/<name>`` route, or the body ``model`` on
    the ``openai`` route. ``None`` on a gateway route that names none, such as a provider's."""
    api_key_env: str
    """The variable it reads its key from."""


def is_gateway_url(url: object) -> TypeGuard[str]:
    """Whether ``url`` calls the Inference Gateway, behind any host and path prefix."""
    return isinstance(url, str) and PLATFORM_IGW_PATH_MARKER in urlsplit(url).path


def model_base_url(model: Mapping[str, Any]) -> str | None:
    """The URL a Fabric model config calls, read as the runtime reads it."""
    base_url = model.get("base_url")
    if isinstance(base_url, str) and base_url:
        return base_url
    settings = model.get("settings")
    legacy = settings.get("base_url") if isinstance(settings, Mapping) else None
    return legacy if isinstance(legacy, str) and legacy else None


def model_configs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A translated Fabric config's shared model configs, then its harness model configs."""
    models = config.get("models")
    found = list(models.values()) if isinstance(models, Mapping) else []
    harnesses = config.get("harnesses")
    if isinstance(harnesses, Mapping):
        found += [harness.get("model") for harness in harnesses.values() if isinstance(harness, Mapping)]
    return [model for model in found if isinstance(model, dict)]


def gateway_model(model: dict[str, Any]) -> GatewayModel | None:
    """``model`` as a :class:`GatewayModel`, or ``None`` when it doesn't call the Inference Gateway."""
    base_url = model_base_url(model)
    if base_url is None or not is_gateway_url(base_url):
        return None
    route = _VIRTUAL_MODEL_ROUTE.search(urlsplit(base_url).path)
    name = (route["name"] or model.get("model")) if route is not None else None
    virtual_model = None
    if route is not None and isinstance(name, str) and name:
        virtual_model = name if "/" in name else f"{route['workspace']}/{name}"
    api_key_env = model.get("api_key_env")
    return GatewayModel(
        config=model,
        base_url=base_url,
        virtual_model=virtual_model,
        api_key_env=api_key_env if isinstance(api_key_env, str) and api_key_env else PLATFORM_IGW_API_KEY_ENV,
    )


def gateway_models(config: Mapping[str, Any]) -> list[GatewayModel]:
    """Each shared and harness model of a translated Fabric config that calls the Inference Gateway, in config order."""
    return [found for model in model_configs(config) if (found := gateway_model(model)) is not None]
