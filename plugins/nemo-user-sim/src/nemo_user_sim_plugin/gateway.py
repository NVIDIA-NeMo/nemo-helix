# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every model call a simulation makes goes through the workspace's inference gateway."""

import json
import os
import re
from pathlib import Path
from typing import Any

import openai
from usersim.cli.model_catalog import INFERENCE_DEFAULTS

TOKEN_ENV = "NHX_USER_SIM_GATEWAY_TOKEN"
PROVIDER = "helix-gateway"
UNAUTHENTICATED = "not-used"


class GatewayEmbeddings:
    """The embedding hook UserSim's retrieval probes call as ``agenerate_text_embeddings``."""

    def __init__(self, client: openai.AsyncOpenAI, model: str):
        self._client = client
        self._model = model

    async def agenerate_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


def export_token(request_headers: dict[str, str] | None) -> None:
    """Expose the client's bearer token to UserSim's Data Designer models, which read keys from the environment."""
    bearer = (request_headers or {}).get("Authorization", "")
    os.environ[TOKEN_ENV] = bearer.removeprefix("Bearer ") or UNAUTHENTICATED


def _entity_name(model_id: str) -> str:
    """The gateway's name for a provider model ID, as the models service's ``normalize_model_entity_name`` derives it."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]", "-", model_id.lower())).strip("-")


# UserSim keys its per-model inference settings by provider ID. The gateway serves the same models by entity name.
_SETTINGS_BY_ENTITY = {_entity_name(model_id): settings for model_id, settings in INFERENCE_DEFAULTS.items()}


# UserSim's default models file gives every chat role this timeout; Data Designer's own default is 60 seconds.
ROLE_TIMEOUT_SECONDS = 300


def _settings(model: str) -> dict[str, Any]:
    """UserSim's settings for a gateway model name, which may carry a provider prefix.

    inference-api.nvidia.com serves nvidia/nemotron-3.5-lightning as nvidia/nvidia/nemotron-3.5-lightning,
    so its entity name is the catalog's with ``nvidia-`` in front.
    """
    matches = [entity for entity in _SETTINGS_BY_ENTITY if model == entity or model.endswith(f"-{entity}")]
    return {"timeout": ROLE_TIMEOUT_SECONDS, **(_SETTINGS_BY_ENTITY[max(matches, key=len)] if matches else {})}


def _toml_value(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items()) + "}"
    return json.dumps(value)


def write_models_file(
    path: Path, *, gateway_url: str, models: dict[str, str], role_settings: dict[str, dict] | None = None
) -> Path:
    """Write a UserSim models TOML whose aliases resolve through the gateway.

    Each row gets UserSim's settings for its model, then any ``role_settings`` for its alias on top.
    """
    lines = [
        "[[providers]]",
        f'name = "{PROVIDER}"',
        f'endpoint = "{gateway_url}"',
        'provider_type = "openai"',
        f'api_key = "{TOKEN_ENV}"',
    ]
    for alias, model in models.items():
        lines += ["", "[[models]]", f'alias = "{alias}"', f'model = "{model}"', f'provider = "{PROVIDER}"']
        settings = {**_settings(model), **(role_settings or {}).get(alias, {})}
        lines += [f"{key} = {_toml_value(value)}" for key, value in settings.items() if value is not None]
        # TOML has no null. UserSim spells a None setting, meaning "don't send it", as drop_params.
        if dropped := [key for key, value in settings.items() if value is None]:
            lines.append(f"drop_params = {_toml_value(dropped)}")
    path.write_text("\n".join(lines) + "\n")
    return path
