# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Harbor agent that runs a NeMo Fabric harness against an OpenAI-compatible model endpoint.

Point :class:`HarborRuntimeConfig.agent_import_path` (or the platform's ``HarborRunnerTarget``) at
``nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent:NemoFabricAgent`` and configure it
through ``agent_kwargs``. It accepts everything ``nemo_fabric.integrations.harbor:FabricAgent`` does,
plus ``fabric_model_api_key_env``, and fills in what a non-OpenAI provider needs: the credential's
environment variable name and, for NVIDIA-hosted models, the endpoint.
"""

from __future__ import annotations

from typing import Any

from nemo_fabric import FabricConfig
from nemo_fabric.integrations.harbor.fabric_agent import FabricAgent

NVIDIA_MODEL_BASE_URL = "https://integrate.api.nvidia.com/v1"
#: Credential variable a provider's adapter reads when ``fabric_model_api_key_env`` is not given.
DEFAULT_API_KEY_ENV: dict[str, str] = {"openai": "OPENAI_API_KEY", "nvidia": "NVIDIA_API_KEY"}
_DEFAULT_BASE_URL: dict[str, str] = {"nvidia": NVIDIA_MODEL_BASE_URL}


class NemoFabricAgent(FabricAgent):  # ty: ignore[unsupported-base]
    """``FabricAgent`` with the model credential and endpoint resolved for the selected provider.

    ``agent_model_name`` keeps Fabric's ``provider/model`` convention. ``nvidia/nemotron-3-super-120b-a12b``
    selects the ``nvidia`` provider and keeps the full id, which is what build.nvidia.com expects;
    ``openai/gpt-5.4`` selects ``openai`` and drops the prefix, as the Codex adapter does.
    """

    def __init__(self, *args: Any, fabric_model_api_key_env: str | None = None, **kwargs: Any) -> None:
        if fabric_model_api_key_env is not None and not fabric_model_api_key_env.strip():
            raise ValueError("fabric_model_api_key_env must not be empty")
        self.fabric_model_api_key_env = fabric_model_api_key_env
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "nemo-fabric"

    def _build_config(self) -> FabricConfig:
        config = super()._build_config()
        model = config.models.get("default")
        if model is None:
            return config
        provider = model.provider
        if provider == "openai":
            model.model = model.model.removeprefix("openai/")
        api_key_env = self.fabric_model_api_key_env or DEFAULT_API_KEY_ENV.get(provider)
        if api_key_env is None:
            raise ValueError(
                f"fabric_model_api_key_env is required for model provider {provider!r}: name the environment "
                "variable holding its API key"
            )
        model.api_key_env = api_key_env
        if model.base_url is None:
            model.base_url = _DEFAULT_BASE_URL.get(provider)
        return config
