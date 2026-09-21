# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for automatic model selection in ``nemo setup --auto``.

Selection has to hold up on a cold start against a real platform: a provider is
created, its models are discovered, the gateway 404s them until the controller
publishes their routes, and only a model that answers a live inference request
may be persisted as the default.

Provider registration from environment credentials and writing the CLI config
are the only stubbed steps; discovery, routing and probing go through the
platform.
"""

import json
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nemo_platform import APIStatusError, NeMoPlatform
from nemo_platform_ext.cli.commands import setup as setup_commands
from nemo_platform_ext.cli.commands.setup import ModelPair, SetupClients
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.secrets.client import SecretsClient
from nmp.common.config import Configuration
from nmp.core.inference_gateway.api.mock_provider import MOCK_RESPONSE_HEADER, MOCK_SERVED_MODELS_HEADER
from nmp.core.inference_gateway.config import InferenceGatewayConfig
from nmp.testing import MockProviderResponse, add_mock_provider

pytestmark = [pytest.mark.timeout(300)]

SETUP_MOD = "nemo_platform_ext.cli.commands.setup"


def _unique_suffix() -> str:
    """Return a suffix that cannot be read as a parameter count (ex. ``12345b``)."""
    return f"x{uuid.uuid4().hex[:6]}"


def _chat_response(content: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _advertise_models_without_routes(sdk: NeMoPlatform, workspace: str, name: str, entities: list[str]) -> str:
    """Create a mock provider that serves *entities* before their routes exist.

    This is ``add_mock_provider`` without the passthrough VirtualModels it
    creates on the caller's behalf, leaving them to the controller as in
    production. ``MOCK_SERVED_MODELS_HEADER`` keeps the entities in the
    provider's discovery response, so the reconciler preserves them.
    """
    provider_name = f"{Configuration.get_service_config(InferenceGatewayConfig).mock_provider_prefix}{name}"
    sdk.inference.providers.create(
        workspace=workspace,
        name=provider_name,
        host_url="http://mock.local",
        default_extra_headers={
            MOCK_RESPONSE_HEADER: json.dumps(_chat_response("OK")),
            MOCK_SERVED_MODELS_HEADER: json.dumps(entities),
        },
    )
    sdk.inference.providers.update_status(
        name=provider_name,
        workspace=workspace,
        served_models=[
            {"model_entity_id": f"{workspace}/{entity}", "served_model_name": entity} for entity in entities
        ],
    )
    return provider_name


def _run_auto_setup(sdk: NeMoPlatform, workspace: str, provider_name: str) -> ModelPair | None:
    """Run ``_run_auto_mode`` for *provider_name* and return the persisted pair."""
    with (
        patch.dict("os.environ", {"NEMO_DEFAULT_MODEL": "", "NEMO_FAST_MODEL": ""}),
        patch(f"{SETUP_MOD}._auto_setup", return_value=provider_name),
        patch(f"{SETUP_MOD}._save_model_pair") as save_pair,
        patch(f"{SETUP_MOD}._maybe_install_skills"),
        patch(f"{SETUP_MOD}._maybe_deploy_agent"),
        patch(f"{SETUP_MOD}._verify_platform_health", return_value=True),
    ):
        clients = SetupClients(
            models=client_from_platform(sdk, ModelsClient),
            secrets=client_from_platform(sdk, SecretsClient),
            gateway=client_from_platform(sdk, InferenceGatewayClient),
        )
        setup_commands._run_auto_mode(
            MagicMock(),
            clients,
            workspace,
            str(sdk.base_url),
            install_skills=False,
            deploy_agent=False,
        )

    if not save_pair.call_args_list:
        return None
    return save_pair.call_args.args[1]


def test_auto_setup_persists_a_default_the_account_can_serve(sdk: NeMoPlatform, workspace: str):
    """The largest model that answers becomes the default, the smallest the fast model."""
    suffix = _unique_suffix()
    ultra = f"nvidia-nemotron-ultra-500b-{suffix}"
    large = f"nvidia-nemotron-super-120b-{suffix}"
    nano = f"nvidia-nemotron-nano-9b-{suffix}"

    provider = add_mock_provider(
        sdk,
        workspace=workspace,
        name=f"selection-{suffix}",
        mock_response_body_by_model={
            # Discovery advertises a model this account cannot run.
            f"{workspace}/{ultra}": [
                MockProviderResponse(
                    response_code=404,
                    response_body={"detail": "Function not found for account"},
                )
            ],
            f"{workspace}/{large}": [MockProviderResponse(response_body=_chat_response("OK"))],
            f"{workspace}/{nano}": [MockProviderResponse(response_body=_chat_response("OK"))],
        },
        served_models={name: name for name in (ultra, large, nano)},
    )

    saved = _run_auto_setup(sdk, workspace, provider.name)

    assert saved == ModelPair(default=f"{workspace}/{large}", fast=f"{workspace}/{nano}")


def test_auto_setup_waits_for_a_late_published_model_route(sdk: NeMoPlatform, workspace: str):
    """Cold start: discovery reports a model before its route is published.

    The gateway 404s the model until the controller creates its passthrough
    VirtualModel, so probing has to retry before a default can be persisted.
    """
    suffix = _unique_suffix()
    entity = f"nvidia-nemotron-nano-9b-{suffix}"
    provider_name = _advertise_models_without_routes(sdk, workspace, f"late-route-{suffix}", [entity])

    with pytest.raises(APIStatusError) as initial_probe:
        sdk.inference.gateway.openai.post(
            "v1/chat/completions",
            workspace=workspace,
            body={
                "model": f"{workspace}/{entity}",
                "messages": [{"role": "user", "content": "Respond with 'OK'"}],
                "max_tokens": 16,
            },
        )
    assert initial_probe.value.status_code == 404

    saved = _run_auto_setup(sdk, workspace, provider_name)

    assert saved == ModelPair(default=f"{workspace}/{entity}", fast=f"{workspace}/{entity}")


def test_auto_setup_saves_nothing_when_no_model_answers(sdk: NeMoPlatform, workspace: str):
    """A provider whose models all fail leaves the default unset rather than broken."""
    suffix = _unique_suffix()
    entity = f"nvidia-nemotron-nano-9b-{suffix}"

    provider = add_mock_provider(
        sdk,
        workspace=workspace,
        name=f"unusable-{suffix}",
        mock_response_body={"detail": "Function not found for account"},
        mock_status=404,
        served_models={entity: entity},
    )

    assert _run_auto_setup(sdk, workspace, provider.name) is None
