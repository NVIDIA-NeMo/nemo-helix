# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Guardrails service endpoint definitions."""

from __future__ import annotations

import json
from typing import get_origin

from nemo_platform_plugin.client.types import Paginated, PreparedRequest
from nemo_platform_plugin.entities.types import DeleteResponse
from nemo_platform_plugin.guardrail import endpoints
from nemo_platform_plugin.guardrail.types import (
    CreateGuardrailConfigRequest,
    GuardrailCheckRequest,
    GuardrailCheckResponse,
    GuardrailConfig,
    RailsConfig,
    UpdateGuardrailConfigRequest,
)

_CONFIGS = "/apis/guardrails/v2/workspaces/{workspace}/configs"
_CHECKS = "/apis/guardrails/v2/workspaces/{workspace}/checks"


def _json_body(prepared: PreparedRequest) -> dict:
    assert isinstance(prepared.content, bytes)
    return json.loads(prepared.content)


def _rails_config() -> RailsConfig:
    return RailsConfig.model_validate(
        {
            "models": [{"type": "content_safety", "engine": "nim", "model": "default/safety"}],
            "rails": {"input": {"flows": ["self check input"]}},
        }
    )


def test_create_guardrail_config() -> None:
    body = CreateGuardrailConfigRequest(name="safe", description="Content safety", data=_rails_config())

    prepared = endpoints.create_guardrail_config(workspace="default", body=body)

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == _CONFIGS
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.content_type == "application/json"
    assert prepared.response_type is GuardrailConfig
    assert _json_body(prepared) == {
        "name": "safe",
        "description": "Content safety",
        "data": {
            "models": [{"engine": "nim", "type": "content_safety", "model": "default/safety"}],
            "rails": {"input": {"flows": ["self check input"]}},
        },
    }


def test_create_guardrail_config_accepts_raw_dict_data() -> None:
    prepared = endpoints.create_guardrail_config(
        body=CreateGuardrailConfigRequest(
            name="safe",
            data={"rails": {"output": {"flows": ["self check output"], "streaming": {"enabled": False}}}},
        )
    )

    assert prepared.path_params == {}
    assert _json_body(prepared) == {
        "name": "safe",
        "data": {"rails": {"output": {"flows": ["self check output"], "streaming": {"enabled": False}}}},
    }


def test_create_guardrail_config_conflict_resolver() -> None:
    prepared = endpoints.create_guardrail_config(
        workspace="default",
        body=CreateGuardrailConfigRequest(name="safe"),
        exist_ok=True,
    )

    assert prepared.on_conflict_get is not None
    assert prepared.on_conflict_get.method == "GET"
    assert prepared.on_conflict_get.path_template == _CONFIGS + "/{name}"
    assert prepared.on_conflict_get.path_params == {"workspace": "default", "name": "safe"}
    assert prepared.client_options == {"exist_ok": True}


def test_list_guardrail_configs() -> None:
    prepared = endpoints.list_guardrail_configs(
        workspace="default",
        query_params={"page": 2, "page_size": 10, "filter": "name:safe", "project": "p"},
    )

    assert prepared.method == "GET"
    assert prepared.path_template == _CONFIGS
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.query_params == {"page": 2, "page_size": 10, "filter": "name:safe", "project": "p"}
    assert get_origin(prepared.response_type) is Paginated


def test_get_update_delete_guardrail_config() -> None:
    get_prepared = endpoints.get_guardrail_config(workspace="default", name="safe")
    update_prepared = endpoints.update_guardrail_config(
        workspace="default",
        name="safe",
        body=UpdateGuardrailConfigRequest(description="updated"),
    )
    delete_prepared = endpoints.delete_guardrail_config(workspace="default", name="safe")

    assert get_prepared.method == "GET"
    assert get_prepared.path_params == {"workspace": "default", "name": "safe"}
    assert get_prepared.response_type is GuardrailConfig

    assert update_prepared.method == "PATCH"
    assert update_prepared.path_params == {"workspace": "default", "name": "safe"}
    assert _json_body(update_prepared) == {"description": "updated"}
    assert update_prepared.response_type is GuardrailConfig

    assert delete_prepared.method == "DELETE"
    assert delete_prepared.content is None
    assert delete_prepared.response_type is DeleteResponse


def test_check_guardrail() -> None:
    prepared = endpoints.check_guardrail(
        workspace="default",
        body=GuardrailCheckRequest(
            model="default/app",
            messages=[{"role": "user", "content": "hello"}],
            guardrails={"config_id": "default/safe"},
        ),
    )

    assert prepared.method == "POST"
    assert prepared.path_template == _CHECKS
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.response_type is GuardrailCheckResponse
    assert _json_body(prepared) == {
        "model": "default/app",
        "messages": [{"role": "user", "content": "hello"}],
        "guardrails": {"config_id": "default/safe"},
    }
