# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Body serialization contract for VirtualModel CRUD endpoints.

Paths, methods, and response types are proven against the real IGW router in
``services/core/inference-gateway/tests/unit/test_virtual_models_router.py``;
endpoint decorator mechanics are covered in ``tests/client/test_endpoint.py``.
This file pins the serialization rules a request body must obey, which neither
of those can observe.
"""

from __future__ import annotations

import json

from nemo_platform_plugin.client.types import PreparedRequest
from nemo_platform_plugin.virtual_models import endpoints
from nemo_platform_plugin.virtual_models.types import (
    CreateVirtualModelRequest,
    UpdateVirtualModelRequest,
    VirtualModelInferenceConfig,
)

PATH = "/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models"


def _json_body(prepared: PreparedRequest) -> dict[str, object]:
    assert isinstance(prepared.content, bytes)
    return json.loads(prepared.content)


def test_create_omits_unset_fields_and_nested_nones() -> None:
    """Defaults stay off the wire so the server applies its own."""
    body = CreateVirtualModelRequest(
        name="router",
        default_model_entity="default/llama",
        models=[VirtualModelInferenceConfig(model="default/llama")],
    )

    prepared = endpoints.create_virtual_model(workspace="default", body=body)

    assert prepared.content_type == "application/json"
    assert _json_body(prepared) == {
        "name": "router",
        "default_model_entity": "default/llama",
        "models": [{"model": "default/llama"}],
    }


def test_create_prebuilds_conflict_retrieve_for_exist_ok() -> None:
    """``exist_ok`` replays the GET for the same name on a 409; off by default."""
    prepared = endpoints.create_virtual_model(workspace="default", body=CreateVirtualModelRequest(name="router"))

    assert prepared.client_options == {"exist_ok": False}
    assert prepared.on_conflict_get is not None
    assert prepared.on_conflict_get.method == "GET"
    assert prepared.on_conflict_get.path_template == PATH + "/{name}"
    assert prepared.on_conflict_get.path_params == {"workspace": "default", "name": "router"}

    prepared = endpoints.create_virtual_model(body=CreateVirtualModelRequest(name="router"), exist_ok=True)
    assert prepared.client_options == {"exist_ok": True}
    assert prepared.on_conflict_get is not None
    assert prepared.on_conflict_get.path_params == {"name": "router"}


def test_delete_sends_expected_db_version_as_query_param() -> None:
    prepared = endpoints.delete_virtual_model(workspace="default", name="router")
    assert prepared.method == "DELETE"
    assert prepared.path_template == PATH + "/{name}"
    assert prepared.query_params is None

    prepared = endpoints.delete_virtual_model(
        workspace="default", name="router", query_params={"expected_db_version": 4}
    )
    assert prepared.query_params == {"expected_db_version": 4}


def test_update_distinguishes_explicit_null_and_empty_list_from_unset() -> None:
    """PATCH must send only what the caller set, so omitted fields stay unchanged."""
    body = UpdateVirtualModelRequest(default_model_entity=None, request_middleware=[])

    prepared = endpoints.update_virtual_model(workspace="default", name="router", body=body)

    assert _json_body(prepared) == {"default_model_entity": None, "request_middleware": []}
