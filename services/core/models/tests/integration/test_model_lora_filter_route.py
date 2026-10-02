# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route-level tests for the lora_enabled cross-entity filter.

These tests go through the full HTTP stack so that ``make_filter_dep``
parses bracket-notation params, validates them against ``ModelEntityFilter``,
and translates field names (``lora_enabled`` → ``data.lora_enabled``) before
the service sees them. The lower-level integration tests in
``test_model_entity_service_integration.py`` build ``ParsedFilter`` directly
with post-translate names and bypass that layer; this file is the regression
guard for the parse/translate pipeline.
"""

import uuid

from nhx.testing import ClientContext

DEFAULT_WORKSPACE = "default"
MODELS_PATH = f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models"
CONFIGS_PATH = f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/deployment-configs"


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _create_model(test_clients: ClientContext, name: str) -> None:
    response = test_clients.test_client.post(MODELS_PATH, json={"name": name})
    assert response.status_code == 201, f"Failed to create model: {response.text}"


def _create_lora_config(test_clients: ClientContext, model_name: str, lora_enabled: bool) -> None:
    body = {
        "name": f"{model_name}-config",
        "model_entity_id": f"{DEFAULT_WORKSPACE}/{model_name}",
        "engine": "nim",
        "model_spec": {
            "model_type": "llm",
            "lora_enabled": lora_enabled,
            "model_namespace": "nvidia",
            "model_name": "llama-3-8b",
        },
        "executor_config": {
            "gpu": 1,
            "disk_size": "50Gi",
            "image_name": "nvcr.io/nvidia/nim/llm",
            "image_tag": "latest",
        },
    }
    response = test_clients.test_client.post(CONFIGS_PATH, json=body)
    assert response.status_code == 201, f"Failed to create deployment config: {response.text}"


def test_lora_enabled_combined_with_name_filter_via_http(test_clients: ClientContext):
    """?filter[lora_enabled]=true&filter[name]=<lora-model> returns just that model.

    Pre-fix, the lora condition rode on ``search`` while the name condition
    rode on ``filter_operation``; ``EntityClient.list`` overwrote one with
    the other and the user got results that ignored their lora intent.
    """
    lora_a = _uid("lora-a")
    lora_b = _uid("lora-b")
    plain = _uid("plain")
    _create_model(test_clients, lora_a)
    _create_model(test_clients, lora_b)
    _create_model(test_clients, plain)
    _create_lora_config(test_clients, lora_a, lora_enabled=True)
    _create_lora_config(test_clients, lora_b, lora_enabled=True)
    _create_lora_config(test_clients, plain, lora_enabled=False)

    response = test_clients.test_client.get(
        MODELS_PATH,
        params={"filter[lora_enabled]": "true", "filter[name]": lora_a},
    )
    assert response.status_code == 200, response.text
    names = sorted(m["name"] for m in response.json()["data"])
    assert names == [lora_a]


def test_lora_enabled_excludes_non_lora_with_name_via_http(test_clients: ClientContext):
    """?filter[lora_enabled]=true&filter[name]=<plain> returns no results.

    Pre-fix, this would return ``plain`` because the lora condition was lost.
    """
    lora_a = _uid("lora-a")
    plain = _uid("plain")
    _create_model(test_clients, lora_a)
    _create_model(test_clients, plain)
    _create_lora_config(test_clients, lora_a, lora_enabled=True)
    _create_lora_config(test_clients, plain, lora_enabled=False)

    response = test_clients.test_client.get(
        MODELS_PATH,
        params={"filter[lora_enabled]": "true", "filter[name]": plain},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"] == []


def test_lora_enabled_inside_or_returns_400(test_clients: ClientContext):
    """lora_enabled nested in $or is rejected with 400 — only top-level $eq is supported."""
    response = test_clients.test_client.get(
        MODELS_PATH,
        params={"filter": '{"$or":[{"lora_enabled":true},{"name":"foo"}]}'},
    )
    assert response.status_code == 400, response.text
    assert "lora_enabled" in response.json()["detail"]


def test_lora_enabled_inside_not_returns_400(test_clients: ClientContext):
    """lora_enabled nested in $not is rejected with 400."""
    response = test_clients.test_client.get(
        MODELS_PATH,
        params={"filter": '{"$not":{"lora_enabled":true}}'},
    )
    assert response.status_code == 400, response.text
    assert "lora_enabled" in response.json()["detail"]
