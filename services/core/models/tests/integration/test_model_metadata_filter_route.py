# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Any

import pytest
from nhx.testing import ClientContext

DEFAULT_WORKSPACE = "default"
MODELS_PATH = f"/apis/models/v2/workspaces/{DEFAULT_WORKSPACE}/models"


@pytest.fixture
def models(test_clients: ClientContext) -> dict[str, str]:
    prefix = f"meta-{uuid.uuid4().hex[:8]}"
    specs: dict[str, dict[str, Any] | None] = {
        "vision": {
            "display_name": "Vision Reasoner",
            "capabilities": ["tool_calling", "reasoning"],
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "context_length": 131072,
            "max_output_tokens": 16384,
            "parameter_count": 70_000_000_000,
        },
        "small": {
            "display_name": "Small Chat",
            "capabilities": ["tool_calling"],
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "context_length": 8192,
            "max_output_tokens": 2048,
            "parameter_count": 8_000_000_000,
        },
        "embedder": {"capabilities": ["embedding"], "input_modalities": ["text"]},
        "bare": None,
    }
    names = {}
    for key, metadata in specs.items():
        name = f"{prefix}-{key}"
        body: dict[str, Any] = {"name": name}
        if metadata is not None:
            body["metadata"] = metadata
        response = test_clients.test_client.post(MODELS_PATH, json=body)
        assert response.status_code == 201, response.text
        names[key] = name
    return names


def _list(test_clients: ClientContext, models: dict[str, str], params: dict[str, str]) -> list[str]:
    response = test_clients.test_client.get(MODELS_PATH, params=params)
    assert response.status_code == 200, response.text
    ours = {name: key for key, name in models.items()}
    return sorted(ours[m["name"]] for m in response.json()["data"] if m["name"] in ours)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"filter[capabilities]": "tool_calling"}, ["small", "vision"]),
        ({"filter[capabilities]": "reasoning"}, ["vision"]),
        ({"filter[capabilities]": "embedding"}, ["embedder"]),
        ({"filter[input_modalities]": "image"}, ["vision"]),
        ({"filter[output_modalities]": "text"}, ["small", "vision"]),
        ({"filter[context_length][$gte]": "100000"}, ["vision"]),
        ({"filter[context_length]": "8192"}, ["small"]),
        ({"filter[max_output_tokens][$lt]": "4096"}, ["small"]),
        ({"filter[parameter_count][$lte]": "10000000000"}, ["small"]),
        ({"filter[display_name]": "Small Chat"}, ["small"]),
        ({"filter[capabilities]": "tool_calling", "filter[context_length][$gte]": "100000"}, ["vision"]),
    ],
)
def test_list_models_filters_on_metadata(
    test_clients: ClientContext, models: dict[str, str], params: dict[str, str], expected: list[str]
):
    assert _list(test_clients, models, params) == expected


def test_list_models_rejects_unknown_capability(test_clients: ClientContext):
    response = test_clients.test_client.get(MODELS_PATH, params={"filter[capabilities]": "telepathy"})
    assert response.status_code == 400, response.text
