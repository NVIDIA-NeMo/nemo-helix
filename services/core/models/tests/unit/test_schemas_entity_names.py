# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create-request names must satisfy the entity store's NAME_PATTERN, not a looser one."""

import pytest
from nhx.common.entities import constants
from nhx.core.models.schemas import (
    CreateModelAdapterRequest,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    CreateModelEntityRequest,
    CreateModelProviderRequest,
    CreatePromptRequest,
)
from pydantic import ValidationError
from pydantic_core import ErrorDetails

REQUEST_MODELS = [
    CreateModelProviderRequest,
    CreatePromptRequest,
    CreateModelEntityRequest,
    CreateModelAdapterRequest,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
]

INVALID_NAMES = [
    "Sparl",
    "1provider",
    "my--provider",
    "myprovider-",
    "a",
    "x" * 64,
    "invalid name!",
    "with/slash",
]


def name_errors(model, name: str) -> list[ErrorDetails]:
    try:
        model(name=name)
    except ValidationError as exc:
        return [err for err in exc.errors() if err["loc"] == ("name",)]
    return []


@pytest.mark.parametrize("model", REQUEST_MODELS)
@pytest.mark.parametrize("name", INVALID_NAMES)
def test_rejects_names_the_entity_store_would_reject(model, name):
    assert name_errors(model, name), f"{model.__name__} accepted {name!r}"


@pytest.mark.parametrize("model", REQUEST_MODELS)
@pytest.mark.parametrize("name", ["my-provider-1", "ab", "llama-3.2-3b-instruct@v1.0.0+a100"])
def test_accepts_valid_names(model, name):
    assert not name_errors(model, name)


@pytest.mark.parametrize("model", REQUEST_MODELS)
def test_advertises_the_entity_store_pattern(model):
    schema = model.model_json_schema()["properties"]["name"]
    assert schema["pattern"] == constants.NAME_PATTERN
    assert schema["maxLength"] == constants.NAME_MAX_LENGTH
