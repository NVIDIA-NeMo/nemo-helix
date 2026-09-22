# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-schema input validation.

Model request bodies reject malformed input rather than letting it through:
- ``name`` fields must satisfy the entity store's NAME_PATTERN;
- unknown fields (e.g. a plugin-style ``executor`` on a model deployment) are
  rejected via ``extra="forbid"`` instead of silently ignored.
"""

import pytest
from nmp.common.entities import constants
from nmp.core.models.schemas import (
    ContainerExecutorConfig,
    CreateModelAdapterRequest,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    CreateModelEntityRequest,
    CreateModelProviderRequest,
    CreatePromptRequest,
    Engine,
    ModelDeploymentConfigModelSpec,
    ModelDeploymentStatus,
    UpdateModelDeploymentConfigRequest,
    UpdateModelDeploymentRequest,
    UpdateModelDeploymentStatusRequest,
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


# Deployment request schemas set extra="forbid", so a plugin-style param like
# `executor` sent to a model deployment endpoint must be rejected, not ignored.
_DEPLOYMENT_BODIES = {
    CreateModelDeploymentRequest: {"name": "dep-1", "config": "cfg"},
    UpdateModelDeploymentRequest: {"config": "cfg"},
    UpdateModelDeploymentStatusRequest: {"status": ModelDeploymentStatus.ERROR},
    CreateModelDeploymentConfigRequest: {
        "name": "cfg-1",
        "engine": Engine.GENERIC,
        "model_spec": ModelDeploymentConfigModelSpec(),
        "executor_config": ContainerExecutorConfig(gpu=0),
    },
    UpdateModelDeploymentConfigRequest: {
        "engine": Engine.GENERIC,
        "model_spec": ModelDeploymentConfigModelSpec(),
        "executor_config": ContainerExecutorConfig(gpu=0),
    },
}


@pytest.mark.parametrize("model", list(_DEPLOYMENT_BODIES))
def test_valid_deployment_body_constructs(model):
    assert model(**_DEPLOYMENT_BODIES[model])


@pytest.mark.parametrize("model", list(_DEPLOYMENT_BODIES))
def test_rejects_unknown_executor_field(model):
    with pytest.raises(ValidationError) as exc:
        model(**_DEPLOYMENT_BODIES[model], executor="openshell-local")
    assert ("executor",) in {err["loc"] for err in exc.value.errors()}
