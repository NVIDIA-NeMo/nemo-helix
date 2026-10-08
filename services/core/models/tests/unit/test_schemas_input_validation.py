# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-schema input validation.

Model request bodies reject malformed input rather than letting it through:
- ``name`` fields must satisfy the entity store's NAME_PATTERN;
- unknown fields (e.g. a plugin-style ``executor`` on a model deployment) are
  rejected via ``extra="forbid"`` instead of silently ignored.
"""

import pytest
from nhx.common.entities import constants
from nhx.core.models.constants import FILESET_REF_MAX_LEN, FILESET_REF_MIN_LEN, FILESET_REF_PATTERN
from nhx.core.models.schemas import (
    AdapterEntityFilter,
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
    ModelEntity,
    ModelEntityFilter,
    UpdateAdapterRequest,
    UpdateModelDeploymentConfigRequest,
    UpdateModelDeploymentRequest,
    UpdateModelDeploymentStatusRequest,
    UpdateModelEntityRequest,
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


FILESET_MODELS = [
    CreateModelEntityRequest,
    UpdateModelEntityRequest,
    CreateModelAdapterRequest,
    UpdateAdapterRequest,
    ModelEntity,
]

VALID_FILESETS = ["ab", "my-fileset", "default/my-fileset", "llama-3.2-3b@v1"]
INVALID_FILESETS = [
    "a",
    "A",
    "has space",
    "ws/name/extra",
    "fileset://default/my-fileset",
    "https://huggingface.co/meta/llama",
    "ab/",
    "/ab",
    "a" * 64,
    "a" * 63 + "/" + "b" * 64,
]


def build_with_fileset(model, fileset: str):
    """Construct *model* with the minimum required fields plus *fileset*."""
    kwargs: dict = {"fileset": fileset}
    if model in (CreateModelEntityRequest, CreateModelAdapterRequest):
        kwargs["name"] = "my-model"
    if model is CreateModelAdapterRequest:
        kwargs["finetuning_type"] = "lora"
    if model is ModelEntity:
        kwargs.update(
            id="id",
            name="my-model",
            workspace="default",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
    return model(**kwargs)


def fileset_string_schema(model):
    schema = model.model_json_schema()["properties"]["fileset"]
    if "anyOf" in schema:
        return next(option for option in schema["anyOf"] if option.get("type") == "string")
    return schema


@pytest.mark.parametrize("model", FILESET_MODELS)
@pytest.mark.parametrize("fileset", VALID_FILESETS)
def test_accepts_bare_and_qualified_fileset_refs(model, fileset):
    assert build_with_fileset(model, fileset).fileset == fileset


@pytest.mark.parametrize("model", FILESET_MODELS)
@pytest.mark.parametrize("fileset", INVALID_FILESETS)
def test_rejects_malformed_fileset_refs(model, fileset):
    with pytest.raises(ValidationError) as exc:
        build_with_fileset(model, fileset)
    assert ("fileset",) in {err["loc"] for err in exc.value.errors()}


@pytest.mark.parametrize("model", FILESET_MODELS)
def test_fileset_schema_advertises_length_and_pattern(model):
    schema = fileset_string_schema(model)
    assert schema["pattern"] == FILESET_REF_PATTERN
    assert schema["minLength"] == FILESET_REF_MIN_LEN
    assert schema["maxLength"] == FILESET_REF_MAX_LEN


def test_fileset_filters_accept_bool_or_valid_ref():
    assert ModelEntityFilter(fileset=True).fileset is True
    assert ModelEntityFilter(fileset=False).fileset is False
    assert ModelEntityFilter(fileset="default/my-fileset").fileset == "default/my-fileset"
    assert AdapterEntityFilter(fileset="my-fileset").fileset == "my-fileset"
    with pytest.raises(ValidationError):
        ModelEntityFilter(fileset="not a ref")
    with pytest.raises(ValidationError):
        AdapterEntityFilter(fileset="ws/name/extra")
