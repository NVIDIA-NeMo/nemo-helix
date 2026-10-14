# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for published API request examples."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nhx.core.auth.oidc_broker.routes import BrokerTokenRequest
from nhx.core.models.schemas import CreateModelDeploymentConfigRequest, UpdateModelDeploymentConfigRequest
from nhx.intake.spans.api.annotations_schemas import AnnotationInput
from nhx.intake.spans.api.evaluator_results_schemas import EvaluatorResultInput
from pydantic import TypeAdapter

from docs._scripts.lint_api_examples import lint_spec

_SPEC_PATH = Path(__file__).resolve().parents[2] / "openapi/openapi.yaml"


def example_spec(value, schema=None):
    return {
        "openapi": "3.1.0",
        "components": {
            "schemas": {
                "Input": schema or {"type": "object", "required": ["spec"], "properties": {"spec": {"type": "object"}}}
            }
        },
        "paths": {
            "/test": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Input"},
                                "examples": {"test": {"value": value}},
                            }
                        }
                    },
                    "responses": {"204": {"description": "No content"}},
                }
            }
        },
    }


def test_required_empty_object_is_rejected_even_when_schema_valid():
    assert "Fern may omit" in " ".join(lint_spec(example_spec({"spec": {}})))
    assert lint_spec(example_spec({"spec": {"message": "hello"}})) == []


def test_optional_empty_object_is_allowed():
    assert lint_spec(example_spec({"spec": {}}, {"type": "object"})) == []


@pytest.mark.parametrize("value", [{}, {"spec": "wrong type"}])
def test_schema_errors_are_reported(value):
    assert lint_spec(example_spec(value))


def test_discriminator_enum_is_checked():
    schema = {"oneOf": [{"type": "object", "required": ["kind"], "properties": {"kind": {"const": "note"}}}]}
    assert lint_spec(example_spec({"kind": "string"}, schema))
    assert lint_spec(example_spec({"kind": "note"}, schema)) == []


@pytest.mark.parametrize("method,status", [("post", "204"), ("post", "205"), ("head", "200")])
@pytest.mark.parametrize(
    "source",
    [
        "print(response.json())",
        "await response.json()",
        "JSON.parse(responseText)",
        "await response.text()",
        "await response.arrayBuffer()",
        "await response.blob()",
        "io.ReadAll(res.Body)",
        "json.NewDecoder(res.Body).Decode(&value)",
        "res.read_body",
        "$response->getBody()",
    ],
)
def test_bodyless_response_samples_do_not_parse_json(method, status, source):
    spec = example_spec({"spec": {"message": "hello"}})
    operation = spec["paths"]["/test"].pop("post")
    spec["paths"]["/test"][method] = operation
    operation["responses"] = {status: {"description": "Success"}}
    operation["x-codeSamples"] = [{"lang": "Python", "source": source}]
    assert "bodyless response" in " ".join(lint_spec(spec))
    operation["x-codeSamples"][0]["source"] = "print(response.status_code)\ndefer res.Body.Close()"
    assert lint_spec(spec) == []


def test_parameter_and_response_examples_are_validated():
    spec = example_spec({"spec": {"message": "hello"}})
    operation = spec["paths"]["/test"]["post"]
    operation["parameters"] = [{"name": "kind", "schema": {"enum": ["note"]}, "example": "string"}]
    operation["responses"]["200"] = {
        "content": {"application/json": {"schema": {"type": "integer"}, "example": "string"}}
    }
    errors = lint_spec(spec)
    assert len(errors) == 2
    assert any("parameter" in error for error in errors)
    assert any("response 200" in error for error in errors)


def test_published_operation_examples():
    assert lint_spec(yaml.safe_load(_SPEC_PATH.read_text())) == []


@pytest.mark.parametrize(
    "path,method,model",
    [
        ("/apis/jobs/v2/workspaces/{workspace}/jobs", "post", CreateHelixJobRequest),
        ("/apis/models/v2/workspaces/{workspace}/deployment-configs", "post", CreateModelDeploymentConfigRequest),
        (
            "/apis/models/v2/workspaces/{workspace}/deployment-configs/{name}",
            "post",
            UpdateModelDeploymentConfigRequest,
        ),
        ("/apis/intake/v2/workspaces/{workspace}/annotations", "post", AnnotationInput),
        ("/apis/intake/v2/workspaces/{workspace}/evaluator-results", "post", EvaluatorResultInput),
        ("/apis/auth/v2/token", "post", BrokerTokenRequest),
    ],
)
def test_regression_examples_validate_with_runtime_models(path, method, model):
    spec = yaml.safe_load(_SPEC_PATH.read_text())
    examples = spec["paths"][path][method]["requestBody"]["content"]["application/json"]["examples"]
    assert examples
    for example in examples.values():
        TypeAdapter(model).validate_python(example["value"])


def test_job_example_runs_a_container_command():
    spec = yaml.safe_load(_SPEC_PATH.read_text())
    example = spec["paths"]["/apis/jobs/v2/workspaces/{workspace}/jobs"]["post"]["requestBody"]["content"][
        "application/json"
    ]["examples"]["job"]["value"]
    step = example["platform_spec"]["steps"][0]
    assert step["executor"]["container"]["command"] == ["python", "-c", "print('Hello from NeMo Helix')"]
    assert "command" not in step["config"]


def test_cyclic_operation_reference_fails_without_hanging():
    spec = example_spec({"spec": {"message": "hello"}})
    spec["paths"]["/test"]["post"]["requestBody"] = {"$ref": "#/components/requestBodies/Input"}
    spec["components"]["requestBodies"] = {"Input": {"$ref": "#/components/requestBodies/Input"}}
    with pytest.raises(ValueError, match="Cyclic OpenAPI reference"):
        lint_spec(spec)


def test_numeric_example_requires_value_at_runtime():
    spec = yaml.safe_load(_SPEC_PATH.read_text())
    example = deepcopy(
        spec["paths"]["/apis/intake/v2/workspaces/{workspace}/evaluator-results"]["post"]["requestBody"]["content"][
            "application/json"
        ]["examples"]["numeric"]["value"]
    )
    del example["value"]
    with pytest.raises(ValueError, match="value.*required"):
        EvaluatorResultInput.model_validate(example)


def test_logout_samples_are_explicit_and_bodyless():
    spec = yaml.safe_load(_SPEC_PATH.read_text())
    operation = spec["paths"]["/apis/auth/v2/logout"]["post"]
    assert "content" not in operation["responses"]["204"]
    assert {sample["lang"] for sample in operation["x-codeSamples"]} == {"Python", "JavaScript"}
    assert all(".json(" not in sample["source"] for sample in operation["x-codeSamples"])
