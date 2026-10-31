# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request serialization and Fern SDK override contract for OpenAPI generation."""

import copy
import json
import shlex
from pathlib import Path

import pytest
import yaml
from nemo_helix_plugin.jobs.types import UpdateHelixJobRequest
from nhx.core.models.schemas import CreateModelProviderRequest, UpsertModelProviderRequest
from nhx.guardrails.entities.values.check import GuardrailCheckRequest
from nhx.intake.api.v2.experiments.schemas import EvaluationRequest

from script.openapi_helper.code_samples import SampleRequest, _curl, add_code_samples, render_samples, sample_request


@pytest.fixture(scope="module")
def spec():
    spec_path = Path(__file__).resolve().parents[2] / "openapi/openapi.yaml"
    return yaml.safe_load(spec_path.read_text())


def test_curl_serializes_headers_and_body_without_shell_expansion():
    request = SampleRequest(
        "POST",
        "http://localhost:8080/example?name=a&other=b",
        headers={"Authorization": "Bearer $TOKEN", "Content-Type": "application/json"},
        body=json.dumps({"text": "it's $HOME `whoami`\nnext line"}),
    )
    args = shlex.split(_curl(request).replace("\\\n", ""))
    assert args[:3] == ["curl", "--fail-with-body", "--request"]
    assert args[4] == request.url
    assert args[args.index("--data-binary") + 1] == request.body
    assert "Authorization: Bearer $TOKEN" in args


def test_default_curl_is_an_sdk_override_not_an_extra_custom_tab():
    spec = {"paths": {"/example": {"delete": {"responses": {"204": {}}}}}}
    operation = add_code_samples(spec)["paths"]["/example"]["delete"]
    samples = operation["x-fern-examples"][0]["code-samples"]
    assert samples == [{"sdk": "curl", "code": _curl(SampleRequest("DELETE", "http://localhost:8080/example"))}]
    assert all(sample["lang"] != "curl" for sample in operation["x-codeSamples"])
    assert add_code_samples(copy.deepcopy(spec)) == spec


def test_binary_response_does_not_use_a_json_decoder():
    operation = {"responses": {"200": {"content": {"application/octet-stream": {}}}}}
    request = sample_request({}, "/files/{path}", "get", operation)
    assert request is not None
    assert request.url.endswith("/files/example.txt")
    samples = {sample["lang"]: sample["source"] for sample in render_samples(request)}
    assert "response.content" in samples["Python"]
    assert "response.json()" not in samples["Python"]
    assert "response.arrayBuffer()" in samples["JavaScript"]
    assert "--output example.txt" in _curl(request)
    assert "BodyHandlers.ofByteArray()" in samples["Java"]
    assert "System.out.write(response.body())" in samples["Java"]
    assert "ReadAsStringAsync" not in samples["C#"]
    assert "CopyToAsync(output)" in samples["C#"]
    assert "String(decoding:" not in samples["Swift"]
    assert "FileHandle.standardOutput.write(data)" in samples["Swift"]


def test_binary_upload_has_nonempty_content_and_content_type():
    operation = {"requestBody": {"content": {"application/octet-stream": {}}, "required": True}}
    request = sample_request({}, "/files/{path}", "put", operation)
    assert request is not None
    assert request.body
    assert request.headers == {"Content-Type": "application/octet-stream"}
    assert "--data-binary" in _curl(request)


def test_generated_fileset_urls_use_the_normalized_path_parameter(spec):
    path = "/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}"
    for method in ("get", "put"):
        samples = spec["paths"][path][method]["x-codeSamples"]
        for sample in samples:
            assert "/-/example.txt" in sample["source"]
            assert "my--" not in sample["source"]


def test_atif_payload_validates_and_empty_201_is_not_decoded():
    from nhx.intake.spans.ingest.atif import AtifIngestRequest

    request = sample_request({}, "/apis/intake/v2/workspaces/{workspace}/ingest/atif", "post", {})
    assert request is not None
    AtifIngestRequest.model_validate_json(request.body)
    assert not request.reads_body
    python = next(sample["source"] for sample in render_samples(request) if sample["lang"] == "Python")
    assert "response.json()" not in python


@pytest.mark.parametrize(
    ("path", "method", "model"),
    [
        ("/apis/models/v2/workspaces/{workspace}/providers", "post", CreateModelProviderRequest),
        ("/apis/models/v2/workspaces/{workspace}/providers/{name}", "put", UpsertModelProviderRequest),
        ("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}", "patch", UpdateHelixJobRequest),
        ("/apis/guardrails/v2/workspaces/{workspace}/checks", "post", GuardrailCheckRequest),
    ],
)
def test_route_examples_pass_runtime_request_validation(path, method, model, spec):
    media = spec["paths"][path][method]["requestBody"]["content"]["application/json"]
    examples = media["examples"]
    assert examples
    for example in examples.values():
        model.model_validate(example["value"])


def test_provider_put_example_documents_replacement_and_preserves_credentials(spec):
    operation = spec["paths"]["/apis/models/v2/workspaces/{workspace}/providers/{name}"]["put"]
    body = operation["requestBody"]
    media = body["content"]["application/json"]
    assert "PUT replaces" in media["schema"]["description"]
    assert "preserve every configuration field" in media["schema"]["description"]
    example = media["examples"]["provider"]["value"]
    assert example["api_key_secret_name"] == "my-provider-api-key"


def test_evaluation_example_covers_create_and_put_without_requiring_patch_membership():
    example = EvaluationRequest.model_json_schema()["examples"][0]
    assert EvaluationRequest.model_validate(example).experiment_ids
    # PATCH intentionally keeps membership optional; it is not the PUT schema.
    from nhx.intake.api.v2.experiments.schemas import EvaluationPatchRequest

    assert EvaluationPatchRequest.model_validate({"description": "updated"}).experiment_ids is None


@pytest.mark.parametrize("status", ["200", "201"])
def test_bodyful_json_responses_are_left_to_fern(status):
    operation = {"responses": {status: {"content": {"application/json": {}}}}}
    assert sample_request({}, "/example", "post", operation) is None
