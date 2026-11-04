# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused request serialization, response handling, and source-example contracts."""

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from nemo_helix_plugin.jobs.types import UpdateHelixJobRequest
from nhx.core.models.schemas import CreateModelProviderRequest, UpsertModelProviderRequest
from nhx.guardrails.entities.values.check import GuardrailCheckRequest
from nhx.intake.api.v2.experiments.schemas import EvaluationRequest
from nhx.intake.spans.ingest.atif import AtifIngestRequest

from script.openapi_helper.code_samples import SampleRequest, _curl, add_code_samples, render_samples, sample_request


@pytest.fixture(scope="module")
def spec():
    return yaml.safe_load((Path(__file__).resolve().parents[2] / "openapi/openapi.yaml").read_text())


def test_curl_arguments_round_trip_through_shell_tokenization():
    request = SampleRequest(
        "POST",
        "http://localhost:8080/example?name=a&other=b",
        headers={"Authorization": "Bearer $TOKEN"},
        body=json.dumps({"text": "it's $HOME `whoami`\nnext line"}),
    )
    args = shlex.split(_curl(request).replace("\\\n", ""))
    assert args == [
        "curl",
        "--fail-with-body",
        "--request",
        "POST",
        request.url,
        "--header",
        "Authorization: Bearer $TOKEN",
        "--data-binary",
        request.body,
    ]


def test_language_policy_covers_authored_samples_and_default_curl(spec):
    docs = yaml.safe_load((Path(__file__).resolve().parents[2] / "docs/fern/docs.yml").read_text())
    assert set(docs["settings"]["http-snippets"]) == {"python", "javascript", "curl"}
    for item in spec["paths"].values():
        for operation in item.values():
            if isinstance(operation, dict):
                assert {s["lang"] for s in operation.get("x-codeSamples", [])} <= {"Python", "JavaScript"}
    operation = add_code_samples({"paths": {"/example": {"delete": {"responses": {"204": {}}}}}})["paths"]["/example"][
        "delete"
    ]
    assert {s["lang"] for s in operation["x-codeSamples"]} == {"Python", "JavaScript"}
    curl = operation["x-fern-examples"][0]["code-samples"][0]
    assert curl["sdk"] == "curl"  # A custom cURL tab does not override Fern's default.
    assert shlex.split(curl["code"].replace("\\\n", ""))[4] == "http://localhost:8080/example"


@pytest.mark.parametrize("binary", [False, True], ids=["empty-201", "binary-download"])
def test_python_response_handling_without_json_decode(binary, capsys):
    request = SampleRequest("GET", "http://localhost:8080/example", reads_body=binary, binary_response=binary)
    response = Mock(status_code=201, content=b"\x00\xff")
    response.json.side_effect = AssertionError("This response is not JSON")
    source = next(s["source"] for s in render_samples(request) if s["lang"] == "Python")
    with patch("requests.request", return_value=response) as send:
        exec(source, {})
    send.assert_called_once_with("GET", request.url)
    response.raise_for_status.assert_called_once()
    response.json.assert_not_called()
    assert capsys.readouterr().out == ("b'\\x00\\xff'\n" if binary else "201\n")


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript renderer contract needs Node")
@pytest.mark.parametrize("binary", [False, True], ids=["empty-201", "binary-download"])
def test_javascript_response_handling_without_json_decode(binary):
    request = SampleRequest("GET", "http://localhost:8080/example", reads_body=binary, binary_response=binary)
    source = next(s["source"] for s in render_samples(request) if s["lang"] == "JavaScript")
    setup = """import assert from "node:assert/strict";
const binary = BINARY;
globalThis.fetch = async (url, options) => {
  assert.equal(url, "http://localhost:8080/example");
  assert.equal(options.method, "GET");
  return {ok: true, status: 201,
    json() { throw new Error("This response is not JSON"); },
    async arrayBuffer() { return new Uint8Array([0, 255]).buffer; }};
};
console.log = value => assert.deepEqual(binary ? Array.from(new Uint8Array(value)) : value, binary ? [0, 255] : 201);
""".replace("BINARY", json.dumps(binary))
    subprocess.run(["node", "--input-type=module", "-e", setup + source], check=True, capture_output=True, text=True)


def test_fileset_requests_use_normalized_paths_and_nonempty_uploads(spec):
    path = "/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}"
    for method in ("get", "put"):
        operation = {**spec["paths"][path][method], "x-codeSamples": []}
        request = sample_request(spec, path, method, operation)
        assert request is not None
        assert request.url == "http://localhost:8080/apis/files/v2/workspaces/default/filesets/my-fileset/-/example.txt"
        if method == "put":
            assert request.body
            assert request.headers["Content-Type"] == "application/octet-stream"


@pytest.mark.parametrize(
    ("path", "method", "model"),
    [
        ("/apis/models/v2/workspaces/{workspace}/providers", "post", CreateModelProviderRequest),
        ("/apis/models/v2/workspaces/{workspace}/providers/{name}", "put", UpsertModelProviderRequest),
        ("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}", "patch", UpdateHelixJobRequest),
        ("/apis/guardrails/v2/workspaces/{workspace}/checks", "post", GuardrailCheckRequest),
    ],
)
def test_source_examples_pass_runtime_request_validation(path, method, model, spec):
    examples = spec["paths"][path][method]["requestBody"]["content"]["application/json"]["examples"]
    assert examples
    for example in examples.values():
        model.model_validate(example["value"])


def test_ingest_and_evaluation_examples_pass_runtime_request_validation():
    request = sample_request({}, "/apis/intake/v2/workspaces/{workspace}/ingest/atif", "post", {})
    assert request is not None
    AtifIngestRequest.model_validate_json(request.body)
    EvaluationRequest.model_validate(EvaluationRequest.model_json_schema()["examples"][0])
