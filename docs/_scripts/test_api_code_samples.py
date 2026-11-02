# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated API reference code samples must run as written."""

from pathlib import Path
from urllib.parse import parse_qs

import pytest
import yaml

from docs._scripts.lint_api_examples import lint_spec
from script.openapi_helper.code_samples import LANGUAGES, add_code_samples, sample_request

_SPEC_PATH = Path(__file__).resolve().parents[2] / "openapi/openapi.yaml"


@pytest.fixture(scope="module")
def spec() -> dict:
    return yaml.safe_load(_SPEC_PATH.read_text())


def _samples(spec: dict, path: str, method: str) -> dict[str, str]:
    return {sample["lang"]: sample["source"] for sample in spec["paths"][path][method]["x-codeSamples"]}


def _bodyless_operations(spec: dict):
    for path, item in spec["paths"].items():
        for method, operation in item.items():
            if not isinstance(operation, dict) or method == "head":
                continue
            success = [str(code) for code in operation.get("responses", {}) if str(code).startswith("2")]
            if success and all(code in {"204", "205"} for code in success):
                yield path, method, operation


def test_every_bodyless_operation_has_explicit_samples(spec):
    operations = list(_bodyless_operations(spec))
    assert len(operations) >= 14
    for path, method, operation in operations:
        assert operation.get("x-codeSamples"), f"{method.upper()} {path} has no explicit samples"


@pytest.mark.parametrize(
    "path",
    [
        "/apis/intake/v2/workspaces/{workspace}/evaluations/{name}",
        "/apis/intake/v2/workspaces/{workspace}/experiments/{name}",
        "/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}",
        "/apis/jobs/v2/workspaces/{workspace}/jobs/{name}",
    ],
)
def test_delete_samples_print_the_status_instead_of_parsing_the_body(spec, path):
    samples = _samples(spec, path, "delete")
    assert set(samples) == set(LANGUAGES)
    assert "print(response.status_code)" in samples["Python"]
    assert "response.json()" not in samples["Python"]
    assert "console.log(response.status)" in samples["JavaScript"]
    assert "/workspaces/default/" in samples["Python"]


def test_lint_accepts_the_generated_spec(spec):
    assert lint_spec(spec) == []


@pytest.mark.parametrize("method", ["get", "post"])
def test_authenticate_samples_send_the_bearer_token(spec, method):
    for lang, source in _samples(spec, "/apis/auth/authenticate", method).items():
        assert "Bearer YOUR_ACCESS_TOKEN" in source, lang


def test_token_exchange_samples_send_every_required_form_field(spec):
    operation = spec["paths"]["/apis/auth/token"]["post"]
    schema = operation["requestBody"]["content"]["application/x-www-form-urlencoded"]["schema"]
    request = sample_request(spec, "/apis/auth/token", "post", {**operation, "x-codeSamples": []})
    assert request is not None
    fields = parse_qs(request.body)
    assert set(fields) == set(schema["required"])
    assert fields["grant_type"] == [schema["properties"]["grant_type"]["enum"][0]]
    for lang, source in _samples(spec, "/apis/auth/token", "post").items():
        assert "application/x-www-form-urlencoded" in source, lang
        assert "grant_type=" in source, lang


def test_otlp_upload_samples_send_a_json_body_the_route_accepts(spec):
    from nhx.core.files.api.v2.otlp.schemas import OtelExportLogsServiceRequest, get_resource_attribute

    path = "/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs"
    request = sample_request(spec, path, "post", {**spec["paths"][path]["post"], "x-codeSamples": []})
    assert request is not None
    assert request.headers["Content-Type"] == "application/json"
    payload = OtelExportLogsServiceRequest.model_validate_json(request.body)
    resource_logs = payload.resourceLogs[0]
    for key in ("job", "job_attempt", "job_step", "job_task"):
        assert get_resource_attribute(resource_logs, key), key
    assert resource_logs.scopeLogs[0].logRecords
    for lang, source in _samples(spec, path, "post").items():
        assert "resourceLogs" in source, lang


def test_authored_samples_are_kept():
    authored = [{"lang": "Python", "source": "print('authored')"}]
    spec = {"paths": {"/x": {"delete": {"responses": {"204": {}}, "x-codeSamples": authored}}}}
    assert add_code_samples(spec)["paths"]["/x"]["delete"]["x-codeSamples"] == authored


def test_operations_fern_handles_are_left_alone():
    spec = {"paths": {"/x": {"get": {"responses": {"200": {"content": {"application/json": {}}}}}}}}
    assert "x-codeSamples" not in add_code_samples(spec)["paths"]["/x"]["get"]


def test_generation_is_idempotent(spec):
    import copy

    once = add_code_samples(copy.deepcopy(spec))
    assert add_code_samples(copy.deepcopy(once)) == once
