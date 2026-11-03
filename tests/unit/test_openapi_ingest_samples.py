# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute explicit Intake samples with mocked transports and validate runtime payloads."""

import base64
import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest
from nhx.intake.api.v2.experiments.schemas import EvaluationPatchRequest
from nhx.intake.spans.ingest.otlp import _parse_export_request, _span_to_domain
from nhx.intake.spans.ingest.spans import DirectSpansIngestRequest

from script.openapi_helper.code_samples import _curl, render_samples, sample_request

CASES = [
    ("post", "/apis/intake/v2/workspaces/{workspace}/ingest/spans", False),
    ("post", "/apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/traces", True),
    ("patch", "/apis/intake/v2/workspaces/{workspace}/evaluations/{name}", True),
]


def _validate_payload(path, body):
    if path.endswith("/ingest/spans"):
        request = DirectSpansIngestRequest.model_validate_json(body)
        assert request.source
        assert len(request.spans) == 1
        assert abs(datetime.now(timezone.utc) - request.spans[0].started_at) < timedelta(seconds=30)
        assert request.spans[0].ended_at is None
    elif path.endswith("/traces"):
        request = _parse_export_request(body)
        assert len(request.resource_spans) == 1
        spans = request.resource_spans[0].scope_spans[0].spans
        assert len(spans) == 1
        span = spans[0]
        assert len(span.trace_id) == 16 and any(span.trace_id)
        assert len(span.span_id) == 8 and any(span.span_id)
        assert span.name
        assert span.start_time_unix_nano == span.end_time_unix_nano == 0
        now = datetime.now(timezone.utc)
        converted = _span_to_domain(
            workspace="default", span=span, resource_attributes={}, scope_data={}, ingested_at=now
        )
        assert converted.start_time == now
        assert converted.end_time is None
    else:
        request = EvaluationPatchRequest.model_validate_json(body)
        assert request.model_fields_set == {"description"}
        assert request.description


@pytest.mark.parametrize(("method", "path", "reads_body"), CASES)
@pytest.mark.parametrize("language", ["Python", "JavaScript", "curl"])
def test_intake_samples_send_runtime_valid_payloads(method, path, reads_body, language):
    request = sample_request({}, path, method, {"responses": {"201" if not reads_body else "200": {}}})
    assert request is not None
    assert request.reads_body is reads_body
    assert request.method == method.upper()
    assert "/workspaces/default/" in request.url
    assert request.headers["Content-Type"] == (
        "application/x-protobuf" if path.endswith("/traces") else "application/json"
    )
    assert request == sample_request({}, path, method, {"responses": {"201" if not reads_body else "200": {}}})

    if language == "Python":
        source = next(s["source"] for s in render_samples(request) if s["lang"] == language)
        response = Mock(status_code=201)
        if not reads_body:
            response.json.side_effect = AssertionError("Empty 201 response must not be decoded")
        with patch("requests.request", return_value=response) as send:
            exec(source, {})
        assert send.call_args.args == (method.upper(), request.url)
        assert send.call_args.kwargs["headers"] == request.headers
        body = send.call_args.kwargs["data"]
        response.raise_for_status.assert_called_once()
        assert response.json.call_count == int(reads_body)
    elif language == "JavaScript":
        if shutil.which("node") is None:
            pytest.skip("JavaScript sample execution needs Node")
        source = next(s["source"] for s in render_samples(request) if s["lang"] == language)
        setup = f"""import assert from 'node:assert/strict';
const readsBody = {json.dumps(reads_body)};
globalThis.fetch = async (url, options) => {{
  assert.equal(url, {json.dumps(request.url)});
  assert.equal(options.method, {json.dumps(method.upper())});
  assert.deepEqual(options.headers, {json.dumps(request.headers)});
  process.stdout.write(Buffer.from(options.body).toString('base64'));
  return {{ok: true, status: 201, async json() {{
    assert.ok(readsBody, 'Empty 201 response must not be decoded');
    return {{}};
  }}}};
}};
console.log = () => {{}};
"""
        result = subprocess.run(["node", "--input-type=module", "-e", setup + source], check=True, capture_output=True)
        body = base64.b64decode(result.stdout)
    else:
        # Capture curl's actual body, including stdin from the protobuf decoding pipeline.
        setup = """curl() {
  while [ "$#" -gt 0 ]; do
    if [ "$1" = --data-binary ]; then
      shift
      if [ "$1" = @- ]; then cat; else printf %s "$1"; fi
      return
    fi
    shift
  done
  return 1
}
"""
        result = subprocess.run(["sh", "-c", setup + _curl(request)], check=True, capture_output=True)
        body = result.stdout
    _validate_payload(path, body)


@pytest.mark.parametrize(("method", "path", "reads_body"), CASES)
def test_intake_samples_preserve_authored_overrides(method, path, reads_body):
    assert sample_request({}, path, method, {"x-codeSamples": [{"lang": "Python", "source": "authored"}]}) is None
