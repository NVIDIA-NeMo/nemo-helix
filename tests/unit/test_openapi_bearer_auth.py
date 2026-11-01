# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native auth metadata and executable demo-credential renderer contracts."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.auth.openapi import BEARER_AUTH_OPENAPI, BEARER_SCHEME_NAME, register_bearer_security_scheme

from script.openapi_helper.code_samples import sample_request
from script.openapi_helper.openapi_tools import merge_specs


@pytest.fixture(scope="module")
def spec():
    return yaml.safe_load((Path(__file__).resolve().parents[2] / "openapi/openapi.yaml").read_text())


@pytest.fixture
def bearer_request(spec):
    path = "/apis/auth/authenticate"
    operation = {**spec["paths"][path]["get"], "x-codeSamples": []}
    request = sample_request(spec, path, "get", operation)
    assert request is not None and request.requires_bearer_token
    return request


@pytest.fixture
def bearer_samples(spec):
    operation = spec["paths"]["/apis/auth/authenticate"]["get"]
    samples = {
        sample["sdk"]: sample["code"] for example in operation["x-fern-examples"] for sample in example["code-samples"]
    }
    assert set(samples) == {"curl", "python", "javascript"}
    return samples


def test_native_security_is_scoped_and_does_not_duplicate_authorization_parameters(spec):
    secured = set()
    for path, item in spec["paths"].items():
        for method, operation in item.items():
            if isinstance(operation, dict) and operation.get("security"):
                secured.add((path, method))
                assert operation["security"] == BEARER_AUTH_OPENAPI["security"]
                assert not any(p.get("name", "").lower() == "authorization" for p in operation.get("parameters", []))
    assert secured == {
        ("/apis/auth/authenticate", "get"),
        ("/apis/auth/authenticate", "post"),
        ("/apis/auth/v2/access-keys", "get"),
        ("/apis/auth/v2/access-keys", "post"),
        ("/apis/auth/v2/access-keys/{jti}", "delete"),
        *((f"/apis/auth/v2/access-keys/{{jti}}/{action}", "post") for action in ("suspend", "unsuspend", "rotate")),
    }
    assert "security" not in spec
    scheme = spec["components"]["securitySchemes"][BEARER_SCHEME_NAME]
    assert (scheme["type"], scheme["scheme"]) == ("http", "bearer")
    token = sample_request(
        spec, "/apis/auth/token", "post", {**spec["paths"]["/apis/auth/token"]["post"], "x-codeSamples": []}
    )
    assert token is not None and not token.requires_bearer_token
    assert token.headers == {"Content-Type": "application/x-www-form-urlencoded"}
    assert token.body is not None and "subject_token=" in token.body


def test_native_component_survives_merging_without_runtime_auth_dependencies():
    app = FastAPI()

    @app.get("/secured", openapi_extra=BEARER_AUTH_OPENAPI)
    def endpoint():
        return {"ok": True}

    # Metadata alone must not install a runtime guard in this prototype.
    assert TestClient(app).get("/secured").json() == {"ok": True}
    secured = register_bearer_security_scheme(app.openapi())
    public = {"paths": {"/public": {"get": {"responses": {"200": {}}}}}}
    merged = merge_specs([(public, "public.yaml"), (secured, "secured.yaml")])
    assert BEARER_SCHEME_NAME in merged["components"]["securitySchemes"]
    assert merged["paths"]["/secured"]["get"]["security"] == BEARER_AUTH_OPENAPI["security"]
    assert "security" not in merged["paths"]["/public"]["get"]


def test_python_uses_runtime_demo_token_and_fails_before_network_when_missing(bearer_request, bearer_samples):
    source = bearer_samples["python"]
    response = Mock()
    with (
        patch.dict(os.environ, {"NHX_ACCESS_TOKEN": "demo-token"}),
        patch("requests.request", return_value=response) as send,
    ):
        exec(source, {})
    send.assert_called_once_with("GET", bearer_request.url, headers={"Authorization": "Bearer demo-token"})
    with patch.dict(os.environ, {}, clear=True), patch("requests.request") as send:
        with pytest.raises(KeyError, match="NHX_ACCESS_TOKEN"):
            exec(source, {})
    send.assert_not_called()


@pytest.mark.parametrize("token", [None, "demo token with spaces and $literal `text`"])
def test_curl_expands_one_quoted_credential_argument_and_guards_missing_token(
    bearer_request, bearer_samples, tmp_path, token
):
    capture = tmp_path / "argv"
    script = 'curl() { printf "%s\\0" "$@" > "$CAPTURE"; }\n' + bearer_samples["curl"]
    env = {"PATH": os.defpath, "CAPTURE": str(capture)}
    if token is not None:
        env["NHX_ACCESS_TOKEN"] = token
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=10)
    if token is None:
        assert result.returncode != 0 and not capture.exists()
        assert "Set NHX_ACCESS_TOKEN" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        args = capture.read_bytes().decode().split("\0")[:-1]
        assert args == [
            "--fail-with-body",
            "--request",
            "GET",
            bearer_request.url,
            "--header",
            f"Authorization: Bearer {token}",
        ]


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript credential contract needs Node")
def test_browser_neutral_javascript_uses_the_application_supplied_demo_token(bearer_request, bearer_samples):
    source = bearer_samples["javascript"]
    setup = """import assert from "node:assert/strict";
const token = "demo-token";
globalThis.fetch = async (url, options) => {
  assert.equal(url, URL);
  assert.deepEqual(options.headers, {Authorization: "Bearer demo-token"});
  return {ok: true, async json() { return {}; }};
};
console.log = () => {};
""".replace("URL", json.dumps(bearer_request.url))
    subprocess.run(
        ["node", "--input-type=module", "-e", setup + source], check=True, capture_output=True, text=True, timeout=10
    )
