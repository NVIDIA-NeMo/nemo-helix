# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native auth metadata and executable demo-credential renderer contracts."""

import builtins
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml
from fastapi.testclient import TestClient
from nemo_helix_plugin.jobs import openapi_utils as plugin_openapi
from nhx.common.api.utils import QueryParamSchemaFastAPI
from nhx.common.auth.openapi import (
    BEARER_AUTH_OPENAPI,
    BEARER_SCHEME_NAME,
    BEARER_SECURITY_SCHEME,
    register_bearer_security_scheme,
)
from nhx.common.service.base import _ServiceFastAPI
from pydantic import BaseModel

from script.openapi_helper.code_samples import sample_request
from script.openapi_helper.openapi_tools import merge_specs, validate_refs


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


class _NestedFilter(BaseModel):
    gt: int | None = None


class _Filter(BaseModel):
    value: _NestedFilter | None = None


@pytest.fixture(params=["platform", "service"])
def metadata_app(request, monkeypatch):
    # Restore the registry afterward without disturbing real service registrations.
    monkeypatch.setattr(plugin_openapi, "_query_param_schemas", {})
    if request.param == "platform":
        return QueryParamSchemaFastAPI()
    return _ServiceFastAPI(service_title="Test", service_version="1", service_openapi_tags=[])


def test_raw_openapi_is_lazy_cached_complete_and_preserves_query_schemas(metadata_app):
    @metadata_app.get(
        "/secured",
        openapi_extra={**BEARER_AUTH_OPENAPI, **plugin_openapi.generate_openapi_extra_params(filter_schema=_Filter)},
    )
    def endpoint():
        return {"ok": True}

    assert metadata_app.openapi_schema is None
    # Documentation metadata must not add runtime authentication dependencies.
    assert TestClient(metadata_app).get("/secured").json() == {"ok": True}
    assert metadata_app.openapi_schema is None
    spec = metadata_app.openapi()
    with (
        patch("nhx.common.auth.openapi.register_bearer_security_scheme", side_effect=AssertionError("cache bypassed")),
        patch("nhx.common.service.base.register_bearer_security_scheme", side_effect=AssertionError("cache bypassed")),
    ):
        assert metadata_app.openapi() is spec
    assert spec["components"]["securitySchemes"][BEARER_SCHEME_NAME] == BEARER_SECURITY_SCHEME
    assert "_Filter" in spec["components"]["schemas"]
    assert "_NestedFilter" in spec["components"]["schemas"]
    assert validate_refs(spec) == []
    assert "security" not in spec

    public = {"paths": {"/public": {"get": {"responses": {"200": {}}}}}}
    merged = merge_specs([(public, "public.yaml"), (spec, "secured.yaml")])
    schemes = merged["components"]["securitySchemes"]
    assert set(merged["paths"]["/secured"]["get"]["security"][0]) <= set(schemes)
    assert "security" not in merged["paths"]["/public"]["get"]


def test_unsecured_raw_openapi_does_not_advertise_bearer(metadata_app):
    @metadata_app.get("/public")
    def endpoint():
        return {"ok": True}

    spec = metadata_app.openapi()
    assert "securitySchemes" not in spec.get("components", {})
    assert "security" not in spec
    assert "security" not in spec["paths"]["/public"]["get"]


def test_registration_failure_clears_cache_and_can_retry(metadata_app):
    @metadata_app.get("/secured", openapi_extra=BEARER_AUTH_OPENAPI)
    def endpoint():
        return {"ok": True}

    # Simulate registration failure after FastAPI itself has generated a schema.
    with patch("nhx.common.auth.openapi.deepcopy", side_effect=ValueError("registration failed")):
        with pytest.raises(ValueError, match="registration failed"):
            metadata_app.openapi()
    assert metadata_app.openapi_schema is None
    assert metadata_app.openapi()["components"]["securitySchemes"][BEARER_SCHEME_NAME] == BEARER_SECURITY_SCHEME


def test_registration_detects_conflicts_and_does_not_alias_the_definition():
    spec = {"paths": {"/secured": {"get": BEARER_AUTH_OPENAPI}}}
    register_bearer_security_scheme(spec)
    exported = spec["components"]["securitySchemes"][BEARER_SCHEME_NAME]
    exported["x-fern-bearer"]["env"] = "MUTATED"
    assert BEARER_SECURITY_SCHEME["x-fern-bearer"]["env"] == "NHX_ACCESS_TOKEN"
    with pytest.raises(ValueError, match="Conflicting security scheme"):
        register_bearer_security_scheme(spec)


def _exec_with_fake_requests(source: str, request: Mock) -> None:
    """Run a Python sample whose ``import requests`` resolves to a fake, inside the sample only."""
    real_import = builtins.__import__

    def sample_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "requests":
            return SimpleNamespace(request=request)
        return real_import(name, globals, locals, fromlist, level)

    exec(source, {"__builtins__": {**vars(builtins), "__import__": sample_import}})


def test_python_uses_runtime_demo_token_and_fails_before_network_when_missing(bearer_request, bearer_samples):
    source = bearer_samples["python"]
    response = Mock()
    send = Mock(return_value=response)
    with patch.dict(os.environ, {"NHX_ACCESS_TOKEN": "demo-token"}):
        _exec_with_fake_requests(source, send)
    send.assert_called_once_with("GET", bearer_request.url, headers={"Authorization": "Bearer demo-token"})
    send = Mock()
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError, match="NHX_ACCESS_TOKEN"):
            _exec_with_fake_requests(source, send)
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
