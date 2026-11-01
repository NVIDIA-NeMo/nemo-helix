# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate ``x-codeSamples`` for operations whose Fern-generated samples cannot run as written.

Fern generates its request snippets from the operation alone. Three shapes come out wrong:

- A bodyless success response (204/205): every tab parses or prints the empty body, so a
  successful call raises (``response.json()`` on Python, ``await response.json()`` in JavaScript).
- A required form body: the snippet sends an empty form (``payload = ""``).
- Raw-request bodies not described by FastAPI, such as OTLP logs uploads.
- Explicit authenticate examples that demonstrate runtime token variables rather than
  embedding bearer placeholders. Their auth requirement comes from native OpenAPI metadata.

For these operations this module writes explicit Python and JavaScript samples and
an SDK override for Fern's default cURL tab. Binary downloads use bytes rather than a JSON decoder;
raw uploads have representative nonempty content. ATIF's empty 201 response is handled explicitly.
Operations that already carry ``x-codeSamples`` (authored on the route) are left alone, and
the output depends only on the spec, so regeneration is idempotent.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

BASE_URL = "http://localhost:8080"
LANGUAGES = ("Python", "JavaScript")
_BODYLESS_STATUSES = {"204", "205"}
_METHODS = ("get", "put", "post", "delete", "patch")

_OTLP_LOGS_BODY: dict[str, Any] = {
    "resourceLogs": [
        {
            "resource": {
                "attributes": [
                    {"key": "job", "value": {"stringValue": "my-job"}},
                    {"key": "job_attempt", "value": {"stringValue": "1"}},
                    {"key": "job_step", "value": {"stringValue": "train"}},
                    {"key": "job_task", "value": {"stringValue": "0"}},
                ]
            },
            "scopeLogs": [
                {
                    "logRecords": [
                        {
                            "timeUnixNano": "1767225600000000000",
                            "severityText": "INFO",
                            "body": {"stringValue": "Training started"},
                        }
                    ]
                }
            ],
        }
    ]
}


@dataclass(frozen=True)
class SampleRequest:
    """Everything a sample needs: one request and whether its success response has a body."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: str | None = None
    reads_body: bool = True
    binary_response: bool = False
    requires_bearer_token: bool = False

    def __post_init__(self) -> None:
        if self.requires_bearer_token and any(key.lower() == "authorization" for key in self.headers):
            raise ValueError("Use bearer-token semantics instead of a literal Authorization header")


# Requests needing a custom body or runtime-credential demonstration (see the module docstring).
_EXPLICIT: dict[tuple[str, str], dict[str, Any]] = {
    ("get", "/apis/auth/authenticate"): {},
    ("post", "/apis/auth/authenticate"): {},
    ("post", "/apis/intake/v2/workspaces/{workspace}/ingest/atif"): {
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "agent": {"name": "my-agent", "version": "1.0"},
                # Omitting the optional timestamp uses ingest time, not a stale literal.
                "steps": [{"step_id": 1, "source": "user", "message": "Hello!"}],
            }
        ),
        "reads_body": False,
    },
    ("post", "/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs"): {
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(_OTLP_LOGS_BODY, separators=(",", ":")),
    },
}


def _resolve(spec: dict, node: Any) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        target: Any = spec
        for part in node["$ref"].removeprefix("#/").split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        node = target
    return node


def _path_value(path: str, param: str) -> str:
    """A readable placeholder for a path parameter, named after the collection it indexes."""
    if param == "workspace":
        return "default"
    if param in {"path", "path:path"}:
        return "example.txt"
    segments = path.split("/")
    index = segments.index("{" + param + "}")
    collection = segments[index - 1] if index > 0 else param
    singular = collection[:-1] if collection.endswith("s") else collection
    return f"my-{singular}"


def _url(path: str) -> str:
    return BASE_URL + re.sub(r"\{([^}]+)\}", lambda m: _path_value(path, m.group(1)), path)


def _is_bodyless(operation: dict) -> bool:
    success = [str(code) for code in operation.get("responses", {}) if str(code).startswith("2")]
    return bool(success) and all(code in _BODYLESS_STATUSES for code in success)


def _required_form_body(spec: dict, operation: dict) -> str | None:
    """Encode the required fields of a required form body: an enum's first value, else a placeholder."""
    body = _resolve(spec, operation.get("requestBody", {}))
    if not body.get("required"):
        return None
    media = body.get("content", {}).get("application/x-www-form-urlencoded")
    if media is None:
        return None
    schema = _resolve(spec, media.get("schema", {}))
    fields = {}
    for name in schema.get("required", []):
        prop = _resolve(spec, schema.get("properties", {}).get(name, {}))
        fields[name] = prop["enum"][0] if prop.get("enum") else f"YOUR_{name.upper()}"
    return urlencode(fields)


def _requires_bearer_token(spec: dict, operation: dict) -> bool:
    """Require bearer credentials only when every documented auth alternative does."""
    schemes = spec.get("components", {}).get("securitySchemes", {})
    requirements = operation.get("security", [])
    return bool(requirements) and all(
        any(
            schemes.get(name, {}).get("type") == "http" and schemes[name].get("scheme", "").lower() == "bearer"
            for name in requirement
        )
        for requirement in requirements
    )


def sample_request(spec: dict, path: str, method: str, operation: dict) -> SampleRequest | None:
    """The request to document for one operation, or None when Fern's own samples are correct."""
    if operation.get("x-codeSamples"):
        return None
    bearer = _requires_bearer_token(spec, operation)
    explicit = _EXPLICIT.get((method, path))
    if explicit is not None:
        return SampleRequest(
            method.upper(),
            _url(path),
            requires_bearer_token=bearer,
            **{"reads_body": not _is_bodyless(operation), **explicit},
        )
    content = _resolve(spec, operation.get("requestBody", {})).get("content", {})
    if "application/octet-stream" in content:
        return SampleRequest(
            method.upper(),
            _url(path),
            headers={"Content-Type": "application/octet-stream"},
            body="Hello, Helix!\n",
            requires_bearer_token=bearer,
        )
    responses = [v for k, v in operation.get("responses", {}).items() if str(k).startswith("2")]
    if responses and all("application/octet-stream" in v.get("content", {}) for v in responses):
        return SampleRequest(method.upper(), _url(path), binary_response=True, requires_bearer_token=bearer)
    form = _required_form_body(spec, operation)
    if form is not None:
        return SampleRequest(
            method.upper(),
            _url(path),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=form,
            reads_body=not _is_bodyless(operation),
            requires_bearer_token=bearer,
        )
    if _is_bodyless(operation):
        return SampleRequest(method.upper(), _url(path), reads_body=False, requires_bearer_token=bearer)
    return None


def _q(value: str) -> str:
    """A double-quoted string literal valid in every target language (ASCII-only escapes)."""
    return json.dumps(value)


def _python(r: SampleRequest) -> str:
    lines = (["import os"] if r.requires_bearer_token else []) + ["import requests", "", f"url = {_q(r.url)}"]
    args = ["url"]
    headers = [f"{_q(k)}: {_q(v)}" for k, v in r.headers.items()]
    if r.requires_bearer_token:
        lines += [
            "# Set NHX_ACCESS_TOKEN to a raw access token before running.",
            'token = os.environ["NHX_ACCESS_TOKEN"]',
        ]
        headers.append('"Authorization": f"Bearer {token}"')
    if headers:
        lines.append("headers = {" + ", ".join(headers) + "}")
        args.append("headers=headers")
    if r.body is not None:
        lines.append(f"payload = {_q(r.body)}")
        args.append("data=payload")
    lines += ["", f"response = requests.request({_q(r.method)}, {', '.join(args)})", "response.raise_for_status()"]
    if r.binary_response:
        lines.append("print(response.content)  # raw file bytes, not JSON")
    else:
        lines.append("print(response.json())" if r.reads_body else "print(response.status_code)  # no response body")
    return "\n".join(lines) + "\n"


def _javascript(r: SampleRequest) -> str:
    lines = []
    headers = [f"{_q(k)}: {_q(v)}" for k, v in r.headers.items()]
    if r.requires_bearer_token:
        lines += [
            "// Supply `token` from your application's authentication flow before running.",
            "// Do not embed a real token in frontend source or build-time environment variables.",
        ]
        headers.append('"Authorization": `Bearer ${token}`')
    lines += [f"const response = await fetch({_q(r.url)}, {{", f"  method: {_q(r.method)},"]
    if headers:
        lines.append("  headers: {" + ", ".join(headers) + "},")
    if r.body is not None:
        lines.append(f"  body: {_q(r.body)},")
    lines += ["});", "if (!response.ok) throw new Error(`HTTP ${response.status}`);"]
    lines.append(
        "console.log(await response.arrayBuffer());"
        if r.binary_response
        else "console.log(await response.json());"
        if r.reads_body
        else "console.log(response.status); // no response body"
    )
    return "\n".join(lines) + "\n"


_RENDERERS = {"Python": _python, "JavaScript": _javascript}


def _curl(r: SampleRequest) -> str:
    args = ["curl --fail-with-body", "--request " + shlex.quote(r.method), shlex.quote(r.url)]
    args += ["--header " + shlex.quote(f"{k}: {v}") for k, v in r.headers.items()]
    guidance = ""
    if r.requires_bearer_token:
        guidance = ': "${NHX_ACCESS_TOKEN:?Set NHX_ACCESS_TOKEN to a raw access token}"\n'
        args.append('--header "Authorization: Bearer ${NHX_ACCESS_TOKEN}"')
    if r.body is not None:
        args.append("--data-binary " + shlex.quote(r.body))
    if r.binary_response:
        args.append("--output example.txt")
    return guidance + " \\\n  ".join(args) + "\n"


def render_samples(request: SampleRequest) -> list[dict[str, str]]:
    return [{"lang": lang, "source": _RENDERERS[lang](request)} for lang in LANGUAGES]


def add_code_samples(spec: dict) -> dict:
    """Attach generated ``x-codeSamples`` to every operation Fern would otherwise get wrong."""
    for path, item in spec.get("paths", {}).items():
        for method in _METHODS:
            operation = item.get(method)
            if not isinstance(operation, dict):
                continue
            request = sample_request(spec, path, method, operation)
            if request is not None:
                operation["x-codeSamples"] = render_samples(request)
                # Redocly x-codeSamples creates custom tabs. Only an SDK sample overrides
                # Fern's default cURL tab; another custom `curl` tab does not replace it.
                sdk_samples = [{"sdk": "curl", "code": _curl(request)}]
                if request.requires_bearer_token:
                    # With native auth, Fern's HTTP fallback can replace custom
                    # language tabs. SDK overrides preserve these runtime bindings.
                    sdk_samples += [
                        {"sdk": language.lower(), "code": _RENDERERS[language](request)} for language in LANGUAGES
                    ]
                operation.setdefault("x-fern-examples", []).append({"code-samples": sdk_samples})
    return spec
