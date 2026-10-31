# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate ``x-codeSamples`` for operations whose Fern-generated samples cannot run as written.

Fern generates its request snippets from the operation alone. Three shapes come out wrong:

- A bodyless success response (204/205): every tab parses or prints the empty body, so a
  successful call raises (``response.json()`` on Python, ``await response.json()`` in JavaScript).
- A required form body: the snippet sends an empty form (``payload = ""``).
- Credentials or bodies the route reads from the raw request, which FastAPI cannot describe:
  ``/apis/auth/authenticate`` needs an ``Authorization`` header and the OTLP logs upload needs a
  JSON body, but neither appears in the spec, so the snippets send neither.

For these operations this module writes explicit samples in all eight custom language tabs and
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
LANGUAGES = ("Python", "JavaScript", "Go", "Ruby", "PHP", "Java", "C#", "Swift")
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


# Requests whose headers or body the spec cannot express (see the module docstring).
_EXPLICIT: dict[tuple[str, str], dict[str, Any]] = {
    ("get", "/apis/auth/authenticate"): {"headers": {"Authorization": "Bearer YOUR_ACCESS_TOKEN"}},
    ("post", "/apis/auth/authenticate"): {"headers": {"Authorization": "Bearer YOUR_ACCESS_TOKEN"}},
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


def sample_request(spec: dict, path: str, method: str, operation: dict) -> SampleRequest | None:
    """The request to document for one operation, or None when Fern's own samples are correct."""
    if operation.get("x-codeSamples"):
        return None
    explicit = _EXPLICIT.get((method, path))
    if explicit is not None:
        return SampleRequest(method.upper(), _url(path), **{"reads_body": not _is_bodyless(operation), **explicit})
    content = _resolve(spec, operation.get("requestBody", {})).get("content", {})
    if "application/octet-stream" in content:
        return SampleRequest(
            method.upper(), _url(path), headers={"Content-Type": "application/octet-stream"}, body="Hello, Helix!\n"
        )
    responses = [v for k, v in operation.get("responses", {}).items() if str(k).startswith("2")]
    if responses and all("application/octet-stream" in v.get("content", {}) for v in responses):
        return SampleRequest(method.upper(), _url(path), binary_response=True)
    form = _required_form_body(spec, operation)
    if form is not None:
        return SampleRequest(
            method.upper(),
            _url(path),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=form,
            reads_body=not _is_bodyless(operation),
        )
    if _is_bodyless(operation):
        return SampleRequest(method.upper(), _url(path), reads_body=False)
    return None


def _q(value: str) -> str:
    """A double-quoted string literal valid in every target language (ASCII-only escapes)."""
    return json.dumps(value)


def _python(r: SampleRequest) -> str:
    lines = ["import requests", "", f"url = {_q(r.url)}"]
    args = ["url"]
    if r.headers:
        lines.append("headers = {" + ", ".join(f"{_q(k)}: {_q(v)}" for k, v in r.headers.items()) + "}")
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
    lines = [f"const response = await fetch({_q(r.url)}, {{", f"  method: {_q(r.method)},"]
    if r.headers:
        lines.append("  headers: {" + ", ".join(f"{_q(k)}: {_q(v)}" for k, v in r.headers.items()) + "},")
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


def _go(r: SampleRequest) -> str:
    imports = ['"fmt"', '"net/http"'] + (['"io"'] if r.reads_body else []) + (['"strings"'] if r.body else [])
    body = f"strings.NewReader({_q(r.body)})" if r.body is not None else "nil"
    lines = ["package main", "", "import ("] + [f"\t{i}" for i in sorted(imports)] + [")", "", "func main() {"]
    lines += [
        f"\treq, err := http.NewRequest({_q(r.method)}, {_q(r.url)}, {body})",
        "\tif err != nil {",
        "\t\tpanic(err)",
        "\t}",
    ]
    lines += [f"\treq.Header.Set({_q(k)}, {_q(v)})" for k, v in r.headers.items()]
    lines += [
        "\tres, err := http.DefaultClient.Do(req)",
        "\tif err != nil {",
        "\t\tpanic(err)",
        "\t}",
        "\tdefer res.Body.Close()",
        "\tfmt.Println(res.Status)",
    ]
    if r.reads_body:
        lines += [
            "\tdata, err := io.ReadAll(res.Body)",
            "\tif err != nil {",
            "\t\tpanic(err)",
            "\t}",
            "\tfmt.Println(string(data))",
        ]
    return "\n".join(lines + ["}"]) + "\n"


def _ruby(r: SampleRequest) -> str:
    cls = r.method.capitalize()
    lines = ['require "net/http"', "", f"uri = URI({_q(r.url)})", f"request = Net::HTTP::{cls}.new(uri)"]
    lines += [f"request[{_q(k)}] = {_q(v)}" for k, v in r.headers.items()]
    if r.body is not None:
        lines.append(f"request.body = {_q(r.body)}")
    lines += [
        'response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: uri.scheme == "https") do |http|',
        "  http.request(request)",
        "end",
        "",
        "puts response.code",
    ]
    if r.reads_body:
        lines.append("puts response.body")
    return "\n".join(lines) + "\n"


def _php(r: SampleRequest) -> str:
    lines = ["<?php", f"$ch = curl_init({_q(r.url)});", f"curl_setopt($ch, CURLOPT_CUSTOMREQUEST, {_q(r.method)});"]
    if r.headers:
        headers = ", ".join(_q(f"{k}: {v}") for k, v in r.headers.items())
        lines.append(f"curl_setopt($ch, CURLOPT_HTTPHEADER, [{headers}]);")
    if r.body is not None:
        lines.append(f"curl_setopt($ch, CURLOPT_POSTFIELDS, {_q(r.body)});")
    lines.append("curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);")
    lines.append("$body = curl_exec($ch);" if r.reads_body else "curl_exec($ch);")
    lines += ["$status = curl_getinfo($ch, CURLINFO_RESPONSE_CODE);", "curl_close($ch);", "", "echo $status, PHP_EOL;"]
    if r.reads_body:
        lines.append("echo $body, PHP_EOL;")
    return "\n".join(lines) + "\n"


def _java(r: SampleRequest) -> str:
    publisher = (
        f"HttpRequest.BodyPublishers.ofString({_q(r.body)})"
        if r.body is not None
        else "HttpRequest.BodyPublishers.noBody()"
    )
    handler, kind = (
        ("ofByteArray()", "byte[]")
        if r.binary_response
        else ("ofString()", "String")
        if r.reads_body
        else ("discarding()", "Void")
    )
    lines = [
        "import java.net.URI;",
        "import java.net.http.HttpClient;",
        "import java.net.http.HttpRequest;",
        "import java.net.http.HttpResponse;",
        "",
        "public class Example {",
        "  public static void main(String[] args) throws Exception {",
        f"    HttpRequest request = HttpRequest.newBuilder(URI.create({_q(r.url)}))",
    ]
    lines += [f"        .header({_q(k)}, {_q(v)})" for k, v in r.headers.items()]
    lines += [
        f"        .method({_q(r.method)}, {publisher})",
        "        .build();",
        f"    HttpResponse<{kind}> response = HttpClient.newHttpClient()",
        f"        .send(request, HttpResponse.BodyHandlers.{handler});",
        "    System.out.println(response.statusCode());",
    ]
    if r.reads_body:
        lines.append(
            "    System.out.write(response.body());"
            if r.binary_response
            else "    System.out.println(response.body());"
        )
    return "\n".join(lines + ["  }", "}"]) + "\n"


def _csharp(r: SampleRequest) -> str:
    lines = [
        "using System;",
        "using System.Net.Http;",
        "using System.Text;",
        "",
        "using var client = new HttpClient();",
        f"using var request = new HttpRequestMessage(new HttpMethod({_q(r.method)}), {_q(r.url)});",
    ]
    content_type = r.headers.get("Content-Type")
    lines += [f"request.Headers.Add({_q(k)}, {_q(v)});" for k, v in r.headers.items() if k != "Content-Type"]
    if r.body is not None:
        lines.append(
            f"request.Content = new StringContent({_q(r.body)}, Encoding.UTF8, {_q(content_type or 'text/plain')});"
        )
    lines += ["using var response = await client.SendAsync(request);", "Console.WriteLine((int)response.StatusCode);"]
    if r.reads_body:
        if r.binary_response:
            lines += ["using var output = Console.OpenStandardOutput();", "await response.Content.CopyToAsync(output);"]
        else:
            lines.append("Console.WriteLine(await response.Content.ReadAsStringAsync());")
    return "\n".join(lines) + "\n"


def _swift(r: SampleRequest) -> str:
    lines = [
        "import Foundation",
        "#if canImport(FoundationNetworking)",
        "import FoundationNetworking",
        "#endif",
        "",
        f"var request = URLRequest(url: URL(string: {_q(r.url)})!)",
        f"request.httpMethod = {_q(r.method)}",
    ]
    lines += [f"request.setValue({_q(v)}, forHTTPHeaderField: {_q(k)})" for k, v in r.headers.items()]
    if r.body is not None:
        lines.append(f"request.httpBody = Data({_q(r.body)}.utf8)")
    data = "data" if r.reads_body else "_"
    lines += [
        "let done = DispatchSemaphore(value: 0)",
        f"URLSession.shared.dataTask(with: request) {{ {data}, response, error in",
        "  defer { done.signal() }",
        "  if let error = error { print(error); return }",
        "  print((response as! HTTPURLResponse).statusCode)",
    ]
    if r.reads_body:
        lines.append(
            "  if let data = data { FileHandle.standardOutput.write(data) }"
            if r.binary_response
            else "  if let data = data { print(String(decoding: data, as: UTF8.self)) }"
        )
    lines += ["}.resume()", "done.wait()"]
    return "\n".join(lines) + "\n"


_RENDERERS = {
    "Python": _python,
    "JavaScript": _javascript,
    "Go": _go,
    "Ruby": _ruby,
    "PHP": _php,
    "Java": _java,
    "C#": _csharp,
    "Swift": _swift,
}


def _curl(r: SampleRequest) -> str:
    args = ["curl --fail-with-body", "--request " + shlex.quote(r.method), shlex.quote(r.url)]
    args += ["--header " + shlex.quote(f"{k}: {v}") for k, v in r.headers.items()]
    if r.body is not None:
        args.append("--data-binary " + shlex.quote(r.body))
    if r.binary_response:
        args.append("--output example.txt")
    return " \\\n  ".join(args) + "\n"


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
                operation.setdefault("x-fern-examples", []).append(
                    {"code-samples": [{"sdk": "curl", "code": _curl(request)}]}
                )
    return spec
