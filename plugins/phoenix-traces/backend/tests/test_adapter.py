# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_phoenix_traces.app import Settings, create_app, next_cursor, page_number, plugin_manifest
from nemo_phoenix_traces.models import PhoenixSpan
from pydantic import SecretStr, ValidationError

KEY = "local-test-adapter-key-not-a-secret"
HEADERS = {"Authorization": f"Bearer {KEY}"}
BASE = "/v1/workspaces/default"
NOW = datetime.now(timezone.utc).isoformat()


def wire_span(**overrides: object) -> dict:
    return {
        "name": "model call",
        "context": {"trace_id": "trace-a", "span_id": "child-a"},
        "parent_id": "root-a",
        "span_kind": "LLM",
        "start_time": NOW,
        "end_time": None,
        "status_code": "UNSET",
        "attributes": {
            "llm.model_name": "test-model",
            "llm.token_count.prompt": 7,
            "provider.raw": {"nested": [1, True, None, "雪"]},
        },
        **overrides,
    }


def envelope(data: list, page: int = 1, total_pages: int = 1) -> dict:
    return {
        "data": data,
        "pagination": {
            "page": page,
            "page_size": 1,
            "current_page_size": len(data),
            "total_pages": total_pages,
            "total_results": total_pages,
        },
    }


@contextmanager
def adapter(tmp_path: Path, handler: Callable[[httpx.Request], httpx.Response]) -> Generator[TestClient]:
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncNemoClient(
        base_url="http://127.0.0.1:8080",
        default_headers={"Authorization": "Bearer intake-test-token"},
        http_client=transport,
        owns_http_client=True,
    )
    settings = Settings(adapter_key=SecretStr(KEY), bundle_dir=tmp_path)
    try:
        with TestClient(create_app(settings, intake_client=AsyncIntakeClient.from_client(client))) as browser:
            yield browser
    finally:
        asyncio.run(client.aclose())


def test_ingest_uses_typed_intake_and_not_caller_credentials(tmp_path: Path):
    seen: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/apis/intake/v2/workspaces/default/ingest/spans"
        assert request.headers["authorization"] == "Bearer intake-test-token"
        assert "cookie" not in request.headers
        body = json.loads(request.content)
        assert body["source"] == "phoenix-plugin"
        span = body["spans"][0]
        assert span["trace_id"] == "trace-a"
        assert span["parent_span_id"] == "root-a"
        assert span["attributes"]["project.name"] == "real project / 雪"
        assert span["attributes"]["provider.raw"] == {"nested": [1, True, None, "雪"]}
        return httpx.Response(201)

    with adapter(tmp_path, upstream) as browser:
        response = browser.post(
            f"{BASE}/spans",
            params={"project": "real project / 雪"},
            headers={**HEADERS, "Cookie": "platform-session=do-not-forward"},
            json={"spans": [wire_span()]},
        )
        assert response.status_code == 201, response.text
        assert response.content == b""
    assert len(seen) == 1


@pytest.mark.parametrize("timestamp", [NOW, NOW.removesuffix("+00:00")])
def test_reads_real_wire_fields_and_query_bound_cursors(tmp_path: Path, timestamp: str):
    seen: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert json.loads(request.url.params["filter"]) == {"project": "real / 雪"}
        assert request.url.params["sort"] == "started_at"
        page = int(request.url.params["page"])
        span = {
            "span_id": f"span-{page}",
            "trace_id": "trace-a",
            "session_id": "session-a",
            "workspace": "default",
            "kind": "LLM",
            "source": "otlp",
            "started_at": timestamp,
            "ingested_at": NOW,
            "status": "success",
            "model": "persisted-model",
            "input_tokens": 9,
            "total_tokens": 12,
            "input": "hello",
            "raw_attributes": '{"custom":{"nested":[1,null]}}',
        }
        return httpx.Response(200, json=envelope([span], page=page, total_pages=2))

    with adapter(tmp_path, upstream) as browser:
        first = browser.get(f"{BASE}/spans", params={"project": "real / 雪", "limit": 1}, headers=HEADERS)
        assert first.status_code == 200, first.text
        span = first.json()["data"][0]
        assert span["context"] == {"trace_id": "trace-a", "span_id": "span-1"}
        assert span["end_time"] is None
        assert span["start_time"].endswith(("Z", "+00:00"))
        assert span["attributes"]["llm.token_count.total"] == 12
        assert span["attributes"]["llm.model_name"] == "persisted-model"
        assert span["attributes"]["custom"] == {"nested": [1, None]}
        cursor = first.json()["next_cursor"]
        second = browser.get(
            f"{BASE}/spans", params={"project": "real / 雪", "limit": 1, "cursor": cursor}, headers=HEADERS
        )
        assert second.status_code == 200, second.text
        assert second.json()["data"][0]["context"]["span_id"] == "span-2"
        assert second.json()["next_cursor"] is None
        for params in (
            {"project": "other", "limit": 1, "cursor": cursor},
            {"project": "real / 雪", "limit": 2, "cursor": cursor},
            {"project": "real / 雪", "limit": 1, "cursor": "invalid!"},
        ):
            assert browser.get(f"{BASE}/spans", params=params, headers=HEADERS).status_code == 400
    assert len(seen) == 2


def test_project_discovery_uses_grouping_and_keeps_unassigned(tmp_path: Path):
    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/spans/groups")
        assert request.url.params["by"] == "project"
        return httpx.Response(
            200,
            json=envelope([{"group": {"project": ""}, "span_count": 2, "started_at": NOW}]),
        )

    with adapter(tmp_path, upstream) as browser:
        response = browser.get(f"{BASE}/projects", headers=HEADERS)
        assert response.status_code == 200, response.text
        assert response.json() == {"data": [{"name": ""}], "next_cursor": None}


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_auth_and_workspace_fail_before_intake(tmp_path: Path, method: str):
    def upstream(_request: httpx.Request) -> httpx.Response:
        pytest.fail("An unauthorized request reached Intake")

    with adapter(tmp_path, upstream) as browser:
        body = {"spans": [wire_span()]} if method == "POST" else None
        for headers, workspace, expected in (({}, "default", 401), (HEADERS, "other", 403)):
            response = browser.request(
                method, f"/v1/workspaces/{workspace}/spans?project=demo", headers=headers, json=body
            )
            assert response.status_code == expected


@pytest.mark.parametrize("status", [401, 403, 422, 503])
def test_intake_errors_are_sanitized(tmp_path: Path, status: int):
    def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "sensitive upstream credential or trace content"})

    with adapter(tmp_path, upstream) as browser:
        response = browser.get(f"{BASE}/projects", headers=HEADERS)
        assert response.status_code == (502 if status == 401 else status)
        assert "sensitive" not in response.text


def test_manifest_contains_only_credential_reference_and_pins_actual_bundle(tmp_path: Path):
    script = b"export const Root = () => null;"
    (tmp_path / "index.js").write_bytes(script)
    (tmp_path / "plugin.manifest.json").write_text(
        json.dumps({"contractVersion": "1.0.0", "specifiers": ["react"], "builtAgainst": {"react": "19.2.7"}})
    )
    settings = Settings(adapter_key=SecretStr(KEY), intake_token=SecretStr("intake-test-token"), bundle_dir=tmp_path)
    manifest = plugin_manifest(settings)
    encoded = json.dumps(manifest)
    assert KEY not in encoded
    assert "intake-test-token" not in encoded
    assert manifest["plugin"]["credential"]["secret_name"] == "phoenix-traces-adapter-key"
    assert manifest["plugin"]["bundle"]["sha256"] == hashlib.sha256(script).hexdigest()
    assert len(manifest["plugin"]["endpoints"]) == 3


def test_api_only_manifest_and_static_example_agree(tmp_path: Path):
    import yaml

    example = Path(__file__).resolve().parents[2] / "plugin.yaml"
    settings = Settings(adapter_key=SecretStr(KEY), bundle_dir=tmp_path)
    actual = plugin_manifest(settings)["plugin"]
    expected = yaml.safe_load(example.read_bytes())["plugin"]
    for field in ("name", "title", "endpoints", "credential"):
        assert actual[field] == expected[field]


@pytest.mark.parametrize(
    "overrides",
    [
        {"intake_url": "http://external.invalid"},
        {"public_url": "http://127.0.0.1:6007/prefix"},
        {"intake_token": KEY},
        {"adapter_key": "too-short"},
        {"workspace": "-"},
    ],
)
def test_rejects_unsafe_configuration(overrides: dict):
    with pytest.raises(ValidationError):
        Settings.model_validate({"adapter_key": KEY, **overrides})


def test_long_unicode_project_does_not_exceed_cursor_size_limit():
    project = "雪" * 4096
    cursor = next_cursor(page=1, total_pages=2, limit=1, workspace="default", project=project)
    assert cursor is not None and len(cursor) < 4096
    assert page_number(cursor, limit=1, workspace="default", project=project) == 2


def test_project_selection_overrides_all_project_aliases():
    span = PhoenixSpan.model_validate(
        wire_span(attributes={"gen_ai.project": "wrong", "project": "also wrong", "project.name": "wrong"})
    )
    assert span.to_intake("chosen").attributes == {"project.name": "chosen"}
    assert span.to_intake("").attributes == {}
