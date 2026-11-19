# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from nemo_observability_plugin.session import COOKIE_NAME

BASE = "/apis/observability/v2/workspaces/default"


def _open_session(client: TestClient, name: str = "mlflow"):
    return client.post(f"{BASE}/connections/{name}/session")


def test_list_connections_exposes_no_upstream_details(make_client):
    body = make_client().get(f"{BASE}/connections").json()
    assert body["data"] == [
        {
            "name": "mlflow",
            "provider": "mlflow",
            "display_name": "MLflow",
            "ui_prefix": "/apis/observability/ui/mlflow",
            "frame_url": "/apis/observability/ui/mlflow/#/experiments/7/traces",
        },
        {
            "name": "phoenix",
            "provider": "phoenix",
            "display_name": "Phoenix",
            "ui_prefix": "/apis/observability/ui/phoenix",
            "frame_url": "/apis/observability/ui/phoenix/projects",
        },
    ]
    assert "test:" not in str(body)


def test_session_sets_path_scoped_httponly_strict_cookie(make_client):
    response = _open_session(make_client())
    assert response.status_code == 200
    assert response.json()["frame_url"] == "/apis/observability/ui/mlflow/#/experiments/7/traces"
    set_cookie = response.headers["set-cookie"]
    assert set_cookie.startswith(f"{COOKIE_NAME}=")
    assert "Path=/apis/observability/ui/mlflow" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=strict" in set_cookie
    assert "Secure" not in set_cookie


def test_session_cookie_is_secure_behind_https_proxy(make_client):
    response = make_client().post(f"{BASE}/connections/mlflow/session", headers={"x-forwarded-proto": "https"})
    assert "Secure" in response.headers["set-cookie"]


def test_unknown_connection_is_404(make_client):
    client = make_client()
    assert _open_session(client, "nope").status_code == 404
    assert client.get(f"{BASE}/connections/nope/otlp-snippet").status_code == 404
    assert client.get("/apis/observability/ui/nope/").status_code == 404


def test_ui_without_session_is_401_and_never_reaches_upstream(make_client, upstream_requests):
    assert make_client().get("/apis/observability/ui/mlflow/").status_code == 401
    assert upstream_requests == []


def test_session_for_one_connection_does_not_open_another(make_client, upstream_requests):
    client = make_client()
    token = _open_session(client, "mlflow").cookies[COOKIE_NAME]
    client.cookies.clear()
    response = client.get("/apis/observability/ui/phoenix/projects", headers={"cookie": f"{COOKIE_NAME}={token}"})
    assert response.status_code == 401
    assert upstream_requests == []


def test_mlflow_proxy_keeps_prefix_and_strips_platform_credentials(make_client, upstream_requests):
    client = make_client()
    _open_session(client)
    client.cookies.set("nhx_session", "platform")
    client.cookies.set("mlflow_ui", "keep")
    response = client.get(
        "/apis/observability/ui/mlflow/ajax-api/2.0/mlflow/experiments/get?experiment_id=7",
        headers={
            "authorization": "Bearer platform-token",
            "x-nhx-principal-id": "spoof",
            "x-forwarded-host": "spoof.example",
            "x-forwarded-for": "10.0.0.1",
            "forwarded": "host=spoof.example",
        },
    )
    assert response.status_code == 200
    sent = upstream_requests[-1]
    assert str(sent.url) == (
        "http://mlflow.test:5000/apis/observability/ui/mlflow/ajax-api/2.0/mlflow/experiments/get?experiment_id=7"
    )
    for dropped in ("authorization", "x-nhx-principal-id", "x-forwarded-host", "x-forwarded-for", "forwarded"):
        assert dropped not in sent.headers
    assert sent.headers["cookie"] == "mlflow_ui=keep"
    assert sent.headers["host"] == "mlflow.test:5000"


def test_upstream_only_offered_encodings_the_caller_accepts(make_client, upstream_requests):
    client = make_client()
    _open_session(client)
    client.get("/apis/observability/ui/mlflow/", headers={"accept-encoding": "gzip"})
    assert upstream_requests[-1].headers["accept-encoding"] == "gzip"
    client.get("/apis/observability/ui/mlflow/", headers={"accept-encoding": ""})
    assert upstream_requests[-1].headers["accept-encoding"] == "identity"


def test_phoenix_proxy_strips_prefix_and_streams_post_body(make_client, upstream_requests):
    client = make_client(lambda request: httpx.Response(200, content=request.content))
    _open_session(client, "phoenix")
    response = client.post(
        "/apis/observability/ui/phoenix/graphql",
        content=b'{"query":"{ projects { edges { node { name } } } }"}',
        headers={"origin": "http://testserver", "content-type": "application/json"},
    )
    assert response.status_code == 200
    assert response.content == b'{"query":"{ projects { edges { node { name } } } }"}'
    sent = upstream_requests[-1]
    assert str(sent.url) == "http://phoenix.test:6006/graphql"
    assert sent.headers["origin"] == "http://phoenix.test:6006"
    # WSGI upstreams read a chunked body as empty, so the length must survive streaming.
    assert sent.headers["content-length"] == str(len(response.content))
    assert "transfer-encoding" not in sent.headers


def test_upstream_credential_injected_from_secret_env(config, make_client, upstream_requests, monkeypatch):
    config.connections[1].secret_env = "PHOENIX_TEST_KEY"
    monkeypatch.setenv("PHOENIX_TEST_KEY", "phx-secret")
    client = make_client()
    _open_session(client, "phoenix")
    client.get("/apis/observability/ui/phoenix/projects")
    assert upstream_requests[-1].headers["authorization"] == "Bearer phx-secret"


def test_response_headers_pass_through_and_redirects_stay_on_studio_origin(make_client):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers=[
                ("location", "http://mlflow.test:5000/apis/observability/ui/mlflow/"),
                ("set-cookie", "a=1; Path=/apis/observability/ui/mlflow"),
                ("set-cookie", "b=2; Path=/apis/observability/ui/mlflow"),
                ("x-frame-options", "SAMEORIGIN"),
            ],
        )

    client = make_client(handler)
    _open_session(client)
    response = client.get("/apis/observability/ui/mlflow", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/apis/observability/ui/mlflow/"
    assert response.headers.get_list("set-cookie") == [
        "a=1; Path=/apis/observability/ui/mlflow",
        "b=2; Path=/apis/observability/ui/mlflow",
    ]
    assert response.headers["x-frame-options"] == "SAMEORIGIN"


@pytest.mark.parametrize(
    ("name", "upstream_location", "expected"),
    [
        ("mlflow", "/apis/observability/ui/mlflow/#/x", "/apis/observability/ui/mlflow/#/x"),
        ("phoenix", "http://phoenix.test:6006/login?next=/", "/apis/observability/ui/phoenix/login?next=/"),
        ("phoenix", "http://phoenix.test:6006", "/apis/observability/ui/phoenix/"),
        ("phoenix", "/login", "/apis/observability/ui/phoenix/login"),
        ("phoenix", "/apis/observability/ui/phoenix/projects", "/apis/observability/ui/phoenix/projects"),
        ("phoenix", "https://idp.example/authorize", "https://idp.example/authorize"),
        ("phoenix", "//idp.example/authorize", "//idp.example/authorize"),
        ("phoenix", "projects", "projects"),
    ],
)
def test_redirects_stay_under_the_connection_prefix(make_client, name, upstream_location, expected):
    client = make_client(lambda request: httpx.Response(302, headers={"location": upstream_location}))
    _open_session(client, name)
    response = client.get(f"/apis/observability/ui/{name}/", follow_redirects=False)
    assert response.headers["location"] == expected


def test_upstream_cookies_are_confined_to_the_connection_prefix(make_client):
    upstream_cookies = [
        "phx=1; Path=/; HttpOnly",
        "deep=2; Path=/apis/observability/ui/phoenix/graphql; Secure",
        "wide=3; Domain=studio.example; Path=/apis",
        "sibling=4; Path=/apis/observability/ui/phoenix-other",
        "bare=5",
        "nhx_session=evil; Path=/",
        f"{COOKIE_NAME}=evil",
    ]
    client = make_client(
        lambda request: httpx.Response(200, headers=[("set-cookie", cookie) for cookie in upstream_cookies])
    )
    _open_session(client, "phoenix")
    response = client.get("/apis/observability/ui/phoenix/")
    assert response.headers.get_list("set-cookie") == [
        "phx=1; HttpOnly; Path=/apis/observability/ui/phoenix",
        "deep=2; Secure; Path=/apis/observability/ui/phoenix/graphql",
        "wide=3; Path=/apis/observability/ui/phoenix",
        "sibling=4; Path=/apis/observability/ui/phoenix",
        "bare=5; Path=/apis/observability/ui/phoenix",
    ]


def test_unreachable_upstream_is_502(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    _open_session(client)
    assert client.get("/apis/observability/ui/mlflow/").status_code == 502


def test_path_traversal_rejected(make_client, upstream_requests):
    client = make_client()
    _open_session(client)
    response = client.get("/apis/observability/ui/mlflow/static/%2e%2e/%2e%2e/admin")
    assert response.status_code == 400
    assert upstream_requests == []


def test_mlflow_otlp_snippet(make_client):
    body = make_client().get(f"{BASE}/connections/mlflow/otlp-snippet").json()
    endpoint = {
        "type": "full",
        "transport": "http_binary",
        "endpoint": "http://mlflow.test:5000/apis/observability/ui/mlflow/v1/traces",
        "headers": {"x-mlflow-experiment-id": "7"},
    }
    assert body["telemetry"] == {
        "enabled": True,
        "opentelemetry": {"enabled": True, "endpoints": [endpoint]},
        "insecure_http_hosts": ["mlflow.test"],
    }
    assert yaml.safe_load(body["yaml"]) == {"telemetry": body["telemetry"]}
    assert body["upstream_setup"] == ["mlflow server --static-prefix /apis/observability/ui/mlflow"]


def test_phoenix_otlp_snippet_on_loopback_needs_no_insecure_host(config, make_client):
    config.connections[1].otlp_endpoint = "http://localhost:6006/v1/traces"
    config.connections[1].default_project = "agents"
    telemetry = make_client().get(f"{BASE}/connections/phoenix/otlp-snippet").json()["telemetry"]
    assert telemetry["opentelemetry"]["endpoints"] == [
        {
            "type": "openinference",
            "transport": "http_binary",
            "endpoint": "http://localhost:6006/v1/traces",
            "resource_attributes": {"openinference.project.name": "agents"},
        }
    ]
    assert "insecure_http_hosts" not in telemetry
