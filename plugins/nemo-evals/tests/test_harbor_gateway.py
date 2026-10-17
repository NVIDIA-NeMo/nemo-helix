# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from urllib.parse import urlsplit

import httpx
import pytest
from nemo_evals.jobs.harbor_forwarder import loopback_forwarder
from nemo_evals.jobs.harbor_gateway import harbor_gateway_config


@pytest.fixture
def loopback_platform():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                (
                    self.path,
                    self.headers.get("X-NHX-Principal-Id"),
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
            )
            payload = json.dumps({"choices": [{"message": {"content": "DONE"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_linux_gateway_routes_model_and_translated_relay_without_mutating_spec(loopback_platform, monkeypatch):
    base, received = loopback_platform
    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway.platform.system", lambda: "Linux")
    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway._docker_bridge_host", lambda: "127.0.0.1")
    config = {
        "target": {
            "kind": "harbor",
            "agent_kwargs": {
                "fabric_config": {
                    "models": {
                        "default": {
                            "base_url": "http://host.docker.internal:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
                        },
                        "external": {"base_url": "https://external.example/v1"},
                    },
                    "relay": {
                        "observability": {
                            "atif": {
                                "storage": [
                                    {
                                        "type": "http",
                                        "endpoint": base + "/apis/intake/v2/workspaces/default/ingest/atif",
                                    }
                                ]
                            }
                        }
                    },
                }
            },
        }
    }
    original = copy.deepcopy(config)
    with harbor_gateway_config(config, platform_base_url=base) as reachable:
        fabric = reachable["target"]["agent_kwargs"]["fabric_config"]
        model = fabric["models"]["default"]["base_url"]
        trace = fabric["relay"]["observability"]["atif"]["storage"][0]["endpoint"]
        port = urlsplit(model).port
        assert urlsplit(trace).port == port
        assert fabric["models"]["external"]["base_url"] == "https://external.example/v1"
        for url in [model + "/chat/completions", trace]:
            local = url.replace("host.docker.internal", "127.0.0.1")
            response = httpx.post(
                local, json={"probe": "ok"}, headers={"X-NHX-Principal-Id": "user:probe"}, trust_env=False
            )
            response.raise_for_status()
    assert len(received) == 2
    assert all(identity == "user:probe" for _, identity, _ in received)
    assert config == original
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1)


@pytest.mark.parametrize("system,base", [("Darwin", "http://127.0.0.1:8080"), ("Linux", "https://remote.example")])
def test_non_local_linux_runs_do_not_start_a_proxy(system, base, monkeypatch):
    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway.platform.system", lambda: system)

    def unexpected():
        pytest.fail("Docker discovery should not run")

    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway._docker_bridge_host", unexpected)
    config = {"target": {"kind": "harbor", "agent_kwargs": {"fabric_config": {}}}}
    with harbor_gateway_config(config, platform_base_url=base) as reachable:
        assert reachable is config


def test_proxy_closes_on_evaluation_failure(loopback_platform):
    base, _ = loopback_platform
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with loopback_forwarder(base, "127.0.0.1") as gateway:
            port = urlsplit(gateway).port
            raise RuntimeError("evaluation failed")
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1)


def test_external_models_do_not_start_a_forwarder(monkeypatch):
    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway.platform.system", lambda: "Linux")

    def unexpected():
        pytest.fail("An external model does not require a local platform forwarder")

    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway._docker_bridge_host", unexpected)
    config = {
        "target": {
            "kind": "harbor",
            "agent_kwargs": {"fabric_config": {"models": {"default": {"base_url": "https://external.example/v1"}}}},
        }
    }
    with harbor_gateway_config(config, platform_base_url="http://127.0.0.1:8080") as reachable:
        assert reachable is config


def test_forwarder_preserves_large_bodies_and_half_closed_connections():
    payload = b"request-body" * 100000
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()

        def exchange():
            with listener.accept()[0] as connection:
                received = bytearray()
                while data := connection.recv(65536):
                    received.extend(data)
                connection.sendall(bytes(received)[::-1])

        thread = Thread(target=exchange, daemon=True)
        thread.start()
        with loopback_forwarder(f"http://127.0.0.1:{listener.getsockname()[1]}", "127.0.0.1") as gateway:
            with socket.create_connection(("127.0.0.1", urlsplit(gateway).port), timeout=5) as client:
                client.sendall(payload)
                client.shutdown(socket.SHUT_WR)
                received = bytearray()
                while data := client.recv(65536):
                    received.extend(data)
                assert received == payload[::-1]
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_job_keeps_forwarder_alive_during_execution(loopback_platform, monkeypatch, mocker):
    from nemo_evals.jobs.agent_evaluate import AgentEvalJob
    from nemo_helix_plugin.client.client import NemoClient

    base, received = loopback_platform
    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway.platform.system", lambda: "Linux")
    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway._docker_bridge_host", lambda: "127.0.0.1")
    config = {
        "target": {
            "kind": "harbor",
            "agent_kwargs": {
                "fabric_config": {
                    "models": {
                        "default": {
                            "base_url": "http://host.docker.internal:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
                        }
                    }
                }
            },
        }
    }
    ports = []

    def execute(self, reachable, **kwargs):
        url = reachable["target"]["agent_kwargs"]["fabric_config"]["models"]["default"]["base_url"]
        ports.append(urlsplit(url).port)
        httpx.post(url + "/chat/completions", json={}, trust_env=False).raise_for_status()
        return {"status": "completed"}

    monkeypatch.setattr(AgentEvalJob, "_run_with_reachable_client", execute)
    with NemoClient(base_url=base) as client:
        result = AgentEvalJob()._run_with_client(
            config, ctx=mocker.Mock(), platform_client=client, async_client=mocker.Mock()
        )
    assert result == {"status": "completed"}
    assert len(received) == 1
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", ports[0]), timeout=1)


def test_forwarder_refuses_wildcard_listeners():
    with pytest.raises(ValueError, match="specific interface"):
        with loopback_forwarder("http://127.0.0.1:8080", "0.0.0.0"):
            pytest.fail("A wildcard listener must never start")


def test_offline_harbor_scoring_does_not_require_docker(monkeypatch):
    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway.platform.system", lambda: "Linux")

    def unexpected():
        pytest.fail("Offline scoring must not require Docker")

    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway._docker_bridge_host", unexpected)
    config = {
        "trials": [],
        "target": {
            "kind": "harbor",
            "agent_kwargs": {
                "fabric_config": {
                    "models": {
                        "default": {
                            "base_url": "http://host.docker.internal:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
                        }
                    }
                }
            },
        },
    }
    with harbor_gateway_config(config, platform_base_url="http://127.0.0.1:8080") as reachable:
        assert reachable is config


@pytest.mark.parametrize(
    "location",
    [
        ("models", "default", "base_url"),
        ("models", "default", "settings", "base_url"),
        ("harnesses", "main", "model", "base_url"),
        ("harnesses", "main", "model", "settings", "base_url"),
        ("telemetry", "atif", "storage"),
        ("relay", "observability", "atif", "storage"),
    ],
)
def test_forwarding_detects_each_supported_local_endpoint(location, loopback_platform, monkeypatch):
    """Reach local services when the only URL lives in any supported config location."""
    base, received = loopback_platform
    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway.platform.system", lambda: "Linux")
    monkeypatch.setattr("nemo_evals.jobs.harbor_gateway._docker_bridge_host", lambda: "127.0.0.1")
    telemetry = location[-1] == "storage"
    url = base + (
        "/apis/intake/v2/workspaces/default/ingest/atif"
        if telemetry
        else "/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )
    fabric = {}
    cursor = fabric
    for key in location[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[location[-1]] = [{"type": "http", "endpoint": url}] if telemetry else url
    config = {"target": {"kind": "harbor", "agent_kwargs": {"fabric_config": fabric}}}
    with harbor_gateway_config(config, platform_base_url=base) as reachable:
        cursor = reachable["target"]["agent_kwargs"]["fabric_config"]
        for key in location:
            cursor = cursor[key]
        endpoint = cursor[0]["endpoint"] if telemetry else cursor
        assert urlsplit(endpoint).port != urlsplit(base).port
        httpx.post(endpoint, json={"probe": "ok"}, trust_env=False).raise_for_status()
    assert len(received) == 1


def test_shutdown_waits_for_handlers_connecting_upstream(loopback_platform, monkeypatch):
    """Drain handlers that have accepted clients but have not connected upstream yet."""
    from nemo_evals.jobs.harbor_forwarder import _Forwarder

    base, _ = loopback_platform
    connecting, release, stop, listening, closing, exited = (Event() for _ in range(6))
    original_connect = socket.create_connection
    original_close = _Forwarder.server_close
    workers = []
    ports = []

    def stalled_connect(address, timeout):
        from threading import current_thread

        workers.append(current_thread())
        connecting.set()
        assert release.wait(5), "Test did not release the upstream handshake"
        return original_connect(address, timeout=timeout)

    def tracked_close(server):
        closing.set()
        original_close(server)

    monkeypatch.setattr("nemo_evals.jobs.harbor_forwarder.socket.create_connection", stalled_connect)
    monkeypatch.setattr(_Forwarder, "server_close", tracked_close)

    def run():
        with loopback_forwarder(base, "127.0.0.1") as gateway:
            ports.append(urlsplit(gateway).port)
            listening.set()
            assert stop.wait(5)
        exited.set()

    thread = Thread(target=run, daemon=True)
    thread.start()
    try:
        assert listening.wait(5)
        with original_connect(("127.0.0.1", ports[0]), timeout=5):
            assert connecting.wait(5)
            stop.set()
            assert closing.wait(5)
            assert not exited.wait(0.2), "Shutdown returned with an upstream handshake still blocked"
            release.set()
            assert exited.wait(5)
            assert all(not worker.is_alive() for worker in workers)
    finally:
        stop.set()
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
