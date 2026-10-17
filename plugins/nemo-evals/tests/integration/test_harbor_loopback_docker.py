# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Linux network namespaces: a loopback-only platform and a separate agent.

Uses a cached Python image, no model credentials, and no platform installation.
The server container represents a Linux host: only its Docker-interface forwarder
is reachable by the agent container. The platform listener remains on 127.0.0.1.
"""

import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.agent_config_formats import resolve_agent_config_for_deployment
from nemo_agents_plugin.fabric.translator import translate_agent_config
from nemo_agents_plugin.runner.deployments_backend import rewrite_fabric_config_base_urls
from nemo_evals.jobs import harbor_forwarder

pytestmark = pytest.mark.integration
_IMAGE = "python:3.12-slim-bookworm"
_SERVER = """
import json, runpy, socket, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
received = []
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        received.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
        self.reply({'choices': [{'message': {'content': 'DONE'}}]})
    def do_GET(self):
        self.reply(received)
    def reply(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args):
        pass
server = ThreadingHTTPServer(('127.0.0.1', 8080), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
host = socket.gethostbyname(socket.gethostname())
forwarder = runpy.run_path('/probe/harbor_forwarder.py')['loopback_forwarder']
with forwarder('http://127.0.0.1:8080', host) as gateway:
    print(json.dumps({'host': host, 'gateway': gateway}), flush=True)
    while True:
        time.sleep(1)
"""
_CLIENT = """
import errno, json, os, urllib.request, urllib.error
host = os.environ['PROBE_HOST']
try:
    urllib.request.urlopen('http://' + host + ':8080/status', timeout=3)
except urllib.error.URLError as error:
    assert isinstance(error.reason, ConnectionRefusedError), repr(error)
else:
    raise AssertionError('The platform must remain unreachable at its Docker-interface address')
config = json.loads(os.environ['PROBE_CONFIG'])
model = config['models']['default']['base_url']
trace = config['relay']['observability']['atif']['storage'][0]['endpoint']
for url, payload in [
    (model + '/chat/completions', {'model': 'probe-model', 'messages': [{'role': 'user', 'content': 'Reply DONE'}]}),
    (trace, {'probe': 'trace-reached-intake'}),
]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200
origin = model.split('/apis/')[0]
with urllib.request.urlopen(origin + '/received', timeout=5) as response:
    received = json.load(response)
assert len(received) == 2, received
assert received[0][0].endswith('/chat/completions'), received
assert received[1][0].endswith('/ingest/atif'), received
assert received[1][1] == {'probe': 'trace-reached-intake'}, received
print('model-and-trace-delivered; direct-platform-connection-refused')
"""


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], text=True, capture_output=True, check=check, timeout=30)


def test_agent_container_reaches_model_and_intake_on_loopback_platform(tmp_path: Path, monkeypatch) -> None:
    if not shutil.which("docker") or _docker("info", check=False).returncode:
        pytest.skip("Docker is required")
    if _docker("image", "inspect", _IMAGE, check=False).returncode:
        pytest.skip(f"Cached {_IMAGE} image is required; this test never pulls images")
    name = f"harbor-loopback-{uuid.uuid4().hex[:10]}"
    script = tmp_path / "server.py"
    script.write_text(_SERVER)
    source = Path(harbor_forwarder.__file__).resolve()
    _docker("network", "create", "--internal", name)
    try:
        _docker(
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--network",
            name,
            "--mount",
            f"type=bind,src={source},dst=/probe/harbor_forwarder.py,readonly",
            "--mount",
            f"type=bind,src={script},dst=/probe/server.py,readonly",
            _IMAGE,
            "python",
            "/probe/server.py",
        )
        deadline = time.monotonic() + 15
        while True:
            logs = _docker("logs", name).stdout.strip()
            if logs:
                ready = json.loads(logs.splitlines()[0])
                break
            if time.monotonic() > deadline:
                pytest.fail(f"Platform probe did not start: {_docker('logs', name).stderr}")
            time.sleep(0.1)
        monkeypatch.delenv("NEMO_BASE_URL", raising=False)
        monkeypatch.setenv("NHX_BASE_URL", "http://127.0.0.1:8080")
        payload = {
            "config_format": "nemo-agents-spec-v1",
            "name": "gateway-probe",
            "instructions": {"system": {"content": "Reply DONE"}},
            "default_harness": "deepagents",
            "harnesses": {"deepagents": {"kind": "deepagents"}},
            "models": {"default": {"provider": "nvidia", "model": "probe-model"}},
            "telemetry": {
                "enabled": True,
                "atif": {
                    "storage": [
                        {
                            "type": "http",
                            "endpoint": "http://127.0.0.1:8080/apis/intake/v2/workspaces/default/ingest/atif",
                        }
                    ]
                },
            },
        }
        resolved = resolve_agent_config_for_deployment(
            "nemo-agents-spec-v1", payload, workspace="default", agent_name="gateway-probe"
        )
        fabric = translate_agent_config(AgentConfig.model_validate(resolved)).model_dump(mode="json", exclude_none=True)
        config = rewrite_fabric_config_base_urls(fabric, ready["gateway"])
        result = _docker(
            "run",
            "--rm",
            "--network",
            name,
            "--add-host",
            f"host.docker.internal:{ready['host']}",
            "--env",
            f"PROBE_HOST={ready['host']}",
            "--env",
            f"PROBE_CONFIG={json.dumps(config)}",
            _IMAGE,
            "python",
            "-c",
            _CLIENT,
        )
        assert "model-and-trace-delivered; direct-platform-connection-refused" in result.stdout
    finally:
        _docker("rm", "--force", name, check=False)
        _docker("network", "rm", name, check=False)
