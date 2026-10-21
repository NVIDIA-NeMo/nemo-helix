# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered Fabric -> Harbor container -> local gateway -> provider, plus Intake.

Start an isolated platform per SETUP.md with auth,entities,files,jobs,secrets,
models,inference-gateway,agents,evals,intake and jobs,entities,models controllers.
Install the Harbor extra with the repo-documented command:

uv sync --frozen --package nemohelix --package nemo-evals-plugin --extra harbor

Use the default subprocess evaluation profile, a Docker platform runtime, and
127.0.0.1:8080. Use a fresh, isolated config and data directory per SETUP.md.
Before startup, check for a listener on port 8080 and existing Helix processes.
Do not start a second platform against an occupied port or shared data directory;
only reuse an existing instance after confirming its configuration and ownership.

Run this on the SAME macOS host or Linux host/network namespace as the platform:

RUN_LOCAL_FABRIC_NETWORK_E2E=1 uv run --frozen --no-sync pytest \
  plugins/nemo-evals/tests/integration/test_local_fabric_network.py -n 0 -v -s

No external model credentials. Explicit opt-in makes missing prerequisites fail.
The provider below is deterministic; the platform gateway, registered agent,
Fabric/DeepAgents, Harbor Docker container, verifier and Intake are real.
"""

import io
import json
import os
import platform
import shutil
import subprocess
import tarfile
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import httpx
import pytest
from nemo_agents_plugin.sdk import AgentsResource
from nemo_evals.api.schemas import TaskInput
from nemo_evals.harbor.publication import publish_harbor_task_archive
from nemo_evals.sdk.resources import Evaluator
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.evals.client import EvaluatorClient
from nemo_helix_plugin.evals.types import SubmitAgentEvalJobRequest
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelProviderRequest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_LOCAL_FABRIC_NETWORK_E2E") != "1", reason="explicit live platform opt-in required"
    ),
    pytest.mark.timeout(900),
]
ROOT = Path(__file__).resolve().parents[4]
BASE = "http://127.0.0.1:8080"


@pytest.fixture
def deterministic_provider(tmp_path: Path) -> Iterator[tuple[str, list[dict]]]:
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def reply(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(
                {"object": "list", "data": [{"id": "network-probe-model", "object": "model", "owned_by": "local"}]}
            )

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append({"path": self.path, "body": body})
            message = {"role": "assistant", "content": "Hello, world!"}
            finish = "stop"
            tools = {tool["function"]["name"] for tool in body.get("tools", [])}
            if tools and not any(item.get("role") == "tool" for item in body["messages"]):
                assert "write_file" in tools, tools
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_network_probe",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps({"file_path": "/hello.txt", "content": "Hello, world!"}),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            self.reply(
                {
                    "id": "chatcmpl-network-probe",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "network-probe-model",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            )

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        (tmp_path / "model-requests.json").write_text(json.dumps(received, indent=2))
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_registered_fabric_evaluation_reaches_local_gateway_and_intake(deterministic_provider, tmp_path):
    assert platform.system() in {"Darwin", "Linux"}, "Run on the same macOS or Linux host as the platform"
    assert shutil.which("docker"), "Harbor requires Docker CLI on the evaluation host"
    subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=15)
    subprocess.run(["docker", "compose", "version"], check=True, capture_output=True, timeout=15)
    with httpx.Client(base_url=BASE, trust_env=False, timeout=30) as http:
        http.get("/health/ready").raise_for_status()
        http.get("/apis/intake/v2/workspaces/default/spans", params={"page_size": 1}).raise_for_status()
        provider_url, requests = deterministic_provider
        name = "network-probe-" + uuid.uuid4().hex[:10]
        with NemoClient(base_url=BASE, workspace="default") as client:
            models_client = ModelsClient.from_client(client)
            models_client.create_provider(body=CreateModelProviderRequest(name=name, host_url=provider_url)).data()
            deadline = time.monotonic() + 90
            model_name = None
            while time.monotonic() < deadline:
                provider = models_client.get_provider(name=name).data()
                for mapping in provider.served_models or []:
                    if mapping.served_model_name == "network-probe-model":
                        model_name = mapping.model_entity_id.removeprefix("default/")
                        break
                if model_name:
                    break
                time.sleep(1)
            assert model_name, "Provider discovery did not register the deterministic model"
            config = {
                "config_format": "nemo-agents-spec-v1",
                "name": name,
                "instructions": {"system": {"content": "Complete the filesystem task in /app and report completion."}},
                "default_harness": "deepagents",
                "harnesses": {"deepagents": {"kind": "deepagents"}},
                "models": {"default": {"provider": "openai-compatible", "model": model_name}},
                "environment": {"workspace": "/app", "artifacts": "/app/artifacts"},
                "runtime": {"max_turns": 20},
                "telemetry": {
                    "enabled": True,
                    "provider": "relay",
                    "output_dir": "/app/traces",
                    "agent_name": name,
                    "atif": {
                        "enabled": True,
                        "storage": [
                            {"type": "http", "endpoint": BASE + "/apis/intake/v2/workspaces/default/ingest/atif"}
                        ],
                    },
                },
            }
            AgentsResource(client).create(name=name, config=config, config_format="nemo-agents-spec-v1")
            files_client = FilesClient.from_client(client)
            files_client.create_fileset(body=CreateFilesetRequest(name=name)).data()
            dataset = (
                ROOT / "packages/nhx_evals_sdk/examples/harbor/fabric_agent/fabric_hello_world_dataset/hello-world"
            )
            task = publish_harbor_task_archive(dataset, files_client=files_client, fileset_ref=f"default/{name}")
            Evaluator.from_client(client).tasks.create(name, task=TaskInput(spec=task))
            evals_client = EvaluatorClient.from_client(client)
            job = evals_client.submit_agent_eval_job(
                body=SubmitAgentEvalJobRequest(
                    spec={
                        "tasks": [f"default/{name}"],
                        "target": {
                            "kind": "harbor",
                            "source": {"agent": name},
                            "agent_kwargs": {"fabric_workspace": "/app"},
                            "agent_setup_timeout_multiplier": 8,
                            "agent_timeout_multiplier": 3,
                            "trace_dir": "/app/traces",
                        },
                    }
                )
            ).data()
            (tmp_path / "job.json").write_text(job.model_dump_json(indent=2))
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                status = evals_client.get_agent_eval_job_status(name=job.name).data()
                if status.status in {"completed", "failed", "error", "cancelled"}:
                    break
                time.sleep(2)
            (tmp_path / "status.json").write_text(status.model_dump_json(indent=2))
            assert status.status == "completed", status
            payload = evals_client.download_agent_eval_job_result(job=job.name, name="agent-eval-results").read()
            (tmp_path / "agent-eval-results.tar.gz").write_bytes(payload)
            with tarfile.open(fileobj=io.BytesIO(payload)) as archive:

                def read(filename):
                    member = next(m for m in archive.getmembers() if Path(m.name).name == filename)
                    stream = archive.extractfile(member)
                    assert stream is not None
                    return stream.read()

                trials = [json.loads(line) for line in read("trials.jsonl").splitlines() if line]
                summary = json.loads(read("summary.json"))
            assert len(trials) == 1 and trials[0]["status"] == "completed", trials
            assert trials[0]["metadata"]["reward"] == 1.0, trials
            assert summary["error_count"] == 0, summary
            assert len(requests) >= 2 and all(r["path"].endswith("/chat/completions") for r in requests), requests
            (tmp_path / "model-requests.json").write_text(json.dumps(requests, indent=2))
            deadline = time.monotonic() + 45
            spans = []
            while time.monotonic() < deadline:
                response = http.get(
                    "/apis/intake/v2/workspaces/default/spans", params={"filter[agent_name]": name, "page_size": 100}
                )
                response.raise_for_status()
                spans = response.json().get("data", [])
                if any("write_file" in json.dumps(span) for span in spans):
                    break
                time.sleep(1)
            (tmp_path / "intake-spans.json").write_text(json.dumps(spans, indent=2))
            assert spans and any("write_file" in json.dumps(span) for span in spans), spans
            session_ids = {span.get("session_id") for span in spans} - {None}
            assert session_ids, spans
            for session_id in session_ids:
                response = http.get(
                    "/apis/intake/v2/workspaces/default/spans",
                    params={"filter[session_id]": session_id, "page_size": 100},
                )
                response.raise_for_status()
                assert response.json().get("data"), response.text
            print(
                f"Verified {job.name}: reward=1, {len(requests)} gateway model calls, {len(spans)} Intake spans; evidence {tmp_path}"
            )
