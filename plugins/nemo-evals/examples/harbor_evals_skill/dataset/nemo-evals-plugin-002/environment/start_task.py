# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap the live loopback agent and its pinned Taskset."""

import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import start_helix
from nemo_evals.api.schemas import TaskRef, TasksetInput
from nemo_evals.sdk.resources import Evaluator
from nemo_helix_plugin.client.client import NemoClient
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric

AGENT_PORT = 8091
NAME = "agent-smoke"
WORKSPACE = "default"


class EchoAgent(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_error(404) if self.path != "/healthz" else self._reply({"status": "ready"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/answer":
            self.send_error(404)
            return
        self.connection.settimeout(5)
        try:
            length = int(self.headers["Content-Length"])
            if not 0 < length <= 4096:
                raise ValueError("invalid body length")
            request = json.loads(self.rfile.read(length))
            answer = request["instruction"]
            if not isinstance(answer, str):
                raise ValueError("instruction must be text")
        except (ValueError, TypeError, KeyError, OSError):
            self.send_error(400)
            return
        self._reply({"answer": answer})

    def _reply(self, content: dict[str, str]) -> None:
        payload = json.dumps(content).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _agent_ready() -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{AGENT_PORT}/healthz", timeout=1) as response:
            return json.load(response) == {"status": "ready"}
    except (URLError, TimeoutError, ValueError):
        return False


def _start_agent() -> None:
    if _agent_ready():
        return
    with (start_helix.STATE / "evidence/loopback-agent.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [sys.executable, __file__, "--serve"],
            user="helix-task",
            group="helix-task",
            extra_groups=[],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    for _ in range(20):
        if process.poll() is not None:
            break
        if _agent_ready():
            return
        time.sleep(0.25)
    process.terminate()
    raise RuntimeError("The fixture agent did not become ready")


def _seed_taskset() -> None:
    with NemoClient(base_url=start_helix.URL, workspace=WORKSPACE) as client:
        evaluator = Evaluator.from_client(client)
        task = evaluator.tasks.replace(
            NAME,
            task=AgentEvalTask(
                id=NAME,
                intent="Echo the readiness instruction.",
                inputs={"instruction": "ready"},
                reference={"expected": "ready"},
                metrics=[ExactMatchMetric(reference="{{reference.expected}}", candidate="{{sample.output_text}}")],
            ),
        )
        task_digest = evaluator.tasks.list_revisions(NAME, page_size=1).data[0].content_hash
        member = f"{WORKSPACE}/{task.name}"
        evaluator.tasksets.replace(NAME, taskset=TasksetInput(tasks=[TaskRef(f"{member}#{task_digest}")]))
        taskset_digest = evaluator.tasksets.list_revisions(NAME, page_size=1).data[0].content_hash

    (start_helix.STATE / "root/expected-task.json").write_text(
        json.dumps({"entity_name": member, "revision_digest": task_digest}), encoding="utf-8"
    )
    spec = json.loads(Path(__file__).with_name("agent-eval-template.json").read_text(encoding="utf-8"))
    spec["tasks"] = f"{WORKSPACE}/{NAME}#{taskset_digest}"
    Path("/workspace/agent-evaluate-spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        with ThreadingHTTPServer(("127.0.0.1", AGENT_PORT), EchoAgent) as server:
            server.serve_forever()
    else:
        start_helix.main()
        _start_agent()
        _seed_taskset()
