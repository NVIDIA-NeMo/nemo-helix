# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real platform/subprocess/container acceptance; requires explicit opt-in and model credentials."""

import io
import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from nemo_evaluator.api.schemas import TaskInput, TaskRef, TasksetInput, TasksetRef
from nemo_evaluator.harbor.agent_source import HarborAgentSource
from nemo_evaluator.harbor.publication import publish_harbor_task_archive
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner, HarborRuntimeConfig
from nemo_evaluator_sdk.values import SecretRef
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.jobs.client import JobsClient
from nemo_platform_plugin.secrets.client import SecretsClient
from nemo_platform_plugin.secrets.types import PlatformSecretCreateRequest
from pydantic import SecretStr

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RUN_HARBOR_AGENT_INTEGRATION"),
        reason="requires platform/subprocess/Docker and explicit model-test opt-in",
    ),
]
CODEX_VERSION = "0.153.0"


@pytest.mark.parametrize("harbor_agent_platform", ["uploaded", "codex"], indirect=True)
@pytest.mark.timeout(1200)
def test_agent_submission(harbor_agent_platform, tmp_path):
    base_url, mode = harbor_agent_platform
    name = f"agent-source-{uuid4().hex[:8]}"
    root = Path(__file__).resolve().parents[4]
    task = tmp_path / "hello-world"
    shutil.copytree(root / "packages/nemo_evaluator_sdk/examples/harbor/hello_world_dataset/hello-world", task)
    # Codex's Alpine installer needs Node/npm; wrapper uses only sh.
    if mode == "codex":
        with (task / "environment/Dockerfile").open("a") as output:
            output.write("\nRUN apk add --no-cache nodejs npm git curl\n")
    files = FilesClient(base_url=base_url, workspace="default")
    definition = publish_harbor_task_archive(task, files_client=files, fileset_ref=f"default/{name}")
    sdk = NeMoPlatform(base_url=base_url, workspace="default")
    sdk.evaluator.tasks.create(name, task=TaskInput(spec=definition))
    sdk.evaluator.tasksets.create(name, taskset=TasksetInput(tasks=[TaskRef(f"default/{name}")]))
    secrets = SecretsClient(base_url=base_url, workspace="default")
    target = {
        "kind": "harbor",
        "n_concurrent_trials": 1,
        "agent_setup_timeout_multiplier": 2.0,
        "agent_timeout_multiplier": 2.0,
        "agent_env": {"SOURCE_TEST_PUBLIC": "delivered"},
    }
    if mode == "uploaded":
        secret_value = "private-" + uuid4().hex
        secrets.create_secret(body=PlatformSecretCreateRequest(name=name, value=SecretStr(secret_value))).data()
        agent = tmp_path / "agent"
        agent.mkdir()
        (agent / "helper.py").write_text(
            "from pathlib import Path\ndef script(): return Path(__file__).parent / 'run.sh'\n"
        )
        script = agent / "run.sh"
        script.write_text(
            '#!/bin/sh\nset -eu\ntest "$SOURCE_TEST_PUBLIC" = delivered\ntest -n "$SOURCE_TEST_SECRET"\nprintf "Hello, world!" > /app/hello.txt\n'
        )
        script.chmod(0o755)
        (agent / "wrapper.py").write_text("""from harbor import BaseAgent
from .helper import script
class UploadedAgent(BaseAgent):
    @staticmethod
    def name(): return "uploaded-only"
    def version(self): return "1.0"
    async def setup(self, environment):
        assert script().stat().st_mode & 0o100
        await environment.upload_file(script(), "/app/run.sh")
        await environment.exec("chmod 755 /app/run.sh")
    async def run(self, instruction, environment, context):
        result = await environment.exec("/app/run.sh", env=self.extra_env)
        if result.return_code: raise RuntimeError("container environment or execution failed")
""")
        runner = HarborAgentTaskRunner(
            config=HarborRuntimeConfig(
                jobs_dir=tmp_path / "local-results",
                agent_dir=agent,
                agent_import_path="wrapper:UploadedAgent",
                agent_env={"SOURCE_TEST_PUBLIC": "delivered"},
                agent_env_from_host=["SOURCE_TEST_SECRET"],
                n_concurrent_trials=1,
                agent_setup_timeout_multiplier=2.0,
                agent_timeout_multiplier=2.0,
            )
        )
        submitted = sdk.evaluator.submit(
            tasks=TasksetRef(f"default/{name}"),
            target=runner,
            env_secrets={"SOURCE_TEST_SECRET": SecretRef(f"default/{name}")},
        )
        job = {"name": submitted.name}
        descriptor = submitted.job.spec.target.agent_source
        assert isinstance(descriptor, HarborAgentSource)
        shutil.rmtree(agent)
    else:
        secret_value = os.environ.get("HARBOR_CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not secret_value:
            pytest.fail("Codex validation blocked: HARBOR_CODEX_API_KEY or OPENAI_API_KEY is required")
        if codex_base_url := os.environ.get("HARBOR_CODEX_BASE_URL"):
            target["agent_env"]["OPENAI_BASE_URL"] = codex_base_url
        secrets.create_secret(body=PlatformSecretCreateRequest(name=name, value=SecretStr(secret_value))).data()
        target.update(
            agent_name="codex",
            agent_model_name=os.environ.get("HARBOR_CODEX_MODEL", "gpt-5.6-luna"),
            agent_kwargs={"version": CODEX_VERSION},
            env_secrets={"OPENAI_API_KEY": f"default/{name}"},
        )
    route = f"{base_url}/apis/evaluator/v2/workspaces/default/agent-evaluate/jobs"
    if mode == "codex":
        response = httpx.post(
            route, json={"profile": "harbor-test", "spec": {"tasks": f"default/{name}", "target": target}}, timeout=60
        )
        assert response.status_code == 201, response.text
        job = response.json()
        assert secret_value not in response.text
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        status = httpx.get(f"{route}/{job['name']}/status", timeout=30).raise_for_status().json()
        if status["status"] in {"completed", "error", "failed", "cancelled"}:
            break
        time.sleep(2)
    assert status["status"] == "completed", status
    payload = (
        JobsClient(base_url=base_url, workspace="default")
        .download_job_result(job=job["name"], name="agent-eval-results")
        .read()
    )
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        contents = {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}
    assert all(secret_value.encode() not in data for data in contents.values())
    trials = [
        json.loads(line)
        for filename, data in contents.items()
        if filename.endswith("trials.jsonl")
        for line in data.splitlines()
        if line
    ]
    scores = [
        json.loads(line)
        for filename, data in contents.items()
        if filename.endswith("scores.jsonl")
        for line in data.splitlines()
        if line
    ]
    assert trials and all(trial["error"] is None for trial in trials), trials
    assert scores and all(score["status"] == "completed" for score in scores), scores
    assert any(
        output["name"] == "reward" and output["value"] == 1.0 for score in scores for output in score["outputs"]
    ), scores
    for config_file in tmp_path.glob("subprocess-jobs/**/job-storage/harbor/**/config.json"):
        assert secret_value not in config_file.read_text(), "Credential leaked into Harbor config"
    if mode == "codex":
        assert {trial["metadata"]["harbor_agent_version"] for trial in trials} == {CODEX_VERSION}
    else:
        assert descriptor.sha256.encode() in b"\n".join(contents.values())
    print(f"Validated {mode}: job={job['name']}, trials={len(trials)}, scores={len(scores)}")
