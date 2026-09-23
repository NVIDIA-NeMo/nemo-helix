# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor taskset E2E with real services/Docker and the deterministic example agent.

Run from the repository root with Python 3.12+ and Docker running::

    uv sync --frozen --package nemoplatform --package nemo-evaluator-plugin --extra harbor
    NEMO_PLUGIN_SERVICES_ALLOWLIST=evaluator \\
    NEMO_PLUGIN_CONTROLLERS_ALLOWLIST='' \\
    RUN_AGENT_EVAL_INTEGRATION=1 UV_NO_SYNC=1 uv run --frozen --no-sync pytest \\
      plugins/nemo-evaluator/tests/integration/test_harbor_taskset_e2e.py -v -s -n 0

CI runs this serially after the general integration suite when the Harbor E2E
path filter or dependency filter matches, and on manual workflow dispatch.
Without RUN_AGENT_EVAL_INTEGRATION=1 it skips before starting any fixtures.
Once opted in, missing Harbor or unavailable Docker FAILS rather than skips.

The shared fixture starts an isolated platform on port 8090 (override with
NMP_AGENT_BASE_URL); the port must be free. Database, files, and job storage are
temporary. Plugin allowlists exclude unrelated plugins while keeping core
services/controllers. UV_NO_SYNC preserves the Harbor extra in child processes.
The deterministic agent needs no API key or OpenAI calls; Docker image builds
may need network access.

Repeat publication must return identical revision pins. The three saved trials
and scores distinguish a correct greeting (reward=1), a completed wrong answer
(reward=0, format_ok=1), and an AgentTimeoutError (partial, reward=0). The summary
must count the error; raw Harbor output must show only the first step executed.
The typed submit API has no profile selector, so the fixture's default subprocess
profile is used; it has the same configuration as harbor-test.

Polling is bounded to 5 minutes; the overall test timeout is 8 minutes, within
the CI step's 10-minute limit (including dependency installation). Job
identity, status, logs, result bundle, and raw debug result are saved under the
printed pytest temporary directory, subject to pytest retention. CI uploads
these and a separate JUnit report, excluding platform storage. The fixture tears
down the platform. Repeat the command in a separate invocation to check isolation;
add tests/integration/test_harbor_plugin_run.py (relative to this plugin) to the
pytest arguments for the existing Harbor coverage.
"""

import importlib
import io
import json
import os
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

import pytest
from nemo_evaluator.sdk.harbor import publish_harbor_tasks
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.evaluator.client import EvaluatorClient
from nemo_platform_plugin.evaluator.types import SubmitAgentEvalJobRequest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_AGENT_EVAL_INTEGRATION") != "1",
        reason="set RUN_AGENT_EVAL_INTEGRATION=1 for real platform/Docker execution",
    ),
    pytest.mark.timeout(480),
]

DATASET = Path(__file__).resolve().parents[2] / "examples/harbor_taskset/harbor_dataset"
EXPECTED = {
    "hello/greet-universe": ("completed", 1.0, 1.0),
    "hello/sum-three": ("completed", 0.0, 1.0),
    "hello/debug-agent-runtime-error": ("partial", 0.0, 0.0),
}


def test_harbor_taskset_e2e(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    importlib.import_module("harbor")
    try:
        docker = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.fail(f"Docker daemon is required for the opted-in E2E test: {exc}")
    if docker.returncode:
        pytest.fail(f"Docker daemon is required: {docker.stderr.decode(errors='replace')}")

    # Resolve only after prerequisite checks, so skipped tests never launch services.
    base_url = request.getfixturevalue("subprocess_platform")
    client = NemoClient(base_url=base_url, workspace="default")
    evaluator = EvaluatorClient.from_client(client)
    name = f"harbor-e2e-{uuid.uuid4().hex[:12]}"
    member_names = {task_id.split("/")[1]: f"{name}-{task_id.split('/')[1]}" for task_id in EXPECTED}

    def publish():
        return publish_harbor_tasks(
            DATASET,
            client=client,
            workspace="default",
            taskset_name=name,
            fileset_ref=f"default/{name}",
            member_names=member_names,
            register=True,
            replace=False,
        )

    upload_details = publish()
    repeated = publish()
    assert upload_details.taskset_ref is not None
    assert "#" in upload_details.taskset_ref.root
    assert repeated.taskset_ref == upload_details.taskset_ref
    pins = {member_details.native_name: member_details.task_ref for member_details in upload_details.members}
    assert len(pins) == 3
    assert all(pin is not None and "#" in pin.root for pin in pins.values())
    assert {member_details.native_name: member_details.task_ref for member_details in repeated.members} == pins
    (tmp_path / "publication.json").write_text(upload_details.model_dump_json(indent=2))

    # The typed submit request exposes only spec. The fixture's default profile
    # uses the same subprocess executor configuration as harbor-test.
    job = evaluator.submit_agent_eval_job(
        workspace="default",
        body=SubmitAgentEvalJobRequest(
            spec={
                "tasks": upload_details.taskset_ref.root,
                "target": {
                    "kind": "harbor",
                    "agent_import_path": "nemo_evaluator.examples.harbor_test_agent:WrappedAgent",
                    "n_attempts": 1,
                    "n_concurrent_trials": 1,
                    "max_retries": 0,
                },
            }
        ),
    ).data()
    (tmp_path / "job.json").write_text(job.model_dump_json(indent=2))
    deadline = time.monotonic() + 300
    try:
        while True:
            status = evaluator.get_agent_eval_job_status(workspace="default", name=job.name).data()
            (tmp_path / "status.json").write_text(status.model_dump_json(indent=2))
            if status.status.value in {"completed", "error", "failed", "cancelled"}:
                break
            if time.monotonic() >= deadline:
                pytest.fail(f"Job {job.name} timed out: {status.model_dump_json()}; diagnostics: {tmp_path}")
            time.sleep(2)
        assert status.status.value == "completed", (
            f"Job {job.name}: {status.model_dump_json()}; diagnostics: {tmp_path}"
        )
    finally:
        # Failure to retrieve logs must not obscure the original polling failure.
        try:
            logs = (
                evaluator.list_agent_eval_job_logs(workspace="default", name=job.name, query_params={"tail": 100})
                .page()
                .items
            )
            log_text = "\n".join(log.model_dump_json() for log in logs)
        except Exception as exc:
            log_text = f"Could not retrieve logs: {exc}"
        (tmp_path / "job-logs.jsonl").write_text(log_text)
        print(f"Job {job.name}; diagnostics: {tmp_path}\nRecent logs:\n{log_text}")

    payload = evaluator.download_agent_eval_job_result(
        workspace="default", job=job.name, name="agent-eval-results"
    ).read()
    (tmp_path / "agent-eval-results.tar.gz").write_bytes(payload)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as artifact:

        def read_member(filename: str) -> bytes:
            members = [member for member in artifact.getmembers() if Path(member.name).name == filename]
            assert len(members) == 1, (filename, members)
            stream = artifact.extractfile(members[0])
            assert stream is not None, filename
            with stream:
                return stream.read()

        trials = [json.loads(line) for line in read_member("trials.jsonl").splitlines() if line.strip()]
        scores = [json.loads(line) for line in read_member("scores.jsonl").splitlines() if line.strip()]
        summary = json.loads(read_member("summary.json"))

    assert len(trials) == len(scores) == 3, (trials, scores)
    by_task = {trial["task_id"]: trial for trial in trials}
    assert by_task.keys() == EXPECTED.keys(), trials
    debug = by_task["hello/debug-agent-runtime-error"]
    raw_path = Path(debug["metadata"]["harbor_trial_dir"]) / "result.json"
    raw_bytes = raw_path.read_bytes()
    (tmp_path / "debug-harbor-result.json").write_bytes(raw_bytes)
    raw = json.loads(raw_bytes)

    by_identity = {(score["task_id"], score["trial_id"]): score for score in scores}
    assert len(by_identity) == 3, scores
    for task_id, (expected_status, reward, format_ok) in EXPECTED.items():
        trial = by_task[task_id]
        assert trial["status"] == expected_status, trial
        assert trial["metadata"]["reward"] == reward, trial
        assert trial["metadata"]["reward_details"]["format_ok"] == format_ok, trial
        if expected_status == "completed":
            assert trial.get("error") is None, trial
        score = by_identity[(task_id, trial["id"])]
        assert score["status"] == "completed", score
        assert len(score["outputs"]) == 2, score
        assert {output["name"]: output["value"] for output in score["outputs"]} == {
            "reward": reward,
            "format_ok": format_ok,
        }, score

    assert debug["error"]["type"] == "AgentTimeoutError", debug
    assert summary["error_count"] == 1, summary
    assert summary["error_trial_ids"] == {"AgentTimeoutError": [debug["id"]]}, summary

    assert raw["exception_info"] is None, raw
    assert [step["step_name"] for step in raw["step_results"]] == ["attempt-answer"], raw
    assert raw["step_results"][0]["exception_info"]["exception_type"] == "AgentTimeoutError", raw
    print(f"Verified {job.name}: 3 trials, 3 scores, 1 AgentTimeoutError; artifacts: {tmp_path}")
