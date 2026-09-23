# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor taskset integration test with a local platform, Docker, and the deterministic example agent.

Run from the repository root with Python 3.12+ and Docker running::

    uv sync --frozen --package nemohelix --package nemo-evaluator-plugin --extra harbor
    NEMO_PLUGIN_SERVICES_ALLOWLIST=evaluator \\
    NEMO_PLUGIN_CONTROLLERS_ALLOWLIST='' \\
    RUN_AGENT_EVAL_INTEGRATION=1 UV_NO_SYNC=1 uv run --frozen --no-sync pytest \\
      plugins/nemo-evaluator/tests/integration/test_harbor_taskset.py -v -s -n 0

CI runs this serially after the general integration suite when the Harbor
integration path filter or dependency filter matches, and on manual workflow
dispatch. Without RUN_AGENT_EVAL_INTEGRATION=1 it skips before starting any
fixtures. Once opted in, missing Harbor or unavailable Docker FAILS rather than skips.

The shared fixture starts an isolated platform on port 8090 (override with
NHX_AGENT_BASE_URL); the port must be free. Database, files, and job storage are
temporary. Plugin allowlists exclude unrelated plugins while keeping core
services/controllers. UV_NO_SYNC preserves the Harbor extra in child processes.
The deterministic agent needs no API key or OpenAI calls; Docker image builds
may need network access.

Publication follows the Harbor runner guide: discover the dataset, replace each
task, replace the taskset from the returned IDs, and submit a HarborAgentTaskRunner.
Repeat publication must reuse the same revisions and archives. The job must
persist one Harbor result record. The three saved trials and
scores distinguish a correct greeting (reward=1), a completed wrong answer
(reward=0, format_ok=1), and an AgentTimeoutError (partial, reward=0). The summary
must count the error; raw Harbor output must show only the first step executed.
Runner submission uses the fixture's default subprocess profile, which has the
same configuration as harbor-test.

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
import uuid
from pathlib import Path

import pytest
from nemo_evaluator.api.schemas import TasksetRef
from nemo_evaluator.sdk.resources import Evaluator
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    HarborAgentTaskRunner,
    HarborRuntimeConfig,
    discover_harbor_tasks,
)
from nemo_helix_plugin.evaluator.client import EvaluatorClient

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


def test_harbor_taskset(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    importlib.import_module("harbor")
    try:
        docker = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.fail(f"Docker daemon is required for the opted-in integration test: {exc}")
    if docker.returncode:
        pytest.fail(f"Docker daemon is required: {docker.stderr.decode(errors='replace')}")

    # Resolve only after prerequisite checks, so skipped tests never launch services.
    base_url = request.getfixturevalue("subprocess_platform")
    evaluator_client = EvaluatorClient(base_url=base_url, workspace="default")
    evaluator = Evaluator(client=evaluator_client)
    name = f"harbor-taskset-{uuid.uuid4().hex[:12]}"

    def publish():
        tasks = [
            evaluator.tasks.replace(f"{name}-{task.id.split('/')[1]}", task=task)
            for task in discover_harbor_tasks(DATASET)
        ]
        return tasks, evaluator.tasksets.replace(name, tasks=[task.id for task in tasks])

    tasks, taskset = publish()
    repeated_tasks, repeated_taskset = publish()
    assert {task.spec.native_task_id for task in tasks} == EXPECTED.keys()
    assert [task.revision for task in repeated_tasks] == [task.revision for task in tasks]
    assert [task.spec.source.fileset_ref for task in repeated_tasks] == [task.spec.source.fileset_ref for task in tasks]
    assert repeated_taskset.revision == taskset.revision
    assert len(taskset.tasks) == 3
    assert all("#" in ref.root for ref in taskset.tasks)
    assert repeated_taskset.tasks == taskset.tasks
    (tmp_path / "publication.json").write_text(
        json.dumps([task.model_dump(mode="json") for task in tasks] + [taskset.model_dump(mode="json")], indent=2)
    )

    # The runner has no profile selector, so the fixture's default subprocess profile is used.
    job = evaluator.submit(
        tasks=TasksetRef(f"default/{name}"),
        target=HarborAgentTaskRunner(
            config=HarborRuntimeConfig(
                agent_import_path="nemo_evaluator.examples.harbor_test_agent:WrappedAgent",
                n_attempts=1,
                n_concurrent_trials=1,
                max_retries=0,
            ),
        ),
    )
    (tmp_path / "job.json").write_text(job.job.model_dump_json(indent=2))
    try:
        job.wait_until_done(poll_interval_seconds=2, job_timeout_seconds=300, pending_timeout_seconds=300)
    finally:
        # Failure to retrieve diagnostics must not obscure the original polling failure.
        try:
            (tmp_path / "status.json").write_text(job.get_job_status().model_dump_json(indent=2))
        except Exception as exc:
            (tmp_path / "status.json").write_text(json.dumps({"error": str(exc)}))
        try:
            logs = (
                evaluator_client.list_agent_eval_job_logs(
                    workspace="default", name=job.name, query_params={"tail": 100}
                )
                .page()
                .items
            )
            log_text = "\n".join(log.model_dump_json() for log in logs)
        except Exception as exc:
            log_text = f"Could not retrieve logs: {exc}"
        (tmp_path / "job-logs.jsonl").write_text(log_text)
        print(f"Job {job.name}; diagnostics: {tmp_path}\nRecent logs:\n{log_text}")

    stored_results = evaluator.agent_eval_results.list(job_id=job.name).data
    assert len(stored_results) == 1, stored_results
    assert stored_results[0].target_kind == "harbor", stored_results[0]

    payload = evaluator_client.download_agent_eval_job_result(
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
