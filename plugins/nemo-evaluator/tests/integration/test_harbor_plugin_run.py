# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-level end-to-end Harbor run through the real local execution machinery.

Unlike the unit tests (which fake the evaluator so the Harbor runner is built but never executed),
this drives ``NemoJobScheduler().run_local(AgentEvalJob, ...)`` against a *real* HarborRunnerTarget:
the scheduler validates the submitter ``AgentEvalInputSpec``, runs ``to_spec`` to the canonical spec,
builds/uses the job context, then runs — the same in-process lifecycle a local job goes through. The
Harbor runner resolves to a native ``HarborAgentTaskRunner``, Harbor runs the bundled hello-world task
in Docker, and the verifier reward is scored (by a cloudpickle-bundled ``HarborRewardMetric``) and
persisted into the run bundle. It is the plugin analog of the SDK's ``test_harbor_runtime_e2e.py``.

Needs the ``harbor`` extra (Python >=3.12; ``pip install nemo-evaluator-sdk[harbor]``) and a working
Docker daemon; ``importorskip('harbor')`` + a Docker check skip it otherwise (so it's inert on the
3.11 workspace and in CI). Marked ``integration`` — a heavy, external-dependency run — but unlike the
sibling tests it stands up no platform and isn't gated on ``RUN_AGENT_EVAL_INTEGRATION``.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

import pytest
from nemo_evaluator.api.schemas import MetadataItem, MetricInline, TaskInputs
from nemo_evaluator.jobs.agent_evaluate import AGENT_BUNDLE_DIR, DEFAULT_RESULT_NAME, AgentEvalJob
from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec, AgentEvalTaskInput, HarborRunnerTarget
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborRewardMetric, discover_harbor_tasks
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from nemo_platform_plugin.scheduler import NemoJobScheduler

pytestmark = [pytest.mark.integration]


@pytest.mark.skipif(
    not os.environ.get("RUN_AGENT_EVAL_INTEGRATION"), reason="requires opt-in platform/subprocess/Docker integration"
)
@pytest.mark.parametrize("direct", [False, True], ids=["taskset", "direct-list"])
@pytest.mark.timeout(900)
def test_publish_stored_harbor_source_and_execute(request, tmp_path, direct):
    pytest.importorskip("harbor")
    if not _docker_available():
        pytest.skip("Docker daemon is required to run a Harbor job")
    import httpx
    from nemo_evaluator.api.schemas import TaskInput, TaskRef, TasksetInput
    from nemo_evaluator.harbor.publication import publish_harbor_task_archive
    from nemo_platform import NeMoPlatform
    from nemo_platform_plugin.files.client import FilesClient
    from nemo_platform_plugin.files.types import CreateFilesetRequest
    from nemo_platform_plugin.jobs.client import JobsClient

    base_url = request.getfixturevalue("subprocess_platform")
    name = f"harbor-bridge-{uuid.uuid4().hex[:8]}"
    task_dir = _DATASET_DIR / "hello-world"
    files_client = FilesClient(base_url=base_url, workspace="default")
    files_client.create_fileset(body=CreateFilesetRequest(name=name)).data()
    definition = publish_harbor_task_archive(task_dir, files_client=files_client, fileset_ref=f"default/{name}")
    sdk = NeMoPlatform(base_url=base_url, workspace="default")
    sdk.evaluator.tasks.create(
        name,
        task=TaskInput(spec=definition),
    )
    sdk.evaluator.tasksets.create(name, taskset=TasksetInput(tasks=[TaskRef(f"default/{name}")]))
    route = f"{base_url}/apis/evaluator/v2/workspaces/default/agent-evaluate/jobs"
    public_tasks = [f"default/{name}"] if direct else f"default/{name}"
    response = httpx.post(
        route,
        json={
            "profile": "harbor-test",
            "spec": {
                "tasks": public_tasks,
                "target": {"kind": "harbor", "agent_name": "oracle"},
            },
        },
        timeout=60,
    )
    assert response.status_code == 201, response.text
    job = response.json()
    assert job["spec"]["tasks"]["kind"] == ("harbor-task-list" if direct else "harbor-taskset")
    jobs_client = JobsClient(base_url=base_url, workspace="default")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        status = httpx.get(f"{route}/{job['name']}/status", timeout=30).raise_for_status().json()
        if status["status"] in {"completed", "error", "failed", "cancelled"}:
            break
        time.sleep(2)
    assert status["status"] == "completed", status
    payload = jobs_client.download_job_result(job=job["name"], name=DEFAULT_RESULT_NAME).read()
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        trial_file = next(member for member in tar.getmembers() if member.name.endswith("trials.jsonl"))
        stream = tar.extractfile(trial_file)
        assert stream is not None
        trials = [json.loads(line) for line in stream if line.strip()]
        assert trials and {trial["task_id"] for trial in trials} == {"harbor/hello-world"}
        score_file = next(member for member in tar.getmembers() if member.name.endswith("scores.jsonl"))
        stream = tar.extractfile(score_file)
        assert stream is not None
        scores = [json.loads(line) for line in stream if line.strip()]
        assert scores and all(score["status"] == "completed" for score in scores), scores
        assert any(
            output["name"] == "reward" and output["value"] == 1.0 for score in scores for output in score["outputs"]
        ), scores


#: The bundled Harbor hello-world dataset (repo root → SDK examples).
_DATASET_DIR = Path(__file__).resolve().parents[4] / "packages/nemo_evaluator_sdk/examples/harbor/hello_world_dataset"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        # Bound the probe: a wedged daemon can make ``docker info`` hang until the
        # test-level timeout. Treat a stalled daemon as unavailable and skip.
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _reward_metric() -> MetricInline:
    # HarborRewardMetric is a custom metric (not in MetricsUnion), so it rides as a cloudpickle bundle.
    bundle = bundle_metric(HarborRewardMetric(), CloudpickleMetricBundlePackager())
    return MetricInline.model_validate(bundle.model_dump(mode="json"))


def _job_context(tmp_path: Path) -> JobContext:
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    return JobContext(
        workspace="dev",
        storage=storage,
        results=LocalJobResults(root=storage.persistent / "results"),
        job_id="harbor-plugin-run",
    )


@pytest.mark.timeout(600)
def test_run_local_runs_a_real_harbor_target(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    if not _docker_available():
        pytest.skip("Docker daemon is required to run a Harbor job")

    # Reuse the SDK's dataset discovery so the task id matches the name Harbor writes into result.json,
    # and the tasks carry the `harbor_dataset_path` metadata the native runner reads. Each task is
    # scored by a HarborRewardMetric, re-bundled here into the plugin's inline-metric wire shape.
    runtime_tasks = discover_harbor_tasks(_DATASET_DIR)
    assert runtime_tasks, "no Harbor tasks discovered in the hello-world dataset"
    input_spec = AgentEvalInputSpec(
        tasks=[
            AgentEvalTaskInput(
                id=rt.id,
                intent=rt.intent,
                inputs=TaskInputs.model_validate(rt.inputs),
                metrics=[_reward_metric()],
                metadata=[MetadataItem(key=key, value=str(value)) for key, value in rt.metadata.items()],
            )
            for rt in runtime_tasks
        ],
        target=HarborRunnerTarget(agent_name="oracle"),
    )
    ctx = _job_context(tmp_path)

    # The real local execution path: the scheduler validates the submitter spec, runs to_spec, and
    # invokes run() with the real evaluator (no fake) → resolve the Harbor target → native
    # HarborAgentTaskRunner → Harbor runs the task in Docker → adapt to trials → score → persist. The
    # explicit ctx keeps storage hermetic under tmp_path so the persisted bundle is readable here.
    result = NemoJobScheduler().run_local(AgentEvalJob, input_spec.model_dump(mode="json"), workspace="dev", ctx=ctx)

    assert result["status"] == "completed", result
    assert result["artifact"]["name"] == DEFAULT_RESULT_NAME

    # The persisted bundle records the Harbor verifier reward: the oracle agent solves hello-world, so
    # the reward metric scores 1.0.
    scores_path = ctx.storage.persistent / AGENT_BUNDLE_DIR / "scores.jsonl"
    assert scores_path.exists(), "run bundle is missing scores.jsonl"
    scores_text = scores_path.read_text(encoding="utf-8")
    assert '"reward"' in scores_text
    assert "1.0" in scores_text
