# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from nemo_agents_plugin.jobs.pipeline import PipelineJob, PipelineJobConfig, PipelineStepFailedError
from nemo_helix_plugin.agents.pipelines import PipelineInline
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.jobs.api_factory import CPUExecutionProviderSpec

EXAMPLES = Path(__file__).parents[2] / "examples" / "pr-pipeline"


class ExecuteJobs:
    def __init__(self, responses=None, statuses=None):
        self.responses = responses or {}
        self.statuses = statuses or {}
        self.created = []
        self.cancelled = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return {"name": f"job-{len(self.created)}"}

    def get(self, name, **kwargs):
        return {"status": self.statuses.get(name, "completed")}

    def cancel(self, name, **kwargs):
        self.cancelled.append(name)

    def download_result(self, name, *, job, **kwargs):
        return json.dumps(self.responses.get(job, {"status": "succeeded", "response": "done"})).encode()

    def list_results(self, job, **kwargs):
        return {
            "data": [
                {
                    "name": "output_workdir",
                    "artifact_storage_type": "fileset",
                    "artifact_url": f"ws/fs#results/{job}/output_workdir",
                }
            ]
        }


class Files:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.paths = []

    def download_file(self, *, path, **kwargs):
        self.paths.append(path)
        value = self.artifacts[path]

        @contextmanager
        def stream(**kwargs):
            yield iter([value if isinstance(value, bytes) else json.dumps(value).encode()])

        return SimpleNamespace(stream=stream)


def run(config, tmp_path, jobs=None, artifacts=None, sleep=lambda _: None):
    jobs = jobs or ExecuteJobs()
    files = Files(artifacts or {})
    ctx = JobContext(workspace="ws", storage=StoragePaths(ephemeral=tmp_path), results=MagicMock())
    with (
        patch("nemo_agents_plugin.jobs.pipeline.AgentsClient"),
        patch("nemo_agents_plugin.jobs.pipeline._ExecuteJobsResource", return_value=jobs),
        patch("nemo_agents_plugin.jobs.pipeline.FilesClient.from_client", return_value=files),
    ):
        result = PipelineJob().run(config, ctx=ctx, sdk=MagicMock(), sleep=sleep)
    return result, jobs, files


def example(mode="patch"):
    definition = yaml.safe_load((EXAMPLES / "pipeline.yaml").read_text())
    definition.pop("name")
    spec = yaml.safe_load((EXAMPLES / "run.pipeline.example.yaml").read_text())
    spec["pipeline"] = definition
    spec["inputs"].update(ticket='Ticket with "quotes" and {{ literal }}', mode=mode)
    return spec


def artifacts(writer_status="pr_created", planner_status="ready"):
    return {
        "results/job-1/output_workdir/plan.json": {"status": planner_status, "summary": 'Quoted "plan" {{ literal }}'},
        "results/job-2/output_workdir/result.json": {
            "status": writer_status,
            "repo": "owner/repo",
            "pr_number": 7,
            "head_sha": "a" * 40,
        },
        "results/job-3/output_workdir/workspace/review-result.json": {"status": "completed", "findings": []},
    }


def test_three_harness_example_chains_json_artifacts_and_runtime_options(tmp_path):
    result, jobs, files = run(example("draft_pr"), tmp_path, artifacts=artifacts())
    assert [s["status"] for s in result["steps"]] == ["completed"] * 3
    plan, write, review = [json.loads(j["spec"]["input"]) for j in jobs.created]
    assert plan["ticket"] == 'Ticket with "quotes" and {{ literal }}'
    assert write["plan"]["summary"] == 'Quoted "plan" {{ literal }}'
    assert review == {"repo": "owner/repo", "pr_number": 7, "head_sha": "a" * 40, "github_token_env": "GH_TOKEN"}
    assert all(j["profile"] == "gpu" for j in jobs.created)
    assert all(j["spec"]["image"] == "pr-compose:local" for j in jobs.created)
    assert [j["spec"]["timeout_seconds"] for j in jobs.created] == [300, 600, 900]
    assert list(jobs.created[0]["spec"]["environment"]["environment_spec"]["secrets"]) == ["LINEAR_API_KEY"]
    assert jobs.created[2]["spec"]["environment"]["environment_spec"]["secrets"] == {
        "GH_TOKEN": "mmogallapall/gh-reviewer-token"
    }
    assert len(files.paths) == 3  # Individual files, not whole repository archives.


def test_patch_mode_skips_pr_review(tmp_path):
    result, jobs, _ = run(example(), tmp_path, artifacts=artifacts(writer_status="patch_ready"))
    assert len(jobs.created) == 2
    assert result["steps"][-1]["status"] == "skipped"


def test_incomplete_plan_stops_before_writer_and_persists_result(tmp_path):
    jobs = ExecuteJobs()
    with pytest.raises(PipelineStepFailedError, match="output check"):
        run(example(), tmp_path, jobs, artifacts(planner_status="needs_clarification"))
    assert len(jobs.created) == 1
    saved = json.loads((tmp_path / "pipeline_result.json").read_text())
    assert saved["status"] == "failed"
    assert saved["steps"][0]["response"]["status"] == "needs_clarification"


@pytest.mark.parametrize("bad", [b"not-json", b"\xff", b"x" * (4 * 1024 * 1024 + 1)])
def test_bad_artifact_stops_without_submitting_writer(tmp_path, bad):
    values = artifacts()
    values["results/job-1/output_workdir/plan.json"] = bad
    jobs = ExecuteJobs()
    with pytest.raises(PipelineStepFailedError):
        run(example(), tmp_path, jobs, values)
    assert len(jobs.created) == 1


def test_missing_artifact_does_not_fall_back_to_successful_text(tmp_path):
    jobs = ExecuteJobs()
    with pytest.raises(PipelineStepFailedError):
        run(example(), tmp_path, jobs, {})
    assert len(jobs.created) == 1


def text_config(retries=0):
    return {
        "pipeline": {
            "version": 1,
            "inputs": {"task": {}},
            "steps": [{"name": "a", "agent": "agent-a", "retries": retries}],
        },
        "inputs": {"task": "hello"},
    }


def test_known_failed_job_can_retry_and_records_both_attempts(tmp_path):
    result, jobs, _ = run(text_config(1), tmp_path, ExecuteJobs(statuses={"job-1": "error"}))
    assert len(jobs.created) == 2
    assert [a["job"] for a in result["steps"][0]["attempts"]] == ["job-1", "job-2"]


def test_timeout_cancels_child_without_retry(tmp_path):
    cfg = text_config(1)
    cfg["step_defaults"] = {"wait_timeout_seconds": 0.001}
    jobs = ExecuteJobs(statuses={"job-1": "active"})
    with patch("nemo_agents_plugin.jobs.pipeline.time", SimpleNamespace(monotonic=MagicMock(side_effect=[0, 1]))):
        with pytest.raises(PipelineStepFailedError, match="did not finish"):
            run(cfg, tmp_path, jobs)
    assert jobs.cancelled == ["job-1"]
    assert len(jobs.created) == 1


def test_cancellation_cancels_child_and_saves_failed_result(tmp_path):
    def stop(_):
        raise KeyboardInterrupt()

    jobs = ExecuteJobs(statuses={"job-1": "active"})
    with pytest.raises(KeyboardInterrupt):
        run(text_config(), tmp_path, jobs, sleep=stop)
    assert jobs.cancelled == ["job-1"]
    assert json.loads((tmp_path / "pipeline_result.json").read_text())["status"] == "failed"


def test_ambiguous_submit_error_is_not_retried(tmp_path):
    jobs = ExecuteJobs()
    jobs.create = MagicMock(side_effect=ConnectionError("connection lost"))
    with pytest.raises(PipelineStepFailedError, match="connection lost"):
        run(text_config(2), tmp_path, jobs)
    assert jobs.create.call_count == 1


@pytest.mark.asyncio
async def test_compile_snapshots_stored_pipeline_and_qualifies_agents():
    definition = PipelineInline.model_validate(text_config()["pipeline"])
    entities = SimpleNamespace(get=AsyncMock(return_value=definition))
    cfg = PipelineJobConfig(pipeline="source/my-pipeline", inputs={"task": "hello"}, image="runner:local")
    compiled = await PipelineJob.compile(
        workspace="caller", spec=cfg, entity_client=entities, job_name=None, async_sdk=None
    )
    snapshot = compiled.steps[0].config
    assert snapshot["pipeline"]["steps"][0]["agent"] == "source/agent-a"
    definition.steps[0].agent = "edited"
    assert snapshot["pipeline"]["steps"][0]["agent"] == "source/agent-a"
    executor = compiled.steps[0].executor
    assert isinstance(executor, CPUExecutionProviderSpec)
    assert executor.container.image == "runner:local"


def test_unknown_override_fails_before_running():
    cfg = text_config()
    cfg["step_overrides"] = {"typo": {"image": "bad"}}
    with pytest.raises(ValueError, match="Unknown step overrides"):
        PipelineJobConfig.model_validate(cfg)
