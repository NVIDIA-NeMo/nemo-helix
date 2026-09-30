# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_tasks import HarborAgentEvalTask, HarborTaskCollection


def test_collection_and_scoring_preserve_source(tmp_path: Path):
    task = HarborAgentEvalTask(id="one", intent="one", inputs={}, source_dir=tmp_path)
    tasks = HarborTaskCollection([task])
    assert list(tasks) == list(tasks)
    assert tasks[-1] is task
    assert isinstance(tasks[:1], HarborTaskCollection)
    assert tasks[:1][0].source_dir == tmp_path.absolute()
    assert len(tasks[:0]) == 0
    with pytest.raises(ValueError):
        task.source_dir = tmp_path / "other"


def _package(root: Path, name: str = "test/native") -> Path:
    root.mkdir()
    (root / "task.toml").write_text(f'[task]\nname = "{name}"\n')
    (root / "instruction.md").write_text("Do it")
    (root / "environment").mkdir()
    (root / "environment" / "Dockerfile").write_text("FROM ubuntu")
    (root / "tests").mkdir()
    (root / "tests" / "test.sh").write_text("exit 0")
    return root


def test_discovery_validates_and_ignores_root_noise(tmp_path):
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks

    root = _package(tmp_path / "one")
    (tmp_path / "README.md").write_text("notes")
    tasks = discover_harbor_tasks(tmp_path)
    assert isinstance(tasks, HarborTaskCollection)
    assert tasks[0].source_dir == root
    assert tasks[0].metadata["harbor_dataset_path"] == str(tmp_path)
    (tmp_path / "invalid").mkdir()
    with pytest.raises(ValueError, match="entry"):
        discover_harbor_tasks(tmp_path)


async def test_native_selection_uses_typed_paths(tmp_path):
    from harbor.models.job.config import DatasetConfig
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
        _dataset_path_from_tasks,
        _harbor_folder_names,
        discover_harbor_tasks,
    )

    root = _package(tmp_path / "one[1]")
    _package(tmp_path / "one1", "test/other")
    tasks = discover_harbor_tasks(root)
    tasks[0].metadata.clear()
    parent = _dataset_path_from_tasks(tasks)
    assert parent == tmp_path
    configs = await DatasetConfig(path=parent, task_names=_harbor_folder_names(tasks)).get_task_configs()
    assert [c.path for c in configs] == [root]


async def test_finalization_and_persisted_scoring_boundary(tmp_path):
    import json

    from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
    from nemo_evaluator_sdk.agent_eval.persistence import persist_run
    from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask, AgentEvalTaskset
    from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput, RunnerInfo
    from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric

    class Runner:
        def runner_info(self):
            return RunnerInfo(name="test", kind="runner")

        async def run_tasks(self, tasks, config=None):
            assert isinstance(tasks[0], HarborAgentEvalTask)
            return [
                AgentEvalTrial(
                    id="trial",
                    task_id=tasks[0].id,
                    status=AgentEvalTrialStatus.COMPLETED,
                    output=AgentOutput(output_text="done"),
                    metadata={"reward_details": {"reward": 1.0}},
                )
            ]

        def scoring_metrics(self, task, trials):
            assert len(trials) == 1
            return [HarborRewardMetric()]

    task = HarborAgentEvalTask(id="one", intent="one", inputs={}, source_dir=tmp_path)
    result = await AgentEvaluator().run(tasks=HarborTaskCollection([task]), target=Runner())
    assert type(result.tasks[0]) is AgentEvalTask
    restored = AgentEvalTaskset.model_validate(AgentEvalTaskset(tasks=[task]).model_dump())
    assert type(restored.tasks[0]) is AgentEvalTask
    persist_run(result, tmp_path / "bundle", write_html_dashboard=False)
    row = json.loads((tmp_path / "bundle/tasks.jsonl").read_text())
    assert "source_dir" not in row
    row["metrics"] = [HarborRewardMetric()]  # Reapply executable scoring configuration.
    scoring_task = AgentEvalTask.model_validate(row)
    from nemo_evaluator_sdk.agent_eval.persistence import read_trials

    rescored = await AgentEvaluator().run(tasks=[scoring_task], trials=read_trials(tmp_path / "bundle"))
    assert rescored.summary.scores.scores


async def test_typed_selection_rejects_redirect_before_touching_cache(tmp_path):
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
        HarborAgentTaskRunner,
        HarborRuntimeConfig,
        discover_harbor_tasks,
    )

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    source = _package(a / "task")
    other = _package(b / "task")
    tasks = discover_harbor_tasks(source)
    tasks[0].metadata.update(harbor_task_dir=str(other), harbor_dataset_path=str(b))
    jobs = tmp_path / "jobs"
    (jobs / "cached").mkdir(parents=True)
    marker = jobs / "cached/keep"
    marker.write_text("existing")
    runner = HarborAgentTaskRunner(config=HarborRuntimeConfig(jobs_dir=jobs, job_name="cached"), dataset_path=b)
    with pytest.raises(ValueError, match="override"):
        await runner.run_tasks(tasks)
    assert marker.read_text() == "existing"
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _task_dirs_for

    assert _task_dirs_for(a, tasks) == {tasks[0].id: source}


@pytest.mark.parametrize("kind", ["root_symlink", "hidden_symlink", "case_duplicate", "empty"])
def test_discovery_root_policy(tmp_path, kind):
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks

    root = tmp_path / "suite"
    root.mkdir()
    if kind != "empty":
        task = _package(root / "one", "test/One")
        if kind == "case_duplicate":
            _package(root / "two", "test/one")
        elif kind == "hidden_symlink":
            (root / ".hidden").symlink_to(task)
        else:
            (tmp_path / "link").symlink_to(root)
            root = tmp_path / "link"
    with pytest.raises(ValueError):
        discover_harbor_tasks(root)


@pytest.mark.parametrize("target", ["test.sh", "../../outside"])
def test_discovery_keeps_only_task_internal_symlinks(tmp_path, target):
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks

    (tmp_path / "outside").write_text("secret")
    (tmp_path / "suite").mkdir()
    task = _package(tmp_path / "suite" / "one")
    (task / "tests" / "run.sh").symlink_to(target)
    if target.startswith(".."):
        with pytest.raises(ValueError, match="Symlink escapes"):
            discover_harbor_tasks(tmp_path / "suite")
        return
    [discovered] = discover_harbor_tasks(tmp_path / "suite")
    assert discovered.source_dir == task
    assert (task / "tests" / "run.sh").readlink() == Path("test.sh")


async def test_views_validate_against_finalized_reward_outputs(tmp_path):
    from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
    from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask, SemanticReducer, SemanticView, ViewSignal
    from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput, RunnerInfo
    from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric

    class Runner:
        def runner_info(self):
            return RunnerInfo(name="test", kind="runner")

        async def run_tasks(self, tasks, config=None):
            return [
                AgentEvalTrial(
                    id="trial",
                    task_id=tasks[0].id,
                    status=AgentEvalTrialStatus.COMPLETED,
                    output=AgentOutput(output_text="done"),
                    metadata={"reward_details": {"reward": 1.0, "accuracy": 0.5}},
                )
            ]

        def scoring_metrics(self, task, trials):
            from nemo_evaluator_sdk.agent_eval.runtimes.harbor_scoring import harbor_scoring_metrics

            return harbor_scoring_metrics(task, trials, reward_key="reward")

    view = SemanticView(reducer=SemanticReducer.MEAN, signals=[ViewSignal(metric="harbor_reward", output="accuracy")])
    base = AgentEvalTask(id="one", intent="one", inputs={}, metrics=[HarborRewardMetric()]).model_copy(
        update={"views": {"quality": view}}
    )
    harbor = HarborAgentEvalTask(id="one", intent="one", inputs={}, metrics=[HarborRewardMetric()], source_dir=tmp_path)
    harbor = harbor.model_copy(update={"views": {"quality": view}})
    for task in (base, harbor):
        result = await AgentEvaluator().run(tasks=[task], target=Runner())
        assert type(result.tasks[0]) is AgentEvalTask
        assert result.tasks[0].views == {"quality": view}
