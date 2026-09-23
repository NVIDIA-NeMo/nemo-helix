# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from nemo_evaluator.api.schemas import EvaluatorTaskDefinition, MetricInline
from nemo_evaluator_sdk.agent_eval.runtimes.gym import discover_gym_tasks
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import discover_harbor_tasks
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_tasks import HarborAgentEvalTask
from nemo_evaluator_sdk.metrics.runner_rewards import HarborRewardMetric


def _package_files(root):
    (root / "environment").mkdir(exist_ok=True)
    (root / "environment" / "Dockerfile").write_text("FROM ubuntu")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test.sh").write_text("exit 0")


def _harbor_task(root: Path, *metrics, views=None) -> HarborAgentEvalTask:
    """A discovered-shape Harbor task for tests that mock the archive upload."""
    task = HarborAgentEvalTask(
        id="task",
        intent="task",
        inputs={"instruction": "Do it"},
        metrics=[HarborRewardMetric(), *metrics],
        source_dir=root,
    )
    # Views may name runner reward outputs that only preparation defers, as with a copied discovered task.
    return task.model_copy(update={"views": views or {}})


def test_harbor_discovery_root_shapes(tmp_path: Path):
    task = tmp_path / "task-a"
    task.mkdir()
    _package_files(task)
    (task / "task.toml").write_text('[task]\nname = "test/native-a"\n')
    (task / "instruction.md").write_text("Do it")
    assert list(discover_harbor_tasks(tmp_path)) == list(discover_harbor_tasks(task))
    assert discover_harbor_tasks(task)[0].source_dir == task.absolute()
    (tmp_path / "junk").mkdir()
    with pytest.raises(ValueError, match="entry"):
        discover_harbor_tasks(tmp_path)


def test_gym_discovery_preserves_row(tmp_path: Path):
    dataset = tmp_path / "gym.jsonl"
    dataset.write_text(json.dumps({"responses_create_params": {"input": "hello"}, "answer": 42}) + "\n")
    source = discover_gym_tasks(dataset)[0]
    assert source.metadata["gym_row_extras"] == {"answer": 42}


def test_prepare_gym_without_remote_writes(tmp_path: Path):
    from nemo_evaluator.sdk.task_resources import EvaluatorTasksResource
    from nemo_helix_plugin.evaluator.client import EvaluatorClient

    dataset = tmp_path / "gym.jsonl"
    dataset.write_text(json.dumps({"responses_create_params": {"input": "hello"}, "answer": 42}) + "\n")
    source = discover_gym_tasks(dataset)[0]
    prepared = EvaluatorTasksResource(EvaluatorClient(base_url="http://localhost:8080", workspace="default")).prepare(
        source
    )
    assert prepared.spec.kind == "evaluator"
    assert isinstance(prepared.spec.metrics[0], MetricInline)
    assert prepared.spec.metrics[0].metric_type == "gym_reward"
    assert {item.key: item.value for item in prepared.metadata} == {"gym_row_extras": {"answer": 42}}


@pytest.mark.parametrize("kind", ["empty", "symlink", "duplicate"])
def test_discovery_rejects_invalid_collection(tmp_path: Path, kind):
    if kind != "empty":
        for name in ("one", "two"):
            root = tmp_path / name
            root.mkdir()
            _package_files(root)
            (root / "task.toml").write_text('[task]\nname = "test/same"\n')
        if kind == "symlink":
            (tmp_path / "link").symlink_to(tmp_path / "one", target_is_directory=True)
    with pytest.raises(ValueError):
        discover_harbor_tasks(tmp_path)


@pytest.mark.parametrize(
    "metrics,message",
    [
        ([HarborRewardMetric(), HarborRewardMetric()], "exactly one HarborRewardMetric"),
        ([], "exactly one HarborRewardMetric"),
        ([HarborRewardMetric(output_name="score")], "Customized mandatory HarborRewardMetric"),
    ],
    ids=["duplicate", "missing", "customized"],
)
def test_harbor_mandatory_metric_must_stay_unmodified(tmp_path: Path, metrics, message):
    from unittest.mock import Mock

    from nemo_evaluator.sdk.task_preparation import prepare_task

    task = _harbor_task(tmp_path).model_copy(update={"metrics": metrics})
    client = Mock()
    with pytest.raises(ValueError, match=message):
        prepare_task(task, files_client=client, workspace="default")
    assert not client.mock_calls


class _CustomMetric:
    type = "publication-custom"

    def output_spec(self):
        from nemo_evaluator_sdk.metrics.protocol import MetricOutputSpec

        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input):
        raise NotImplementedError


@pytest.mark.parametrize("harbor", [False, True])
async def test_custom_metric_opt_in_and_async_preparation(tmp_path: Path, monkeypatch, harbor):
    from unittest.mock import AsyncMock

    from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskDefinition, HarborTaskHash
    from nemo_evaluator.sdk.task_resources import AsyncEvaluatorTasksResource
    from nemo_evaluator.shared.metric_bundles.hybrid import HybridMetricBundlePackager
    from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
    from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient

    upload = AsyncMock(
        return_value=HarborTaskDefinition(
            kind="harbor",
            native_task_id="task",
            source=HarborArchiveSource(fileset_ref="default/files#archive", files_hash="a" * 64),
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="test"),
        )
    )
    monkeypatch.setattr("nemo_evaluator.sdk.task_preparation.publish_harbor_task_archive_async", upload)
    source = (
        _harbor_task(tmp_path, _CustomMetric())
        if harbor
        else AgentEvalTask(id="task", intent="test", inputs={}, reference={"answer": 42}, metrics=[_CustomMetric()])
    )
    resource = AsyncEvaluatorTasksResource(AsyncEvaluatorClient(base_url="http://test", workspace="other"))
    with pytest.raises(RuntimeError, match="explicit metric_bundle_packager"):
        await resource.prepare(source)
    upload.assert_not_awaited()
    result = await resource.prepare(source, metric_bundle_packager=HybridMetricBundlePackager())
    assert isinstance(result.spec.metrics[0], MetricInline)
    assert result.spec.metrics[0].payload.kind == "cloudpickle"
    if harbor:
        assert upload.call_args.kwargs["fileset_ref"] == "other/harbor-tasks"
    else:
        assert isinstance(result.spec, EvaluatorTaskDefinition)
        assert result.spec.reference == {"answer": 42}


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_prepare_harbor_builtin_metric(tmp_path: Path, monkeypatch, asynchronous):
    from unittest.mock import AsyncMock, Mock

    from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskDefinition, HarborTaskHash
    from nemo_evaluator.sdk.task_resources import AsyncEvaluatorTasksResource, EvaluatorTasksResource
    from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
    from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient

    definition = HarborTaskDefinition(
        kind="harbor",
        native_task_id="task",
        source=HarborArchiveSource(fileset_ref="default/files#archive", files_hash="a" * 64),
        harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="test"),
    )
    source = _harbor_task(tmp_path, ExactMatchMetric(reference="Done", candidate="{{sample.output_text}}"))
    if asynchronous:
        monkeypatch.setattr(
            "nemo_evaluator.sdk.task_preparation.publish_harbor_task_archive_async",
            AsyncMock(return_value=definition),
        )
        result = await AsyncEvaluatorTasksResource(AsyncEvaluatorClient(base_url="http://test")).prepare(source)
    else:
        monkeypatch.setattr(
            "nemo_evaluator.sdk.task_preparation.publish_harbor_task_archive", Mock(return_value=definition)
        )
        result = EvaluatorTasksResource(EvaluatorClient(base_url="http://test")).prepare(source)
    assert isinstance(result.spec.metrics[0], MetricInline)
    assert result.spec.metrics[0].metric_type == "exact-match"
    assert result.spec.metrics[0].payload.kind == "inline"


def test_discovery_validates_nested_files_before_returning(tmp_path: Path):
    for name in ("first", "last"):
        root = tmp_path / name
        root.mkdir()
        _package_files(root)
        (root / "task.toml").write_text("")
        (root / "instruction.md").write_text("Do it")
    (tmp_path / "last" / "unsafe").symlink_to(tmp_path / "first" / "instruction.md")
    with pytest.raises(ValueError, match="[Ss]ymlink|[Uu]nsupported"):
        discover_harbor_tasks(tmp_path)


def test_harbor_views_defer_primary_reward_key_validation(tmp_path):
    from nemo_evaluator.sdk.task_preparation import _scoring
    from nemo_evaluator_sdk.agent_eval.tasks import SemanticReducer, SemanticView, ViewSignal
    from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric

    view = SemanticView(
        reducer=SemanticReducer.MEAN,
        signals=[
            ViewSignal(metric="harbor_reward", output="success"),
            ViewSignal(metric="exact-match", output="exact-match"),
        ],
    )
    source = _harbor_task(tmp_path, ExactMatchMetric(reference="Done"), views={"quality": view})
    _, metrics, _ = _scoring(source, None)
    assert len(metrics) == 1
    assert source.views["quality"].signals[0].output == "success"
    view.signals[1] = ViewSignal(metric="exact-match", output="missing")
    with pytest.raises(ValueError, match="unknown output"):
        _scoring(source, None)


def test_restored_harbor_record_rejected_before_bundling(tmp_path):
    from unittest.mock import Mock

    from nemo_evaluator.sdk.task_preparation import prepare_task
    from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTaskset

    _package_files(tmp_path)
    (tmp_path / "task.toml").write_text("")
    (tmp_path / "instruction.md").write_text("Do it")
    task = discover_harbor_tasks(tmp_path)[0]
    task.metrics.clear()  # Raw metric JSON is not an executable metric round trip.
    restored = AgentEvalTaskset.model_validate(AgentEvalTaskset(tasks=[task]).model_dump()).tasks[0]
    client = Mock()
    with pytest.raises(ValueError, match="Harbor tasks must be of type HarborAgentEvalTask"):
        prepare_task(restored, files_client=client, workspace="default")
    assert not client.mock_calls


def test_discovered_views_defer_runner_reward_outputs(tmp_path):
    from nemo_evaluator.sdk.task_preparation import _scoring
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _typed_task_dirs
    from nemo_evaluator_sdk.agent_eval.tasks import SemanticReducer, SemanticView, ViewSignal

    _package_files(tmp_path)
    (tmp_path / "task.toml").write_text("")
    (tmp_path / "instruction.md").write_text("Do it")
    task = discover_harbor_tasks(tmp_path)[0]
    task.views["quality"] = SemanticView(
        reducer=SemanticReducer.MEAN, signals=[ViewSignal(metric="harbor_reward", output="success")]
    )
    assert _typed_task_dirs([task]) == [tmp_path]
    validated, metrics, _ = _scoring(task, None)
    assert validated.views == task.views
    assert metrics == []
