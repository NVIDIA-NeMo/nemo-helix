# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline stored Harbor jobs must score without native Harbor or archive access."""

import builtins
import json

import pytest
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash, ResolvedHarborTaskDefinition
from nemo_evaluator.api.task_definitions.provenance import TaskProvenance
from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
from nemo_evaluator.jobs.agent_spec import AgentEvalSpec, ResolvedTask
from nemo_evaluator.jobs.kinds.registry import KIND_ADAPTERS
from nemo_evaluator.jobs.metric_resolution import to_inline
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator.task_refs import validate_scoring
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner, HarborTasksetLoader
from nemo_evaluator_sdk.agent_eval.tasks import SemanticView
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentOutput, TrialError
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults


def _source(reward_key="grade"):
    """Build a Harbor snapshot whose quality view combines a saved reward with an exact-match metric."""
    metric = to_inline(
        bundle_metric(
            ExactMatchMetric(reference="{{inputs.instruction}}", candidate="{{sample.output_text}}"),
            CloudpickleMetricBundlePackager(),
        )
    )
    return [
        ResolvedTask(
            id="native/task",
            spec=ResolvedHarborTaskDefinition(
                provenance=TaskProvenance(entity_name="default/stored", revision_digest="a" * 64),
                kind="harbor",
                native_task_id="native/task",
                instruction="<!-- SPDX-License-Identifier: Apache-2.0 -->\n\n Do it \n",
                source=HarborArchiveSource(fileset_ref="default/files#archive", files_hash="b" * 64),
                harbor_hash=HarborTaskHash(digest="c" * 64, harbor_version="test"),
                metrics=[metric],
                views={
                    "quality": SemanticView.model_validate(
                        {
                            "reducer": "mean",
                            "signals": [
                                {"metric": "harbor_reward", "output": reward_key},
                                {"metric": "exact-match", "output": "exact-match"},
                            ],
                        }
                    )
                },
            ),
        )
    ]


def _trial(reward_key="grade", **changes):
    return AgentEvalTrial.model_validate(
        {
            "id": "trial",
            "task_id": "native/task",
            "status": "completed",
            "output": AgentOutput(output_text="Do it"),
            "metadata": {
                "harbor_primary_reward_key": reward_key,
                "reward": 0.5,
                "reward_details": {reward_key: 0.5, "secondary": 0.25},
            },
            **changes,
        }
    )


@pytest.fixture
def offline_worker(monkeypatch, tmp_path):
    """Provide worker storage while failing any attempt to resolve entities, fetch archives, or execute Harbor."""
    monkeypatch.setattr("nemo_evaluator.jobs.agent_evaluate.persist_agent_eval_result", lambda *a, **kw: None)

    def forbidden(*args, **kwargs):
        pytest.fail("Offline scoring attempted task resolution, archive access or Harbor execution")

    from nemo_helix_plugin.entities import EntityClient, SyncEntityClient
    from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient

    for entity_client in (EntityClient, SyncEntityClient):
        monkeypatch.setattr(entity_client, "get", forbidden)
        monkeypatch.setattr(entity_client, "list", forbidden)
    monkeypatch.setattr("nemo_evaluator.harbor.preparation.run_with_isolated_async_client", forbidden)
    monkeypatch.setattr(FilesClient, "download_file", forbidden)
    monkeypatch.setattr(AsyncFilesClient, "download_file", forbidden)
    monkeypatch.setattr(HarborTasksetLoader, "load", forbidden)
    monkeypatch.setattr(HarborAgentTaskRunner, "__init__", forbidden)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "harbor" or name.startswith("harbor.") or name == "nemo_evaluator.harbor.materialization":
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    ctx = JobContext(
        workspace="default",
        job_id="rescore",
        storage=StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent"),
        results=LocalJobResults(root=tmp_path / "results"),
    )
    return ctx


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reward_key", ["reward", "grade"])
@pytest.mark.parametrize("failed", [False, True])
def test_rescore_executes_metrics_views_without_harbor(offline_worker, tmp_path, asynchronous, reward_key, failed):
    """Rescore saved trials through the worker and verify rewards, custom metrics, views, and error counts."""
    ctx = offline_worker
    changes = {"status": "partial", "error": TrialError(type="RuntimeError", message="agent failed")} if failed else {}
    trial = _trial(reward_key, **changes)
    spec = AgentEvalSpec(tasks=_source(reward_key), trials=[trial])
    client = NemoClient(base_url="http://unused.test", workspace="default")
    async_client = AsyncNemoClient(base_url="http://unused.test", workspace="default") if asynchronous else None
    result = AgentEvalJob()._run_with_client(
        spec.model_dump(mode="json"),
        ctx=ctx,
        platform_client=client,
        async_client=async_client,
    )
    assert result["status"] == "completed"
    assert not (tmp_path / "persistent" / "harbor-inputs").exists()
    summary = json.loads(next((tmp_path / "persistent").rglob("summary.json")).read_text())
    scores = {score["name"]: score for score in summary["scores"]["scores"]}
    assert scores[f"harbor_reward.{reward_key}"]["mean"] == 0.5
    assert scores["harbor_reward.secondary"]["mean"] == 0.25
    assert scores["exact-match.exact-match"]["mean"] == 1.0
    assert scores["view.quality"]["mean"] == 0.75
    assert summary["error_count"] == int(failed)


@pytest.mark.parametrize(
    "metadata,error",
    [
        ({}, "requires harbor_primary_reward_key"),
        ({"harbor_primary_reward_key": ""}, "invalid Harbor reward_key"),
        ({"harbor_primary_reward_key": 42}, "requires harbor_primary_reward_key"),
        ({"harbor_primary_reward_key": "other"}, "consistent primary reward_key"),
    ],
)
def test_canonical_rescore_requires_explicit_consistent_key(metadata, error):
    with pytest.raises(ValueError, match=error):
        spec = AgentEvalSpec(tasks=_source(), trials=[_trial(), _trial(metadata=metadata)])
        validate_scoring(spec.tasks, target=spec.target, trials=spec.trials, adapters=KIND_ADAPTERS)


@pytest.mark.parametrize("unknown", [False, True])
def test_rescore_preserves_sdk_trial_coverage_checks(offline_worker, unknown):
    """Verify offline preparation still lets the evaluator reject missing or unknown task trials."""
    ctx = offline_worker
    source = _source()
    trials = [_trial(task_id="unknown" if unknown else "native/task")]
    if not unknown:
        missing = source[0].model_copy(deep=True)
        missing.spec.provenance.entity_name = "default/missing"
        missing.spec.native_task_id = "missing"
        missing.id = "missing"
        source.append(missing)
    with pytest.raises(ValueError, match="unknown task" if unknown else "no trials produced"):
        AgentEvalJob()._run_with_client(
            AgentEvalSpec(tasks=source, trials=trials).model_dump(mode="json"),
            ctx=ctx,
            platform_client=NemoClient(base_url="http://unused.test"),
            async_client=None,
        )


def test_offline_preparation_needs_no_platform_client(offline_worker, tmp_path):
    from nemo_evaluator.harbor.preparation import prepare_stored_harbor_tasks

    tasks = prepare_stored_harbor_tasks(
        _source(),
        destination_root=tmp_path / "unused",
        client=None,
        async_client=None,
        reward_key="grade",
        trials=[_trial()],
    )
    assert [task.id for task in tasks] == ["native/task"]
    assert not (tmp_path / "unused").exists()
