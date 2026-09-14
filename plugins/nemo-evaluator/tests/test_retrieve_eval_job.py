# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.retrieve_eval import (
    EVAL_RESULTS_FILE_NAME,
    EVAL_RESULTS_RESULT_NAME,
    RetrievalInputSpec,
    RetrieveEvalInputSpec,
    RetrieveEvalJob,
    RetrieveEvalSpec,
)
from nemo_evaluator_sdk.metrics.retrieval import RetrievalNDCGMetric, RetrievalRecallMetric
from nemo_evaluator_sdk.retrieval.beir import BeirDataset
from nemo_evaluator_sdk.retrieval.nim_ranking import NimRankingError
from nemo_evaluator_sdk.values.models import Model, ModelRef, RankingInference
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.results import AggregatedMetricResult, AggregateRangeScore
from nemo_evaluator_sdk.values.retrieval import Retrieval
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from nemo_platform_plugin.jobs.api_factory import CPUExecutionProviderSpec
from nemo_platform_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_platform_plugin.sdk import AsyncNeMoPlatform, NeMoPlatform
from pydantic import ValidationError
from pytest_mock import MockerFixture


def _context(tmp_path: Path) -> JobContext:
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    return JobContext(
        workspace="default",
        storage=storage,
        results=LocalJobResults(root=storage.persistent / "results"),
    )


def _spec() -> RetrieveEvalSpec:
    return RetrieveEvalSpec(
        dataset=FilesetRef("default/retrieval-data"),
        target=Retrieval(embeddings=Model(url="https://igw.example.test/v1/chat/completions", name="embed")),
        k=[1, 10],
    )


def _result(*, ndcg: float = 0.75, recall: float = 1.0) -> BenchmarkEvaluationResult:
    return BenchmarkEvaluationResult(
        row_scores=[],
        aggregate_scores=AggregatedMetricResult(
            scores=[
                AggregateRangeScore(
                    name="retrieval-ndcg.ndcg_cut_10",
                    count=2,
                    nan_count=0,
                    mean=ndcg,
                ),
                AggregateRangeScore(
                    name="retrieval-recall.recall_10",
                    count=2,
                    nan_count=0,
                    mean=recall,
                ),
                AggregateRangeScore(
                    name="retrieval-precision.P_10",
                    count=2,
                    nan_count=0,
                    mean=0.1,
                ),
                AggregateRangeScore(
                    name="retrieval-map.map_cut_10",
                    count=2,
                    nan_count=0,
                    mean=0.7,
                ),
            ]
        ),
        per_metric={},
    )


@pytest.mark.parametrize("k", [[], [0], [1, 1]])
def test_input_spec_rejects_invalid_cutoffs(k: list[int]) -> None:
    with pytest.raises(ValidationError):
        RetrieveEvalInputSpec(dataset=FilesetRef("default/data"), target=_spec().target.embeddings, k=k)


async def test_to_spec_forwards_retrieval_pipeline_fields() -> None:
    submit = RetrieveEvalInputSpec(
        dataset=FilesetRef("default/data"),
        target=RetrievalInputSpec(
            embeddings=_spec().target.embeddings,
            first_stage_k=50,
            truncate_long_documents=None,
            batch_size=16,
            embedding_dimensions=1024,
        ),
    )

    canonical = await RetrieveEvalJob.to_spec(
        submit,
        workspace="default",
        entity_client=object(),
        async_sdk=None,
        is_local=True,
    )

    assert isinstance(canonical, RetrieveEvalSpec)
    assert canonical.target.first_stage_k == 50
    assert canonical.target.truncate_long_documents is None
    assert canonical.target.batch_size == 16
    assert canonical.target.embedding_dimensions == 1024


async def test_to_spec_preflights_and_stamps_reranker_model_ref(mocker: MockerFixture) -> None:
    resolved = Model(
        url="https://igw.example.test/model/reranker/-/v1",
        name="reranker",
        served_model_name="publisher/reranker",
    )
    stamped = resolved.model_copy(update={"inference": RankingInference(contract="hosted-rerank-v1", path="/rerank")})
    resolve = mocker.patch(
        "nemo_evaluator.jobs.retrieve_eval.PlatformMetricModelResolver.resolve_model",
        new=mocker.AsyncMock(return_value=resolved),
    )
    ranker = mocker.patch("nemo_evaluator.jobs.retrieve_eval.NimRankingClient")
    ranker.return_value.preflight = mocker.AsyncMock(return_value=stamped)
    submit = RetrieveEvalInputSpec(
        dataset=FilesetRef("default/data"),
        target=RetrievalInputSpec(
            embeddings=_spec().target.embeddings,
            reranker=ModelRef("default/reranker"),
        ),
    )

    async with AsyncNeMoPlatform(base_url="https://platform.example.test", workspace="default") as async_sdk:
        canonical = await RetrieveEvalJob.to_spec(
            submit,
            workspace="default",
            entity_client=object(),
            async_sdk=async_sdk,
            is_local=False,
        )

    resolve.assert_awaited_once()
    ranker.return_value.preflight.assert_awaited_once()
    assert isinstance(canonical, RetrieveEvalSpec)
    assert canonical.target.reranker == stamped


async def test_to_spec_rejects_incompatible_reranker_model_ref(mocker: MockerFixture) -> None:
    resolved = Model(url="https://igw.example.test/v1", name="reranker")
    mocker.patch(
        "nemo_evaluator.jobs.retrieve_eval.PlatformMetricModelResolver.resolve_model",
        new=mocker.AsyncMock(return_value=resolved),
    )
    ranker = mocker.patch("nemo_evaluator.jobs.retrieve_eval.NimRankingClient")
    ranker.return_value.preflight = mocker.AsyncMock(side_effect=NimRankingError("no compatible ranking contract"))
    submit = RetrieveEvalInputSpec(
        dataset=FilesetRef("default/data"),
        target=RetrievalInputSpec(
            embeddings=_spec().target.embeddings,
            reranker=ModelRef("default/reranker"),
        ),
    )

    async with AsyncNeMoPlatform(base_url="https://platform.example.test", workspace="default") as async_sdk:
        with pytest.raises(ValueError, match="Reranker ModelRef 'default/reranker'.*no compatible"):
            await RetrieveEvalJob.to_spec(
                submit,
                workspace="default",
                entity_client=object(),
                async_sdk=async_sdk,
                is_local=False,
            )


def test_truncation_json_schema_marks_null_as_allowed() -> None:
    schema = RetrievalInputSpec.model_json_schema()["properties"]["truncate_long_documents"]
    assert schema.get("nullable") is True
    assert schema.get("enum") == ["end", "start"] or any(
        member.get("enum") == ["end", "start"] for member in schema.get("anyOf", []) if isinstance(member, dict)
    )


async def test_compile_builds_cpu_retrieve_eval_task() -> None:
    job = await RetrieveEvalJob.compile(
        workspace="default",
        spec=_spec(),
        entity_client=object(),
        job_name=None,
        async_sdk=None,
    )

    step = job.steps[0]
    assert step.name == "retrieve-eval"
    assert isinstance(step.executor, CPUExecutionProviderSpec)
    assert step.executor.container.command == ["nemo_evaluator.tasks.retrieve_eval"]
    environment = step.environment
    assert environment is not None
    assert any(variable.name == PERSISTENT_JOB_STORAGE_PATH_ENVVAR for variable in environment)


def test_run_validates_fileset_and_persists_nemotron_keys(tmp_path: Path, mocker: MockerFixture) -> None:
    ctx = _context(tmp_path)
    downloaded = tmp_path / "downloaded"
    download = mocker.patch(
        "nemo_evaluator.jobs.retrieve_eval.download_dataset_sync",
        return_value=downloaded,
    )
    dataset = mocker.Mock(spec=BeirDataset)
    load = mocker.patch("nemo_evaluator.jobs.retrieve_eval.load_beir_dataset", return_value=dataset)
    evaluator = mocker.Mock()
    evaluator.run_sync.return_value = _result()
    mocker.patch("nemo_evaluator.jobs.retrieve_eval.Evaluator", return_value=evaluator)
    sdk = SimpleNamespace()

    output = RetrieveEvalJob().run(_spec().model_dump(mode="json"), ctx=ctx, sdk=cast(NemoClient, sdk))

    download.assert_called_once()
    load.assert_called_once_with(downloaded)
    assert evaluator.run_sync.call_args.kwargs["dataset"] is dataset
    assert isinstance(evaluator.run_sync.call_args.kwargs["target"], Retrieval)
    assert output["eval_results"] == {
        "ndcg_cut_10": 0.75,
        "recall_10": 1.0,
        "P_10": 0.1,
        "map_cut_10": 0.7,
    }
    assert json.loads((ctx.storage.persistent / EVAL_RESULTS_FILE_NAME).read_text()) == output["eval_results"]
    assert (ctx.storage.persistent / "results" / EVAL_RESULTS_RESULT_NAME).exists()


def test_run_reports_relative_baseline_scores(tmp_path: Path, mocker: MockerFixture) -> None:
    ctx = _context(tmp_path)
    mocker.patch(
        "nemo_evaluator.jobs.retrieve_eval.download_dataset_sync",
        return_value=tmp_path / "downloaded",
    )
    mocker.patch(
        "nemo_evaluator.jobs.retrieve_eval.load_beir_dataset",
        return_value=mocker.Mock(spec=BeirDataset),
    )
    evaluator = mocker.Mock()
    evaluator.run_sync.side_effect = [
        _result(ndcg=0.75, recall=0.9),
        _result(ndcg=0.5, recall=0.75),
    ]
    mocker.patch("nemo_evaluator.jobs.retrieve_eval.Evaluator", return_value=evaluator)
    spec = _spec().model_copy(
        update={"baseline": Retrieval(embeddings=Model(url="https://igw.example.test/v1", name="baseline"))}
    )

    output = RetrieveEvalJob().run(
        spec.model_dump(mode="json"),
        ctx=ctx,
        sdk=cast(NemoClient, SimpleNamespace()),
    )

    assert evaluator.run_sync.call_count == 2
    assert output["relative"] == pytest.approx({"ndcg_cut_10": 0.5, "recall_10": 0.2})


def test_run_includes_cutoff_10_when_baseline_omits_it(tmp_path: Path, mocker: MockerFixture) -> None:
    ctx = _context(tmp_path)
    mocker.patch(
        "nemo_evaluator.jobs.retrieve_eval.download_dataset_sync",
        return_value=tmp_path / "downloaded",
    )
    mocker.patch(
        "nemo_evaluator.jobs.retrieve_eval.load_beir_dataset",
        return_value=mocker.Mock(spec=BeirDataset),
    )
    evaluator = mocker.Mock()
    evaluator.run_sync.side_effect = [
        _result(ndcg=0.75, recall=0.9),
        _result(ndcg=0.5, recall=0.75),
    ]
    mocker.patch("nemo_evaluator.jobs.retrieve_eval.Evaluator", return_value=evaluator)
    spec = _spec().model_copy(
        update={
            "k": [1],
            "baseline": Retrieval(embeddings=Model(url="https://igw.example.test/v1", name="baseline")),
        }
    )

    output = RetrieveEvalJob().run(
        spec.model_dump(mode="json"),
        ctx=ctx,
        sdk=cast(NemoClient, SimpleNamespace()),
    )

    metrics = evaluator.run_sync.call_args_list[0].kwargs["metrics"]
    ndcg = next(metric for metric in metrics if isinstance(metric, RetrievalNDCGMetric))
    recall = next(metric for metric in metrics if isinstance(metric, RetrievalRecallMetric))
    assert 10 in ndcg.k
    assert 10 in recall.k
    assert output["relative"] == pytest.approx({"ndcg_cut_10": 0.5, "recall_10": 0.2})


def test_run_records_started_at_before_evaluation(tmp_path: Path, mocker: MockerFixture) -> None:
    ctx = _context(tmp_path)
    mocker.patch(
        "nemo_evaluator.jobs.retrieve_eval.download_dataset_sync",
        return_value=tmp_path / "downloaded",
    )
    mocker.patch(
        "nemo_evaluator.jobs.retrieve_eval.load_beir_dataset",
        return_value=mocker.Mock(spec=BeirDataset),
    )
    started = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    mocker.patch("nemo_evaluator.jobs.retrieve_eval.datetime", wraps=datetime).now.return_value = started
    evaluator = mocker.Mock()
    evaluator.run_sync.return_value = _result()
    mocker.patch("nemo_evaluator.jobs.retrieve_eval.Evaluator", return_value=evaluator)

    RetrieveEvalJob().run(_spec().model_dump(mode="json"), ctx=ctx, sdk=cast(NemoClient, SimpleNamespace()))

    metadata = json.loads((ctx.storage.persistent / "artifacts" / "run-metadata.json").read_text())
    assert metadata["started_at"] == started.isoformat()
    evaluator.run_sync.assert_called_once()


def test_run_adapts_the_generated_sdk_the_local_cli_injects(tmp_path: Path, mocker: MockerFixture) -> None:
    """``nemo evaluator retrieve-eval run`` injects a generated ``NeMoPlatform``, but the BEIR
    fileset download only accepts a typed client."""
    download = mocker.patch(
        "nemo_evaluator.jobs.retrieve_eval.download_dataset_sync",
        return_value=tmp_path / "downloaded",
    )
    mocker.patch("nemo_evaluator.jobs.retrieve_eval.load_beir_dataset", return_value=mocker.Mock(spec=BeirDataset))
    evaluator = mocker.Mock()
    evaluator.run_sync.return_value = _result()
    mocker.patch("nemo_evaluator.jobs.retrieve_eval.Evaluator", return_value=evaluator)
    platform = NeMoPlatform(base_url="http://platform.test", workspace="dev", http_client=httpx.Client())

    output = RetrieveEvalJob().run(_spec().model_dump(mode="json"), ctx=_context(tmp_path), sdk=platform)

    assert isinstance(download.call_args.kwargs["client"], NemoClient)
    assert output["eval_results"]["ndcg_cut_10"] == 0.75
