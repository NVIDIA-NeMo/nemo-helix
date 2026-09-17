# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for jobs.result_persistence: target-trait mapping + best-effort entity writes.

These cover the pure mapping helpers and the persist_* entry points, with the async ``EntityClient``
stubbed (``_entity_client`` patched at its usage site) so no real SDK or event loop wiring is needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from nemo_evaluator.entities import AgentEvalResultEntity, EvaluateResultEntity
from nemo_evaluator.jobs import result_persistence
from nemo_evaluator.jobs.agent_spec import (
    AgentTarget,
    FabricRunnerTarget,
    GymRunnerTarget,
    HarborRunnerTarget,
    ModelTarget,
)
from nemo_evaluator.jobs.result_persistence import (
    _agent_target_fields,
    _row_target_fields,
    _safe_target_url,
    persist_agent_eval_result,
    persist_evaluate_result,
)
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary
from nemo_evaluator_sdk.enums import AgentFormat
from nemo_evaluator_sdk.values import Agent, GenericAgent, Model
from nemo_evaluator_sdk.values.results import AggregatedMetricResult, EvaluationResult
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.entities import EntityBase, EntityClient
from nemo_platform_plugin.entities.client import AsyncEntitiesClient
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import LocalJobResults
from pytest_mock import MockerFixture

_ASYNC_SDK = AsyncNemoClient(
    base_url="http://platform.test",
    workspace="dev",
    http_client=AsyncMock(spec=httpx.AsyncClient),
)


def test_entity_client_adapts_async_sdk(mocker: MockerFixture) -> None:
    wrapped_client = mocker.Mock(spec=EntityClient)
    constructor = mocker.patch.object(result_persistence, "EntityClient", return_value=wrapped_client)

    assert result_persistence._entity_client(_ASYNC_SDK) is wrapped_client
    constructor.assert_called_once()
    typed_client = constructor.call_args.args[0]
    assert isinstance(typed_client, AsyncEntitiesClient)
    assert typed_client._http is _ASYNC_SDK._http


def _model() -> Model:
    return Model(url="https://model.test/v1/chat/completions", name="my-model")


def _agent() -> Agent:
    return GenericAgent(
        url="http://agent.test",
        name="my-agent",
        format=AgentFormat.GENERIC,
        body={"question": "{{item.prompt}}"},
        response_path="$.answer",
    )


# ---- target-trait mapping --------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (ModelTarget(model=_model()), ("model", "my-model", "https://model.test/v1/chat/completions")),
        (AgentTarget(agent=_agent()), ("agent", "my-agent", "http://agent.test")),
        (
            FabricRunnerTarget(
                config={"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}},
                model="openai/gpt-5.4",
            ),
            ("fabric", "openai/gpt-5.4", None),
        ),
        (
            GymRunnerTarget(agent="simple_agent", agent_config="conf/agent.yaml", resources_server="mcqa"),
            ("gym", "simple_agent", None),
        ),
        (HarborRunnerTarget(agent_name="oracle"), ("harbor", "oracle", None)),
        (HarborRunnerTarget(agent_import_path="wrapper:Agent"), ("harbor", "wrapper:Agent", None)),
        (None, (None, None, None)),
    ],
)
def test_agent_target_fields(target, expected) -> None:
    assert _agent_target_fields(target) == expected


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (_model(), ("model", "my-model", "https://model.test/v1/chat/completions")),
        (_agent(), ("agent", "my-agent", "http://agent.test")),
        (None, (None, None, None)),
    ],
)
def test_row_target_fields(target, expected) -> None:
    assert _row_target_fields(target) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # Plain endpoints round-trip unchanged.
        ("https://model.test/v1/chat/completions", "https://model.test/v1/chat/completions"),
        ("http://agent.test:8080/infer", "http://agent.test:8080/infer"),
        # Userinfo (credentials) is stripped from the netloc. Assembled from parts so no literal
        # basic-auth userinfo appears contiguously in source (the secret scanner flags that pattern).
        ("https://" + "u:p" + "@model.test/v1", "https://model.test/v1"),
        # Sensitive query values are redacted; benign ones survive.
        ("https://model.test/v1?api_key=sekret&region=us", "https://model.test/v1?api_key=REDACTED&region=us"),
        ("https://model.test/v1?access_token=abc&x=1", "https://model.test/v1?access_token=REDACTED&x=1"),
        # Unparseable / host-less inputs are omitted rather than stored raw.
        ("not a url", None),
        (None, None),
    ],
)
def test_safe_target_url_strips_credentials(url, expected) -> None:
    assert _safe_target_url(url) == expected


# ---- persist_* entity construction + best-effort write ---------------------


class _FakeClient:
    """Records saved entities; optionally fails to exercise the best-effort path."""

    def __init__(self, fail: bool = False) -> None:
        self.saved: list[EntityBase] = []
        self._fail = fail

    async def save(self, entity: EntityBase) -> EntityBase:
        if self._fail:
            raise RuntimeError("entity store unavailable")
        self.saved.append(entity)
        return entity


def _ctx(tmp_path: Path, job_id: str | None) -> JobContext:
    storage = StoragePaths(ephemeral=tmp_path / "e", persistent=tmp_path / "p")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    return JobContext(
        workspace="dev",
        storage=storage,
        results=LocalJobResults(root=storage.persistent / "results"),
        job_id=job_id,
    )


def _agent_result() -> AgentEvalResult:
    return AgentEvalResult(run_id="run-1", tasks=[], trials=[], scores=[], summary=AgentEvalSummary())


def _eval_result() -> EvaluationResult:
    return EvaluationResult(row_scores=[], aggregate_scores=AggregatedMetricResult(scores=[]))


def test_persist_agent_eval_result_builds_entity_and_saves(tmp_path: Path, mocker: MockerFixture) -> None:
    client = _FakeClient()
    entity_client_factory = mocker.patch.object(result_persistence, "_entity_client", return_value=client)

    persist_agent_eval_result(
        _agent_result(),
        target=FabricRunnerTarget(
            config={"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}},
            model="openai/gpt-5.4",
        ),
        ctx=_ctx(tmp_path, "job-1"),
        bundle_ref="fileset://dev/agent-eval-results#b",
        async_client=_ASYNC_SDK,
    )

    entity_client_factory.assert_called_once()
    cloned_sdk = entity_client_factory.call_args.args[0]
    assert isinstance(cloned_sdk, AsyncNemoClient)
    assert cloned_sdk is not _ASYNC_SDK
    (entity,) = client.saved
    assert isinstance(entity, AgentEvalResultEntity)
    assert entity.name == "job-1"
    assert entity.job_id == "job-1"
    assert entity.workspace == "dev"
    assert entity.target_kind == "fabric"
    assert entity.target_name == "openai/gpt-5.4"
    assert entity.target_url is None
    assert entity.bundle_ref == "fileset://dev/agent-eval-results#b"


def test_persist_evaluate_result_records_dataset_and_metric_types(tmp_path: Path, mocker: MockerFixture) -> None:
    client = _FakeClient()
    mocker.patch.object(result_persistence, "_entity_client", return_value=client)

    persist_evaluate_result(
        _eval_result(),
        target=_model(),
        dataset_ref="dev/my-dataset",
        metric_types=["exact_match"],
        ctx=_ctx(tmp_path, "job-2"),
        bundle_ref="fileset://dev/eval-results#b",
        async_client=_ASYNC_SDK,
    )

    (entity,) = client.saved
    assert isinstance(entity, EvaluateResultEntity)
    assert entity.job_id == "job-2"
    assert entity.target_kind == "model"
    assert entity.dataset_ref == "dev/my-dataset"
    assert entity.metric_types == ["exact_match"]


def test_persist_skips_when_no_job_id(tmp_path: Path, mocker: MockerFixture) -> None:
    client = _FakeClient()
    mocker.patch.object(result_persistence, "_entity_client", return_value=client)

    # Without a job id, there is no run to key the result on, so skip.
    persist_agent_eval_result(
        _agent_result(),
        target=None,
        ctx=_ctx(tmp_path, None),
        bundle_ref="x",
        async_client=_ASYNC_SDK,
    )

    assert client.saved == []


def test_persist_skips_when_no_async_client(tmp_path: Path) -> None:
    # No async SDK injected (offline run): _entity_client returns None and persistence is skipped.
    # Runs the real _entity_client(None) path; must not raise.
    persist_evaluate_result(
        _eval_result(),
        target=None,
        dataset_ref=None,
        metric_types=[],
        ctx=_ctx(tmp_path, "job-3"),
        bundle_ref="x",
        async_client=None,
    )


def test_persist_is_best_effort_on_save_failure(tmp_path: Path, mocker: MockerFixture) -> None:
    client = _FakeClient(fail=True)
    mocker.patch.object(result_persistence, "_entity_client", return_value=client)

    # The eval already succeeded and the bundle is saved; a store error must not fail the job.
    persist_agent_eval_result(
        _agent_result(),
        target=ModelTarget(model=_model()),
        ctx=_ctx(tmp_path, "job-4"),
        bundle_ref="x",
        async_client=_ASYNC_SDK,
    )
    assert client.saved == []
