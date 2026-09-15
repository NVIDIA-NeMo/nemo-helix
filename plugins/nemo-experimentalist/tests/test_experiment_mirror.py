# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Generic, Protocol, TypeVar
from unittest.mock import AsyncMock

import httpx
import pytest
from doubles import make_candidate
from nemo_experimentalist_plugin.entities import Candidate, ExperimentRun, RewardRecord
from nemo_experimentalist_plugin.experimentalist.experiment_mirror import ExperimentMirror, group_metadata
from nemo_platform_plugin.client.errors import ConflictError, NotFoundError
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import EvaluationCreateRequest, ExperimentCreateRequest, ExperimentUpdateRequest

pytestmark = pytest.mark.asyncio

_CONFLICT = ConflictError(httpx.Response(409, request=httpx.Request("POST", "http://x/y")))
_NOTFOUND = NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://x/y")))

T = TypeVar("T")


@dataclass(frozen=True)
class _Response(Generic[T]):
    body: T

    def data(self) -> T:
        return self.body


class _GroupClient(Protocol):
    async def create(self, *, workspace: str, body: ExperimentCreateRequest) -> _Response[object]: ...

    async def retrieve(self, name: str, *, workspace: str) -> _Response[object]: ...

    async def update(self, name: str, *, workspace: str, body: ExperimentUpdateRequest) -> _Response[object]: ...


class _EvaluationClient(Protocol):
    async def create(self, *, workspace: str, body: EvaluationCreateRequest) -> _Response[object]: ...

    async def retrieve(self, name: str, *, workspace: str) -> _Response[object]: ...

    async def update(self, name: str, *, workspace: str, body: EvaluationCreateRequest) -> _Response[object]: ...


class _MirrorClient(AsyncIntakeClient):
    def __init__(self, groups: _GroupClient, experiments: _EvaluationClient) -> None:
        super().__init__(base_url="http://test")
        self._groups = groups
        self._experiments = experiments

    async def create_experiment(
        self,
        *,
        workspace: str | None = None,
        body: ExperimentCreateRequest,
    ) -> _Response[object]:
        return await self._groups.create(workspace=workspace or "default", body=body)

    async def get_experiment(
        self,
        *,
        workspace: str | None = None,
        name: str,
    ) -> _Response[object]:
        return await self._groups.retrieve(name, workspace=workspace or "default")

    async def update_experiment(
        self,
        *,
        workspace: str | None = None,
        name: str,
        body: ExperimentUpdateRequest,
    ) -> _Response[object]:
        return await self._groups.update(name, workspace=workspace or "default", body=body)

    async def create_evaluation(
        self,
        *,
        workspace: str | None = None,
        body: EvaluationCreateRequest,
    ) -> _Response[object]:
        return await self._experiments.create(workspace=workspace or "default", body=body)

    async def get_evaluation(
        self,
        *,
        workspace: str | None = None,
        name: str,
    ) -> _Response[object]:
        return await self._experiments.retrieve(name, workspace=workspace or "default")

    async def update_evaluation(
        self,
        *,
        workspace: str | None = None,
        name: str,
        body: EvaluationCreateRequest,
    ) -> _Response[object]:
        return await self._experiments.update(name, workspace=workspace or "default", body=body)


def _client(groups: _GroupClient, experiments: _EvaluationClient) -> AsyncIntakeClient:
    if isinstance(groups, AsyncMock) and isinstance(groups.retrieve.return_value, AsyncMock):
        groups.retrieve.return_value = _Response(SimpleNamespace(id="grp-1", name="opt-run-1"))
    return _MirrorClient(groups, experiments)


class _StatefulGroups:
    """Models experiments as a full-replace PUT store: create/update set the
    stored body to exactly the fields passed, treating ``omit`` (or an absent kwarg)
    as "field became None". This is what catches accidental field-wiping on update."""

    _TRACKED = ("insight_id", "summary", "metadata")
    calls: list[dict[str, object]]

    def __init__(self) -> None:
        self.name: str | None = None
        self.body: dict[str, object | None] = {}
        self.calls = []

    def _apply(self, body: ExperimentCreateRequest | ExperimentUpdateRequest) -> None:
        dumped = body.model_dump(exclude_unset=True)
        for field in self._TRACKED:
            self.body[field] = dumped.get(field)

    async def create(self, *, workspace: str, body: ExperimentCreateRequest) -> _Response[object]:
        self.calls.append({"workspace": workspace, "body": body})
        self.name = body.name
        self._apply(body)
        return _Response(SimpleNamespace(id="grp-1", name=body.name, **self.body))

    async def retrieve(self, name: str, *, workspace: str) -> _Response[object]:
        self.calls.append({"workspace": workspace, "name": name})
        return _Response(SimpleNamespace(id="grp-1", name=self.name, **self.body))

    async def update(self, name: str, *, workspace: str, body: ExperimentUpdateRequest) -> _Response[object]:
        self.calls.append({"workspace": workspace, "name": name, "body": body})
        self._apply(body)
        return _Response(SimpleNamespace(id="grp-1", name=self.name, **self.body))


def _run(**kw: Any) -> ExperimentRun:
    base: dict[str, Any] = dict(
        workspace="default",
        agent="a",
        insight="ins-1",
        config_snapshot={},
        status="running",
        rounds_completed=0,
        winner_agent=None,
    )
    base.update(kw)
    run = ExperimentRun(**base)
    run._id = "run-1"  # store id (private attr, as the entity client sets it)
    return run


def _cand(**kw: Any) -> Candidate:
    base: dict[str, Any] = dict(run_id="run-1", label="agent-0", generation=0, description="baseline")
    base.update(kw)
    return make_candidate(**base)


async def test_ensure_group_creates_with_insight_and_metadata():
    groups = AsyncMock()
    groups.create.return_value = _Response(SimpleNamespace(id="grp-1", name="opt-run-1"))
    mirror = ExperimentMirror(_client(groups, AsyncMock()), workspace="default")
    await mirror.ensure_group(_run())
    body = groups.create.await_args.kwargs["body"]
    assert body.name == "opt-run-1" and body.insight_id == "ins-1"
    assert body.metadata["agent"] == "a"


async def test_ensure_group_conflict_retrieves():
    groups = AsyncMock()
    groups.create.side_effect = _CONFLICT
    groups.retrieve.return_value = _Response(SimpleNamespace(id="grp-1", name="opt-run-1"))
    mirror = ExperimentMirror(_client(groups, AsyncMock()), workspace="default")
    await mirror.ensure_group(_run())
    groups.retrieve.assert_awaited_once()


async def test_project_candidate_upserts_experiment_per_evaluated_split():
    experiments = AsyncMock()
    experiments.create.return_value = _Response(SimpleNamespace(id="exp-train"))
    mirror = ExperimentMirror(_client(AsyncMock(), experiments), workspace="default")
    cand = _cand(generation=0, rewards={"train": RewardRecord(metrics={"reward": 1.0}, trials=[])})
    await mirror.project_candidate(cand)
    body = experiments.create.await_args.kwargs["body"]
    assert body.name == "opt-run-1-agent-0-train"
    assert body.status == "baseline"
    # metadata is identity-only — reward/trials are NOT copied (§4.3)
    assert body.metadata == {
        "generation": "0",
        "candidate_id": "id-agent-0",
        "candidate_label": "agent-0",
        "split": "train",
    }
    assert "aggregate_metrics" not in body.metadata and "trials" not in body.metadata
    # validation not evaluated → not created
    assert experiments.create.await_count == 1


async def test_project_candidate_skips_when_no_reward():
    experiments = AsyncMock()
    mirror = ExperimentMirror(_client(AsyncMock(), experiments), workspace="default")
    await mirror.project_candidate(_cand())  # no reward set
    experiments.create.assert_not_awaited()


async def test_project_candidate_creates_experiment_for_custom_reward_channel():
    experiments = AsyncMock()
    experiments.create.return_value = _Response(SimpleNamespace(id="exp-custom"))
    mirror = ExperimentMirror(_client(AsyncMock(), experiments), workspace="default")
    candidate = _cand(rewards={"custom": RewardRecord(metrics={"custom_score": 0.5}, trials=[])})

    await mirror.project_candidate(candidate)

    body = experiments.create.await_args.kwargs["body"]
    assert body.name == "opt-run-1-agent-0-custom"
    assert body.dataset_name == "custom"
    assert body.metadata == {
        "generation": "0",
        "candidate_id": "id-agent-0",
        "candidate_label": "agent-0",
        "split": "custom",
    }


async def test_project_candidate_conflict_updates_experiment():
    groups = AsyncMock()
    groups.retrieve.return_value = _Response(SimpleNamespace(id="grp-1", name="opt-run-1"))
    experiments = AsyncMock()
    experiments.create.side_effect = _CONFLICT
    experiments.update.return_value = _Response(SimpleNamespace(id="exp-train"))
    mirror = ExperimentMirror(_client(groups, experiments), workspace="default")
    cand = _cand(generation=0, rewards={"train": RewardRecord(metrics={"reward": 1.0}, trials=[])})
    await mirror.project_candidate(cand)
    experiments.update.assert_awaited_once()
    # update is a full-replace: dataset_version must be re-supplied (symmetric with create)
    body = experiments.update.await_args.kwargs["body"]
    assert body.dataset_version == "v1"
    assert body.metadata == {
        "generation": "0",
        "candidate_id": "id-agent-0",
        "candidate_label": "agent-0",
        "split": "train",
    }


async def test_parent_experiment_id_cache_hit():
    groups = AsyncMock()
    groups.retrieve.return_value = _Response(SimpleNamespace(id="grp-1", name="opt-run-1"))
    experiments = AsyncMock()
    experiments.create.side_effect = [
        _Response(SimpleNamespace(id="exp-ancestor-train")),  # ancestor train experiment
        _Response(SimpleNamespace(id="exp-child-train")),  # child train experiment
    ]
    mirror = ExperimentMirror(_client(groups, experiments), workspace="default")
    ancestor = _cand(label="agent-0", generation=0, rewards={"train": RewardRecord(metrics={"reward": 1.0}, trials=[])})
    await mirror.project_candidate(ancestor)
    child = _cand(
        label="agent-1",
        ancestor="id-agent-0",
        generation=1,
        rewards={"train": RewardRecord(metrics={"reward": 1.0}, trials=[])},
    )
    await mirror.project_candidate(child)
    child_body = experiments.create.await_args.kwargs["body"]  # last call == child
    assert child_body.parent_evaluation_id == "exp-ancestor-train"
    experiments.retrieve.assert_not_awaited()  # resolved from cache, no lookup


async def test_parent_experiment_id_omits_the_link_for_an_unprojected_ancestor():
    """``ancestor`` is an id and Experiment names are built from labels.

    Without having projected the ancestor, this mirror cannot name its Experiment, so it
    omits the lineage link rather than composing a name from an id that was never used
    to build one. Projection is best-effort and one-way, so a missing link is the
    honest outcome.
    """
    groups = AsyncMock()
    groups.retrieve.return_value = _Response(SimpleNamespace(id="grp-1", name="opt-run-1"))
    experiments = AsyncMock()
    experiments.create.return_value = _Response(SimpleNamespace(id="exp-child-train"))
    mirror = ExperimentMirror(_client(groups, experiments), workspace="default")
    child = _cand(
        label="agent-1",
        ancestor="id-agent-0",
        generation=1,
        rewards={"train": RewardRecord(metrics={"reward": 1.0}, trials=[])},
    )
    await mirror.project_candidate(child)
    experiments.retrieve.assert_not_awaited()
    assert experiments.create.await_args.kwargs["body"].parent_evaluation_id is None


async def test_parent_experiment_id_not_found_omits():
    groups = AsyncMock()
    groups.retrieve.return_value = _Response(SimpleNamespace(id="grp-1", name="opt-run-1"))
    experiments = AsyncMock()
    experiments.retrieve.side_effect = _NOTFOUND
    experiments.create.return_value = _Response(SimpleNamespace(id="exp-child-train"))
    mirror = ExperimentMirror(_client(groups, experiments), workspace="default")
    child = _cand(
        label="agent-1",
        ancestor="id-agent-0",
        generation=1,
        rewards={"train": RewardRecord(metrics={"reward": 1.0}, trials=[])},
    )
    await mirror.project_candidate(child)
    body = experiments.create.await_args.kwargs["body"]
    assert body.parent_evaluation_id is None


async def test_finalize_round0_winner_preserves_existing_source_link():
    # A round-0 winner with no PR must keep the source_link written during the run (e.g. a
    # real {repo}@{ref}), not have it clobbered with a pseudo link by the full-replace update.
    groups = AsyncMock()
    groups.retrieve.return_value = _Response(
        SimpleNamespace(id="grp-1", name="opt-run-1", insight_id=None, metadata=None)
    )
    experiments = AsyncMock()
    experiments.retrieve.return_value = _Response(SimpleNamespace(id="exp-w", source_link="https://git/repo.git@main"))
    experiments.create.return_value = _Response(SimpleNamespace(id="exp-w"))
    mirror = ExperimentMirror(_client(groups, experiments), workspace="default")
    winner = _cand(label="agent-0", generation=0, rewards={"train": RewardRecord(metrics={"reward": 1.0}, trials=[])})
    await mirror.finalize(run_id="run-1", summary="done", winner=winner, pr_url=None)
    body = experiments.create.await_args.kwargs["body"]
    assert body.source_link == "https://git/repo.git@main"  # preserved, not pseudo
    assert body.status == "winner"


async def test_group_update_and_finalize_do_not_wipe_insight_or_metadata():
    # experiments.update is a full-replace PUT; ensure_group → update_group →
    # finalize must all preserve insight_id + metadata rather than reset them to None.
    groups = _StatefulGroups()
    mirror = ExperimentMirror(_client(groups, AsyncMock()), workspace="default")
    run = _run(insight="ins-1")
    await mirror.ensure_group(run)
    await mirror.update_group(run)
    await mirror.finalize(run_id="run-1", summary="done", winner=None)
    assert groups.body["insight_id"] == "ins-1"
    assert groups.body["metadata"] == group_metadata(run)
    assert groups.body["metadata"] is not None
    assert groups.body["summary"] == "done"


async def test_mirror_projects_every_measured_channel_without_an_allowlist() -> None:
    """A new reward channel must reach Studio with no entity or mirror edit.

    The mirror used to iterate a hardcoded SPLITS tuple and look rewards up through an
    explicit per-field dict, so an unknown channel was invisible.
    """
    experiments = AsyncMock()
    experiments.create.return_value = _Response(SimpleNamespace(id="exp-x"))
    mirror = ExperimentMirror(_client(AsyncMock(), experiments), workspace="default")
    candidate = _cand(
        rewards={
            "validation": RewardRecord(metrics={"reward": 1.0}),
            "some-new-channel": RewardRecord(metrics={"score": 0.25}),
        }
    )

    await mirror.project_candidate(candidate)

    # The channel the mirror never knew about must still reach Studio. Asserting on the
    # entity alone would pass even if project_candidate went back to a fixed allowlist.
    projected = {call.kwargs["body"].metadata["split"] for call in experiments.create.await_args_list}
    assert projected == {"validation", "some-new-channel"}
    assert candidate.rewards["never-measured"].metrics == {}
