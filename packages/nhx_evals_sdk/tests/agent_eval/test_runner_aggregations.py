# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The evaluator merges run-level aggregations only from runners that opt into
:class:`RunAggregationsProvider`, and only under the runner's own namespace."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from nhx_evals_sdk.agent_eval.evaluator import _collect_runner_aggregate_scores
from nhx_evals_sdk.agent_eval.results import AgentEvalSummary
from nhx_evals_sdk.agent_eval.trials import RunnerInfo
from nhx_evals_sdk.values.results import AggregateScalarScore, AggregateScore


class _Provider:
    def run_aggregate_scores(self) -> Sequence[AggregateScore]:
        return [AggregateScalarScore(name="runner.gym.pass@1", count=None, nan_count=0, value=0.6)]

    def runner_info(self) -> RunnerInfo:
        return RunnerInfo(name="gym", kind="runner")


class _SloppyProvider:
    """A runner that ignores the namespace contract — including a name the SDK itself computes."""

    def run_aggregate_scores(self) -> Sequence[AggregateScore]:
        return [
            AggregateScalarScore(name="gym_reward.reward", count=None, nan_count=0, value=0.9),
            AggregateScalarScore(name="runner.harbor.pass@1", count=None, nan_count=0, value=0.4),
            AggregateScalarScore(name="runner.gym.pass@1", count=None, nan_count=0, value=0.6),
        ]

    def runner_info(self) -> RunnerInfo:
        return RunnerInfo(name="gym", kind="runner")


class _EmptyProvider:
    def run_aggregate_scores(self) -> Sequence[AggregateScore]:
        return []


class _PlainRunner:
    """A runner that does not implement RunAggregationsProvider."""


def test_collect_empty_when_the_provider_had_nothing_to_map() -> None:
    assert _collect_runner_aggregate_scores(_EmptyProvider()) == []


def test_collect_empty_when_runner_does_not_implement_protocol() -> None:
    assert _collect_runner_aggregate_scores(_PlainRunner()) == []


def test_runner_scores_are_merged_into_the_summary_under_their_own_namespace() -> None:
    # Merging (rather than parking them in a separate untyped bag) is the point: a backend's own
    # numbers become addressable by name the same way ours are, while the `runner.` prefix keeps it
    # obvious they were imported rather than computed here.
    scores = _collect_runner_aggregate_scores(_Provider())
    summary = AgentEvalSummary.from_scores([], tasks=[], extra_scores=scores)

    by_name = {score.name: score for score in summary.scores.scores}
    imported = by_name["runner.gym.pass@1"]
    assert isinstance(imported, AggregateScalarScore) and imported.value == 0.6
    assert all(name.startswith("runner.") for name in by_name)


def test_scores_outside_the_runners_own_namespace_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    # The namespace is enforced, not merely documented. `summary.scores` is a flat list, so an
    # un-namespaced `gym_reward.reward` would not overwrite the SDK's own aggregate — it would sit
    # beside it under the same name and leave any lookup picking one arbitrarily. A name under some
    # *other* runner's namespace is equally wrong: it misattributes the number's provenance.
    with caplog.at_level("WARNING"):
        collected = _collect_runner_aggregate_scores(_SloppyProvider())

    assert [score.name for score in collected] == ["runner.gym.pass@1"]
    # Dropped rather than raised: this runs after run_tasks, so a naming bug must not sink a completed
    # run. Silence would be worse than either — the warning names what was discarded.
    assert "gym_reward.reward" in caplog.text
    assert "runner.harbor.pass@1" in caplog.text
