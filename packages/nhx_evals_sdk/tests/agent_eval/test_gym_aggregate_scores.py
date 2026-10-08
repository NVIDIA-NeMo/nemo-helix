# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mapping Gym's flattened ``agent_metrics`` onto typed aggregate scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from nhx_evals_sdk.agent_eval.runtimes.gym.results import aggregate_scores_from_gym
from nhx_evals_sdk.values.results import AggregateRangeScore, AggregateScalarScore, AggregateScore


def _gym(agent_metrics: Mapping[str, object], agent: str = "simple_agent") -> dict[str, object]:
    """A Gym aggregate-metrics payload. Imports read ``agent_metrics`` (the full run-level set), not
    ``key_metrics`` — which by default holds only the ``mean/*`` subset."""
    return {agent: {"agent_metrics": agent_metrics}}


def _by_name(scores: Sequence[AggregateScore]) -> dict[str, AggregateScore]:
    return {score.name: score for score in scores}


def test_a_full_stat_family_is_reassembled_into_one_range_score() -> None:
    scores = _by_name(
        aggregate_scores_from_gym(
            _gym(
                {
                    "mean/total_tokens": 120.0,
                    "max/total_tokens": 200.0,
                    "min/total_tokens": 50.0,
                    "median/total_tokens": 110.0,
                    "std/total_tokens": 12.0,
                }
            )
        )
    )

    assert list(scores) == ["runner.gym.total_tokens"]
    score = scores["runner.gym.total_tokens"]
    assert isinstance(score, AggregateRangeScore)
    assert (score.mean, score.min, score.max, score.median) == (120.0, 50.0, 200.0, 110.0)
    # Gym computes std with pandas (ddof=1), so it lands on the sample field, not the population one.
    assert score.sample_std_dev == 12.0
    assert score.std_dev is None


def test_a_partial_stat_family_is_left_as_standalone_scalars() -> None:
    # A resources-server may define a metric literally named `mean`; re-assembling on a partial match
    # would rename someone's real metric into a statistic of a distribution that never existed.
    scores = _by_name(aggregate_scores_from_gym(_gym({"mean/accuracy": 0.8, "max/accuracy": 1.0})))

    assert set(scores) == {"runner.gym.mean/accuracy", "runner.gym.max/accuracy"}
    assert all(isinstance(score, AggregateScalarScore) for score in scores.values())
    mean_accuracy = scores["runner.gym.mean/accuracy"]
    assert isinstance(mean_accuracy, AggregateScalarScore) and mean_accuracy.value == 0.8


def test_environment_specific_metrics_survive_as_scalars() -> None:
    # 36 of Gym's ~97 resources-servers override compute_metrics, emitting keys in their own shapes.
    scores = _by_name(aggregate_scores_from_gym(_gym({"arena_elo/score": 1523.0, "easy/pass@1/accuracy": 0.42})))

    assert set(scores) == {"runner.gym.arena_elo/score", "runner.gym.easy/pass@1/accuracy"}
    easy = scores["runner.gym.easy/pass@1/accuracy"]
    assert isinstance(easy, AggregateScalarScore) and easy.value == 0.42


def test_reward_is_skipped_because_the_sdk_scores_it_natively() -> None:
    scores = _by_name(
        aggregate_scores_from_gym(
            _gym(
                {
                    "mean/reward": 0.6,
                    "max/reward": 1.0,
                    "min/reward": 0.0,
                    "median/reward": 0.5,
                    "std/reward": 0.4,
                    "mean/steps": 3.0,
                    "max/steps": 5.0,
                    "min/steps": 1.0,
                    "median/steps": 3.0,
                    "std/steps": 1.2,
                }
            )
        )
    )

    # gym_reward.reward already carries this, derived from the same rollouts.
    assert list(scores) == ["runner.gym.steps"]


def test_an_incomplete_reward_family_is_skipped_rather_than_kept_as_scalars() -> None:
    # Redundancy is a property of the metric, not of how complete its family is. A resources-server
    # that emits only part of the reward family would otherwise leave `mean/reward` behind as a scalar
    # -- the very duplicate of the natively-computed `gym_reward.reward` that skipping reward prevents.
    scores = _by_name(aggregate_scores_from_gym(_gym({"mean/reward": 0.6, "max/reward": 1.0, "accuracy": 0.9})))

    assert list(scores) == ["runner.gym.accuracy"]


def test_nothing_numeric_is_dropped_from_a_custom_environment_payload() -> None:
    """Every numeric key Gym reported must be represented — as a re-assembled family or as a scalar.

    The invariant that matters: importing is a *renaming*, never a filter. Only ``reward`` (redundant
    by construction) and non-numeric values are allowed to disappear.
    """
    key_metrics = {
        "mean/latency_s": 1.0,
        "max/latency_s": 2.0,
        "min/latency_s": 0.5,
        "median/latency_s": 0.9,
        "std/latency_s": 0.3,
        "mean/partial": 0.9,  # incomplete family
        "arena_elo/score": 1200.0,
        "easy/pass@1/accuracy": 0.5,
        "hard/pass@1/accuracy": 0.1,
        "num_rollouts": 40,
        "notes": "not a measurement",  # non-numeric: stays only in the opaque payload
        "converged": True,  # a bool is a flag, not a measurement
    }
    scores = aggregate_scores_from_gym(_gym(key_metrics))

    numeric_keys = {key for key, value in key_metrics.items() if isinstance(value, (int, float)) and value is not True}
    represented = set()
    for score in scores:
        name = score.name.removeprefix("runner.gym.")
        if isinstance(score, AggregateRangeScore):
            represented.update(f"{stat}/{name}" for stat in ("mean", "max", "min", "median", "std"))
        else:
            represented.add(name)

    assert represented == numeric_keys


def test_names_are_qualified_by_agent_only_when_a_run_produced_several() -> None:
    # One agent per run is the norm, so the agent name adds nothing; with two it prevents a collision.
    one = _by_name(aggregate_scores_from_gym(_gym({"score": 1.0})))
    assert list(one) == ["runner.gym.score"]

    two = _by_name(
        aggregate_scores_from_gym({"a": {"agent_metrics": {"score": 1.0}}, "b": {"agent_metrics": {"score": 2.0}}})
    )
    assert set(two) == {"runner.gym.a.score", "runner.gym.b.score"}


def test_absent_or_malformed_aggregations_yield_nothing() -> None:
    assert aggregate_scores_from_gym(None) == []
    assert aggregate_scores_from_gym({}) == []
    assert aggregate_scores_from_gym({"agent": {"group_level_metrics": []}}) == []  # no agent_metrics
    assert aggregate_scores_from_gym({"agent": "not a mapping"}) == []


def test_imported_scores_report_an_unknown_sample_size_rather_than_zero() -> None:
    # Gym reports statistics without the n behind them. count=0 would assert that nothing was
    # evaluated — false, and a landmine for anything that divides by it.
    scores = aggregate_scores_from_gym(
        _gym(
            {
                "mean/steps": 3.0,
                "max/steps": 5.0,
                "min/steps": 1.0,
                "median/steps": 3.0,
                "std/steps": 1.2,
                "elo": 1200.0,
            }
        )
    )

    assert scores
    assert all(score.count is None for score in scores)


def test_imports_read_agent_metrics_not_the_key_metrics_subset() -> None:
    """Reading ``key_metrics`` would silently degrade every distribution into a lone mean.

    Gym's ``get_key_metrics`` defaults to selecting only the ``mean/*`` entries of ``agent_metrics``, so
    a payload's ``key_metrics`` never carries the max/min/median/std that make a stat family. Sourcing
    from it would leave every metric as a scalar named ``runner.gym.mean/<name>``.
    """
    payload = {
        "simple_agent": {
            "agent_metrics": {
                "mean/steps": 3.0,
                "max/steps": 5.0,
                "min/steps": 1.0,
                "median/steps": 3.0,
                "std/steps": 1.2,
            },
            # what Gym's default get_key_metrics would select — a strict subset, means only
            "key_metrics": {"mean/steps": 3.0},
        }
    }

    scores = _by_name(aggregate_scores_from_gym(payload))

    assert list(scores) == ["runner.gym.steps"]
    steps = scores["runner.gym.steps"]
    assert isinstance(steps, AggregateRangeScore)
    assert steps.max == 5.0  # would be unreachable if key_metrics were the source


def test_median_matches_p50_wherever_both_are_reported() -> None:
    """`median` is documented as equal to `percentiles.p50`, so the two must not drift.

    They are produced by different call sites (agent-eval's `_aggregate_range_score` and the
    deterministic-metric `aggregate_metrics`), which is exactly how one silently stops matching.
    """
    from nhx_evals_sdk.agent_eval.results import AgentEvalSummary
    from nhx_evals_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
    from nhx_evals_sdk.metrics.protocol import MetricOutput

    scores = [
        AgentEvalTaskScore(
            id=f"r:t{i}:tr:m",
            run_id="r",
            task_id=f"t{i}",
            trial_id=f"tr{i}",
            metric_type="m",
            status=AgentEvalScoreStatus.COMPLETED,
            outputs=[MetricOutput(name="score", value=value)],
        )
        for i, value in enumerate([0.1, 0.4, 0.9, 0.2])
    ]

    summary = AgentEvalSummary.from_scores(scores)
    aggregate = summary.score("m.score")

    assert isinstance(aggregate, AggregateRangeScore)
    assert aggregate.percentiles is not None
    assert aggregate.median == aggregate.percentiles.p50
