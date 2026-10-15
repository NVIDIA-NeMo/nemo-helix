# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx_evals_sdk.agent_eval.dashboard import render_dashboard
from nhx_evals_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary
from nhx_evals_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nhx_evals_sdk.metrics.protocol import MetricOutput
from nhx_evals_sdk.values.results import AggregatedMetricResult, AggregateRangeScore, AggregateScalarScore


def test_dashboard_contains_metric_rollups_and_outputs() -> None:
    result = AgentEvalResult(
        run_id="run-1",
        tasks=[],
        trials=[],
        scores=[
            AgentEvalTaskScore(
                id="run-1:task-1:trial-1:example_metric",
                run_id="run-1",
                task_id="task-1",
                trial_id="trial-1",
                metric_type="example_metric",
                status=AgentEvalScoreStatus.COMPLETED,
                outputs=[
                    MetricOutput(name="score", value=0.5),
                    MetricOutput(name="label", value="partial"),
                ],
            )
        ],
        summary=AgentEvalSummary(
            scores=AggregatedMetricResult(
                scores=[AggregateRangeScore(name="example_metric.score", count=1, nan_count=0, mean=0.5)]
            ),
            task_count=1,
            trial_count=1,
            score_count=1,
        ),
    )

    html = render_dashboard(result)

    assert "0.500" in html
    assert "example_metric" in html
    assert "trial-1" in html
    assert "partial" in html
    assert "Scores" in html


def test_dashboard_renders_a_runner_imported_scalar_as_its_value_not_a_blank_mean() -> None:
    # A scalar has no distribution, so reading the table straight off `mean` would leave the one number
    # worth seeing blank, and print a sample size of 0 as though the metric had failed everywhere.
    result = AgentEvalResult(
        run_id="run-1",
        tasks=[],
        trials=[],
        scores=[],
        summary=AgentEvalSummary(
            scores=AggregatedMetricResult(
                scores=[AggregateScalarScore(name="runner.gym.arena_elo/score", count=None, nan_count=0, value=1523.0)]
            ),
        ),
    )

    html = render_dashboard(result)

    assert "1523.000" in html
    assert "runner.gym.arena_elo/score" in html
    assert "&mdash;" in html  # count is "not reported" (None), not a zero sample size


def test_dashboard_renders_an_imported_median_that_has_no_percentile_distribution() -> None:
    # The case `median` was added to AggregateScoreBase for: Gym reports a median without the samples
    # behind it, so no percentiles are set. Reading `percentiles.p50` alone blanked the very column
    # added to show it. Natively-computed scores populate both fields identically.
    result = AgentEvalResult(
        run_id="run-1",
        tasks=[],
        trials=[],
        scores=[],
        summary=AgentEvalSummary(
            scores=AggregatedMetricResult(
                scores=[
                    AggregateRangeScore(
                        name="runner.gym.pass@1/accuracy",
                        count=None,
                        nan_count=0,
                        mean=62.5,
                        min=0.0,
                        max=100.0,
                        median=75.0,
                    )
                ]
            ),
        ),
    )

    html = render_dashboard(result)

    assert "75.000" in html  # the median, from the field rather than a distribution
    assert "62.500" in html  # ...alongside the mean, so the two columns aren't confused
