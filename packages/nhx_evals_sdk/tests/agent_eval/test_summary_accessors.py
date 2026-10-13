# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""By-name aggregate lookup on AgentEvalSummary and the AggregatedMetricResult it delegates to."""

from __future__ import annotations

import pytest
from nhx_evals_sdk.agent_eval.results import AgentEvalSummary
from nhx_evals_sdk.values.results import AggregatedMetricResult, AggregateRangeScore


def _aggregates(*names: str) -> AggregatedMetricResult:
    return AggregatedMetricResult(
        scores=[AggregateRangeScore(name=name, count=2, nan_count=0, mean=0.5) for name in names]
    )


def _summary(*names: str) -> AgentEvalSummary:
    return AgentEvalSummary(scores=_aggregates(*names))


def test_score_returns_the_aggregate_with_that_name() -> None:
    summary = _summary("gym_reward.reward", "gym_reward.reward.pass@2")

    assert summary.score("gym_reward.reward.pass@2").mean == 0.5


def test_score_finds_a_name_that_is_not_the_first_in_the_list() -> None:
    # Guards the scan itself: returning self.scores[0] regardless of name would satisfy a
    # single-aggregate test but is plainly wrong.
    summary = _summary("a.first", "b.second", "c.third")

    assert summary.score("c.third").name == "c.third"


def test_score_suggests_the_intended_name_when_the_lookup_looks_like_a_typo() -> None:
    summary = _summary("gym_reward.reward", "view.solved")

    with pytest.raises(KeyError, match="did you mean") as excinfo:
        summary.score("gym_reward.rewrad")

    # The suggestion is the whole point: a transposed name should be fixable from the message alone,
    # without going back to the aggregation code to find out what was produced.
    assert "gym_reward.reward" in str(excinfo.value)


@pytest.mark.parametrize(
    ("extra_names", "expected"),
    [
        # A suggestion that misses shouldn't be a dead end: the count tells the caller there is more
        # to look at, rather than implying the offered names are the whole set.
        (20, "(20 other aggregates in this result)"),
        (1, "(1 other aggregate in this result)"),
        (0, None),
    ],
)
def test_score_reports_how_many_other_names_exist_alongside_a_suggestion(
    extra_names: int, expected: str | None
) -> None:
    aggregates = _aggregates("reward.reward", *(f"metric_{index:02d}.zzz" for index in range(extra_names)))

    with pytest.raises(KeyError) as excinfo:
        aggregates.score("reward.rewrad")

    message = str(excinfo.value)
    assert "'reward.reward'" in message
    if expected is None:
        assert "in this result" not in message
    else:
        assert expected in message


def test_miss_message_names_a_repeated_aggregate_only_once() -> None:
    # A duplicate name is one name, not two near-misses: suggesting it twice, listing it twice, or
    # counting it twice in the "other aggregates" tally all misrepresent what the result holds.
    duplicated = _aggregates("reward.reward", "reward.reward", "view.solved")

    with pytest.raises(KeyError) as excinfo:
        duplicated.score("reward.rewrad")

    message = str(excinfo.value)
    assert message.count("'reward.reward'") == 1
    assert "(1 other aggregate in this result)" in message


def test_miss_message_does_not_repeat_a_duplicate_in_the_fallback_listing() -> None:
    duplicated = _aggregates("alpha.one", "alpha.one", "beta.two")

    with pytest.raises(KeyError) as excinfo:
        duplicated.score("zzzzzz")

    message = str(excinfo.value)
    assert message.count("'alpha.one'") == 1


def test_score_lists_available_names_when_nothing_is_close() -> None:
    summary = _summary("gym_reward.reward", "view.solved")

    with pytest.raises(KeyError) as excinfo:
        summary.score("totally_unrelated")

    message = str(excinfo.value)
    assert "did you mean" not in message
    assert "gym_reward.reward" in message
    assert "view.solved" in message


def test_score_truncates_a_long_name_list_rather_than_dumping_all_of_them() -> None:
    # A real run carries several metrics times pass@k values; an untruncated dump buries the answer.
    aggregates = _aggregates(*(f"metric_{index:02d}.zzz" for index in range(25)))

    with pytest.raises(KeyError) as excinfo:
        aggregates.score("qqq")

    message = str(excinfo.value)
    assert "(15 more)" in message
    assert "metric_00.zzz" in message
    assert "metric_24.zzz" not in message


def test_score_says_so_when_the_result_has_no_aggregates_at_all() -> None:
    # Distinct from a typo: nothing was produced, so no name would have worked.
    with pytest.raises(KeyError, match="no aggregates at all"):
        _summary().score("anything")


def test_scores_by_name_supports_membership_and_get_for_optional_aggregates() -> None:
    summary = _summary("gym_reward.reward")

    assert "gym_reward.reward" in summary.scores_by_name
    assert summary.scores_by_name.get("never_ran") is None


def test_scores_by_name_keeps_the_first_of_a_repeated_name() -> None:
    # Names are expected unique, but runner-contributed extras are appended as-is. First-wins matches
    # the `next(...)` scans this replaced, so a collision behaves as it did before.
    duplicated = AggregatedMetricResult(
        scores=[
            AggregateRangeScore(name="m.score", count=2, nan_count=0, mean=0.25),
            AggregateRangeScore(name="m.score", count=2, nan_count=0, mean=0.75),
        ]
    )

    assert duplicated.score("m.score").mean == 0.25
    assert duplicated.scores_by_name["m.score"].mean == 0.25
