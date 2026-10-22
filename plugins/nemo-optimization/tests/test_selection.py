# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import optuna
import pytest
from nemo_optimization.backends.optuna.selection import pick_trial
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


def _study_with_trials(values_list: list[tuple[float, float]]) -> optuna.Study:
    study = optuna.create_study(
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
    )
    for values in values_list:
        trial = optuna.trial.create_trial(values=list(values), params={}, distributions={})
        study.add_trial(trial)
    return study


def _values(trial: FrozenTrial) -> tuple[float, ...]:
    assert trial.values is not None, "a completed trial has values"
    return tuple(trial.values)


def test_pick_trial_sum_and_chebyshev_select_center_point() -> None:
    study = _study_with_trials([(0.1, 0.9), (0.2, 0.2), (0.9, 0.1)])
    assert _values(pick_trial(study, mode="sum")) == (0.2, 0.2)
    assert _values(pick_trial(study, mode="chebyshev")) == (0.2, 0.2)


def test_pick_trial_harmonic_selects_compromise_not_zero_cost_extreme() -> None:
    # Normalized costs [0,1] vs [0.4,0.4]: argmin(hmean(cost)) wrongly prefers the extreme.
    study = _study_with_trials([(0.0, 1.0), (0.4, 0.4)])
    assert _values(pick_trial(study, mode="harmonic")) == (0.4, 0.4)


@pytest.mark.parametrize("mode", ["sum", "weighted_sum", "WEIGHTED_SUM"])
@pytest.mark.parametrize("weights, expected", [(None, (0.2, 0.2)), ([10.0, 1.0], (0.1, 0.9))])
def test_pick_trial_weighted_sum_alias(mode: str, weights: list[float] | None, expected: tuple[float, float]) -> None:
    study = _study_with_trials([(0.1, 0.9), (0.2, 0.2), (0.9, 0.1)])
    assert _values(pick_trial(study, mode=mode, weights=weights)) == expected


def test_pick_trial_weighted_sum_rejects_wrong_weight_count() -> None:
    study = _study_with_trials([(0.1, 0.9)])
    with pytest.raises(ValueError, match="length must equal"):
        pick_trial(study, mode="weighted_sum", weights=[1.0])


def test_pick_trial_harmonic_returns_pareto_member() -> None:
    study = _study_with_trials([(0.1, 0.9), (0.2, 0.2), (0.9, 0.1)])
    trial = pick_trial(study, mode="harmonic")
    assert _values(trial) == (0.2, 0.2)


def test_pick_trial_rejects_hypervolume() -> None:
    study = _study_with_trials([(0.1, 0.9), (0.2, 0.2)])
    with pytest.raises(ValueError, match="hypervolume"):
        pick_trial(study, mode="hypervolume")


def test_pick_trial_empty_front_raises() -> None:
    study = optuna.create_study(directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE])
    with pytest.raises(ValueError, match="empty"):
        pick_trial(study, mode="sum")
