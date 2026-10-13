# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml
from nemo_optimization.backends.optuna.early_stop import maybe_stop_if_target_met
from nemo_optimization.backends.optuna.study_driver import (
    SyntheticTrialEvaluator,
    average_metric_vectors,
    create_sampler,
    parse_numeric_study_config,
    resolve_n_trials,
    run_numeric_study,
)
from nemo_optimization.optimizer_config import MetricDirection
from optuna.samplers import GridSampler, NSGAIISampler, TPESampler
from optuna.study import StudyDirection


def _payload(**overrides: Any) -> dict[str, Any]:
    base = {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "demo"},
        "models": {"default": {"temperature": 0.0, "top_p": 1.0}},
        "optimizer": {
            "numeric": {"enabled": True, "n_trials": 4, "sampler": None},
            "reps_per_param_set": 2,
            "eval_metrics": {
                "average_score": {"evaluator_name": "average_score", "direction": "maximize", "weight": 1.0},
            },
            "search_space": {
                "temperature": {
                    "type": "fabric",
                    "path": "models.default.temperature",
                    "low": 0.0,
                    "high": 0.4,
                    "step": 0.2,
                },
                "top_p": {
                    "type": "fabric",
                    "path": "models.default.top_p",
                    "values": [0.7, 0.85],
                },
            },
        },
    }
    if overrides:
        base.update(overrides)
    return base


def test_parse_numeric_study_config() -> None:
    config = parse_numeric_study_config(_payload()["optimizer"])
    assert config.n_trials == 4
    assert config.reps_per_param_set == 2
    assert len(config.search_space) == 2
    assert config.metrics[0].direction == MetricDirection.MAXIMIZE
    assert config.sampler == "bayesian"
    assert isinstance(create_sampler(config), TPESampler)


@pytest.mark.parametrize("sampler_name", ["bayesian", "tpe", None])
def test_bayesian_sampler_aliases_to_explicit_tpe(sampler_name: str | None) -> None:
    optimizer = _payload()["optimizer"]
    optimizer = {**optimizer, "numeric": {**optimizer["numeric"], "sampler": sampler_name}}
    config = parse_numeric_study_config(optimizer)
    assert config.sampler == "bayesian"
    assert isinstance(create_sampler(config, seed=0), TPESampler)


def test_bayesian_multi_objective_uses_nsgaii() -> None:
    optimizer = _payload()["optimizer"]
    optimizer = {
        **optimizer,
        "numeric": {**optimizer["numeric"], "sampler": "bayesian"},
        "eval_metrics": {
            "coverage": {"direction": "maximize", "weight": 0.5},
            "latency": {"direction": "minimize", "weight": 0.5},
        },
    }
    config = parse_numeric_study_config(optimizer)
    assert config.sampler == "bayesian"
    assert isinstance(create_sampler(config, seed=0), NSGAIISampler)


def test_grid_sampler_trial_count() -> None:
    optimizer = _payload()["optimizer"]
    optimizer = {**optimizer, "numeric": {**optimizer["numeric"], "sampler": "grid"}}
    config = parse_numeric_study_config(optimizer)
    assert isinstance(create_sampler(config), GridSampler)
    assert resolve_n_trials(config) == 3 * 2


def test_average_metric_vectors() -> None:
    averaged = average_metric_vectors(
        [{"m1": 0.8, "m2": 0.2}, {"m1": 0.6, "m2": 0.4}],
        ["m1", "m2"],
    )
    assert averaged == pytest.approx([0.7, 0.3])


def test_run_numeric_study_writes_configs(tmp_path: Path) -> None:
    payload = _payload()
    config = parse_numeric_study_config(payload["optimizer"])
    evaluator = SyntheticTrialEvaluator([metric.name for metric in config.metrics])

    result = run_numeric_study(payload, tmp_path, evaluator, seed=0)

    assert result.n_trials == 4
    assert (tmp_path / "optimized_config.yml").is_file()
    assert (tmp_path / "trials_dataframe_params.csv").is_file()
    assert len(list(tmp_path.glob("config_numeric_trial_*.yml"))) == 4
    optimized = yaml.safe_load((tmp_path / "optimized_config.yml").read_text(encoding="utf-8"))
    assert "optimizer" not in optimized
    assert result.optimized_payload["optimizer"] == payload["optimizer"]
    assert result.best_trial.params
    # Logical Optuna names must be mapped onto Fabric dotted paths in the export.
    for name, value in result.best_trial.params.items():
        path = payload["optimizer"]["search_space"][name]["path"]
        cursor = optimized
        for part in path.split("."):
            cursor = cursor[part]
        assert cursor == value
    assert "temperature" not in optimized  # must not write logical top-level keys

    with (tmp_path / "trials_dataframe_params.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "values_average_score" in rows[0]
    assert "params_temperature" in rows[0]
    assert "rep_scores" in rows[0]
    assert "pareto_optimal" in rows[0]
    assert json.loads(rows[0]["rep_scores"])


def test_run_numeric_study_grid_exhaustive(tmp_path: Path) -> None:
    payload = _payload()
    payload["optimizer"]["numeric"]["sampler"] = "grid"
    config = parse_numeric_study_config(payload["optimizer"])
    evaluator = SyntheticTrialEvaluator([metric.name for metric in config.metrics])

    result = run_numeric_study(payload, tmp_path, evaluator, seed=0)

    assert result.n_trials == 6
    assert len(result.study.trials) == 6


def test_run_numeric_study_multi_objective(tmp_path: Path) -> None:
    payload = _payload()
    payload["optimizer"]["eval_metrics"] = {
        "coverage": {"direction": "maximize", "weight": 0.5},
        "latency": {"direction": "minimize", "weight": 0.5},
    }
    config = parse_numeric_study_config(payload["optimizer"])
    evaluator = SyntheticTrialEvaluator([metric.name for metric in config.metrics])

    result = run_numeric_study(payload, tmp_path, evaluator, seed=0)

    assert len(result.best_trial.values or []) == 2
    with (tmp_path / "trials_dataframe_params.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "values_coverage" in rows[0]
    assert "values_latency" in rows[0]
    assert any(row["pareto_optimal"] == "True" for row in rows)


def test_maybe_stop_if_target_met_maximize() -> None:
    study = optuna.create_study(direction="maximize")
    completed: list[float] = []

    def objective(trial: optuna.Trial) -> float:
        score = 0.95
        maybe_stop_if_target_met(study, [score], target=0.9, directions=[StudyDirection.MAXIMIZE])
        completed.append(score)
        return score

    study.optimize(objective, n_trials=3)
    assert completed == [0.95]


def test_maybe_stop_if_target_met_ignored_for_multi_objective() -> None:
    study = optuna.create_study(directions=["maximize", "minimize"])
    maybe_stop_if_target_met(
        study,
        [0.95, 0.1],
        target=0.9,
        directions=[StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE],
    )
    assert not study._stop_flag  # noqa: SLF001


def test_run_numeric_study_all_trials_failed(tmp_path: Path) -> None:
    class AlwaysFailEvaluator:
        def evaluate(self, *, trial_number: int, suggestions: dict, trial_overlay: dict, rep: int) -> dict[str, float]:
            raise StudyDriverError("simulated trial failure")

    from nemo_optimization.backends.optuna.study_driver import StudyDriverError

    payload = _payload()
    payload["optimizer"]["numeric"]["n_trials"] = 2
    with pytest.raises(StudyDriverError, match="no completed trials") as exc_info:
        run_numeric_study(payload, tmp_path, AlwaysFailEvaluator(), seed=0)
    assert exc_info.value.trial_count == 2


def test_sanitize_config_for_artifact_redacts_secrets() -> None:
    from nemo_optimization.backends.optuna.study_driver import sanitize_config_for_artifact

    sanitized = sanitize_config_for_artifact(
        {
            "models": {
                "default": {
                    "api_key": "super-secret",
                    "base_url": "http://example/v1",
                    "api_key_env": "NVIDIA_API_KEY",
                }
            }
        }
    )
    assert sanitized["models"]["default"]["api_key"] == "${REDACTED}"
    assert sanitized["models"]["default"]["base_url"] == "http://example/v1"
    # Unexpanded env refs are left intact.
    assert sanitized["models"]["default"]["api_key_env"] == "NVIDIA_API_KEY"


def test_optuna_backend_missing_optimizer_message() -> None:
    from unittest.mock import MagicMock

    from nemo_optimization.backends.optuna.backend import OptunaBackend
    from nemo_optimization.backends.optuna.study_driver import StudyDriverError

    ctx = MagicMock()
    ctx.storage.persistent = MagicMock()
    with pytest.raises(StudyDriverError, match="must include an 'optimizer' section"):
        OptunaBackend().run_study({"schema_version": "fabric.agent/v1alpha1"}, ctx=ctx)
