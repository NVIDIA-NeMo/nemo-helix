# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optuna study loop for numeric/categorical HPO.

Ported from https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/main/packages/nvidia_nat_config_optimizer/src/nat/plugins/config_optimizer/parameters/optimizer.py
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import optuna
import yaml
from optuna.samplers import GridSampler
from optuna.study import StudyDirection

from nemo_optimization.backends.optuna.artifacts import maybe_write_pareto_plots, write_trials_dataframe
from nemo_optimization.backends.optuna.config_overlay import (
    apply_suggestions,
    suggestions_to_profile_overlay,
)
from nemo_optimization.backends.optuna.early_stop import maybe_stop_if_target_met
from nemo_optimization.backends.optuna.search_space import (
    SearchSpaceError,
    SearchSpaceSpec,
    grid_trial_count,
    parse_search_space,
    suggestions_by_path,
)
from nemo_optimization.backends.optuna.selection import pick_trial

logger = logging.getLogger(__name__)


class StudyDriverError(RuntimeError):
    """Raised when study configuration or execution fails."""


class TrialEvaluator(Protocol):
    """Evaluate one repetition of a trial (wired to AgentEvaluator in Phase B2)."""

    def evaluate(
        self,
        *,
        trial_number: int,
        suggestions: dict[str, Any],
        trial_overlay: dict[str, Any],
        rep: int,
    ) -> dict[str, float]:
        """Return metric name to score for one repetition.

        ``suggestions`` must be keyed by Fabric dotted paths (the output of
        ``suggestions_by_path``), matching ``apply_suggestions`` / trial YAML.
        """


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: StudyDirection
    weight: float


@dataclass(frozen=True)
class NumericStudyConfig:
    n_trials: int
    sampler: str
    reps_per_param_set: int
    target: float | None
    multi_objective_mode: str
    metrics: tuple[MetricSpec, ...]
    search_space: dict[str, SearchSpaceSpec]


@dataclass(frozen=True)
class NumericStudyResult:
    study: optuna.Study
    best_trial: optuna.trial.FrozenTrial
    metric_names: tuple[str, ...]
    n_trials: int
    output_dir: Path
    #: The winning trial's suggestions, keyed by the Fabric dotted paths declared in
    #: ``optimizer.search_space`` (the same mapping ``apply_suggestions`` used to build
    #: ``optimized_config.yml``). Lets a caller apply the tuned values to something other
    #: than the Fabric payload — e.g. the stored platform agent spec — without recomputing
    #: the suggestion mapping itself.
    best_params_by_path: dict[str, Any] = field(default_factory=dict)


def parse_numeric_study_config(optimizer: Mapping[str, Any]) -> NumericStudyConfig:
    numeric = optimizer.get("numeric")
    if not isinstance(numeric, Mapping):
        raise StudyDriverError("optimizer.numeric must be a mapping.")
    if not numeric.get("enabled"):
        raise StudyDriverError("optimizer.numeric.enabled must be true.")

    eval_metrics = optimizer.get("eval_metrics")
    if not isinstance(eval_metrics, Mapping) or not eval_metrics:
        raise StudyDriverError("optimizer.eval_metrics must declare at least one metric.")

    metrics: list[MetricSpec] = []
    for name, raw in eval_metrics.items():
        if not isinstance(raw, Mapping):
            raise StudyDriverError(f"optimizer.eval_metrics[{name!r}] must be a mapping.")
        direction_raw = str(raw.get("direction", "maximize")).lower()
        if direction_raw not in {"maximize", "minimize"}:
            raise StudyDriverError(f"Metric {name!r} direction must be 'maximize' or 'minimize'.")
        metric_name = str(raw.get("evaluator_name") or name)
        metrics.append(
            MetricSpec(
                name=metric_name,
                direction=StudyDirection.MAXIMIZE if direction_raw == "maximize" else StudyDirection.MINIMIZE,
                weight=float(raw.get("weight", 1.0)),
            )
        )

    sampler = numeric.get("sampler")
    # "bayesian" / "tpe" / omitted → Optuna TPE (Bayesian optimization). Keep the
    # canonical name so configs and study metadata are not silently rewritten to None.
    if sampler is None:
        sampler_name = "bayesian"
    else:
        sampler_name = str(sampler).lower()
        if sampler_name in {"bayesian", "tpe"}:
            sampler_name = "bayesian"
        elif sampler_name != "grid":
            raise StudyDriverError(
                f"Unsupported optimizer.numeric.sampler: {sampler!r}. "
                "Supported values: 'bayesian' (TPE), 'tpe', 'grid'."
            )

    return NumericStudyConfig(
        n_trials=int(numeric.get("n_trials", 20)),
        sampler=sampler_name,
        reps_per_param_set=max(1, int(optimizer.get("reps_per_param_set", 1))),
        target=float(optimizer["target"]) if optimizer.get("target") is not None else None,
        multi_objective_mode=str(optimizer.get("multi_objective_combination_mode", "harmonic")),
        metrics=tuple(metrics),
        search_space=parse_search_space(optimizer),
    )


def create_sampler(config: NumericStudyConfig, *, seed: int | None = None) -> optuna.samplers.BaseSampler:
    if config.sampler == "grid":
        grid = {name: spec.to_grid_values() for name, spec in config.search_space.items()}
        return GridSampler(grid, seed=seed)
    # bayesian / default: TPE for single-objective, NSGA-II for multi-objective.
    if len(config.metrics) > 1:
        return optuna.samplers.NSGAIISampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def resolve_n_trials(config: NumericStudyConfig) -> int:
    if config.sampler == "grid":
        return grid_trial_count(config.search_space)
    return config.n_trials


def average_metric_vectors(rep_scores: Sequence[Mapping[str, float]], metric_names: Sequence[str]) -> list[float]:
    if not rep_scores:
        raise StudyDriverError("Cannot average scores from zero repetitions.")
    return [sum(rep[name] for rep in rep_scores) / len(rep_scores) for name in metric_names]


def scores_to_objective_values(scores: Mapping[str, float], metric_names: Sequence[str]) -> list[float]:
    return [float(scores[name]) for name in metric_names]


def run_numeric_study(
    payload: Mapping[str, Any],
    output_dir: Path,
    evaluator: TrialEvaluator,
    *,
    seed: int | None = None,
) -> NumericStudyResult:
    """Execute one numeric Optuna study for a Fabric-native optimize payload."""
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise StudyDriverError("payload must include an optimizer mapping.")

    config = parse_numeric_study_config(optimizer)
    metric_names = tuple(metric.name for metric in config.metrics)
    directions = [metric.direction for metric in config.metrics]
    weights = [metric.weight for metric in config.metrics]

    sampler = create_sampler(config, seed=seed)
    n_trials = resolve_n_trials(config)
    study = optuna.create_study(
        directions=directions,
        sampler=sampler,
        study_name=str(payload.get("metadata", {}).get("name") or "optimize"),
    )

    base_config = copy.deepcopy(dict(payload))
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_id_width = max(1, len(str(max(0, n_trials - 1))))

    def objective(trial: optuna.Trial) -> float | list[float]:
        suggestions = {name: spec.suggest(trial, name) for name, spec in config.search_space.items()}
        path_suggestions = suggestions_by_path(config.search_space, suggestions)
        trial_overlay = suggestions_to_profile_overlay(path_suggestions, trial.number)
        write_trial_config(
            output_dir,
            trial.number,
            apply_suggestions(base_config, path_suggestions),
            width=trial_id_width,
        )

        rep_scores = [
            evaluator.evaluate(
                trial_number=trial.number,
                suggestions=dict(path_suggestions),
                trial_overlay=trial_overlay,
                rep=rep,
            )
            for rep in range(config.reps_per_param_set)
        ]
        for rep_index, rep_score in enumerate(rep_scores):
            missing = [name for name in metric_names if name not in rep_score]
            if missing:
                raise StudyDriverError(f"Trial {trial.number} rep {rep_index} missing metric scores: {missing}")

        trial.set_user_attr(
            "rep_scores",
            [scores_to_objective_values(rep, metric_names) for rep in rep_scores],
        )
        averaged = average_metric_vectors(rep_scores, metric_names)
        objective_values = scores_to_objective_values(dict(zip(metric_names, averaged, strict=True)), metric_names)
        maybe_stop_if_target_met(
            study,
            objective_values,
            target=config.target,
            directions=directions,
        )
        return objective_values[0] if len(objective_values) == 1 else objective_values

    logger.info("Starting numeric Optuna study (%d trials, %d metrics)", n_trials, len(metric_names))
    # Agent-eval / audit failures raise StudyDriverError; fail that Optuna trial and continue.
    # Do not catch broader Exception — programming errors should still abort the study.
    study.optimize(objective, n_trials=n_trials, catch=(StudyDriverError,))
    logger.info("Numeric Optuna study finished")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        n_failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
        n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        raise StudyDriverError(
            f"Numeric study finished with no completed trials "
            f"({n_failed} failed, {n_pruned} pruned, {len(study.trials)} total)."
        )

    if len(metric_names) == 1:
        best_trial = study.best_trial
    else:
        best_trial = pick_trial(
            study,
            mode=config.multi_objective_mode,
            weights=weights,
        )

    # best_trial.params is keyed by logical search-space names; map to Fabric paths
    # the same way trial configs do before writing optimized_config.yml.
    best_params_by_path = suggestions_by_path(config.search_space, best_trial.params)
    optimized_config = apply_suggestions(base_config, best_params_by_path)
    write_optimized_config(output_dir, optimized_config)
    write_trials_dataframe(study=study, metric_names=metric_names, output_dir=output_dir)
    maybe_write_pareto_plots(study=study, metric_names=metric_names, directions=directions, output_dir=output_dir)

    return NumericStudyResult(
        study=study,
        best_trial=best_trial,
        metric_names=metric_names,
        n_trials=n_trials,
        output_dir=output_dir,
        best_params_by_path=best_params_by_path,
    )


def write_trial_config(
    output_dir: Path,
    trial_number: int,
    trial_config: Mapping[str, Any],
    *,
    width: int,
) -> Path:
    path = output_dir / f"config_numeric_trial_{trial_number:0{width}d}.yml"
    path.write_text(
        yaml.safe_dump(sanitize_config_for_artifact(trial_config), sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_optimized_config(output_dir: Path, optimized_config: Mapping[str, Any]) -> Path:
    path = output_dir / "optimized_config.yml"
    path.write_text(
        yaml.safe_dump(sanitize_config_for_artifact(optimized_config), sort_keys=False),
        encoding="utf-8",
    )
    return path


_SECRET_VALUE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "password",
        "passwd",
        "secret",
        "token",
        "authorization",
        "access_token",
        "refresh_token",
        "client_secret",
        "nvidia_api_key",
    }
)


def _is_secret_value_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    # Reference fields hold env/secret *names*, not credentials.
    if lowered.endswith("_env") or lowered in {"api_key_secret", "api_key_env"}:
        return False
    if lowered in _SECRET_VALUE_KEYS:
        return True
    return any(lowered.endswith(f"_{suffix}") for suffix in ("api_key", "password", "token", "secret"))


def sanitize_config_for_artifact(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy with secret-bearing fields redacted for persistent YAML."""

    def _redact(value: Any, *, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {k: _redact(v, key=str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [_redact(v, key=key) for v in value]
        if key is not None and isinstance(value, str) and value and not value.startswith("${"):
            if _is_secret_value_key(key):
                return "${REDACTED}"
        return value

    return _redact(dict(config))


class SyntheticTrialEvaluator:
    """Deterministic evaluator for unit tests (sum of numeric suggestion values)."""

    def __init__(self, metric_names: Sequence[str]) -> None:
        self._metric_names = tuple(metric_names)

    def evaluate(
        self,
        *,
        trial_number: int,
        suggestions: dict[str, Any],
        trial_overlay: dict[str, Any],
        rep: int,
    ) -> dict[str, float]:
        del trial_number, trial_overlay, rep
        score = _numeric_suggestion_score(suggestions)
        return {name: score for name in self._metric_names}


def _numeric_suggestion_score(suggestions: Mapping[str, Any]) -> float:
    total = 0.0
    for value in suggestions.values():
        if isinstance(value, bool):
            total += float(value)
        elif isinstance(value, (int, float)):
            total += float(value)
    return total


__all__ = [
    "MetricSpec",
    "NumericStudyConfig",
    "NumericStudyResult",
    "SearchSpaceError",
    "StudyDriverError",
    "SyntheticTrialEvaluator",
    "TrialEvaluator",
    "average_metric_vectors",
    "create_sampler",
    "parse_numeric_study_config",
    "resolve_n_trials",
    "run_numeric_study",
    "sanitize_config_for_artifact",
    "write_optimized_config",
    "write_trial_config",
]
