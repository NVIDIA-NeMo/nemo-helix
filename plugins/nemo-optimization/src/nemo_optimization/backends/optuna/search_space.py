# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optuna search-space helpers for numeric Fabric dimensions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np

from nemo_optimization.search_space import (
    NumericSearchSpaceSpec,
    PromptSearchSpaceSpec,
    SearchSpaceError,
    parse_numeric_search_space,
    parse_search_space_entry,
    suggestions_by_path,
)


class _TrialLike(Protocol):
    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any: ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int = 1,
    ) -> int: ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float: ...


@dataclass(frozen=True)
class SearchSpaceSpec(NumericSearchSpaceSpec):
    """One Optuna-compatible numeric search-space dimension."""

    @classmethod
    def from_mapping(cls, name: str, spec: Mapping[str, Any]) -> SearchSpaceSpec:
        parsed = parse_search_space_entry(name, spec)
        if isinstance(parsed, PromptSearchSpaceSpec):
            raise SearchSpaceError(f"Search space entry {name!r} is prompt-only; enable optimizer.prompt for GA.")
        return _to_optuna_spec(parsed)

    def suggest(self, trial: _TrialLike, name: str) -> Any:
        if self.values is not None:
            return trial.suggest_categorical(name, list(self.values))
        if isinstance(self.low, int) and isinstance(self.high, int):
            if self.step is None:
                return trial.suggest_int(name, self.low, self.high, log=self.log)
            return trial.suggest_int(name, self.low, self.high, log=self.log, step=int(self.step))
        return trial.suggest_float(
            name,
            float(cast(float, self.low)),
            float(cast(float, self.high)),
            log=self.log,
            step=float(self.step) if self.step is not None else None,
        )

    def to_grid_values(self) -> list[Any]:
        if self.values is not None:
            return list(self.values)
        if self.low is None or self.high is None:
            raise SearchSpaceError("Grid search requires 'values' or both 'low' and 'high'.")
        if self.step is None:
            raise SearchSpaceError(f"Grid search with range (low={self.low}, high={self.high}) requires 'step'.")

        step_float = float(self.step)
        if step_float <= 0:
            raise SearchSpaceError(f"Grid search 'step' must be positive; got {self.step}.")

        if isinstance(self.low, int) and isinstance(self.high, int) and step_float.is_integer():
            if self.log:
                raise SearchSpaceError("Log scale is not supported for integer grid ranges.")
            step = int(step_float)
            values = list(range(self.low, self.high + 1, step))
            if values and values[-1] != self.high:
                values.append(self.high)
            return values

        if self.log:
            raise SearchSpaceError("Log scale is not supported for float grid ranges; use explicit 'values'.")

        low_val = float(self.low)
        high_val = float(self.high)
        values = np.arange(low_val, high_val, step_float).tolist()
        if not values or abs(values[-1] - high_val) > 1e-9:
            values.append(high_val)
        return [round(v, 12) for v in values]


def parse_search_space(optimizer: Mapping[str, Any]) -> dict[str, SearchSpaceSpec]:
    """Parse numeric dimensions from ``optimizer.search_space`` for Optuna."""

    return {name: _to_optuna_spec(spec) for name, spec in parse_numeric_search_space(optimizer).items()}


def grid_trial_count(space: Mapping[str, SearchSpaceSpec]) -> int:
    """Cartesian product size for an exhaustive grid study."""
    count = 1
    for spec in space.values():
        count *= len(spec.to_grid_values())
    return count


def _to_optuna_spec(spec: NumericSearchSpaceSpec) -> SearchSpaceSpec:
    return SearchSpaceSpec(
        path=spec.path,
        param_type=spec.param_type,
        values=spec.values,
        low=spec.low,
        high=spec.high,
        log=spec.log,
        step=spec.step,
    )


__all__ = [
    "SearchSpaceError",
    "SearchSpaceSpec",
    "grid_trial_count",
    "parse_search_space",
    "suggestions_by_path",
]
