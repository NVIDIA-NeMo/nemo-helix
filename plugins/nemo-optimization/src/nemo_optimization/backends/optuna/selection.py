# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-objective Pareto front collapse (harmonic / sum / chebyshev).

Ported from https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/main/packages/nvidia_nat_config_optimizer/src/nat/plugins/config_optimizer/parameters/selection.py
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import optuna
from optuna.study import Study, StudyDirection

_SUPPORTED_MODES = frozenset({"harmonic", "weighted_sum", "sum", "chebyshev"})


def pick_trial(
    study: Study,
    mode: str = "harmonic",
    *,
    weights: Sequence[float] | None = None,
    eps: float = 1e-12,
) -> optuna.trial.FrozenTrial:
    """Collapse ``study.best_trials`` to a single compromise trial."""
    front = study.best_trials
    if not front:
        raise ValueError("`study.best_trials` is empty — no Pareto-optimal trials found.")

    vals = _to_minimisation_matrix(front, study.directions)
    span = np.ptp(vals, axis=0)
    norm = (vals - vals.min(axis=0)) / (span + eps)

    normalized_mode = mode.lower()
    if normalized_mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}. Choose from {sorted(_SUPPORTED_MODES)} (hypervolume is intentionally unsupported)."
        )

    if normalized_mode == "harmonic":
        # Harmonic mean of *utilities* (1 - normalized cost). Using costs directly
        # lets a single zero cost dominate via argmin(hmean).
        utility = 1.0 - norm
        hmean = norm.shape[1] / (1.0 / (utility + eps)).sum(axis=1)
        best_idx = int(hmean.argmax())
    elif normalized_mode in {"weighted_sum", "sum"}:
        w = np.ones(norm.shape[1]) if weights is None else np.asarray(weights, float)
        if w.size != norm.shape[1]:
            raise ValueError("`weights` length must equal number of objectives.")
        best_idx = int((norm @ w).argmin())
    else:  # chebyshev
        best_idx = int(norm.max(axis=1).argmin())

    return front[best_idx]


def _to_minimisation_matrix(
    trials: Sequence[optuna.trial.FrozenTrial],
    directions: Sequence[StudyDirection],
) -> np.ndarray:
    vals = np.asarray([t.values for t in trials], dtype=float)
    for index, direction in enumerate(directions):
        if direction == StudyDirection.MAXIMIZE:
            vals[:, index] *= -1.0
    return vals
