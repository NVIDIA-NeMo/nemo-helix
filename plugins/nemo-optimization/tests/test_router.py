# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from nemo_optimization.backends.ga.backend import GaBackendError
from nemo_optimization.router import OptimizeRouter, OptimizeRouterError
from nemo_helix_plugin.job_context import JobContext


def test_dispatch_routes_numeric_to_optuna_study(ctx: JobContext) -> None:
    payload = {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "demo"},
        "optimizer": {
            "numeric": {"enabled": True, "n_trials": 2},
            "eval_metrics": {
                "average_score": {"direction": "maximize", "weight": 1.0},
            },
            "search_space": {
                "temperature": {
                    "type": "fabric",
                    "path": "models.default.temperature",
                    "values": [0.0, 0.2],
                },
            },
        },
    }
    result = OptimizeRouter.dispatch_payload(payload, ctx=ctx)
    assert result["status"] == "completed"
    assert result["backend"] == "optuna"
    assert result["phase"] == "core"
    assert result["n_trials"] == 2

    out_dir = ctx.storage.persistent / "results" / "optimizer_results"
    summary = json.loads((out_dir / "study_summary.json").read_text(encoding="utf-8"))
    assert summary["backend"] == "optuna"
    debug = json.loads((out_dir / "study_debug.json").read_text(encoding="utf-8"))
    assert debug["n_trials"] == 2
    assert len(debug["trials"]) == 2
    assert {t["state"] for t in debug["trials"]} == {"COMPLETE"}
    assert (out_dir / "optimized_config.yml").is_file()


def test_dispatch_prompt_enabled_fails_fast(ctx: JobContext) -> None:
    payload = {
        "schema_version": "fabric.agent/v1alpha1",
        "optimizer": {"prompt": {"enabled": True}},
    }
    with pytest.raises(GaBackendError, match="not supported yet"):
        OptimizeRouter.dispatch_payload(payload, ctx=ctx)


def test_dispatch_requires_enabled_backend(ctx: JobContext) -> None:
    payload = {"schema_version": "fabric.agent/v1alpha1", "optimizer": {}}
    with pytest.raises(OptimizeRouterError, match="No Tune backend selected"):
        OptimizeRouter.dispatch_payload(payload, ctx=ctx)
