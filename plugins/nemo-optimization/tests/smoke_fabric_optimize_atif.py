# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in smoke: Fabric optimize study with ATIF trajectory capture (Hermes).

Requires a reachable OpenAI-compatible inference endpoint and NeMo Fabric with Hermes:

  NEMO_FABRIC_REPO=/path/to/NeMo-Fabric \\
  RUN_NEMO_OPTIMIZE_ATIF_E2E=1 \\
  FABRIC_QWEN_BASE_URL=http://.../v1 \\
  FABRIC_QWEN_MODEL=<your-model-id> \\
  uv run --package nemo-optimization-plugin pytest \\
    plugins/nemo-optimization/tests/smoke_fabric_optimize_atif.py -q

Install Fabric first: ``NEMO_FABRIC_REPO=... script/dev-install-fabric.sh``

Golden path shape matches Hermes (``nvidia.fabric.hermes``), inspired by the
email-phishing-analyzer harnesses. MCP AnalyzerRunBinding agents need an
extra trial-path bridge and are not covered by this smoke.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml
from nemo_optimization.router import OptimizeRouter
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "hermes-optimize"
_FABRIC_REPO = Path(os.environ.get("NEMO_FABRIC_REPO", ""))
_BASE_URL = os.environ.get("FABRIC_QWEN_BASE_URL", "")
_MODEL = os.environ.get("FABRIC_QWEN_MODEL", "")
_LIVE_READY = bool(
    os.environ.get("RUN_NEMO_OPTIMIZE_ATIF_E2E") == "1"
    and _FABRIC_REPO.is_dir()
    and _BASE_URL
    and _MODEL
    and importlib.util.find_spec("nemo_fabric") is not None
)

requires_live_optimize_atif = pytest.mark.skipif(
    not _LIVE_READY,
    reason=(
        "set RUN_NEMO_OPTIMIZE_ATIF_E2E=1, NEMO_FABRIC_REPO, FABRIC_QWEN_BASE_URL, "
        "FABRIC_QWEN_MODEL, and install nemo-fabric (script/dev-install-fabric.sh)"
    ),
)


def _build_payload(dataset_path: Path) -> dict:
    agent = yaml.safe_load((_EXAMPLE / "optimize-chatonly.yaml").read_text(encoding="utf-8"))

    agent["models"]["default"].update(
        {
            "model": _MODEL,
            "base_url": _BASE_URL,
            "api_key_env": "NEMO_AGENTS_IGW_API_KEY",
        }
    )
    agent["models"]["judge"].update(
        {
            "model": _MODEL,
            "base_url": _BASE_URL,
            "api_key_env": "NEMO_AGENTS_IGW_API_KEY",
        }
    )
    agent["eval"]["general"]["dataset"] = {"file_path": str(dataset_path)}
    agent["eval"]["fabric"] = {
        "base_dir": str(_EXAMPLE),
        "capture_trajectory": True,
        "timeout_s": 300,
    }
    agent["optimizer"]["numeric"]["n_trials"] = int(os.environ.get("NEMO_OPTIMIZE_ATIF_TRIALS", "2"))
    return agent


@requires_live_optimize_atif
@pytest.mark.timeout(300)
def test_optimize_study_writes_trial_trace_map(tmp_path: Path) -> None:
    dataset = tmp_path / "rows.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "capital-france",
                    "question": "In one short sentence, what is the capital of France?",
                    "answer": "Answer must state that the capital of France is Paris.",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"
    persistent.mkdir()
    ephemeral.mkdir()
    ctx = JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
        results=LocalJobResults(root=persistent / "results"),
    )

    result = OptimizeRouter.dispatch_payload(_build_payload(dataset), ctx=ctx)
    assert result["status"] == "completed"
    assert result["n_trials"] >= 2

    out_dir = persistent / "results" / "optimizer_results"
    summary = json.loads((out_dir / "study_summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_id"]

    trace_map = json.loads((out_dir / "trial_trace_map.json").read_text(encoding="utf-8"))
    assert len(trace_map) >= 2, trace_map
    trial_numbers = {entry["trial_number"] for entry in trace_map}
    assert len(trial_numbers) >= 2, trial_numbers
    for entry in trace_map:
        assert entry["experiment_id"] == summary["experiment_id"]
        assert entry["row_id"] == "capital-france"
        assert entry["trace_format"] == "atif"

    first_trace = trace_map[0]
    atif_path = Path(first_trace["trace_ref"])
    assert atif_path.is_file(), first_trace["trace_ref"]
    trajectory = json.loads(atif_path.read_text(encoding="utf-8"))
    assert trajectory.get("steps"), trajectory
