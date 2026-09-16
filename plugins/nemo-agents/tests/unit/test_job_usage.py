# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from nemo_agents_plugin.jobs.job_usage import evaluation_batch_token_usage, fabric_output_token_usage


def test_fabric_output_accepts_openai_usage_names() -> None:
    usage = fabric_output_token_usage(
        {"response": "done", "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}}
    )

    assert usage is not None
    assert usage.input_tokens == 12
    assert usage.output_tokens == 5


def test_fabric_output_rejects_total_only_or_invalid_counts() -> None:
    assert fabric_output_token_usage({"usage": {"total_tokens": 17}}) is None
    assert fabric_output_token_usage({"usage": {"input_tokens": True, "output_tokens": -1}}) is None


def test_nat_batch_sums_complete_raw_executions(tmp_path: Path) -> None:
    _write_nat_result(tmp_path / "run-a", prompt=10, completion=4)
    _write_nat_result(tmp_path / "run-b", prompt=20, completion=6)

    usage = evaluation_batch_token_usage(tmp_path, runner="nat", expected_executions=2)

    assert usage is not None
    assert usage.input_tokens == 30
    assert usage.output_tokens == 10


def test_nat_batch_does_not_publish_partial_totals(tmp_path: Path) -> None:
    _write_nat_result(tmp_path / "run-a", prompt=10, completion=4)
    _write_nat_result(tmp_path / "run-b", prompt=20, completion=None)

    assert evaluation_batch_token_usage(tmp_path, runner="nat", expected_executions=2) is None
    assert evaluation_batch_token_usage(tmp_path, runner="nat", expected_executions=3) is None


def test_harbor_batch_sums_each_raw_trial_and_includes_cache_input(tmp_path: Path) -> None:
    _write_harbor_result(tmp_path / "batch__eval__trial-0", input_tokens=10, output_tokens=4, cache_tokens=3)
    _write_harbor_result(tmp_path / "batch__eval__trial-1", input_tokens=20, output_tokens=6, cache_tokens=5)

    usage = evaluation_batch_token_usage(tmp_path, runner="harbor", expected_executions=2)

    assert usage is not None
    assert usage.input_tokens == 38
    assert usage.output_tokens == 10


def test_harbor_batch_rejects_missing_trial_usage(tmp_path: Path) -> None:
    _write_harbor_result(tmp_path / "batch__eval", input_tokens=None, output_tokens=None, cache_tokens=None)

    assert evaluation_batch_token_usage(tmp_path, runner="harbor", expected_executions=1) is None


def _write_nat_result(run_dir: Path, *, prompt: int | None, completion: int | None) -> None:
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        json.dumps({"task": run_dir.name, "metrics": {"prompt_tokens": prompt, "completion_tokens": completion}})
    )


def _write_harbor_result(
    job_dir: Path,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_tokens: int | None,
) -> None:
    trial_dir = job_dir / "trial"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "eval",
                "agent_result": {
                    "n_input_tokens": input_tokens,
                    "n_output_tokens": output_tokens,
                    "n_cache_tokens": cache_tokens,
                },
            }
        )
    )
