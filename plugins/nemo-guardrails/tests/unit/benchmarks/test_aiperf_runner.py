# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import yaml
from nemo_guardrails_plugin.benchmarks.aiperf_runner import (
    SweepRunResult,
    collect_sweep_results,
    prepare_runtime_aiperf_config,
)


def _write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "batch_name": "nhx_igw_guardrails_sweep_concurrency",
                "output_base_dir": "plugins/nemo-guardrails/benchmarks/artifacts/aiperf_results",
                "base_config": {"model": "benchmark/guardrails-vm"},
                "sweeps": {"concurrency": [1, 2, 4]},
            }
        ),
        encoding="utf-8",
    )


class TestPrepareRuntimeAiperfConfig:
    def test_overrides_output_base_dir(self, tmp_path: Path) -> None:
        template_path = tmp_path / "template.yaml"
        _write_template(template_path)
        runtime_config_path = tmp_path / "out" / "runtime.yaml"
        aiperf_output_dir = tmp_path / "results"

        config = prepare_runtime_aiperf_config(
            template_path=template_path,
            runtime_config_path=runtime_config_path,
            aiperf_output_dir=aiperf_output_dir,
        )

        assert config["output_base_dir"] == str(aiperf_output_dir)
        written = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8"))
        assert written["output_base_dir"] == str(aiperf_output_dir)
        assert written["base_config"]["model"] == "benchmark/guardrails-vm"
        assert written["sweeps"]["concurrency"] == [1, 2, 4]

    def test_missing_template_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            prepare_runtime_aiperf_config(
                template_path=tmp_path / "absent.yaml",
                runtime_config_path=tmp_path / "out.yaml",
                aiperf_output_dir=tmp_path / "results",
            )

    def test_non_mapping_template_raises(self, tmp_path: Path) -> None:
        template_path = tmp_path / "bad.yaml"
        template_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            prepare_runtime_aiperf_config(
                template_path=template_path,
                runtime_config_path=tmp_path / "out.yaml",
                aiperf_output_dir=tmp_path / "results",
            )


def _make_sweep_dir(parent: Path, sweep_label: str, *, returncode: int | None, duration: float | None) -> Path:
    sweep_dir = parent / sweep_label
    sweep_dir.mkdir(parents=True)
    if returncode is not None:
        (sweep_dir / "process_result.json").write_text(json.dumps({"returncode": returncode}), encoding="utf-8")
    if duration is not None:
        (sweep_dir / "run_metadata.json").write_text(json.dumps({"duration_seconds": duration}), encoding="utf-8")
    return sweep_dir


class TestCollectSweepResults:
    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        assert collect_sweep_results(tmp_path / "missing") == []

    def test_collects_all_sweeps_with_status(self, tmp_path: Path) -> None:
        batch = tmp_path / "nhx_igw_guardrails_sweep_concurrency" / "20260527_120000"
        _make_sweep_dir(batch, "concurrency1", returncode=0, duration=70.5)
        _make_sweep_dir(batch, "concurrency2", returncode=1, duration=70.5)

        results = collect_sweep_results(tmp_path)

        assert len(results) == 2
        results_by_label = {r.sweep_label: r for r in results}
        assert results_by_label["concurrency1"].passed
        assert results_by_label["concurrency1"].duration_seconds == 70.5
        assert not results_by_label["concurrency2"].passed
        assert results_by_label["concurrency2"].return_code == 1

    def test_missing_process_result_is_failure(self, tmp_path: Path) -> None:
        batch = tmp_path / "batch" / "ts"
        _make_sweep_dir(batch, "concurrency1", returncode=None, duration=10.0)

        results = collect_sweep_results(tmp_path)

        assert len(results) == 1
        assert not results[0].passed

    def test_malformed_json_treated_as_failure_without_crashing(self, tmp_path: Path) -> None:
        batch = tmp_path / "batch" / "ts"
        sweep = _make_sweep_dir(batch, "concurrency1", returncode=None, duration=None)
        (sweep / "process_result.json").write_text("not-json", encoding="utf-8")
        (sweep / "run_metadata.json").write_text("not-json", encoding="utf-8")

        results = collect_sweep_results(tmp_path)
        assert results[0].return_code == 1
        assert results[0].duration_seconds == 0.0


class TestSweepRunResult:
    def test_passed_property(self) -> None:
        passing = SweepRunResult(
            sweep_label="x",
            output_dir=Path("."),
            return_code=0,
            duration_seconds=1.0,
            metadata_path=None,
            process_result_path=None,
        )
        assert passing.passed
        assert not SweepRunResult(
            sweep_label="x",
            output_dir=Path("."),
            return_code=2,
            duration_seconds=1.0,
            metadata_path=None,
            process_result_path=None,
        ).passed
