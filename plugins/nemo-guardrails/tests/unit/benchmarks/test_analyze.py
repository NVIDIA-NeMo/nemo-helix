# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The strict baseline check must fail when a CI shard's results are lost."""

from pathlib import Path

import pytest
from nemo_guardrails_plugin.benchmarks.analyze import (
    CONCURRENCIES_TO_VALIDATE,
    DELTA_P50_BASELINE_BY_CONCURRENCY,
    VARIANT_WITH_GUARDRAILS,
    VARIANT_WITHOUT_GUARDRAILS,
    check_against_baseline,
    compare,
    find_missing_levels,
    load_variant_results,
    main,
)

_WITHOUT_GUARDRAILS_P50_MS = 4100.0
_SHARDS = ((1, 2), (4, 8), (16, 32))


def _write_level(run_dir: Path, variant: str, timestamp: str, concurrency: int) -> None:
    """Write one level's AIPerf CSV as `<variant>/<batch>/<timestamp>/concurrency<N>/`."""
    p50 = _WITHOUT_GUARDRAILS_P50_MS
    if variant == VARIANT_WITH_GUARDRAILS:
        p50 += DELTA_P50_BASELINE_BY_CONCURRENCY[concurrency]
    sweep_dir = run_dir / "aiperf_results" / variant / "sweep" / timestamp / f"concurrency{concurrency}"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "profile_export_aiperf.csv").write_text(
        f"Metric,avg,p50,p90,p99,std\nRequest Latency (ms),{p50},{p50},{p50},{p50},1.0\n",
        encoding="utf-8",
    )


def _write_sharded_run(run_dir: Path, *, skip: tuple[str, int] | None = None) -> None:
    """Lay out a run as CI's merged shard artifacts do: one timestamp dir per shard."""
    for variant in (VARIANT_WITH_GUARDRAILS, VARIANT_WITHOUT_GUARDRAILS):
        for shard_index, shard in enumerate(_SHARDS):
            for concurrency in shard:
                if skip != (variant, concurrency):
                    _write_level(run_dir, variant, f"20261127_10000{shard_index}", concurrency)


def test_shards_of_one_variant_merge_into_one_result_set(tmp_path: Path) -> None:
    _write_sharded_run(tmp_path)

    results = load_variant_results(tmp_path / "aiperf_results" / VARIANT_WITH_GUARDRAILS)

    assert sorted(results) == sorted(CONCURRENCIES_TO_VALIDATE)


def test_every_checked_level_has_a_baseline() -> None:
    assert set(CONCURRENCIES_TO_VALIDATE) <= set(DELTA_P50_BASELINE_BY_CONCURRENCY)


def test_find_missing_levels_names_the_variant(tmp_path: Path) -> None:
    _write_sharded_run(tmp_path, skip=(VARIANT_WITHOUT_GUARDRAILS, 16))
    aiperf_dir = tmp_path / "aiperf_results"

    missing = find_missing_levels(
        {
            VARIANT_WITH_GUARDRAILS: load_variant_results(aiperf_dir / VARIANT_WITH_GUARDRAILS),
            VARIANT_WITHOUT_GUARDRAILS: load_variant_results(aiperf_dir / VARIANT_WITHOUT_GUARDRAILS),
        }
    )

    assert missing == {VARIANT_WITHOUT_GUARDRAILS: [16]}


def test_check_against_baseline_passes_with_every_level(tmp_path: Path) -> None:
    _write_sharded_run(tmp_path)
    aiperf_dir = tmp_path / "aiperf_results"
    rows = compare(
        load_variant_results(aiperf_dir / VARIANT_WITH_GUARDRAILS),
        load_variant_results(aiperf_dir / VARIANT_WITHOUT_GUARDRAILS),
    )

    report, failed_count = check_against_baseline(rows)

    assert failed_count == 0
    assert "MISSING" not in report


def test_check_against_baseline_fails_on_a_missing_level(tmp_path: Path) -> None:
    _write_sharded_run(tmp_path, skip=(VARIANT_WITH_GUARDRAILS, 32))
    aiperf_dir = tmp_path / "aiperf_results"
    rows = compare(
        load_variant_results(aiperf_dir / VARIANT_WITH_GUARDRAILS),
        load_variant_results(aiperf_dir / VARIANT_WITHOUT_GUARDRAILS),
    )

    report, failed_count = check_against_baseline(rows)

    assert failed_count == 1
    assert "MISSING" in report
    assert "[32]" in report


@pytest.mark.parametrize("variant", [VARIANT_WITH_GUARDRAILS, VARIANT_WITHOUT_GUARDRAILS])
@pytest.mark.parametrize("concurrency", CONCURRENCIES_TO_VALIDATE)
def test_strict_main_fails_when_any_checked_level_is_missing(tmp_path: Path, variant: str, concurrency: int) -> None:
    _write_sharded_run(tmp_path, skip=(variant, concurrency))

    assert main([str(tmp_path), "--strict"]) == 1
    # Without --strict the report still prints and the exit code stays 0.
    assert main([str(tmp_path)]) == 0


def test_strict_main_fails_when_a_whole_shard_is_missing(tmp_path: Path) -> None:
    _write_sharded_run(tmp_path)
    # Drop the with-guardrails c=4,8 shard: its two levels share a timestamp dir.
    for concurrency in (4, 8):
        sweep_dir = tmp_path / "aiperf_results" / VARIANT_WITH_GUARDRAILS / "sweep" / "20261127_100001"
        (sweep_dir / f"concurrency{concurrency}" / "profile_export_aiperf.csv").unlink()

    assert main([str(tmp_path), "--strict"]) == 1


def test_strict_main_passes_with_every_shard(tmp_path: Path) -> None:
    _write_sharded_run(tmp_path)

    assert main([str(tmp_path), "--strict"]) == 0
