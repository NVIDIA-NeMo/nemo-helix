# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-run analyzer for the nemo-guardrails IGW benchmark.

Reads ``profile_export_aiperf.csv`` files from both variants in one run dir
and prints a with-vs-without latency comparison. The delta isolates
middleware overhead since the only difference between variants is whether
middleware is attached to the targeted VirtualModel.

Used both as a script (``python -m ... <run-dir>``) and auto-invoked from
``run.py`` after a multi-variant sweep.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Duplicated from `constants.py` so this module stays import-free and can
# run on bare `python3` in CI without bootstrapping the uv workspace.
VARIANT_WITH_GUARDRAILS = "with-guardrails"
VARIANT_WITHOUT_GUARDRAILS = "without-guardrails"

# --- CI baseline gate ---------------------------------------------------------
# For each concurrency level we list:
#   - The expected p50 latency delta between requests with guardrails vs.
#     without guardrails.
#   - The allowed regression tolerance in CI. Benchmark jobs whose p50
#     latency exceeds the baseline by more than this tolerance will fail.

# Concurrency levels we check in CI.
CONCURRENCIES_TO_VALIDATE: list[int] = [1, 2, 4, 8, 16, 32]

# Tolerance (ms) used for every concurrency level unless overridden below.
DEFAULT_DELTA_P50_TOLERANCE_MS: int = 150

# Looser tolerance (ms) for higher concurrencies. With more requests in
# flight at once, they contend for shared resources (the IGW event loop,
# the mock-LLM workers, the CI runner's CPU), so we see more variance in
# latency values.
DELTA_P50_TOLERANCE_OVERRIDES_MS: dict[int, int] = {16: 200, 32: 450}

# Estimated expected delta_p50 (ms) at each concurrency level, based on
# a few sample runs in CI.
DELTA_P50_BASELINE_BY_CONCURRENCY: dict[int, int] = {
    1: 1070,
    2: 1110,
    4: 1190,
    8: 1230,
    16: 1390,
    32: 2110,
}

log = logging.getLogger(__name__)

_LATENCY_METRIC = "Request Latency (ms)"

# Mock-LLM time per request, subtracted to isolate platform overhead. Mirrors
# `E2E_LATENCY_MEAN_SECONDS` in configs/mock_llm/*.env and the 2 CS calls
# (input + output rails) of `content_safety_local`. Update in lock-step.
_APP_MOCK_LATENCY_MS = 4000.0
_CONTENT_SAFETY_MOCK_LATENCY_MS = 500.0
_CONTENT_SAFETY_CALLS_PER_GUARDED_REQUEST = 2
_MOCK_TIME_PER_REQUEST_WITHOUT_GUARDRAILS_MS = _APP_MOCK_LATENCY_MS
_MOCK_TIME_PER_REQUEST_WITH_GUARDRAILS_MS = (
    _APP_MOCK_LATENCY_MS + _CONTENT_SAFETY_CALLS_PER_GUARDED_REQUEST * _CONTENT_SAFETY_MOCK_LATENCY_MS
)


@dataclass(frozen=True)
class LatencyRow:
    """Per-concurrency latency stats parsed from one AIPerf CSV."""

    concurrency: int
    avg: float
    p50: float
    p90: float
    p99: float
    std: float


@dataclass(frozen=True)
class ComparisonRow:
    """Side-by-side comparison of one concurrency level across variants."""

    concurrency: int
    with_guardrails: LatencyRow
    without_guardrails: LatencyRow

    @property
    def delta_p50(self) -> float:
        return self.with_guardrails.p50 - self.without_guardrails.p50

    @property
    def delta_p90(self) -> float:
        return self.with_guardrails.p90 - self.without_guardrails.p90

    @property
    def delta_avg(self) -> float:
        return self.with_guardrails.avg - self.without_guardrails.avg


def load_variant_results(variant_output_dir: Path) -> dict[int, LatencyRow]:
    """Load per-concurrency latency stats for one variant.

    Walks the ``<batch>/<timestamp>/concurrency<N>/`` layout produced by
    ``collect_sweep_results``. Missing CSVs are skipped, not raised, so
    partial runs still produce a table.
    """
    if not variant_output_dir.is_dir():
        return {}

    latency_by_concurrency: dict[int, LatencyRow] = {}
    for batch_dir in sorted(p for p in variant_output_dir.iterdir() if p.is_dir()):
        for timestamp_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
            for sweep_dir in sorted(p for p in timestamp_dir.iterdir() if p.is_dir()):
                concurrency = _parse_concurrency_from_label(sweep_dir.name)
                if concurrency is None:
                    continue
                csv_path = sweep_dir / "profile_export_aiperf.csv"
                row = _read_latency_row(csv_path, concurrency)
                if row is not None:
                    latency_by_concurrency[concurrency] = row
    return latency_by_concurrency


def compare(
    latency_by_concurrency_with_guardrails: dict[int, LatencyRow],
    latency_by_concurrency_without_guardrails: dict[int, LatencyRow],
) -> list[ComparisonRow]:
    """Build per-concurrency comparison rows, sorted by concurrency.

    Only levels present in both variants are compared; asymmetric levels are
    logged at WARNING and excluded.
    """
    concurrencies_with_guardrails = set(latency_by_concurrency_with_guardrails)
    concurrencies_without_guardrails = set(latency_by_concurrency_without_guardrails)
    concurrencies_in_both_variants = sorted(concurrencies_with_guardrails & concurrencies_without_guardrails)

    concurrencies_in_only_one_variant = sorted(concurrencies_with_guardrails ^ concurrencies_without_guardrails)
    if concurrencies_in_only_one_variant:
        log.warning(
            "Concurrency levels present in only one variant, excluded from comparison: %s",
            concurrencies_in_only_one_variant,
        )

    return [
        ComparisonRow(
            concurrency,
            latency_by_concurrency_with_guardrails[concurrency],
            latency_by_concurrency_without_guardrails[concurrency],
        )
        for concurrency in concurrencies_in_both_variants
    ]


def format_table(rows: list[ComparisonRow]) -> str:
    """Render the comparison as a fixed-width text table."""
    if not rows:
        return "No comparable sweep results found (need both variants to share concurrency levels)."

    header = (
        "conc",
        "with p50",
        "w/o p50",
        "delta p50",
        "with p90",
        "w/o p90",
        "delta p90",
        "with avg",
        "w/o avg",
        "delta avg",
    )
    fmt = "{:>4}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}"
    header_line = fmt.format(*header)
    lines = ["Measured Latencies (ms), with and without guardrails:", header_line, "-" * len(header_line)]
    for r in rows:
        lines.append(
            fmt.format(
                r.concurrency,
                f"{r.with_guardrails.p50:.0f}",
                f"{r.without_guardrails.p50:.0f}",
                f"{r.delta_p50:+.0f}",
                f"{r.with_guardrails.p90:.0f}",
                f"{r.without_guardrails.p90:.0f}",
                f"{r.delta_p90:+.0f}",
                f"{r.with_guardrails.avg:.0f}",
                f"{r.without_guardrails.avg:.0f}",
                f"{r.delta_avg:+.0f}",
            )
        )
    lines.append("")
    lines.append("delta = with-guardrails minus without-guardrails.")
    return "\n".join(lines)


def format_platform_overhead_table(rows: list[ComparisonRow]) -> str:
    """Render a table with mock-LLM time subtracted from p50/p90/avg.

    Isolates NHX + IGW + shim + middleware overhead from the much larger
    mock sleeps. The delta columns are the middleware's own cost over the
    bare path.
    """
    if not rows:
        return "No comparable sweep results found (need both variants to share concurrency levels)."

    header = (
        "conc",
        "with p50",
        "w/o p50",
        "delta p50",
        "with p90",
        "w/o p90",
        "delta p90",
        "with avg",
        "w/o avg",
        "delta avg",
    )
    fmt = "{:>4}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}"
    header_line = fmt.format(*header)
    lines = ["Platform Overhead (ms), with and without guardrails:", header_line, "-" * len(header_line)]

    for r in rows:
        with_p50 = r.with_guardrails.p50 - _MOCK_TIME_PER_REQUEST_WITH_GUARDRAILS_MS
        without_p50 = r.without_guardrails.p50 - _MOCK_TIME_PER_REQUEST_WITHOUT_GUARDRAILS_MS
        with_p90 = r.with_guardrails.p90 - _MOCK_TIME_PER_REQUEST_WITH_GUARDRAILS_MS
        without_p90 = r.without_guardrails.p90 - _MOCK_TIME_PER_REQUEST_WITHOUT_GUARDRAILS_MS
        with_avg = r.with_guardrails.avg - _MOCK_TIME_PER_REQUEST_WITH_GUARDRAILS_MS
        without_avg = r.without_guardrails.avg - _MOCK_TIME_PER_REQUEST_WITHOUT_GUARDRAILS_MS
        lines.append(
            fmt.format(
                r.concurrency,
                f"{with_p50:+.0f}",
                f"{without_p50:+.0f}",
                f"{with_p50 - without_p50:+.0f}",
                f"{with_p90:+.0f}",
                f"{without_p90:+.0f}",
                f"{with_p90 - without_p90:+.0f}",
                f"{with_avg:+.0f}",
                f"{without_avg:+.0f}",
                f"{with_avg - without_avg:+.0f}",
            )
        )
    lines.append("")
    lines.append(
        f"Minus mock-LLM time "
        f"(with-guardrails: {_MOCK_TIME_PER_REQUEST_WITH_GUARDRAILS_MS:.0f} ms; "
        f"without-guardrails: {_MOCK_TIME_PER_REQUEST_WITHOUT_GUARDRAILS_MS:.0f} ms)."
    )
    return "\n".join(lines)


def analyze_run(run_dir: Path) -> str:
    """Read both variants from one run dir and return a printable report.

    Output is the raw comparison table followed by a platform-overhead table
    (mock time subtracted). Falls back to a single-variant table if only one
    variant has results.
    """
    aiperf_dir = run_dir / "aiperf_results"
    latency_by_concurrency_with_guardrails = load_variant_results(aiperf_dir / VARIANT_WITH_GUARDRAILS)
    latency_by_concurrency_without_guardrails = load_variant_results(aiperf_dir / VARIANT_WITHOUT_GUARDRAILS)

    if not latency_by_concurrency_with_guardrails and not latency_by_concurrency_without_guardrails:
        return f"No AIPerf results found under {aiperf_dir}"
    if not latency_by_concurrency_with_guardrails or not latency_by_concurrency_without_guardrails:
        if latency_by_concurrency_with_guardrails:
            return _format_single_variant(VARIANT_WITH_GUARDRAILS, latency_by_concurrency_with_guardrails)
        return _format_single_variant(VARIANT_WITHOUT_GUARDRAILS, latency_by_concurrency_without_guardrails)

    rows = compare(latency_by_concurrency_with_guardrails, latency_by_concurrency_without_guardrails)
    return f"{format_table(rows)}\n\n{format_platform_overhead_table(rows)}"


def _load_comparison_rows(run_dir: Path) -> list[ComparisonRow]:
    """Reload comparison rows from a run dir; returns ``[]`` if either variant is absent."""
    aiperf_dir = run_dir / "aiperf_results"
    with_guardrails = load_variant_results(aiperf_dir / VARIANT_WITH_GUARDRAILS)
    without_guardrails = load_variant_results(aiperf_dir / VARIANT_WITHOUT_GUARDRAILS)
    if not with_guardrails or not without_guardrails:
        return []
    return compare(with_guardrails, without_guardrails)


@dataclass(frozen=True)
class LatencyReport:
    """Latency results for a single concurrency level, rendered as one row of the report.

    Each instance represents a single concurrency level from the benchmark
    run: what we measured (observed_ms), what we expected from the
    baseline (baseline_ms), and how much slower it is allowed to be
    (tolerance_ms).
    The check passes when observed_ms - baseline_ms <= tolerance_ms, so
    improvements over the baseline do not fail the regression gate.
    """

    concurrency: int
    metric: str
    baseline_ms: float
    observed_ms: float
    tolerance_ms: float

    @property
    def diff_ms(self) -> float:
        return self.observed_ms - self.baseline_ms

    @property
    def passed(self) -> bool:
        return self.diff_ms <= self.tolerance_ms


def check_against_baseline(rows: list[ComparisonRow]) -> tuple[str, int]:
    """Compare the delta_p50 for each concurrency level against the baseline latencies.

    Returns ``(report_text, failed_count)``. Concurrencies missing from
    either the run or the baseline are skipped with a note.
    """
    rows_by_concurrency = {r.concurrency: r for r in rows}

    latency_reports: list[LatencyReport] = []
    skipped_concurrencies: list[int] = []

    for concurrency in sorted(CONCURRENCIES_TO_VALIDATE):
        if concurrency not in rows_by_concurrency or concurrency not in DELTA_P50_BASELINE_BY_CONCURRENCY:
            skipped_concurrencies.append(concurrency)
            continue
        latency_reports.append(
            LatencyReport(
                concurrency=concurrency,
                metric="delta_p50",
                baseline_ms=float(DELTA_P50_BASELINE_BY_CONCURRENCY[concurrency]),
                observed_ms=rows_by_concurrency[concurrency].delta_p50,
                tolerance_ms=float(DELTA_P50_TOLERANCE_OVERRIDES_MS.get(concurrency, DEFAULT_DELTA_P50_TOLERANCE_MS)),
            )
        )

    fmt = "{:>9}  {:>4}  {:>10}  {:>10}  {:>9}  {:>11}  {:>6}"
    header_line = fmt.format("metric", "conc", "baseline", "observed", "diff", "tolerance", "status")
    lines = [
        "Guardrails Overhead vs. Baseline (ms):",
        header_line,
        "-" * len(header_line),
    ]
    failed_count = 0
    for report in latency_reports:
        status = "PASS" if report.passed else "FAIL"
        if not report.passed:
            failed_count += 1
        lines.append(
            fmt.format(
                report.metric,
                report.concurrency,
                f"{report.baseline_ms:.0f}",
                f"{report.observed_ms:.0f}",
                f"{report.diff_ms:+.0f}",
                f"+{report.tolerance_ms:.0f}ms",
                status,
            )
        )
    if skipped_concurrencies:
        lines.append("")
        lines.append(f"Skipped (missing from results or baseline): {skipped_concurrencies}")
    if failed_count:
        lines.append("")
        lines.append(f"FAIL: {failed_count} of {len(latency_reports)} check(s) exceeded regression tolerance.")

    return "\n".join(lines), failed_count


def _format_single_variant(variant: str, latency_by_concurrency: dict[int, LatencyRow]) -> str:
    """Render one variant's table when the other variant didn't run."""
    fmt = "{:>4}  {:>9}  {:>9}  {:>9}  {:>9}"
    header_line = fmt.format("conc", "avg", "p50", "p90", "std")
    lines = [
        f"Only one variant present: {variant}",
        header_line,
        "-" * len(header_line),
    ]
    for concurrency in sorted(latency_by_concurrency):
        row = latency_by_concurrency[concurrency]
        lines.append(fmt.format(concurrency, f"{row.avg:.0f}", f"{row.p50:.0f}", f"{row.p90:.0f}", f"{row.std:.0f}"))
    lines.append("")
    lines.append("All values in milliseconds.")
    return "\n".join(lines)


def _parse_concurrency_from_label(label: str) -> int | None:
    """Extract N from a sweep label like ``concurrency16``; ``None`` otherwise."""
    if not label.startswith("concurrency"):
        return None
    try:
        return int(label.removeprefix("concurrency"))
    except ValueError:
        return None


def _read_latency_row(csv_path: Path, concurrency: int) -> LatencyRow | None:
    """Pull the ``Request Latency (ms)`` row from an AIPerf CSV's first block."""
    if not csv_path.is_file():
        log.debug("Missing CSV at %s; skipping", csv_path)
        return None

    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or header[0] != "Metric":
                log.warning("Unexpected header in %s: %s", csv_path, header)
                return None
            try:
                col = {name: header.index(name) for name in ("avg", "p50", "p90", "p99", "std")}
            except ValueError as exc:
                log.warning("Missing expected column in %s: %s", csv_path, exc)
                return None
            for row in reader:
                if not row:
                    break  # end of first block
                if row[0] == _LATENCY_METRIC:
                    return LatencyRow(
                        concurrency=concurrency,
                        avg=float(row[col["avg"]]),
                        p50=float(row[col["p50"]]),
                        p90=float(row[col["p90"]]),
                        p99=float(row[col["p99"]]),
                        std=float(row[col["std"]]),
                    )
    except (OSError, ValueError, IndexError) as exc:
        log.warning("Failed to parse %s: %s", csv_path, exc)
        return None

    log.warning("Did not find '%s' row in %s", _LATENCY_METRIC, csv_path)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nemo-guardrails-benchmark-analyze",
        description=__doc__,
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a run directory under `plugins/nemo-guardrails/benchmarks/artifacts/runs/<timestamp>/`.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any baseline check exceeds tolerance. CI sets this; local runs default off so you can iterate without the gate failing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        return 2

    print(analyze_run(run_dir))

    rows = _load_comparison_rows(run_dir)
    if not rows:
        print("Skipping baseline check: no comparable rows from this run.", file=sys.stderr)
        return 0 if not args.strict else 2

    report, failed_count = check_against_baseline(rows)
    print()
    print(report)
    return 1 if (args.strict and failed_count) else 0


if __name__ == "__main__":
    sys.exit(main())
