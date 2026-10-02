# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and display the results of an agent-eval run.

Companion to ``run_gym_eval.py``: that script *produces* a run bundle, this one *reads* it and shows
how to get at each kind of result — headline aggregates, ``pass@k``, per-task outcomes, and the
runner's own aggregations.

Per-task results are read through the SDK's own typed view, ``summary.task_outcomes(metric_name)``,
rather than by walking the nested ``summary.task_metric_values`` dict here — that is the accessor to
lift into your own code. Everything shown here also works on the in-memory ``AgentEvalResult``
returned by ``AgentEvaluator().run(...)`` — reading from a bundle just makes the example runnable
without a live run.

There is no bundle checked into the repo — ``run_gym_eval.py`` makes one. It writes to a fresh
temporary directory by default, so give it an explicit ``--output-dir`` and point this script at the
same path. From the repository root::

    uv run python -m packages.nemo_evaluator_sdk.examples.gym.run_gym_eval \\
        --output-dir /tmp/gym-eval
    uv run python -m packages.nemo_evaluator_sdk.examples.gym.inspect_results --bundle /tmp/gym-eval

Any agent-eval bundle works, not just a Gym one: pass ``--metric-type``/``--output-name`` for the
metric it was scored with. Only the ``runner.gym.`` section is Gym-specific, and it is skipped when a
run has no imported aggregations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.results import (
    AgentEvalSummary,
    PerTaskOutcomes,
    numeric_metric_values,
)
from nemo_evaluator_sdk.values.results import AggregateScore
from pydantic import ValidationError

#: Value at which a trial counts as a pass, matching the SDK's pass@k definition (full credit).
PASS_VALUE = 1.0

#: Namespace the Gym runner's own aggregations are imported under, so they never collide with ours.
RUNNER_PREFIX = "runner.gym."


class BundleFormatError(Exception):
    """A run bundle this script cannot read (wrong directory, or written by an older SDK)."""


# --------------------------------------------------------------------------------------------------
# Accessors — lift these into your own code.
# --------------------------------------------------------------------------------------------------


def aggregate(summary: AgentEvalSummary, name: str) -> AggregateScore:
    """Look up one aggregate by name, e.g. ``"gym_reward.reward.pass@2"``.

    Aggregates are a flat list, so this is a scan. Raises with the available names on a miss, which is
    the failure you actually want when a metric or output was renamed.
    """
    for score in summary.scores.scores:
        if score.name == name:
            return score
    available = ", ".join(sorted(score.name for score in summary.scores.scores))
    raise KeyError(f"no aggregate named {name!r}; available: {available}")


# --------------------------------------------------------------------------------------------------
# Bundle loading (see the run.json manifest for the full artifact list).
# --------------------------------------------------------------------------------------------------


def load_bundle(bundle: Path) -> AgentEvalSummary:
    """Load the persisted summary, including native and runner aggregates and per-task values.

    Every way a bundle can be unreadable raises :class:`BundleFormatError` and nothing else, so
    :func:`main` turns all of them into an exit code rather than a traceback. In particular it
    rejects a bundle written before ``task_metric_values`` existed rather than reading one: the
    field defaults to empty, so an older bundle would otherwise load cleanly and simply show no
    per-task section — the reader would conclude the run had no per-task outcomes rather than that
    this script cannot see them.
    """
    summary_path = bundle / "summary.json"
    if not summary_path.exists():
        raise BundleFormatError(f"{bundle} is not a run bundle (no summary.json). Run run_gym_eval.py first.")
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleFormatError(f"{summary_path} is not readable JSON: {exc}") from exc
    # Before the membership test below, which raises TypeError on a root that is not a container and
    # matches a substring on a bare JSON string.
    if not isinstance(payload, dict):
        raise BundleFormatError(f"{summary_path} is not a JSON object (found {type(payload).__name__}).")
    # Checked explicitly because `model_validate` would *not* catch this: the field defaults to an
    # empty dict, so an older bundle loads cleanly and simply shows no per-task section.
    if "task_metric_values" not in payload:
        raise BundleFormatError(
            f"{summary_path} predates summary.task_metric_values, which this script reads "
            "per-task outcomes from. Re-run the eval to produce a current bundle."
        )
    try:
        return AgentEvalSummary.model_validate(payload)
    except ValidationError as exc:
        raise BundleFormatError(f"{summary_path} is not a valid run summary: {exc}") from exc


# --------------------------------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------------------------------


def show_aggregates(summary: AgentEvalSummary) -> None:
    print("Aggregates ('runner.*' are the runner's own numbers, imported)")
    print(f"  {'name':<40} {'value':>8} {'count':>6} {'nan':>5}")
    for score in sorted(summary.scores.scores, key=lambda item: item.name):
        value = score.headline_value
        shown = "—" if value is None else f"{value:.3f}"
        # None means the producer didn't report a sample size; a real 0 means every sample was NaN.
        count = "—" if score.count is None else str(score.count)
        print(f"  {score.name:<40} {shown:>8} {count:>6} {score.nan_count:>5}")
    print(f"\n  {summary.task_count} tasks · {summary.trial_count} trials · {summary.score_count} scores")


def show_per_task(outcomes: list[PerTaskOutcomes]) -> None:
    """Per-task outcomes: which tasks were solved, and how consistently.

    Takes the SDK's typed view, so every row already knows its own task and metric and no dict has
    to be re-keyed here. Already sorted by task, hence no ``sorted()``.

    A trial passes on full credit (``>= PASS_VALUE``), matching how the SDK computes pass@k. A
    ``None`` is a trial that died: it counts toward ``n`` and never as a pass, so a task that passed
    once and crashed once reads as flaky rather than solved.
    """
    print("\nPer-task outcomes (trial values; a trial passes at full credit)")
    solved = flaky = failed = unmeasured = 0
    for per_task in outcomes:
        for outcome in per_task.outcomes:
            # Projected to floats only here, where the work is genuinely arithmetic: `>=` and `:g`
            # both raise on a judge's label, and numeric_metric_values drops those while keeping a
            # dead trial's None. Everything above reads the records as they were recorded.
            values = numeric_metric_values(outcome.trials)
            if not values:
                verdict, marker = "unmeasured", "?"
                unmeasured += 1
                shown = ""
            else:
                passes = sum(1 for value in values if value is not None and value >= PASS_VALUE)
                if passes == len(values):
                    verdict, marker = "solved", "+"
                    solved += 1
                elif passes:
                    verdict, marker = f"flaky ({passes}/{len(values)})", "~"
                    flaky += 1
                else:
                    verdict, marker = "failed", "-"
                    failed += 1
                shown = ", ".join("died" if value is None else f"{value:g}" for value in values)
            print(f"  {marker} {per_task.task_id[:16]}…  [{shown}]  {verdict}")
    print(f"\n  {solved} solved · {flaky} flaky · {failed} failed · {unmeasured} unmeasured")


def show_runner_aggregations(summary: AgentEvalSummary) -> None:
    """The runner's own numbers, plus a cross-check against the SDK's native aggregates.

    Imported figures sit in the same ``summary.scores`` list as everything else, distinguished only by
    the ``runner.`` prefix — so they are read exactly like the natively-computed ones. Units and names
    stay the runner's own: Gym reports accuracy on a 0-100 scale where the SDK uses 0-1.
    """
    imported = [score for score in summary.scores.scores if score.name.startswith(RUNNER_PREFIX)]
    if not imported:
        print("\nNo runner-provided aggregations (this runner doesn't supply any).")
        return

    print("\nRunner-provided aggregations (imported into summary.scores)")
    for score in sorted(imported, key=lambda item: item.name):
        value = score.headline_value
        shown = "—" if value is None else f"{value:g}"
        print(f"      {score.name[len(RUNNER_PREFIX) :]:<34} {shown}")

    # Cross-check: the SDK computes pass@k natively from the trials; Gym computes its own. They should
    # agree once you normalise the scale.
    try:
        native = aggregate(summary, "gym_reward.reward.pass@1").mean
        reported = aggregate(summary, f"{RUNNER_PREFIX}pass@1/accuracy").headline_value
    except KeyError:
        return
    if native is not None and reported is not None:
        agreement = "agree" if abs(native - reported / 100) < 1e-9 else "DIFFER"
        print(f"\n  cross-check pass@1: native={native:.3f} · runner={reported / 100:.3f} (0-100 scale) -> {agreement}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=Path("/tmp/gym-eval"), help="Run bundle directory to read.")
    parser.add_argument("--metric-type", default="gym_reward", help="Metric type to break down per task.")
    parser.add_argument("--output-name", default="reward", help="Metric output to break down per task.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # The CLI boundary is where a bad bundle becomes an exit code; the accessors above just raise.
    try:
        summary = load_bundle(args.bundle)
    except BundleFormatError as exc:
        raise SystemExit(str(exc)) from exc

    show_aggregates(summary)
    # Empty when no task was measured by this metric at all -- typically a wrong --metric-type.
    outcomes = summary.task_outcomes(f"{args.metric_type}.{args.output_name}")
    if outcomes:
        show_per_task(outcomes)
    show_runner_aggregations(summary)

    print(f"\nFull report: {args.bundle / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
