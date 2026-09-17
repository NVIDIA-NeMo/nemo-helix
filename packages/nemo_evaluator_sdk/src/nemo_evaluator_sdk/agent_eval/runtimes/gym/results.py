# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What Gym produced, read back as trials and scores.

Covers the rollout and failure records, the aggregate metrics Gym computes for itself, and the
judgement calls in between: which records represent an agent that actually ran, and which reward
is a measurement rather than an artefact of one that did not.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from google.protobuf.json_format import MessageToDict
from nemo_evaluator_sdk.agent_eval.runtimes.gym.config import DEFAULT_REWARD_KEY, model_call_capture_dir
from nemo_evaluator_sdk.agent_eval.runtimes.gym.records import (
    ENV_LOG_NAME,
    NG_ATTEMPT_INDEX,
    NG_ROLLOUT_INDEX,
    NG_TASK_INDEX,
    read_jsonl,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput, TrialMeasurements
from nemo_evaluator_sdk.agent_eval.usage_keys import (
    CACHE_CREATION_INPUT_TOKENS_KEY,
    CACHE_READ_INPUT_TOKENS_KEY,
    CACHED_TOKENS_KEY,
    COMPLETION_TOKEN_KEYS,
    PROMPT_TOKEN_KEYS,
    SEPARATE_CACHE_KEYS,
    USAGE_COUNT_KEYS,
    USAGE_DETAILS_KEYS,
    _first_nonnegative_int,
    _first_usage_details,
)
from nemo_evaluator_sdk.ng_trajectory_otlp import rollout_to_resource_spans
from nemo_evaluator_sdk.values.evidence import (
    EVIDENCE_FORMAT_OTLP,
    EVIDENCE_TRACE,
    CandidateEvidence,
    EvidenceDescriptor,
)
from nemo_evaluator_sdk.values.results import AggregateRangeScore, AggregateScalarScore, AggregateScore
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans

logger = logging.getLogger(__name__)

#: Evidence key for Gym's own model-call capture, kept beside the OTLP view projected from it.
#: Named for NeMo Gym's documented `ng_trajectory` schema; 0.5.0 writes it as one
#: `<rollout_id>.capture.jsonl` per rollout rather than as a field on the rollout record.
NG_TRAJECTORY_EVIDENCE = "ng_trajectory"


_TOKEN_USAGE_KEYS = ("total_tokens", "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens")
#: The subset that proves a call happened. A model call always consumes input tokens, so zero on any
#: of these is evidence the model was never reached; zero output tokens only means an empty answer.
_INPUT_TOKEN_KEYS = ("total_tokens", "input_tokens", "prompt_tokens")


def _agent_never_ran(record: Mapping[str, Any]) -> bool:
    """True when a result record shows the agent produced nothing *and* never called the model.

    A verifier-scored runner reports whatever score the environment computed, so an agent that
    silently failed to start is indistinguishable from one that tried and scored 0.0 — the record
    looks normal, no failure is reported alongside it, and the run succeeds. Observed on
    ``legal_agent_bench``, whose agent died locating its task tree and returned an empty response:
    two "completed" trials scoring 0.0, no model call, no failure recorded.

    Deliberately conservative — **both** conditions must hold:

    * no output, and
    * a stated zero on the input side of token usage.

    An empty answer alone is a legitimate result (a model can return nothing and earn 0.0), so it is
    not sufficient. Zero *input* tokens is what says the model was never called: a call always sends
    a prompt, whereas zero output tokens only describes an empty answer. A record carrying no
    ``response`` key at all is left alone — some runners populate it differently, and this must not
    reclassify trials it does not understand.
    """
    response = record.get("response")
    if not isinstance(response, Mapping):
        return False
    if "output" not in response:
        # Nothing to judge: some runners shape `response` differently, and absence is not emptiness.
        return False
    # Any non-empty value counts as output, whatever its shape. Checking `len(...) > 0` on a list
    # alone would treat a plain string answer — or a single mapping — as "produced nothing" and fail
    # a trial that ran perfectly well.
    if response["output"]:
        return False
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        # No usage block to corroborate with; an empty output on its own is not enough to judge.
        return False
    # Zero usage has to be *stated*, not inferred from a key being absent. Two vocabularies are in
    # play — the Responses API's `input_tokens`/`output_tokens` and Chat Completions'
    # `prompt_tokens`/`completion_tokens` — and Gym's model servers speak both depending on the
    # adapter. Reading only one set would make a real call reported in the other read as zero.
    reported = [usage[key] for key in _TOKEN_USAGE_KEYS if isinstance(usage.get(key), (int, float))]
    if any(value > 0 for value in reported):
        return False
    # The discriminating evidence is *input*-side, and only input-side. Calling a model always sends
    # a prompt, so zero input tokens cannot describe a call that happened; zero **output** tokens
    # merely describes an empty answer, which is a legitimate result. Requiring an input-side count
    # rules out a whole class of false positives — a provider reporting only `output_tokens: 0`, or
    # only `completion_tokens: 0` — rather than excluding those shapes one at a time.
    input_side = [usage[key] for key in _INPUT_TOKEN_KEYS if isinstance(usage.get(key), (int, float))]
    return bool(input_side) and all(value == 0 for value in input_side)


def _usage_measurements(record: Mapping[str, Any]) -> TrialMeasurements:
    """Normalize Gym response usage for both rollout and failure records."""
    response = record.get("response")
    usage = response.get("usage") if isinstance(response, Mapping) else None
    if not isinstance(usage, Mapping):
        return TrialMeasurements()
    _warn_invalid_usage_counts(record, usage)

    separate_cache = any(usage.get(key) is not None for key in SEPARATE_CACHE_KEYS)
    details = _first_usage_details(usage)
    inclusive_cache = details is not None and details.get(CACHED_TOKENS_KEY) is not None
    completion = _first_nonnegative_int(usage, *COMPLETION_TOKEN_KEYS)
    if separate_cache and inclusive_cache:
        logger.warning(
            "Gym task=%r rollout=%r usage contains both inclusive cache details and separate cache fields; "
            "omitting ambiguous prompt/cache measurements",
            record.get(NG_TASK_INDEX),
            record.get(NG_ROLLOUT_INDEX),
        )
        return TrialMeasurements(completion_tokens=completion)

    prompt = _first_nonnegative_int(usage, *PROMPT_TOKEN_KEYS)
    cache_read = _first_nonnegative_int(usage, CACHE_READ_INPUT_TOKENS_KEY) if separate_cache else None
    cache_creation = _first_nonnegative_int(usage, CACHE_CREATION_INPUT_TOKENS_KEY) if separate_cache else None
    invalid_separate_cache = any(
        key in usage and usage[key] is not None and _first_nonnegative_int(usage, key) is None
        for key in SEPARATE_CACHE_KEYS
    )
    if separate_cache and prompt is not None:
        prompt = None if invalid_separate_cache else prompt + (cache_creation or 0) + (cache_read or 0)
    elif details is not None and CACHED_TOKENS_KEY in details:
        cache_read = _first_nonnegative_int(details, CACHED_TOKENS_KEY)

    return TrialMeasurements(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cache_creation_tokens=cache_creation,
        cache_read_tokens=cache_read,
    )


def _warn_invalid_usage_counts(record: Mapping[str, Any], usage: Mapping[str, Any]) -> None:
    trial_label = f"task={record.get(NG_TASK_INDEX)!r} rollout={record.get(NG_ROLLOUT_INDEX)!r}"
    for key in USAGE_COUNT_KEYS:
        if key in usage and usage[key] is not None and _first_nonnegative_int(usage, key) is None:
            logger.warning(
                "Gym %s has invalid usage %s=%r; expected a non-negative integer and omitted it",
                trial_label,
                key,
                usage[key],
            )
    for details_key in USAGE_DETAILS_KEYS:
        details = usage.get(details_key)
        if (
            isinstance(details, Mapping)
            and CACHED_TOKENS_KEY in details
            and details[CACHED_TOKENS_KEY] is not None
            and _first_nonnegative_int(details, CACHED_TOKENS_KEY) is None
        ):
            logger.warning(
                "Gym %s has invalid usage %s.cached_tokens=%r; expected a non-negative integer and omitted it",
                trial_label,
                details_key,
                details[CACHED_TOKENS_KEY],
            )


def require_full_coverage(tasks: Sequence[AgentEvalTask], *, covered_task_ids: set[str], rollouts_path: Path) -> None:
    """Fail the run when Gym produced no trial at all for some requested task.

    :class:`AgentEvaluator` already refuses to score a task with no trial
    (``evaluator._score_trials``), and deliberately so: an incomplete run must not read as a
    successful one with quieter counts. We keep that contract rather than papering over it with
    synthesized FAILED trials — an unscored trial is excluded from the mean, so fabricating them
    would report a plausible-looking average over an unannounced subset of the dataset. A rollout
    that failed in a *diagnosable* way already becomes a FAILED trial via the failures sidecar;
    reaching here means the attempt is unaccounted for entirely.

    What this adds over letting the evaluator raise is diagnosis: by the time the run fails the full
    collection cost is already paid, so the error names the evidence instead of just the task ids.
    """
    missing = sorted({task.id for task in tasks} - covered_task_ids)
    if not missing:
        return
    raise RuntimeError(
        f"Gym produced no rollout for {len(missing)} of {len(tasks)} requested task(s), so the run is "
        f"incomplete and will not be scored (a partial run must not be reported as a whole one).\n"
        f"  rollouts:  {rollouts_path}\n"
        f"  failures:  {_failures_path_for(rollouts_path)}\n"
        f"  gym logs:  {rollouts_path.parent} (gym_env.log, gym_eval.stdout.log, gym_eval.stderr.log)\n"
        f"  unrepresented task id(s): {missing}"
    )


def _rollout_evidence(
    record: Mapping[str, Any],
    *,
    rollouts_path: Path,
    task_id: str,
    trial_id: str,
    capture_dir: Path | None = None,
) -> CandidateEvidence:
    """The rollouts file every trial shares, plus this rollout's own trace.

    The trace is derived from the record rather than captured, so it is carried inline rather than
    written beside the rollouts file. It is registered under the format-qualified key as well as the
    plain one, so a metric asking for ``format="otlp"`` and one asking for whatever is primary reach
    the same handle.
    """
    descriptors: dict[str, EvidenceDescriptor] = {
        "rollouts": EvidenceDescriptor(kind="filesystem", format="file", ref=str(rollouts_path))
    }
    capture_path = _capture_path(record, capture_dir)
    if capture_dir is not None and capture_path is None:
        # Capture is switched on by the runner, so an absent one is Gym not doing what it was
        # asked. The trial still scores; its trace just loses the timing capture exists to provide.
        logger.warning(
            "No model-call capture for trial %s under %s; its trace will carry no per-call timing.",
            trial_id,
            capture_dir,
        )
    model_calls = _read_model_calls(capture_path, trial_id=trial_id)
    if capture_path is not None:
        # Kept as its own evidence, by reference: the OTLP view below is a projection of it, and a
        # projection is not a reason to lose what it was projected from.
        descriptors[NG_TRAJECTORY_EVIDENCE] = EvidenceDescriptor(
            kind="filesystem", format="file", ref=str(capture_path)
        )
    resource_spans = _rollout_resource_spans(record, task_id=task_id, trial_id=trial_id, model_calls=model_calls)
    if resource_spans is not None:
        # Rendered to JSON here and only here, because an ``EvidenceDescriptor`` holds JSON.
        trace = EvidenceDescriptor(
            kind=EVIDENCE_TRACE,
            format=EVIDENCE_FORMAT_OTLP,
            data=[MessageToDict(entry) for entry in resource_spans],
        )
        descriptors[EVIDENCE_TRACE] = trace
        descriptors[f"{EVIDENCE_TRACE}:{EVIDENCE_FORMAT_OTLP}"] = trace
    return CandidateEvidence(descriptors=descriptors)


def _rollout_resource_spans(
    record: Mapping[str, Any], *, task_id: str, trial_id: str, model_calls: Sequence[Mapping[str, Any]] = ()
) -> list[ResourceSpans] | None:
    """Project one rollout onto OTLP resource spans, or ``None`` when it will not convert.

    Seeded on the trial id because span identity has to be unique per attempt: Gym does not always
    record ``_ng_rollout_index``, and ``_rollout_trial_id`` is what already resolves that.
    """
    try:
        return rollout_to_resource_spans(record, rollout_id=trial_id, task_id=task_id, model_calls=model_calls)
    except (TypeError, ValueError) as error:
        # A trace is a second view of a result Gym already reported; losing it must not lose the
        # trial, whose reward and output stand on their own.
        logger.warning("Could not build an OTLP trace for trial %s: %s", trial_id, error)
        return None


def _capture_path(record: Mapping[str, Any], capture_dir: Path | None) -> Path | None:
    """The model-call capture Gym wrote for this rollout, or ``None`` when there is not one.

    Gym keys a capture by ``"{task}-{rollout}"``, suffixed ``-a{attempt}`` past the first attempt
    (``nemo_gym.rollout_correlation``). Both indices are on the record, so the join is exact rather
    than a search.
    """
    if capture_dir is None:
        return None
    name = capture_filename(record)
    if name is None:
        return None
    path = capture_dir / name
    return path if path.exists() else None


def capture_filename(record: Mapping[str, Any]) -> str | None:
    """Gym's capture filename for this rollout, or ``None`` when its indices do not name one.

    The single definition of that name: the sandboxed runner writes captures under it and
    :func:`_capture_path` reads them back, so the two cannot drift apart.
    """
    task, rollout = record.get(NG_TASK_INDEX), record.get(NG_ROLLOUT_INDEX)
    if not isinstance(task, int) or not isinstance(rollout, int):
        return None
    rollout_id = f"{task}-{rollout}"
    attempt = record.get(NG_ATTEMPT_INDEX)
    if isinstance(attempt, int) and attempt > 0:
        rollout_id = f"{rollout_id}-a{attempt}"
    return f"{rollout_id}.capture.jsonl"


def _read_model_calls(capture_path: Path | None, *, trial_id: str) -> list[dict[str, Any]]:
    """This rollout's captured model exchanges, or none when unreadable.

    Tolerant on purpose: the capture is a bonus view over a result Gym already reported, so a
    truncated file costs the trial its per-call timing, not the trial.
    """
    if capture_path is None:
        return []
    try:
        return read_jsonl(capture_path, tolerant=True)
    except OSError as error:
        logger.warning("Could not read the model-call capture for trial %s: %s", trial_id, error)
        return []


def _failures_path_for(rollouts_path: Path) -> Path:
    """Sidecar Gym writes failed rollouts to (mirrors Gym's own ``_failures_path_for``)."""
    return rollouts_path.with_name(rollouts_path.stem + "_failures.jsonl")


def _aggregate_metrics_path_for(rollouts_path: Path) -> Path:
    """Sidecar Gym writes run-level aggregate metrics to (``<stem>_aggregate_metrics.json``)."""
    return rollouts_path.with_name(rollouts_path.stem + "_aggregate_metrics.json")


def read_run_aggregations(rollouts_path: Path) -> dict[str, Any] | None:
    """Parse Gym's ``rollouts_aggregate_metrics.json``, or ``None`` when absent/unparseable.

    Gym's file is a list with one entry per agent (``agent_ref`` / ``agent_metrics`` / ``key_metrics`` /
    ``group_level_metrics``), so it is returned keyed by agent name. Carried through as-is — Gym's schema
    isn't contractual, so the SDK does not type it.
    """
    path = _aggregate_metrics_path_for(rollouts_path)
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        # UnicodeDecodeError is a ValueError, not an OSError: a sidecar truncated mid-codepoint would
        # otherwise raise straight out of run_tasks and discard a collection that already succeeded.
        logger.warning("Could not parse Gym aggregate metrics at %s; skipping run aggregations.", path)
        return None
    if not isinstance(parsed, list):
        logger.warning("Unexpected Gym aggregate-metrics shape at %s (%s); skipping.", path, type(parsed).__name__)
        return None
    # Gym's schema is not contractual, and this runs after a collection that already succeeded — an
    # entry in an unexpected shape must not raise out of run_tasks and discard every trial with it.
    aggregations: dict[str, Any] = {}
    for entry in parsed:
        agent_ref = entry.get("agent_ref") if isinstance(entry, Mapping) else None
        name = agent_ref.get("name") if isinstance(agent_ref, Mapping) else None
        if not isinstance(name, str):
            logger.warning(
                "Skipping a Gym aggregate-metrics entry in %s with no usable agent_ref.name (got %r).", path, agent_ref
            )
            continue
        aggregations[name] = {key: value for key, value in entry.items() if key != "agent_ref"}
    return aggregations or None


#: Gym flattens a distribution into ``<stat>/<metric>`` keys. Its RewardProfiler emits this exact set
#: together for every numeric column (``describe_dataframe``), minus ``histogram``, which
#: ``prepare_for_serialization`` strips before the file is written. All five must be present before we
#: treat a group of keys as one distribution: a resources-server is free to define a metric literally
#: named ``mean`` (36 of Gym's ~97 servers override ``compute_metrics``), and re-assembling on a partial
#: match would rename someone's standalone metric into a statistic of a distribution that never existed.
_GYM_STAT_FAMILY = ("mean", "max", "min", "median", "std")

#: Metric Gym reports that the SDK already computes natively from the same rollouts (``gym_reward.reward``).
_GYM_REDUNDANT_METRICS = frozenset({"reward"})


def aggregate_scores_from_gym(aggregations: Mapping[str, Any] | None) -> list[AggregateScore]:
    """Map Gym's run-level ``agent_metrics`` onto typed aggregate scores named ``runner.gym.<metric>``.

    Reads ``agent_metrics``, not ``key_metrics``: ``key_metrics`` is a *subset* of it chosen by the
    resources-server, and the default selection (``get_key_metrics``) keeps only the ``mean/*`` entries —
    so the max/min/median/std that make a distribution never appear there, and every metric would arrive
    as a lone ``mean/<name>`` scalar.

    Keys forming a full stat family are re-assembled into one :class:`AggregateRangeScore`; every other
    numeric key becomes an :class:`AggregateScalarScore`. Non-numeric values are skipped — they are
    labels or notes, not measurements.

    Names are namespaced by runner (not by agent): each run is instrumented with a single agent, so the
    agent adds no disambiguation, and a reader needs to know which *backend* produced a number.
    """
    if not aggregations:
        return []
    multi_agent = len(aggregations) > 1
    scores: list[AggregateScore] = []
    for agent_name, payload in sorted(aggregations.items()):
        agent_metrics = payload.get("agent_metrics") if isinstance(payload, Mapping) else None
        if not isinstance(agent_metrics, Mapping):
            continue
        # One agent per run is the norm, so `runner.gym.<metric>` reads cleanly; qualify by agent only
        # when a run really did produce several, where the unqualified names would collide.
        prefix = f"runner.gym.{agent_name}." if multi_agent else "runner.gym."
        scores.extend(_scores_from_agent_metrics(agent_metrics, prefix=prefix))
    return scores


def _scores_from_agent_metrics(agent_metrics: Mapping[str, Any], *, prefix: str) -> list[AggregateScore]:
    families: dict[str, dict[str, float]] = {}
    scalars: dict[str, float] = {}
    for key, value in agent_metrics.items():
        number = _as_float(value)
        if number is None:
            continue
        stat, _, metric = key.partition("/")
        if metric and stat in _GYM_STAT_FAMILY:
            families.setdefault(metric, {})[stat] = number
        else:
            scalars[key] = number

    scores: list[AggregateScore] = []
    for metric, stats in sorted(families.items()):
        # Redundancy is a property of the metric, not of how complete its family is. Checking after the
        # fallback below would let a partial `reward` family survive as `mean/reward` scalars, since the
        # scalar filter matches the bare name -- reintroducing the duplicate of `gym_reward.reward` that
        # skipping reward exists to prevent.
        if metric in _GYM_REDUNDANT_METRICS:
            continue
        if set(stats) < set(_GYM_STAT_FAMILY):
            # Not a distribution we can vouch for; keep each key as the standalone number it may well be.
            scalars.update({f"{stat}/{metric}": value for stat, value in stats.items()})
            continue
        scores.append(
            AggregateRangeScore(
                name=f"{prefix}{metric}",
                # Gym reports the statistics but not the sample size behind them, and inventing one
                # would misreport coverage. `None` says "unknown"; 0 would assert nothing was evaluated.
                count=None,
                nan_count=0,
                mean=stats["mean"],
                min=stats["min"],
                max=stats["max"],
                median=stats["median"],
                # Gym computes this with pandas (ddof=1), so it is the sample standard deviation.
                sample_std_dev=stats["std"],
                sample_variance=stats["std"] ** 2,
            )
        )
    scores.extend(
        AggregateScalarScore(name=f"{prefix}{key}", count=None, nan_count=0, value=value)
        for key, value in sorted(scalars.items())
        if key not in _GYM_REDUNDANT_METRICS
    )
    return scores


def _as_float(value: Any) -> float | None:
    """``float`` for a numeric Gym metric value; None for anything else (bools included: not measurements)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def ensure_fresh_output(rollouts_path: Path) -> None:
    """Enforce one Gym run per output dir (the AgentEvaluator convention).

    Gym appends to the failures sidecar (``open("ab")``) and doesn't clear it between runs, so reusing
    a populated directory would silently mix this run's failures with a prior run's. Rather than clear
    (which would clobber an earlier run's results — infra failures are useful signal), refuse to run
    into a directory that already holds Gym rollout output: rollouts, failures, or model-call
    captures.
    """
    preexisting = [path for path in (rollouts_path, _failures_path_for(rollouts_path)) if path.is_file()]
    # Model-call captures count as output too, and the sandboxed runner writes them *before*
    # `rollouts.jsonl`, so a run that died in between leaves captures with no file above to catch
    # it. Their names -- `{task}-{rollout}` -- repeat across runs of one dataset, so the next run
    # would attach a previous run's timing to its own trials.
    captures = model_call_capture_dir(rollouts_path.parent)
    if captures.is_dir() and any(captures.iterdir()):
        preexisting.append(captures)
    if preexisting:
        names = ", ".join(path.name for path in preexisting)
        raise FileExistsError(
            f"{rollouts_path.parent} already holds Gym rollout output ({names}); give each run a fresh "
            "work_dir (AgentEvalRunConfig.work_dir). Gym appends to the failures sidecar, so reusing a "
            "directory would mix runs and could obscure a prior run's results."
        )


def _coerce_reward(value: Any) -> float | None:
    """Best-effort ``float`` for a reward; returns None (never raises) on a missing/malformed value."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Gym rollout carried a non-numeric reward %r; recording the trial as unscored.", value)
        return None


def _resolve_task_id(
    record: Mapping[str, Any], index_to_task_id: Mapping[int, str], *, strict: bool = True
) -> str | None:
    """Map a rollout record to a task id via the ``_ng_task_index`` we stamped in the input dataset.

    Returns None when the record carries no usable index (kill-shaped failures can lack one).

    An index we never stamped is handled by ``strict``. In the **successes** file (``strict=True``)
    it is fatal: every row Gym read came from :func:`materialize_dataset`, so an unknown index means
    Gym reindexed the dataset rather than honoring our stamp, and every reward attribution is
    therefore suspect. In the **failures sidecar** (``strict=False``) it is merely unattributable and
    is counted instead — that file is written during abnormal termination and read tolerantly by
    design, so one odd record must not discard the successes already parsed. That relaxation cannot
    inflate a result: failure records carry no reward, and a task left with no trial still trips
    :func:`require_full_coverage`.
    """
    index = record.get(NG_TASK_INDEX)
    if not isinstance(index, int) or isinstance(index, bool):
        return None
    task_id = index_to_task_id.get(index)
    if task_id is None:
        if not strict:
            logger.warning(
                "Gym failure record carried %s=%s, which was not stamped on the materialized dataset; "
                "counting it as unattributable rather than failing the run.",
                NG_TASK_INDEX,
                index,
            )
            return None
        raise ValueError(
            f"Gym emitted {NG_TASK_INDEX}={index}, which is not one of the {len(index_to_task_id)} index value(s) "
            f"stamped on the materialized dataset. Gym appears to have reassigned {NG_TASK_INDEX} instead "
            "of honoring the supplied one, so rollout-to-task attribution is unsafe."
        )
    return task_id


def _rollout_trial_id(
    task_id: str, raw_rollout_index: Any, synth_seq: dict[str, int], *, missing_label: str, attempt: Any = None
) -> str:
    """Per-attempt trial id: use the record's ``_ng_rollout_index`` when it's a real int, else a
    per-task-unique ``{missing_label}{n}`` suffix so records lacking an index don't collide on one id.

    A retry reuses its task and rollout indices, so the attempt index is part of the identity too:
    without it a failed attempt and the retry that replaced it land on one id, and since span ids are
    derived from this one, a consumer keyed on span identity would keep only the last. Suffixed
    ``-a{attempt}`` past the first attempt, matching how Gym keys the capture in ``_capture_path`` so
    the two agree on what "this attempt" means.
    """
    suffix = f"-a{attempt}" if isinstance(attempt, int) and attempt > 0 else ""
    if isinstance(raw_rollout_index, int) and not isinstance(raw_rollout_index, bool):
        return f"{task_id}:{raw_rollout_index}{suffix}"
    seq = synth_seq[task_id] = synth_seq.get(task_id, 0) + 1
    return f"{task_id}:{missing_label}{seq}{suffix}"


def trials_from_rollouts(
    rollouts_path: Path,
    tasks: Sequence[AgentEvalTask],
    index_to_task_id: Mapping[int, str],
    *,
    reward_key: str = DEFAULT_REWARD_KEY,
    capture_dir: Path | None = None,
) -> list[AgentEvalTrial]:
    """Fan Gym's rollout records out into one :class:`AgentEvalTrial` per attempt.

    Reads *both* Gym output files: successes from ``rollouts.jsonl`` (COMPLETED, carrying the reward)
    and failures from the ``*_failures.jsonl`` sidecar (FAILED — so a failed attempt is counted and
    diagnosed rather than silently dropped). Every record is attributed to its task by
    ``_ng_task_index`` → ``index_to_task_id``, the map we stamped in :func:`materialize_dataset`.
    """
    if not rollouts_path.exists():
        # `ensure_fresh_output` removed any stale file before the run, and `_collect_rollouts`
        # already raised on a non-zero exit — so reaching here means Gym reported success and wrote
        # nothing. Opening the path regardless would surface that as a bare FileNotFoundError
        # naming a file the reader has no reason to know about, which is the same illegibility the
        # rest of this change set exists to remove.
        raise RuntimeError(
            f"`gym eval run` reported success but wrote no results to {rollouts_path}, so there is "
            "nothing to score. This usually means collection stopped before any attempt completed.\n"
            f"See {rollouts_path.parent / 'gym_eval.stdout.log'} for what collection did, and "
            f"{rollouts_path.parent / ENV_LOG_NAME} for a traceback from the environment's own "
            "servers."
        )
    known_task_ids = {task.id for task in tasks}
    # Defence in depth: the index map and the task list must describe the same run. Gym is an external
    # project on its own release cadence, and people make agent-optimization decisions from these
    # numbers — so any disagreement about *which task a result belongs to* is a hard error, never a
    # best-effort join. Cheap to check once; impossible to notice downstream if it silently drifts.
    mapped_task_ids = set(index_to_task_id.values())
    if mapped_task_ids != known_task_ids:
        raise ValueError(
            f"the {NG_TASK_INDEX} map does not match the task list it will be joined against "
            f"({len(mapped_task_ids - known_task_ids)} mapped task(s) absent from the list, "
            f"{len(known_task_ids - mapped_task_ids)} listed task(s) absent from the map); "
            "attributing rollouts across a mismatch would report results against the wrong tasks"
        )
    trials: list[AgentEvalTrial] = []
    # Shared per-task counter for records lacking a usable _ng_rollout_index (success + failure); a
    # missing/null index must not collapse multiple attempts for one task onto the same trial id.
    synth_seq: dict[str, int] = {}

    unattributed_successes = 0
    empty_output_count = 0
    #: Tasks with at least one trial where the agent never ran. Tracked separately from trial
    #: status, since the failures sidecar also produces FAILED trials for unrelated reasons.
    empty_output_task_ids: set[str] = set()
    for record in read_jsonl(rollouts_path):
        task_id = _resolve_task_id(record, index_to_task_id)
        if task_id is None:
            # No usable index (an unknown one raises). Skipping is right, but silence is not: with
            # num_repeats > 1 the task keeps its other trials, so require_full_coverage won't fire and
            # the task would simply be scored on fewer attempts than were actually run.
            unattributed_successes += 1
            continue
        raw_rollout_index = record.get(NG_ROLLOUT_INDEX)
        reward = _coerce_reward(record.get(reward_key))
        # An agent that never ran did not earn its reward — reporting 0.0 as a completed trial would
        # let a broken environment read as a poorly-performing one.
        never_ran = _agent_never_ran(record)
        if never_ran:
            empty_output_count += 1
            empty_output_task_ids.add(task_id)
            status = AgentEvalTrialStatus.FAILED
        elif reward is not None:
            status = AgentEvalTrialStatus.COMPLETED
        else:
            status = AgentEvalTrialStatus.PARTIAL
        metadata: dict[str, Any] = {
            "reward": None if never_ran else reward,
            NG_TASK_INDEX: record.get(NG_TASK_INDEX),
            NG_ROLLOUT_INDEX: raw_rollout_index,
        }
        if never_ran:
            # Kept alongside the reward Gym reported, so the discarded value is still auditable.
            metadata["gym_failure"] = (
                "the agent produced no output and consumed no tokens, so the model was never called; "
                f"the reported score of {reward!r} is not a measurement of this agent"
            )
        trial_id = _rollout_trial_id(
            task_id, raw_rollout_index, synth_seq, missing_label="noidx", attempt=record.get(NG_ATTEMPT_INDEX)
        )
        trials.append(
            AgentEvalTrial(
                id=trial_id,
                task_id=task_id,
                status=status,
                output=AgentOutput(response=record.get("response"), metadata={"agent_ref": record.get("agent_ref")}),
                evidence=_rollout_evidence(
                    record,
                    rollouts_path=rollouts_path,
                    task_id=task_id,
                    trial_id=trial_id,
                    capture_dir=capture_dir,
                ),
                measurements=_usage_measurements(record),
                metadata=metadata,
            )
        )

    failures_path = _failures_path_for(rollouts_path)
    unattributed_failures = 0
    if failures_path.exists():
        failure_evidence = CandidateEvidence(
            descriptors={
                "rollout_failures": EvidenceDescriptor(kind="filesystem", format="file", ref=str(failures_path))
            }
        )
        # tolerant: a truncated line in the abnormal-termination sidecar must not sink the good trials.
        for record in read_jsonl(failures_path, tolerant=True):
            # strict=False to match this loop's tolerant read: an odd record is counted, not fatal.
            task_id = _resolve_task_id(record, index_to_task_id, strict=False)
            if task_id is None:
                # kill-shaped failures (SIGTERM/OOM/actor-death) can lack a usable index — count, don't drop.
                unattributed_failures += 1
                continue
            raw_rollout_index = record.get(NG_ROLLOUT_INDEX)
            trials.append(
                AgentEvalTrial(
                    id=_rollout_trial_id(
                        task_id,
                        raw_rollout_index,
                        synth_seq,
                        missing_label="fail",
                        attempt=record.get(NG_ATTEMPT_INDEX),
                    ),
                    task_id=task_id,
                    status=AgentEvalTrialStatus.FAILED,
                    output=AgentOutput(
                        response=record.get("response"), metadata={"agent_ref": record.get("agent_ref")}
                    ),
                    evidence=failure_evidence,
                    measurements=_usage_measurements(record),
                    metadata={
                        "reward": None,
                        # best-effort: Gym's failure-record schema isn't contractual, so probe common keys.
                        "gym_failure": record.get("error") or record.get("exception") or record.get("failure_reason"),
                        NG_TASK_INDEX: record.get(NG_TASK_INDEX),
                        NG_ROLLOUT_INDEX: raw_rollout_index,
                    },
                )
            )
    if empty_output_count:
        # Every trial empty is a configuration failure, not a result: nothing was measured, so there
        # is no score to report and letting the run "succeed" would publish zeros as findings. A
        # partial count is left as failed trials — a flaky environment is still worth the attempts
        # that did run.
        #
        # Measured against *every* trial, including those the failures sidecar contributed. A run
        # that also recorded real failures is not the uniform "nothing ran" picture this raise is
        # for: those trials carry their own diagnosis (a verifier timeout, say, from an attempt that
        # did reach the model), and raising would discard it for a less specific message.
        #
        # The unattributed counts need the same treatment and are easy to miss, because those
        # records never became trials and so are invisible to `len(trials)`. Each is still a result
        # the environment produced — an unattributed *success* may have run perfectly well, and an
        # unattributed *failure* is a diagnosis of its own (an OOM-killed agent, say). Claiming
        # nothing was measured while either exists would be false.
        if empty_output_count == len(trials) and not unattributed_successes and not unattributed_failures:
            affected_tasks = {trial.task_id for trial in trials}
            raise RuntimeError(
                f"No agent ran: all {empty_output_count} trial(s) across all {len(affected_tasks)} "
                "task(s) produced no output and consumed no tokens, so the model was never called "
                "and nothing was measured. The scores reported for them describe an agent that did "
                "not run.\n"
                "This is normally the environment failing to start its agent rather than a bad "
                f"model: check the per-server startup output in {rollouts_path.parent / ENV_LOG_NAME} "
                "for a traceback from the environment's own agent server.\n"
                f"Results as collected: {rollouts_path}"
            )
        logger.warning(
            "%d of %d trial(s), across %d of %d task(s), produced no agent output and consumed no "
            "tokens; those are marked FAILED rather than scored, since the model was never called "
            "for them.",
            empty_output_count,
            len(trials),
            len(empty_output_task_ids),
            len({trial.task_id for trial in trials}),
        )
    if unattributed_successes:
        logger.warning(
            "%d Gym rollout record(s) in %s carried no usable %s and could not be attributed to a task; "
            "affected tasks are scored on fewer attempts than were collected.",
            unattributed_successes,
            rollouts_path,
            NG_TASK_INDEX,
        )
    if unattributed_failures:
        logger.warning(
            "%d Gym failure record(s) lacked a usable %s and could not be attributed to a task.",
            unattributed_failures,
            NG_TASK_INDEX,
        )

    # Reported here, enforced by require_full_coverage in run_tasks (which knows the run's log paths).
    missing = known_task_ids - {trial.task_id for trial in trials}
    if missing:
        logger.warning("No Gym rollout produced a trial for %d requested task(s): %s", len(missing), sorted(missing))
    return trials
