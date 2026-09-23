# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training progress callback shared by the customization backends.

Wraps a :class:`~nhx.customization_common.training.progress.JobsServiceProgressReporter`
and turns a backend's metric dict into job status. Every chartable metric is
reported as a current value; the ones matching ``time_series_metrics`` also
accumulate a history under a ``metrics`` key.

Naming
------
One rule, no exceptions: a metric is stored and reported as ``<phase>_<name>``,
the backend supplying its framework's ``<name>`` (``loss``, ``lr``, ``accuracy``)
and the phase supplying the prefix. ``train_loss`` and ``val_loss`` are what that
produces for ``loss``, not special cases.

The prefix is load-bearing. DPO reports ``accuracy`` on both train and
validation, so bare names would interleave two different quantities into one
series. It also keeps metrics clear of the keys this class owns -- ``phase``,
``step``, ``epoch``, ``metrics``.

What a report costs
-------------------
The Jobs service merges ``status_details`` key-wise but shallowly: a key that is
sent replaces the stored value wholesale, and one left out is left standing. So a
report either resends a series in full or omits the key entirely.

One report is three writes -- client, task, job attempt -- and the two
server-side ones PUT the whole entity blob whether or not the report mentioned
the series. Measured at ~43 bytes a point and 46ms + 0.31ms/KB a report, and that
latency lands on the training thread between optimizer steps.

This class does not thin or cap what it records, deliberately. The quadratic is a
property of ``status_details`` being a blob that is replaced wholesale, and
capping points here only made that cheaper to leave unfixed; a point cap was
tried and removed. The limiter below bounds how *often* a report goes, not how
much it carries, and nothing here bounds a long run -- that ceiling belongs to
the transport. See the AALGO-497 design note for the measurements and the
argument.

Seeding
-------
The accumulator lives in this process, so it seeds itself from the server.
Without that, a process taking over a task would report from empty and -- the
merge being wholesale per key -- erase the curves already stored.

Known issue, left alone deliberately: when a pod is replaced mid-task the new
process seeds from the old one's points and appends its own, so one series can
carry two runs. Consumers see the last value written at a given step plus a tail
from the abandoned attempt. Fixing it means telling two restart shapes apart --
a from-scratch restart, whose earlier points are a real history worth keeping,
and a resume-from-checkpoint, whose replayed range is work that was rolled back
-- which is more than a passing change.

Backends subclass this and set :attr:`_default_backend`; callers may also pass
``backend`` per call.
"""

import logging
import math
import numbers
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, ClassVar, cast

from nhx.customization_common.training.progress import JobsServiceProgressReporter
from nhx.customization_common.training.reporting import ALL_METRICS, DEFAULT_MIN_REPORT_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

__all__ = ["DatasetQualifier", "ReportRateLimiter", "TrainingProgressCallback", "is_chartable"]


def is_chartable(value: Any) -> bool:
    """Whether ``value`` is a finite scalar that can enter a metric series.

    Backends hand over whatever their framework produced, which is not always a
    number -- NeMo-RL's dicts interleave ``Histogram`` objects, tables and nested
    dicts with the scalars, and ``math.isfinite`` raises on those rather than
    returning False.

    NaN and the infinities are rejected: one infinity flattens every real point
    in the series against it. ``bool`` is rejected despite being an ``int``: no
    metric here is a flag, and charting one as 0/1 is worse than dropping it.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        # An unbounded Python int overflows float(). The contract here is that
        # any value a backend hands us gets classified without raising, so an
        # absurd counter loses its series rather than the run losing reporting.
        return False


def _qualify_metric_names(phase: str, metrics: Mapping[str, object]) -> dict[str, float | int]:
    """The chartable subset of ``metrics``, keyed by ``<phase>_<name>``.

    The naming rule, applied in one place. See the module docstring for why the
    prefix matters.

    Unchartable values are dropped from the report as well as the series, so one
    bad metric costs its own curve rather than the whole report: a ``Histogram``
    left in ``status_details`` makes the update fail to serialize, and
    ``update_task`` swallows that, losing every metric while the job goes on
    looking healthy.
    """
    return {f"{phase}_{name}": _coerce(value) for name, value in metrics.items() if is_chartable(value)}


def _coerce(value: object) -> float | int:
    """Narrow a chartable value to a JSON-serializable builtin.

    numpy scalars satisfy ``numbers.Real`` but are not serializable. Counts stay
    ``int`` rather than becoming ``64.0``.
    """
    real = cast(numbers.Real, value)
    return int(real) if isinstance(real, numbers.Integral) else float(real)


def _point_step(point: object) -> float | int | None:
    """The step a stored series point records, or ``None`` if it records none.

    Defensive because the seed is whatever the server had stored: a malformed
    blob should cost the points it corrupted, not raise into the training loop.
    """
    if not isinstance(point, dict):
        return None
    step = point.get("step")
    return step if isinstance(step, (int, float)) else None


class DatasetQualifier:
    """Fold a dataset's name into its metric names, past the first dataset seen.

    Both LLM backends validate over a *dict* of dataloaders and log once per
    entry, all at the same step. Forwarded as-is, two datasets' ``loss`` become
    two points at one step in a single ``val_loss`` series -- the collision the
    ``<phase>_`` rule prevents, one level further down, and invisible because
    Studio keys its chart by step, so one silently wins.

    The first dataset seen keeps the bare names, so the ordinary single-dataset
    run still reports ``val_loss``. Both frameworks name that dataloader too, so
    keying on "did a name arrive" would rename the common case and take Studio's
    curve with it.
    """

    def __init__(self) -> None:
        self._primary: str | None = None

    def qualify(self, key: str, label: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Qualify ``metrics`` for the dataset identified by ``key``.

        ``key`` identifies the dataset, ``label`` is what gets folded into the
        names. They differ for NeMo-RL, whose key is the whole
        ``validation-<name>`` prefix while only ``<name>`` belongs in the series.
        """
        if self._primary is None:
            self._primary = key
        if key == self._primary:
            return dict(metrics)
        return {f"{label}_{name}": value for name, value in metrics.items()}


def _clean_patterns(patterns: Collection[str] | None) -> tuple[str, ...]:
    """Normalise a configured pattern list into something safe to match against.

    These come off a config file, on the NeMo-RL path straight out of YAML with
    no schema in between, and matching runs inline in a training step. So
    anything unusable is handled here rather than raising out of
    ``report_train_step`` and taking the run with it.

    A bare string is the quiet trap: ``str`` satisfies ``Collection[str]`` as a
    collection of its *characters*, so ``time_series_metrics: train_loss`` in
    YAML became ten single-letter patterns matching nothing, with no error. It
    can only have meant one pattern, so it is read as one.

    A non-empty input that yields nothing usable is a broken config, and falls
    back to recording everything -- a misconfiguration should cost noise, not
    data. An input that was *already* empty is left alone: ``[]`` legitimately
    means "no series at all".
    """
    if patterns is None:
        return ALL_METRICS
    if isinstance(patterns, str):
        return (patterns,)

    try:
        given = tuple(patterns)
    except TypeError:
        logger.warning(f"progress_reporting.time_series_metrics is not a list ({patterns!r}); recording every metric.")
        return ALL_METRICS

    usable = tuple(pattern for pattern in given if isinstance(pattern, str))
    unusable = [pattern for pattern in given if not isinstance(pattern, str)]
    if unusable:
        logger.warning(
            f"Ignoring {len(unusable)} non-string entr{'y' if len(unusable) == 1 else 'ies'} in "
            f"progress_reporting.time_series_metrics: {unusable!r}."
        )
    if given and not usable:
        logger.warning("No usable entry in progress_reporting.time_series_metrics; recording every metric.")
        return ALL_METRICS
    return usable


class ReportRateLimiter:
    """Decides when a metric report may go on the wire.

    Points are never at stake: every metric is recorded the moment it arrives,
    and this only decides how often the accumulated state is *sent*. A withheld
    report loses nothing, because what it would have carried goes out with the
    next one.

    What it buys is bounded, and worth stating so nobody expects more. On a
    594-step run over half an hour it takes 594 requests down to 180 and the time
    blocked inside the training loop from 36s to 11s. It does not bound a long
    run: the cost of one request grows with the blob, which no rate limit can fix.

    ``monotonic`` rather than wall-clock: a training process outlives NTP
    corrections, and a clock stepping backwards would stall reporting until it
    caught up.
    """

    def __init__(
        self,
        min_interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        #: Negative is meaningless and would make every check pass; clamp rather
        #: than raise, since this arrives from a job config and a reporting knob
        #: must not be able to stop a run from starting.
        self._min_interval = max(float(min_interval_seconds), 0.0)
        self._clock = clock
        self._last_sent: float | None = None

    def allow_request(self) -> bool:
        """Whether to send now, recording the send if so.

        The first call always allows, so a curve starts at the beginning of the
        run rather than one interval into it.
        """
        now = self._clock()
        if self._last_sent is not None and now - self._last_sent < self._min_interval:
            return False
        self._last_sent = now
        return True


@dataclass(frozen=True, slots=True)
class _Progress:
    """Where training was when a metric was last observed.

    The part of a report that is not a metric. Unlike the metrics, which merge by
    name, this is genuinely replaced: there is one current step, and a newer
    observation supersedes an older one outright.
    """

    phase: str
    step: int
    epoch: int
    backend: str | None


class TrainingProgressCallback:
    """Report training progress to the Jobs service."""

    #: Backend name stamped on each report when a per-call ``backend`` isn't given.
    #: ``None`` means no ``backend`` field is added.
    _default_backend: ClassVar[str | None] = None

    def __init__(
        self,
        reporter: JobsServiceProgressReporter,
        *,
        time_series_metrics: Collection[str] | None = None,
        min_report_interval_seconds: float = DEFAULT_MIN_REPORT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Build the callback over ``reporter``.

        Args:
            reporter: Transport to the Jobs service.
            time_series_metrics: Qualified metric names or glob patterns to
                accumulate, or None for all of them.
            min_report_interval_seconds: Least time between metric reports going
                on the wire. 0 sends every one. Points are recorded regardless.
            clock: Monotonic time source, injectable so the limiter can be tested
                without sleeping.
        """
        self._reporter = reporter
        self._limiter = ReportRateLimiter(min_report_interval_seconds, clock)
        #: Where training was when a metric was last observed, cleared once sent.
        #: Doubles as the "a report is owed" flag ``close()`` reads.
        self._current_progress: _Progress | None = None
        #: Whether a point has been appended since the last send, which decides
        #: whether the next report carries the series at all. Only the series are
        #: gated: they are the expensive part, the scalars are a few numbers.
        self._series_dirty = False
        #: None becomes "*", which is what it already meant, so there is one code
        #: path. ``[]`` stays empty -- a different and equally valid request.
        self._time_series_metrics: tuple[str, ...] = _clean_patterns(time_series_metrics)
        #: Qualified names seen *this run*. Not the same as ``_metrics``, which
        #: also holds seeded names never seen again, and the difference is what
        #: makes an unmatched pattern worth warning about.
        self._metrics_seen: set[str] = set()
        #: Patterns that selected at least one metric, so close() can name the
        #: ones that never did.
        self._patterns_matched: set[str] = set()
        self._closed = False

        seeded = reporter.fetch_current_metrics()

        #: The seed *read failed*, as opposed to finding nothing stored. The
        #: accumulator is then known to be incomplete, and since a sent key
        #: replaces the stored one wholesale, reporting it would overwrite the
        #: history that could not be read. So ``metrics`` is withheld for the life
        #: of the process: this run's curves stall, which is recoverable, rather
        #: than destroying previous ones, which is not. Values and progress still
        #: report normally.
        self._seed_unavailable = seeded is None
        if self._seed_unavailable:
            logger.warning(
                "Could not read the stored metric series; this run will report progress and current "
                "values but will not update the accumulated curves, to avoid overwriting them."
            )

        #: Qualified metric name -> what the server should hold for it. State,
        #: not a log of events: a send transmits this, it does not replay what
        #: happened since the last one.
        #:
        #: **The type is the discriminator.** A list is a series and appends; a
        #: bare number is a scalar and is replaced. Which one a name is gets
        #: decided once, on first arrival, by :meth:`_records_history`, and is
        #: never re-checked.
        #:
        #: Seeded from the server (see the module docstring). Everything seeded
        #: arrives as a list and so stays a series, including a name the current
        #: patterns would not select -- its history exists, and freezing it
        #: half-written serves nobody. A seeded point recording no step is
        #: dropped: nothing can place it on a curve.
        self._metrics: dict[str, float | int | list[dict[str, float | int]]] = {
            name: [point for point in points if _point_step(point) is not None]
            for name, points in (seeded or {}).items()
        }

        if any(self._metrics.values()):
            logger.info(
                "Seeded %d metric series from server (%d points): %s",
                len(self._metrics),
                sum(len(cast(list, v)) for v in self._metrics.values()),
                ", ".join(sorted(self._metrics)),
            )

    def _resolve_backend(self, backend: str | None) -> str | None:
        return backend if backend is not None else self._default_backend

    def _send(self, phase: str, details: dict[str, object], backend: str | None) -> None:
        """Stamp the backend field, if there is one, and hand the report over.

        In one place because a change that missed a copy would be near-invisible:
        ``_default_backend`` is ``None`` for two of the three backends, so only
        unsloth's payload would show the difference.
        """
        resolved = self._resolve_backend(backend)
        if resolved is not None:
            details["backend"] = resolved
        self._reporter.report_running(phase=phase, **details)

    def _report_metrics(
        self,
        phase: str,
        report_phase: str,
        step: int,
        epoch: int,
        metrics: Mapping[str, object],
        backend: str | None,
    ) -> None:
        """Fold every metric into the store, then send it if the limiter allows.

        The one path both ``report_train_step`` and ``report_validation`` take.
        ``phase`` qualifies the metric names (``train``/``val``);
        ``report_phase`` is what the Jobs service records as the task's phase.

        Every report is *recorded*; the limiter decides only which are *sent*.
        Merging by name is what makes that safe -- a withheld report's values are
        already in the store. Holding withheld reports whole instead loses them:
        a train step and then a validation pass, both withheld, and the
        validation report's disjoint ``val_`` keys displace the train scalars
        entirely. That is the ordinary end-of-run sequence.
        """
        qualified = _qualify_metric_names(phase, metrics)
        self._metrics_seen.update(qualified)
        for name, value in qualified.items():
            existing = self._metrics.get(name)
            if isinstance(existing, list):
                existing.append({"step": step, "epoch": epoch, "value": value})
                self._series_dirty = True
            elif existing is None and self._records_history(name):
                self._metrics[name] = [{"step": step, "epoch": epoch, "value": value}]
                self._series_dirty = True
            else:
                # A scalar: the newest value is the whole of what anyone reads.
                self._metrics[name] = value

        self._current_progress = _Progress(report_phase, step, epoch, backend)
        if self._limiter.allow_request():
            self._flush()

    def _flush(self) -> None:
        """Send the store as it stands, under the latest progress.

        Scalars ride on every report rather than only when observed. They are a
        few single numbers, and restating them makes a dropped report cost
        nothing -- ``update_task`` swallows its failures, so the next send is
        what repairs it. The series are the expensive term and ship only when a
        point was added, since a key left unmentioned is left standing.
        """
        progress = self._current_progress
        if progress is None:
            return

        details: dict[str, object] = {"step": progress.step, "epoch": progress.epoch}
        series: dict[str, list[dict[str, float | int]]] = {}
        for name, value in self._metrics.items():
            if isinstance(value, list):
                # A series reports both: the history under `metrics`, and its
                # newest point as a current value like any other metric. Being a
                # series adds a history; it does not cost the value.
                series[name] = list(value)
                if value and (latest := value[-1].get("value")) is not None:
                    details[name] = latest
            else:
                details[name] = value

        if not self._seed_unavailable and self._series_dirty:
            details["metrics"] = self._series_payload(series)

        self._current_progress = None
        self._series_dirty = False
        self._send(progress.phase, details, progress.backend)

    @staticmethod
    def _series_payload(
        series: dict[str, list[dict[str, float | int]]],
    ) -> dict[str, list[dict[str, float | int]]]:
        """The ``metrics`` blob: every series, with the two Studio charts present.

        ``train_loss``/``val_loss`` are always keys, even when empty, so the shape
        stays stable for consumers that index them directly. The lists arrive
        already copied -- the payload must not mutate after it is handed over.
        """
        return {"train_loss": [], "val_loss": [], **series}

    def _records_history(self, name: str) -> bool:
        """Whether *name* gets a stored series rather than only a current value.

        Asked once, on a metric's first arrival; the answer is then carried by its
        type in ``_metrics``. That is why the log below needs no seen-set to stay
        at one line per name -- it is reached once per name by construction.

        The one bound on the payload this class still applies, and a
        data-modelling choice rather than a workaround: throughput and accounting
        counters (``tps``, ``mem``, ``num_label_tokens``) are worth a number on a
        status page, not two hundred stored points each.

        Answering no costs a metric its history, never its visibility -- every
        chartable metric is still reported as a current value. That is what makes
        this safe where the old report-time allow-list was not: that one dropped
        metrics outright, and DPO's ``accuracy``, ``sft_loss`` and
        ``rewards_chosen_mean`` went missing for a release.

        Patterns match the **qualified** name, so what a user writes is what they
        read back, and NeMo-RL's second validation set is reachable at all --
        its loss arrives as ``val_heldout_loss``, which ``*_loss`` covers without
        knowing the dataset name. ``fnmatchcase`` and not ``fnmatch``: the latter
        applies ``os.path.normcase``, so it would match case-insensitively on
        some platforms and not others.
        """
        matched = [p for p in self._time_series_metrics if fnmatchcase(name, p)]
        if matched:
            self._patterns_matched.update(matched)
            return True
        logger.info(
            "Reporting %s as a current value only, with no stored history. Recording a series "
            "for anything matching: %s. Add a name or pattern to "
            "progress_reporting.time_series_metrics to record one of these too.",
            name,
            ", ".join(self._time_series_metrics) or "(nothing)",
        )
        return False

    def _warn_on_patterns_that_matched_nothing(self) -> None:
        """Say so when a configured pattern never matched a metric that arrived.

        Catches a misspelling, most often an unqualified name: ``loss`` selects
        nothing, because the metric is ``train_loss``. Without this the run
        reports no history and looks exactly like one configured not to.

        Left to ``close()`` because a name that has not arrived *yet* is not yet
        wrong -- backends omit metrics on some steps, and validation names do not
        appear until the first pass.

        Gated on whether any metric arrived, not on whether any was *excluded*.
        Those differ exactly when everything that arrived matched something,
        which is the ordinary case a typo has to be caught in.
        """
        unmatched = [p for p in self._time_series_metrics if p not in self._patterns_matched]
        if not unmatched or not self._metrics_seen:
            # No metric arrived at all: every pattern trivially matched nothing,
            # and the run reporting nothing is the larger problem to look at.
            return
        logger.warning(
            "These progress_reporting.time_series_metrics entries matched no metric this run: %s. "
            "Names are qualified by phase, so 'loss' matches nothing and 'train_loss', 'val_loss' "
            "or '*_loss' is meant. Metrics this run reported: %s.",
            ", ".join(unmatched),
            ", ".join(sorted(self._metrics_seen)),
        )

    def report_training_start(
        self,
        max_steps: int,
        num_epochs: int,
        *,
        run_facts: Mapping[str, object] | None = None,
        backend: str | None = None,
    ) -> None:
        """Report that training has started with schedule information.

        States neither ``metrics`` nor ``step``, and for the same reason: this
        fires before the first step, and the merge is wholesale per key. An empty
        accumulator would replace the stored series with empty lists, and a
        literal step 0 would overwrite the stored ``percentage_done`` -- the
        harder one to spot, since the epoch beside it goes on reading correctly.
        The first train step states the position instead.

        Args:
            run_facts: Constants describing the run rather than its progress --
                which algorithm, how many rollouts a step generates. Sent once
                and never restated, which works for the same reason the omissions
                above do: a key left out of a later report keeps the value it
                already has.

                They share the flat ``status_details`` namespace with everything
                else reported, so a name here can collide. Most collisions are
                self-correcting: ``phase``, ``step``, ``epoch``, ``metrics`` and
                every metric name go out again on the next report, so the real
                value lands within an interval. ``max_steps`` and ``num_epochs``
                are the exception -- this is the only method that writes them, so
                a collision would stand for the whole run. Hence the ordering
                below rather than a note here.
        """
        self._reporter.configure_progress_tracking(max_steps, num_epochs)
        # Schedule last, so a run fact cannot shadow it. The same two numbers went
        # to configure_progress_tracking above, and `percentage_done` is derived
        # from those and not from the keys below -- so a shadowed `max_steps` would
        # leave the two disagreeing, rendering as `step 30 / 999` at 100% with
        # nothing raised and the wrong-looking number being the correct one.
        details: dict[str, object] = {
            **(run_facts or {}),
            "max_steps": max_steps,
            "num_epochs": num_epochs,
        }
        self._send("training", details, backend)

    def report_train_step(
        self, step: int, epoch: int, metrics: Mapping[str, object], *, backend: str | None = None
    ) -> None:
        """Report one training step.

        Hand over the framework's metric dict as-is, under its own names. Each
        chartable entry becomes a ``train_<name>`` current value, and a
        ``train_<name>`` series if it is configured for one.

        No metric is required and none is privileged: a step producing no loss
        reports no ``train_loss``, rather than a null that charts as a real zero.

        A dict rather than ``**kwargs`` so metric names cannot collide with this
        method's own parameters -- a framework is free to call something ``step``.
        """
        self._report_metrics("train", "training", step, epoch, metrics, backend)

    def report_validation(
        self, step: int, epoch: int, metrics: Mapping[str, object], *, backend: str | None = None
    ) -> None:
        """Report one validation pass.

        The same rule under the ``val_`` prefix. An algorithm that validates on
        task metrics alone and reports no ``loss`` just leaves ``val_loss`` empty.
        """
        self._report_metrics("val", "validation", step, epoch, metrics, backend)

    def report_checkpoint_saved(
        self,
        step: int,
        epoch: int,
        checkpoint_path: str | None = None,
        *,
        backend: str | None = None,
    ) -> None:
        """Report that a checkpoint was saved.

        ``checkpoint_path`` is omitted rather than sent as null when the backend
        has none to state: the merge is key-wise, so an explicit null overwrites
        the last known checkpoint while omitting the key leaves it standing.
        """
        details: dict[str, object] = {
            "step": step,
            "epoch": epoch,
        }
        if checkpoint_path:
            details["checkpoint_path"] = checkpoint_path
        self._send("checkpoint_saved", details, backend)

    def report_epoch_end(self, step: int, epoch: int, *, backend: str | None = None) -> None:
        """Report that an epoch has completed."""
        details: dict[str, object] = {
            "step": step,
            "epoch": epoch,
        }
        self._send("epoch_end", details, backend)

    def close(self) -> None:
        """Flush whatever the limiter withheld, then clean up resources.

        Idempotent: NeMo-RL reaches this from a driver ``finally``, from the
        composite logger's ``finish()`` and from ``__del__``.
        """
        if self._closed:
            return
        self._closed = True
        if self._current_progress is not None:
            # Load-bearing for data, not just freshness: points recorded since
            # the last send exist nowhere but this process until something
            # carries them.
            try:
                self._flush()
            except Exception as exc:  # pragma: no cover - defensive, shutdown path
                logger.warning(f"Failed to send the final metric report: {exc}")
        self._warn_on_patterns_that_matched_nothing()
        self._reporter.close()
