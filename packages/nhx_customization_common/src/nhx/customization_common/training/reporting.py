# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How much detail training progress is reported with.

One fragment, embedded in every backend's schedule section under one field name,
so the docs, the SDK and Studio can treat the knob uniformly rather than learning
three spellings of it. Deliberately *not* a
:class:`~nhx.customization_common.schema.NamespacedModel`: that base exists to
keep two backends' same-named-but-different models from colliding in the merged
``/apis/customization`` spec, and this is the opposite case -- one model, shared
on purpose, which should emit one component.

Kept in its own module rather than beside the callback that consumes it so the
API layer can import the schema without pulling in the platform SDK that
:mod:`nhx.customization_common.training.progress` needs.
"""

from pydantic import BaseModel, ConfigDict, Field

#: The time-series metrics every backend has in common: the loss family, the
#: learning rate, and the gradient norm. Written as patterns rather than literal
#: names so one entry covers both phases (``train_loss`` and ``val_loss``), the
#: algorithm-specific members of a family (``train_sft_loss``,
#: ``val_preference_loss``), and the dataset-qualified names NeMo-RL produces for
#: a second validation set (``val_heldout_loss``).
DIAGNOSTIC_TIME_SERIES = ("*_loss", "*_lr", "*_grad_norm")

#: Everything, spelled explicitly. This is what a user passes to opt out of a
#: backend's default and record a curve for every metric it emits; leaving the
#: field unset means "the backend's default" instead.
ALL_METRICS = ("*",)

#: Least time between metric reports reaching the Jobs service. Points are always
#: recorded; this only bounds how often the accumulator is sent.
#:
#: Ten seconds is about as often as anyone reads a progress bar, and rare enough
#: that a run is not spending its time reporting.
DEFAULT_MIN_REPORT_INTERVAL_SECONDS = 10.0


class ProgressReportingConfig(BaseModel):
    """How a training job reports its progress to the Jobs service.

    Two controls over what lands in the job's status details: how often an update
    is sent, and which metrics keep a full history rather than just their latest
    value. Both have defaults that suit most jobs.
    """

    # Validates and ensures that only the fields we define below are accepted.
    model_config = ConfigDict(extra="forbid")

    min_report_interval_seconds: float = Field(
        default=DEFAULT_MIN_REPORT_INTERVAL_SECONDS,
        ge=0.0,
        description=(
            "Least number of seconds between progress reports reaching the Jobs service. Every "
            "metric the training library logs is still recorded at full resolution -- this only "
            "buffers them in memory and decides how often the accumulated set is sent, which is "
            "what the reporting actually costs. 0 sends a report for every step the library logs. "
            "Raising it reduces the time training spends blocked on reporting, at the cost of a "
            "progress bar and charts that update less often."
        ),
    )
    time_series_metrics: list[str] | None = Field(
        default=None,
        description=(
            "Which metrics are recorded as a time series, named exactly as they appear in the "
            "job's status details and so qualified by phase: 'train_loss', 'val_accuracy'. "
            "Glob patterns are accepted, so '*_loss' covers every loss on both phases and '*' "
            "records everything. Omit the field to take the backend's default, which keeps the "
            "loss, learning rate and gradient norm. "
            "Every metric the run produces is reported either way -- one not matched here is "
            "still sent as a current value on every report, it simply has no history, so a "
            "throughput counter like 'train_tps' costs one number rather than several hundred. "
            "The job log names what was matched and what was not."
        ),
    )
