#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a customization job's training health from ``nemo jobs get-status -f json``.

Reads the status JSON on stdin and prints one summary line, then one line per finding:

- ``ALERT:`` the run is broken and more GPU time will not fix it. Stop and ask the user.
- ``WARN:`` worth reporting, but the run can continue.

Uses only the standard library, so it runs wherever the skill is installed.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

# Training loss below this usually means the model is memorizing a small dataset.
OVERFIT_TRAIN_LOSS = 0.2
# Consecutive points a pattern must hold before it is reported.
RUN_LENGTH = 3
GRPO_MIXED_GROUPS = "train_baseline_reward/pct_mixed"
GRPO_SERIES = (
    "train_reward",
    "val_accuracy",
    GRPO_MIXED_GROUPS,
    "train_approx_entropy",
    "train_truncation_rate",
)
LOSS_SERIES = ("train_loss", "val_loss")


def training_details(status: dict[str, Any]) -> dict[str, Any]:
    """The status details of the task that reports training metrics, or ``{}`` before it starts."""
    candidates = [
        (step.get("name"), task.get("status_details") or {})
        for step in status.get("steps") or []
        for task in step.get("tasks") or []
    ]
    reporting = [(name, details) for name, details in candidates if _reports_metrics(details)]
    for name, details in reporting:
        if name == "training":
            return details
    return reporting[0][1] if reporting else {}


def _reports_metrics(details: dict[str, Any]) -> bool:
    return bool(details.get("metrics")) or any(key.startswith(("train_", "val_")) for key in details)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latest(details: dict[str, Any], name: str) -> float | None:
    """The latest value of a metric: its top-level field, else the end of its history."""
    value = _number(details.get(name))
    if value is not None:
        return value
    history = series(details, name)
    return history[-1] if history else None


def series(details: dict[str, Any], name: str) -> list[float]:
    """A metric's history in step order; empty when the job keeps no history for it."""
    points = (details.get("metrics") or {}).get(name) or []
    ordered = sorted(points, key=lambda point: point.get("step", 0))
    values = [_number(point.get("value")) for point in ordered]
    return [value for value in values if value is not None]


def is_grpo(details: dict[str, Any]) -> bool:
    return any(latest(details, name) is not None for name in ("train_reward", GRPO_MIXED_GROUPS))


def summary_line(status: dict[str, Any], details: dict[str, Any]) -> str:
    phase = details.get("phase") or (status.get("status_details") or {}).get("phase", "")
    parts = [f"status={status.get('status')}", f"phase={phase}"]
    if details.get("step") is not None:
        parts.append(f"step={details['step']}/{details.get('max_steps', '?')}")
    if details.get("epoch") is not None:
        parts.append(f"epoch={details['epoch']}/{details.get('num_epochs', '?')}")
    for name in GRPO_SERIES if is_grpo(details) else LOSS_SERIES:
        value = latest(details, name)
        if value is not None:
            parts.append(f"{name}={value:.4g}")
    return " ".join(parts)


def findings(details: dict[str, Any]) -> list[str]:
    """``ALERT:`` and ``WARN:`` lines for the current training state."""
    lines: list[str] = []
    for name in LOSS_SERIES:
        value = latest(details, name)
        if value is not None and not math.isfinite(value):
            lines.append(f"ALERT: {name} is {value}; training has diverged.")

    if is_grpo(details):
        mixed = series(details, GRPO_MIXED_GROUPS)
        if len(mixed) >= RUN_LENGTH and all(value == 0 for value in mixed[-RUN_LENGTH:]):
            lines.append(
                "ALERT: every prompt group got the same reward for all its rollouts "
                f"({GRPO_MIXED_GROUPS} is 0), so GRPO has no gradient to learn from."
            )
        elif not mixed and latest(details, GRPO_MIXED_GROUPS) == 0 and (details.get("step") or 0) >= RUN_LENGTH:
            # One reading cannot show a sustained run, so this does not stop the poll.
            lines.append(
                f"WARN: {GRPO_MIXED_GROUPS} is 0 at this step. If it stays 0, every prompt group "
                "gets the same reward and GRPO has no gradient to learn from."
            )
        return lines

    train_loss = latest(details, "train_loss")
    if train_loss is not None and math.isfinite(train_loss) and train_loss < OVERFIT_TRAIN_LOSS:
        lines.append(f"WARN: train_loss is {train_loss:.3g}, below {OVERFIT_TRAIN_LOSS}; the model may be memorizing.")
    val = series(details, "val_loss")[-(RUN_LENGTH + 1) :]
    if len(val) == RUN_LENGTH + 1 and all(later > earlier for earlier, later in zip(val, val[1:])):
        lines.append(f"WARN: val_loss rose for {RUN_LENGTH} validation passes in a row; the model may be overfitting.")
    return lines


def main() -> int:
    status = json.load(sys.stdin)
    details = training_details(status)
    print(summary_line(status, details))
    for line in findings(details):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
