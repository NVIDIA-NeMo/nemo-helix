# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notebook display for customization-job progress and loss."""

from __future__ import annotations

import time
from typing import Any

_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "error"}
_LOSS_PREFERENCE = ("train_loss", "val_loss", "loss")
_PLOT_WARNED = False


def status_text(value: Any) -> str:
    """Render a status enum as its value, for example ``active``."""
    return str(getattr(value, "value", value))


def show_training_progress(status: Any) -> None:
    """Print step completion and the latest loss, and chart loss by step."""
    print(describe_training_progress(status))
    _plot_loss(status)


def wait_for_training(
    client: Any,
    job_name: str,
    *,
    workspace: str = "default",
    timeout_seconds: float | None = None,
    poll_interval: float = 10,
) -> Any:
    """Poll a customization job until it finishes, refreshing the notebook output."""
    start = time.monotonic()
    while True:
        if timeout_seconds is not None and time.monotonic() - start > timeout_seconds:
            raise TimeoutError(f"{job_name} took longer than {timeout_seconds} seconds")
        status = client.jobs.get_job_status(name=job_name, workspace=workspace).data()
        _clear_output()
        print(f"Job: {job_name}")
        print(f"Status: {status_text(status.status)}")
        show_training_progress(status)
        state = status_text(status.status)
        if state in _TERMINAL_STATUSES:
            print(f"\nJob finished: {state}")
            if state != "completed":
                raise RuntimeError(f"Training job finished with status: {state}")
            return status
        time.sleep(poll_interval)


def describe_training_progress(status: Any) -> str:
    """Return the step, percentage, and latest loss lines for a job status."""
    lines: list[str] = []
    steps = status.steps or []
    if not steps:
        lines.append("Waiting for the job to report steps...")
    for job_step in steps:
        lines.append(f"  {status_text(job_step.status):<10} {job_step.name}")
        for task in job_step.tasks or []:
            progress = _task_progress(task.status_details or {})
            if progress:
                lines.append(f"             {progress}")
    for key, value in _latest_losses(_training_details(status)).items():
        lines.append(f"{key}: {value:.4f}")
    return "\n".join(lines)


def _task_progress(details: dict[str, Any]) -> str | None:
    step, max_steps = details.get("step"), details.get("max_steps")
    if isinstance(step, (int, float)) and isinstance(max_steps, (int, float)) and max_steps:
        shown = 100.0 * float(step) / float(max_steps)
        extra = f" ({details['phase']})" if details.get("phase") else ""
        epoch, num_epochs = details.get("epoch"), details.get("num_epochs")
        if isinstance(epoch, (int, float)) and isinstance(num_epochs, (int, float)):
            extra += f", epoch {int(epoch)}/{int(num_epochs)}"
        return f"step {int(step)}/{int(max_steps)} — {shown:.1f}%{extra}"
    pct = details.get("percentage_done", details.get("progress_pct"))
    if not isinstance(pct, (int, float)):
        return None
    current = details.get("current_file")
    suffix = f" — {current}" if current else ""
    return f"{float(pct):.1f}%{suffix}"


def _training_details(status: Any) -> dict[str, Any]:
    fallback: dict[str, Any] = {}
    for job_step in status.steps or []:
        for task in job_step.tasks or []:
            details = task.status_details or {}
            if not isinstance(details, dict):
                continue
            if details.get("metrics") or _latest_losses(details):
                return details
            if details.get("step") is not None:
                fallback = details
    return fallback


def _loss_keys(details: dict[str, Any]) -> list[str]:
    metrics = details.get("metrics") or {}
    keys = [key for key in _LOSS_PREFERENCE if key in details or key in metrics]
    for key in (*details, *metrics):
        if isinstance(key, str) and key.endswith("_loss") and key not in keys:
            keys.append(key)
    return keys


def _latest_losses(details: dict[str, Any]) -> dict[str, float]:
    found: dict[str, float] = {}
    metrics = details.get("metrics") or {}
    for key in _loss_keys(details):
        value = details.get(key)
        if not isinstance(value, (int, float)):
            for point in reversed(metrics.get(key) or []):
                if isinstance(point, dict) and isinstance(point.get("value"), (int, float)):
                    value = point["value"]
                    break
        if isinstance(value, (int, float)):
            found[key] = float(value)
    return found


def _loss_series(details: dict[str, Any]) -> dict[str, list[tuple[int, float]]]:
    metrics = details.get("metrics") or {}
    series: dict[str, list[tuple[int, float]]] = {}
    for key in _loss_keys(details):
        points: list[tuple[int, float]] = []
        for point in metrics.get(key) or []:
            if (
                isinstance(point, dict)
                and isinstance(point.get("step"), (int, float))
                and isinstance(point.get("value"), (int, float))
            ):
                points.append((int(point["step"]), float(point["value"])))
        if points:
            series[key] = points
    return series


def _plot_loss(status: Any) -> None:
    global _PLOT_WARNED
    series = _loss_series(_training_details(status))
    if not series:
        return
    try:
        import matplotlib.pyplot as plt
        from IPython.display import display
    except ImportError:
        if not _PLOT_WARNED:
            print("Install matplotlib to chart loss: pip install matplotlib")
            _PLOT_WARNED = True
        return
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for key, points in series.items():
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", markersize=3, label=key.replace("_", " "))
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss by step")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    display(fig)
    plt.close(fig)


def _clear_output() -> None:
    try:
        from IPython.display import clear_output
    except ImportError:
        return
    clear_output(wait=True)
