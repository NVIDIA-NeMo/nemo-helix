# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-Job slots: how the run controller counts its caps and hands out free slots.

Each cycle's settle phase settles every open run and reports a :class:`RunTally`; the grant phase hands out what is left
with :func:`allocate`: each free slot goes to the creator holding the fewest, and within that creator to its
oldest run that is below its caps and has a queued task.
"""

from collections import Counter
from dataclasses import dataclass

from nemo_evals.config import RunControllerConfig
from nemo_evals.taskset_runs.entities import TasksetRunEntity
from nemo_evals.taskset_runs.specs import TasksetRunPlan


@dataclass(frozen=True)
class RunTally:
    """What the settle phase learned about one open run."""

    run: TasksetRunEntity
    creator: str  # run.created_by or DEVELOPMENT_CALLER
    held: int  # registered + image_ready with task_job (create intents)
    queued: int  # image_ready without task_job; 0 for a run that gets no grant this cycle
    cap: int | None  # run_cap(plan, config); 0 for a run that gets no grant this cycle


def task_slots(tallies: list[RunTally], stuck_by_creator: Counter[str], config: RunControllerConfig) -> int:
    """Free task-Job slots: the cluster cap minus held members of open runs and stuck Jobs still running."""
    used = sum(t.held for t in tallies) + stuck_by_creator.total()
    return max(0, config.max_cluster_concurrent_jobs - used)


def allocate(
    tallies: list[RunTally], available_slots: int, creator_cap: int | None, stuck_by_creator: Counter[str]
) -> list[str]:
    """Grant phase: each free slot goes to the creator holding the fewest; that creator's oldest eligible run takes it.

    Returns one run per slot, by entity id (run names are unique only within a workspace), in grant order.
    Registration follows that order, so ending the grant phase at an ambiguous create leaves the creates made so far
    spread across creators (A, B, A, …), not one run's batch.

    Tallies come oldest run first (list_open_runs order). Pending creates (intents whose Job wasn't found) are
    retried by the caller after these grants; they already hold a slot and are in each run's and creator's
    held count, so a create that always fails ambiguously idles only its own slot.
    """
    grants = {t.run.id: 0 for t in tallies}
    order: list[str] = []
    run_held = {t.run.id: t.held for t in tallies}
    creator_held = Counter(stuck_by_creator)  # stuck Jobs still running count against their creator
    for t in tallies:
        creator_held[t.creator] += t.held
    while len(order) < available_slots:  # allocate() is the only place slots are taken
        eligible = [
            t
            for t in tallies
            if grants[t.run.id] < t.queued
            and (t.cap is None or run_held[t.run.id] < t.cap)
            and (creator_cap is None or creator_held[t.creator] < creator_cap)
        ]
        if not eligible:
            break
        t = min(eligible, key=lambda t: creator_held[t.creator])  # ties keep list order = oldest run
        grants[t.run.id] += 1
        order.append(t.run.id)
        run_held[t.run.id] += 1
        creator_held[t.creator] += 1
    return order


def run_cap(plan: TasksetRunPlan, config: RunControllerConfig) -> int | None:
    """The operator's value takes precedence; a request may only ask for fewer."""
    caps = [c for c in (config.default_max_concurrent_task_jobs, plan.max_concurrent_task_jobs) if c is not None]
    return min(caps) if caps else None
