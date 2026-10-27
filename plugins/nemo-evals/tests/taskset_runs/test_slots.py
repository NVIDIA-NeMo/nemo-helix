# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slot counting and hand-out (pure rules), and the create-grace arithmetic."""

from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest
from nemo_evals.config import RunControllerConfig
from nemo_evals.taskset_runs.entities import TasksetRunEntity
from nemo_evals.taskset_runs.slots import RunTally, allocate, run_cap, task_slots
from nemo_evals.taskset_runs.task_jobs import past_grace, sent_at_after_refusal
from pydantic import ValidationError

NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


def tally(name: str, creator: str = "user:a", *, held: int = 0, queued: int = 0, cap: int | None = None) -> RunTally:
    run = TasksetRunEntity.model_construct(name=name)
    run._id = f"id-{name}"
    return RunTally(run=run, creator=creator, held=held, queued=queued, cap=cap)


def names(order: list[str]) -> list[str]:
    return [run_id.removeprefix("id-") for run_id in order]


def test_task_slots_subtract_held_members_and_stuck_jobs():
    config = RunControllerConfig(max_cluster_concurrent_jobs=10)
    tallies = [tally("r1", held=3), tally("r2", held=4)]
    assert task_slots(tallies, Counter({"user:a": 2}), config) == 1
    assert task_slots(tallies, Counter({"user:a": 5}), config) == 0  # never negative


def test_one_creator_without_a_creator_cap_is_first_in_first_out():
    tallies = [tally("r1", queued=3, cap=2), tally("r2", queued=3, cap=2), tally("r3", queued=3)]
    assert names(allocate(tallies, 5, None, Counter())) == ["r1", "r1", "r2", "r2", "r3"]


def test_slots_go_to_the_least_loaded_creator_then_its_oldest_run():
    tallies = [
        tally("a1", "user:a", held=3, queued=5),
        tally("b1", "user:b", held=1, queued=5),
        tally("b2", "user:b", queued=5),
        tally("c1", "user:c", queued=5),
    ]
    # b: 1, c: 0 → c; then b and c tie at 1: oldest run first (b1) …
    assert names(allocate(tallies, 6, None, Counter())) == ["c1", "b1", "c1", "b1", "c1", "a1"]


def test_run_and_creator_caps_count_intents_and_stuck_jobs():
    tallies = [tally("a1", "user:a", held=2, queued=5, cap=3), tally("a2", "user:a", queued=5)]
    assert names(allocate(tallies, 10, 4, Counter())) == ["a1", "a2"]  # a1 hits its run cap, A its creator cap
    assert names(allocate(tallies, 10, 4, Counter({"user:a": 2}))) == []  # stuck Jobs count against their creator


def test_free_slots_run_out_mid_list_and_no_grant_runs_are_skipped():
    tallies = [tally("r1", queued=0, cap=0, held=2), tally("r2", queued=3), tally("r3", queued=3)]
    assert names(allocate(tallies, 2, None, Counter())) == ["r2", "r2"]


@pytest.mark.parametrize(
    ("operator", "request_cap", "expected"),
    [(20, None, 20), (None, 5, 5), (20, 40, 20), (20, 5, 5), (None, None, None)],
)
def test_the_operator_cap_takes_precedence(harbor_plan, operator, request_cap, expected):
    plan = harbor_plan.model_copy(update={"max_concurrent_task_jobs": request_cap})
    assert run_cap(plan, RunControllerConfig(default_max_concurrent_task_jobs=operator)) == expected


def test_past_grace_and_the_refusal_restore():
    grace = timedelta(seconds=600)
    assert not past_grace(NOW, NOW + grace - timedelta(seconds=1), grace)
    assert past_grace(NOW, NOW + grace, grace)
    assert past_grace(NOW.replace(tzinfo=None), NOW + grace, grace)  # stored naive, read as UTC
    assert past_grace(None, NOW, grace)
    assert sent_at_after_refusal(None, NOW) == NOW  # the first POST keeps its own
    assert sent_at_after_refusal(NOW - grace, NOW) == NOW - grace  # a retry writes back the one it overwrote


def test_phase_1_concurrency_and_the_create_budget_default_to_8_and_10():
    config = RunControllerConfig()
    assert (config.settle_concurrency, config.max_creates_per_cycle) == (8, 10)
    with pytest.raises(ValidationError):
        RunControllerConfig(settle_concurrency=0)
    with pytest.raises(ValidationError):
        RunControllerConfig(max_creates_per_cycle=1)  # 1 // 2 would reserve nothing for retries
