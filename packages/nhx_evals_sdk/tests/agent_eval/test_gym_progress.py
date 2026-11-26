# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A colocated Gym run reports its rollouts as Gym writes them, not only once collection ends."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from nhx_evals_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig, discover_gym_tasks
from nhx_evals_sdk.agent_eval.runtimes.gym.progress import RolloutFileFollower
from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig

EXAMPLE = Path(__file__).parent / "fixtures" / "gym_mcqa_example.jsonl"


class RecordingProgress:
    def __init__(self) -> None:
        self.planned_rollouts: list[int] = []
        self.batches: list[list[dict[str, Any]]] = []
        self.stops = 0

    async def planned(self, rollouts: int) -> None:
        self.planned_rollouts.append(rollouts)

    async def finished(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.batches.append([dict(record) for record in records])

    async def stopped(self) -> None:
        self.stops += 1


class FailingProgress:
    async def planned(self, rollouts: int) -> None:
        raise RuntimeError("status details unreachable")

    async def finished(self, records: Sequence[Mapping[str, Any]]) -> None:
        raise RuntimeError("status details unreachable")

    async def stopped(self) -> None:
        raise RuntimeError("status details unreachable")


def _config(**kwargs: Any) -> GymRuntimeConfig:
    return GymRuntimeConfig(agent="simple_agent", agent_config="cfg.yaml", resources_server="mcqa", **kwargs)


async def test_the_follower_reports_each_complete_record_once(tmp_path: Path) -> None:
    rollouts, failures = tmp_path / "out.jsonl", tmp_path / "out_failures.jsonl"
    progress = RecordingProgress()
    follower = RolloutFileFollower([rollouts, failures], progress)

    rollouts.write_text('{"n": 1}\n{"n": 2}\n{"n": ', encoding="utf-8")
    await follower.report_new()
    with rollouts.open("a", encoding="utf-8") as file:
        file.write("3}\n")
    failures.write_text('{"n": 4, "_ng_failure_class": "agent_error"}\n', encoding="utf-8")
    await follower.report_new()
    await follower.report_new()

    assert [[record["n"] for record in batch] for batch in progress.batches] == [[1, 2], [3, 4]]


async def test_the_follower_skips_lines_that_are_not_records_and_files_not_yet_written(tmp_path: Path) -> None:
    rollouts = tmp_path / "out.jsonl"
    rollouts.write_text('not json\n[1, 2]\n\n{"n": 1}\n', encoding="utf-8")
    progress = RecordingProgress()

    await RolloutFileFollower([rollouts, tmp_path / "missing.jsonl"], progress).report_new()

    assert progress.batches == [[{"n": 1}]]


async def test_a_progress_callback_that_raises_never_fails_the_report(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rollouts = tmp_path / "out.jsonl"
    rollouts.write_text('{"n": 1}\n', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        await RolloutFileFollower([rollouts], FailingProgress()).report_new()

    assert "Could not report Gym rollout progress" in caplog.text


async def test_collection_reports_rollouts_while_gym_is_still_running(tmp_path: Path) -> None:
    """Each record reaches progress as Gym writes it, and the last one before the run returns."""
    output = tmp_path / "out.jsonl"
    gym = tmp_path / "stub-gym"
    gym.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys, time\n"
        f"output = pathlib.Path({str(output)!r})\n"
        "failures = output.with_name(output.stem + '_failures.jsonl')\n"
        "with output.open('a') as file:\n"
        '    file.write(\'{"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0}\\n\')\n'
        "time.sleep(1)\n"
        "with failures.open('a') as file:\n"
        '    file.write(\'{"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_failure_class": "agent_error"}\\n\')\n',
        encoding="utf-8",
    )
    gym.chmod(0o755)
    progress = RecordingProgress()
    runner = GymAgentTaskRunner(config=_config(), progress=progress)
    runner.follow_interval_s = 0.05

    await runner._collect_rollouts(str(gym), [], tmp_path / "in.jsonl", output, tmp_path, {})

    assert [[record["_ng_task_index"] for record in batch] for batch in progress.batches] == [[0], [1]]


async def test_a_run_tells_progress_its_planned_rollouts_and_when_it_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = discover_gym_tasks(EXAMPLE)
    progress = RecordingProgress()
    runner = GymAgentTaskRunner(config=_config(num_repeats=3), progress=progress)

    async def stop(*args: object) -> None:
        raise RuntimeError("stop before Gym starts")

    monkeypatch.setattr(runner, "_run_two_step", stop)
    with pytest.raises(RuntimeError, match="stop before Gym starts"):
        await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    assert progress.planned_rollouts == [len(tasks) * 3]
    # Gym never started, so none of the planned rollouts will finish.
    assert progress.stops == 1


async def test_a_failing_progress_callback_never_fails_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = GymAgentTaskRunner(config=_config(), progress=FailingProgress())
    reached: list[bool] = []

    async def collect(*args: object) -> None:
        reached.append(True)
        raise RuntimeError("stop before Gym starts")

    monkeypatch.setattr(runner, "_run_two_step", collect)
    with pytest.raises(RuntimeError, match="stop before Gym starts"):
        await runner.run_tasks(discover_gym_tasks(EXAMPLE), AgentEvalRunConfig(work_dir=tmp_path))

    assert reached == [True]


async def test_a_rollout_file_that_cannot_be_read_still_reports_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    unreadable = tmp_path / "out.jsonl"
    unreadable.mkdir()  # opening a directory raises an OSError other than FileNotFoundError
    failures = tmp_path / "out_failures.jsonl"
    failures.write_text('{"n": 1}\n', encoding="utf-8")
    progress = RecordingProgress()

    with caplog.at_level(logging.WARNING):
        await RolloutFileFollower([unreadable, failures], progress).report_new()

    assert progress.batches == [[{"n": 1}]]
    assert "Could not read Gym's rollouts" in caplog.text
