# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import types
from typing import Any, cast

import pytest
from nhx.customization_common.service.context import NHXJobContext
from nhx.rl.tasks.training.backends.nemo_rl import pause as pause_module
from nhx.rl.tasks.training.backends.nemo_rl.pause import PauseStop, planned_steps, stop_for_pause


class _TimeoutChecker:
    """The shape of NeMo-RL's checker: ``check_save`` asks ``would_save``, then latches."""

    def __init__(self, timeout: str | None = None, fit_last_save_time: bool = False) -> None:
        self.last_saved = False
        self.deadline_passed = False

    def would_save(self) -> bool:
        return not self.last_saved and self.deadline_passed

    def check_save(self) -> bool:
        if not self.would_save():
            return False
        self.last_saved = True
        return True


@pytest.fixture
def algorithm() -> types.ModuleType:
    module = types.ModuleType("fake_algorithm")
    setattr(module, "TimeoutChecker", _TimeoutChecker)
    return module


def test_a_requested_pause_makes_the_loop_save_once(algorithm: types.ModuleType) -> None:
    requested = threading.Event()
    stop = PauseStop(requested)

    with stop.patch(algorithm):
        checker = algorithm.TimeoutChecker(timeout=None, fit_last_save_time=True)
        assert checker.check_save() is False
        requested.set()
        assert checker.would_save() is True
        assert checker.check_save() is True
        # The loop returns after this save; the latch keeps a second check from saving again.
        assert checker.check_save() is False

    assert stop.saved is True


def test_the_deadline_save_still_works_and_is_not_a_pause(algorithm: types.ModuleType) -> None:
    stop = PauseStop(threading.Event())

    with stop.patch(algorithm):
        checker = algorithm.TimeoutChecker()
        checker.deadline_passed = True
        assert checker.check_save() is True

    assert stop.saved is False


def test_the_original_checker_is_restored_even_when_training_fails(algorithm: types.ModuleType) -> None:
    with pytest.raises(RuntimeError, match="boom"), PauseStop(threading.Event()).patch(algorithm):
        assert algorithm.TimeoutChecker is not _TimeoutChecker
        raise RuntimeError("boom")

    assert algorithm.TimeoutChecker is _TimeoutChecker


@pytest.mark.parametrize(
    "saved,completed,planned,expected",
    [
        (True, 12, 30, True),
        # The pause came at the last step: training finished, so it publishes.
        (True, 30, 30, False),
        (False, 12, 30, False),
    ],
)
def test_stopped_training(saved: bool, completed: int, planned: int, expected: bool) -> None:
    stop = PauseStop(threading.Event())
    stop.saved = saved

    assert stop.stopped_training(completed, planned) is expected


@pytest.mark.parametrize(
    "max_num_steps,max_num_epochs,steps_per_epoch,expected",
    [
        (100, 3, 20, 60),
        (50, 3, 20, 50),
        # Epochs do not bound the run.
        (50, 3, None, 50),
    ],
)
def test_planned_steps(max_num_steps: int, max_num_epochs: int, steps_per_epoch: int | None, expected: int) -> None:
    assert planned_steps(max_num_steps, max_num_epochs, steps_per_epoch) == expected


def test_stop_for_pause_watches_the_step_while_training_runs(
    algorithm: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    pollers: list[Any] = []

    class _Poller:
        def __init__(self, job_ctx: NHXJobContext, service_name: str) -> None:
            self.service_name = service_name
            self.requested = threading.Event()
            self.stopped = False
            pollers.append(self)

        def __enter__(self) -> "_Poller":
            return self

        def __exit__(self, *exc: object) -> None:
            self.stopped = True

    monkeypatch.setattr(pause_module, "PausePoller", _Poller)

    with stop_for_pause(algorithm, cast(NHXJobContext, object())) as stop:
        (poller,) = pollers
        assert poller.service_name == "rl" and not poller.stopped
        poller.requested.set()
        assert algorithm.TimeoutChecker().check_save() is True

    assert stop.saved is True
    assert poller.stopped
    assert algorithm.TimeoutChecker is _TimeoutChecker
