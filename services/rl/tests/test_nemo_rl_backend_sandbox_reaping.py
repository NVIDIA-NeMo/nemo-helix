# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GRPO step reaps the sandboxes its job created, however the driver ends.

The Gym sandbox is created by a Ray actor, and Ray kills that actor without running its teardown
when a job is cancelled. The training task is the process the container runtime signals, so it has
to name and delete the job's sandboxes itself.
"""

from __future__ import annotations

import importlib.util
import signal
import sys
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from nhx.customization_common.service.context import NHXJobContext
from nhx.customization_common.training.errors.exceptions import CustomizerTrainingError
from nhx.rl.entities.values import TrainingType
from nhx.rl.tasks.training.protocol import LibraryConfig

# NeMo-RL is only installed in the training image; stub the one module backend imports from it.
_stubbed: list[str] = []
if importlib.util.find_spec("nemo_rl") is None:  # pragma: no cover - env dependent
    _checkpoint_mod = types.ModuleType("nemo_rl.utils.checkpoint")
    setattr(_checkpoint_mod, "CheckpointingConfig", dict)
    setattr(_checkpoint_mod, "CheckpointManager", object)
    for _name, _module in (
        ("nemo_rl", types.ModuleType("nemo_rl")),
        ("nemo_rl.utils", types.ModuleType("nemo_rl.utils")),
        ("nemo_rl.utils.checkpoint", _checkpoint_mod),
    ):
        if _name not in sys.modules:
            sys.modules[_name] = _module
            _stubbed.append(_name)

from nhx.rl.tasks.training.backends.nemo_rl import backend as backend_module  # noqa: E402

for _name in _stubbed:
    del sys.modules[_name]

JOB_ID = "rl-job-1"


class _Recorder:
    """Records reaper and driver-termination calls in the order they happen, on a fake clock."""

    def __init__(self) -> None:
        """Start with no recorded events or calls, and a clock that only the fakes advance."""
        self.events: list[str] = []
        self.reap_calls: list[dict[str, Any]] = []
        self.terminate_calls: list[tuple[int, int]] = []
        self.signal_actions_during_sweep: list[tuple[Any, Any]] = []
        self.reap_error: Exception | None = None
        self.now = 1000.0
        self.sweep_cost_s = 0.0
        self.driver_uses_whole_wait = False

    def monotonic(self) -> float:
        """The fake clock the backend reads."""
        return self.now

    def terminate(self, signum: int, timeout: int = 30) -> None:
        """Record the driver termination, spending its whole wait when asked to."""
        self.events.append("terminate_driver")
        self.terminate_calls.append((signum, timeout))
        if self.driver_uses_whole_wait:
            self.now += timeout

    def reap(self, job_id: str, **kwargs: Any) -> tuple[str, ...]:
        """Record a sweep, spend its cost on the clock, and raise the configured error if any."""
        self.events.append("reap")
        self.reap_calls.append({"job_id": job_id, **kwargs})
        self.signal_actions_during_sweep.append((signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)))
        self.now += self.sweep_cost_s
        if self.reap_error is not None:
            raise self.reap_error
        return ()


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Install a fake ``sandboxed_gym.job_reaper`` and a fake clock on the backend module."""
    recorded = _Recorder()
    module = types.ModuleType("sandboxed_gym.job_reaper")
    setattr(module, "reap_job_sandboxes", recorded.reap)
    monkeypatch.setitem(sys.modules, "sandboxed_gym", types.ModuleType("sandboxed_gym"))
    monkeypatch.setitem(sys.modules, "sandboxed_gym.job_reaper", module)
    # Replace the backend's own ``time`` name rather than the shared stdlib module.
    monkeypatch.setattr(backend_module, "time", SimpleNamespace(monotonic=recorded.monotonic))
    return recorded


def _execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorder: _Recorder,
    *,
    exit_code: int = 0,
    gym: Any = None,
    during_driver: Any = None,
) -> None:
    """Run ``execute_training`` against a fake Ray bootstrap, with an optional hook while the driver runs."""

    def run_with_driver(script: str, args: list[str]) -> int:
        """Run the in-flight hook, then return the configured exit code."""
        if during_driver is not None:
            during_driver()
        return exit_code

    bootstrap = SimpleNamespace(
        progress=None,
        node_env={},
        driver_output=deque(["RuntimeError: driver failed"]),
        run_with_driver=run_with_driver,
        terminate_driver=recorder.terminate,
    )
    monkeypatch.setattr(backend_module, "create_bootstrap_from_env", lambda: bootstrap)
    backend = backend_module.NemoRLBackend(cast(NHXJobContext, SimpleNamespace(job_id=JOB_ID)))
    config: Any = SimpleNamespace(
        parallelism=SimpleNamespace(num_gpus_per_node=1, num_nodes=1),
        integrations=None,
        model=SimpleNamespace(name="out"),
        training=SimpleNamespace(training_type=TrainingType.GRPO),
        gym=gym,
    )
    library = LibraryConfig(config_dict={}, config_path=tmp_path / "nemo_rl_config.yaml")
    progress: Any = SimpleNamespace(report_running=lambda *args, **kwargs: None)
    backend.execute_training(config, library, progress)


def _sandboxed_gym(**overrides: Any) -> SimpleNamespace:
    """A Gym config that runs in a sandbox, with optional overrides."""
    values: dict[str, Any] = {"sandboxed": True, "sandbox_server_protocol": "http", "sandbox_resources": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_a_finished_sandboxed_job_is_swept_with_its_own_job_id_and_connection_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """A normal exit sweeps once with the job id and connection options."""
    _execute(tmp_path, monkeypatch, recorder, gym=_sandboxed_gym())

    assert recorder.reap_calls == [
        {
            "job_id": JOB_ID,
            "host_provider": "opensandbox",
            "host_provider_options": {"connection": {"protocol": "http"}},
            "timeout_s": backend_module.SANDBOX_REAP_TIMEOUT_S,
        }
    ]


def test_a_driver_crash_still_sweeps_and_still_reports_the_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """A crashing driver is still swept and its failure is still raised."""
    with pytest.raises(CustomizerTrainingError):
        _execute(tmp_path, monkeypatch, recorder, exit_code=1, gym=_sandboxed_gym())

    assert len(recorder.reap_calls) == 1


def test_a_cancel_reaps_before_it_waits_on_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """The driver wait can use the whole termination grace period, so the sweep has to come first."""

    def cancel() -> None:
        """Simulate a cancel arriving while the driver runs."""
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)

    with pytest.raises(SystemExit):
        _execute(tmp_path, monkeypatch, recorder, gym=_sandboxed_gym(), during_driver=cancel)

    # The handler reaps then terminates the driver, then the ``finally`` sweeps once more for
    # anything created while the driver was shutting down.
    assert recorder.events == ["reap", "terminate_driver", "reap"]


@pytest.mark.parametrize("gym", [None, _sandboxed_gym(sandboxed=False)], ids=["no-gym", "not-sandboxed"])
def test_jobs_without_a_sandbox_are_not_swept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder, gym: Any
) -> None:
    """Jobs with no sandboxed Gym have nothing to sweep."""
    _execute(tmp_path, monkeypatch, recorder, gym=gym)

    assert recorder.reap_calls == []


def test_a_failing_sweep_does_not_change_the_training_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """A sweep that raises is logged and the run still succeeds."""
    recorder.reap_error = RuntimeError("opensandbox unreachable")

    _execute(tmp_path, monkeypatch, recorder, gym=_sandboxed_gym())

    assert len(recorder.reap_calls) == 1


def test_a_task_environment_without_the_reaper_module_degrades_to_a_logged_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """An image that ships an older sandboxed_gym must not turn a clean run into a failure."""
    monkeypatch.setitem(sys.modules, "sandboxed_gym.job_reaper", None)

    _execute(tmp_path, monkeypatch, recorder, gym=_sandboxed_gym())

    assert recorder.reap_calls == []


def _cancel() -> None:
    """Deliver a SIGTERM to the handler the backend installed."""
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)
    handler(signal.SIGTERM, None)


def test_a_cancel_restores_default_signal_actions_before_any_blocking_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """A second signal must end the process, not re-enter the handler mid-teardown."""
    with pytest.raises(SystemExit):
        _execute(tmp_path, monkeypatch, recorder, gym=_sandboxed_gym(), during_driver=_cancel)

    # The first sweep runs inside the handler, before it waits on the driver.
    assert recorder.signal_actions_during_sweep[0] == (signal.SIG_DFL, signal.SIG_DFL)


def test_a_cancel_shares_one_shutdown_budget_between_the_sweeps_and_the_driver_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """Every sweep and the driver wait are bounded by what is left of one deadline."""
    recorder.sweep_cost_s = 8.0
    recorder.driver_uses_whole_wait = True

    with pytest.raises(SystemExit):
        _execute(tmp_path, monkeypatch, recorder, gym=_sandboxed_gym(), during_driver=_cancel)

    first, second = recorder.reap_calls
    assert first["timeout_s"] == backend_module.SANDBOX_REAP_TIMEOUT_S
    # 25s budget - 8s spent - 5s kill allowance = 12s for the driver to exit.
    assert recorder.terminate_calls == [(signal.SIGTERM, 12)]
    # The driver used its 12s after the first 8s sweep, leaving 5s for the second.
    assert second["timeout_s"] == pytest.approx(5.0)


def test_a_sweep_after_the_shutdown_budget_is_spent_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    """The final sweep must not push teardown past the grace period."""
    recorder.sweep_cost_s = backend_module.SHUTDOWN_GRACE_S + 1

    with pytest.raises(SystemExit):
        _execute(tmp_path, monkeypatch, recorder, gym=_sandboxed_gym(), during_driver=_cancel)

    assert len(recorder.reap_calls) == 1
