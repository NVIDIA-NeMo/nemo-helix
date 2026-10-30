# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend-neutral half of telling why a sandbox stopped, against a fake backend."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from fnmatch import fnmatch
from pathlib import Path

import pytest
from nhx_sandbox.core.errors import SandboxBackendError, SandboxNotFound
from nhx_sandbox.core.termination import (
    TERMINATION_GLOB,
    SandboxTermination,
    TerminationCause,
    TerminationDiagnosis,
    check_termination,
    diagnose_gone,
    termination_filename,
    write_termination,
)

LOGGER = logging.getLogger(__name__)
NOW = datetime(2026, 10, 30, 20, 0, tzinfo=UTC)


class FakeDiagnostics:
    """A backend that answers with ``answer``: a diagnosis, ``None`` for alive, or an error to raise."""

    backend = "fake"

    def __init__(self, answer: TerminationDiagnosis | BaseException | None, *, delay_sec: float = 0) -> None:
        self._answer = answer
        self._delay_sec = delay_sec

    async def diagnose_termination(self, sandbox_id: str) -> TerminationDiagnosis | None:
        await asyncio.sleep(self._delay_sec)
        if isinstance(self._answer, BaseException):
            raise self._answer
        return self._answer


async def _check(diagnostics: FakeDiagnostics, *, ends_at: datetime | None = None) -> SandboxTermination | None:
    return await check_termination(diagnostics, "sbx-1", ends_at=ends_at, role="agent", budget_sec=1, logger=LOGGER)


async def test_an_alive_sandbox_has_no_termination() -> None:
    assert await _check(FakeDiagnostics(None)) is None


async def test_the_backends_diagnosis_is_recorded_with_its_name() -> None:
    diagnosis = TerminationDiagnosis(TerminationCause.EVICTED, "evicted for disk", {"phase": "ERROR"})

    termination = await _check(FakeDiagnostics(diagnosis))

    assert termination is not None
    assert (termination.backend, termination.sandbox_id, termination.role) == ("fake", "sbx-1", "agent")
    assert (termination.cause, termination.detail, termination.evidence) == (
        "evicted",
        "evicted for disk",
        {"phase": "ERROR"},
    )


async def test_a_sandbox_the_backend_no_longer_knows_is_judged_by_its_lifetime() -> None:
    """Shared, so a backend with no lifetime of its own (its sandboxes reaped by label) gets the same answer."""
    gone = FakeDiagnostics(SandboxNotFound("no such sandbox"))

    before = await _check(gone, ends_at=datetime.now(UTC) + timedelta(hours=1))
    after = await _check(gone, ends_at=datetime.now(UTC) - timedelta(seconds=1))

    assert before is not None and before.cause is TerminationCause.DELETED
    assert after is not None and after.cause is TerminationCause.LIFETIME_REACHED


async def test_a_backend_that_cant_say_records_nothing(caplog: pytest.LogCaptureFixture) -> None:
    assert await _check(FakeDiagnostics(SandboxBackendError("server unavailable"))) is None
    assert "Could not check whether sandbox sbx-1 stopped early" in caplog.text


async def test_a_slow_backend_gives_up_at_the_budget() -> None:
    started = asyncio.get_running_loop().time()

    termination = await check_termination(
        FakeDiagnostics(None, delay_sec=10), "sbx-1", ends_at=None, role=None, budget_sec=0.05, logger=LOGGER
    )

    assert termination is None
    assert asyncio.get_running_loop().time() - started < 5


def test_gone_with_no_known_lifetime_was_deleted() -> None:
    assert diagnose_gone(ends_at=None, now=NOW).cause is TerminationCause.DELETED


def test_a_record_round_trips_through_its_file(tmp_path: Path) -> None:
    termination = SandboxTermination(
        backend="fake", sandbox_id="a/b", cause=TerminationCause.UNKNOWN, detail="stopped", observed_at=NOW
    )

    path = write_termination(tmp_path / "trial", termination)

    assert path.name == "nemo-sandbox-termination-a_b.json"
    assert SandboxTermination.model_validate(json.loads(path.read_text())) == termination
    assert list(path.parent.iterdir()) == [path]


def test_the_trial_adapter_reads_the_files_this_writes() -> None:
    """The SDK's trial adapter matches these files by name without importing nhx_sandbox; this keeps the two in step."""
    trial_adapter = pytest.importorskip("nhx_evals_sdk.agent_eval.runtimes.harbor.trial_adapter")

    assert trial_adapter._SANDBOX_TERMINATION_GLOB == TERMINATION_GLOB
    assert fnmatch(termination_filename("41747334-5b52-4ea3-aab1-203bdd3ad86c"), TERMINATION_GLOB)
