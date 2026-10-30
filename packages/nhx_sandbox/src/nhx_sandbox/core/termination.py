# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Why a sandbox stopped before its caller was done with it, whatever the backend.

A sandbox that dies mid-trial (out of memory, evicted, out of lifetime, deleted) surfaces as
whatever operation happened to run next, such as a failed download, which names neither the
sandbox nor the cause. Callers check once, when they release the sandbox, and write the result
next to their own output; NeMo's Harbor environment does it in ``stop``.

Each backend reads its own status and diagnoses the cause (``TerminationDiagnostics``). What is
the same on every backend lives here: the causes, the record and its file, telling a sandbox whose
lifetime ended from one deleted early, and a check that never raises and never waits long.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from nhx_sandbox.core.errors import SandboxNotFound
from pydantic import BaseModel, ConfigDict, Field

# The trial adapter in nhx_evals_sdk matches this name without importing nhx_sandbox, which the
# published SDK doesn't ship; keep the two in step.
TERMINATION_FILENAME_PREFIX = "nemo-sandbox-termination"
TERMINATION_GLOB = f"{TERMINATION_FILENAME_PREFIX}-*.json"


class TerminationCause(StrEnum):
    """Why a sandbox stopped. ``UNKNOWN`` when the backend reported it stopped but not why."""

    LIFETIME_REACHED = "lifetime_reached"
    DELETED = "deleted"
    EVICTED = "evicted"
    OUT_OF_MEMORY = "out_of_memory"
    EXITED = "exited"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TerminationDiagnosis:
    """A backend's diagnosis of a sandbox that stopped."""

    cause: TerminationCause
    # One line for people: the cause and the backend's own words for it.
    detail: str
    # What the backend reported, in its own terms and JSON-serializable, kept as evidence.
    evidence: dict[str, Any] = field(default_factory=dict)


class TerminationDiagnostics(Protocol):
    """One backend's way of telling why a sandbox stopped."""

    #: The backend's name, recorded with each termination.
    backend: str

    async def diagnose_termination(self, sandbox_id: str) -> TerminationDiagnosis | None:
        """The diagnosis for a sandbox that was ready once, or ``None`` while it is still alive.

        Raises ``SandboxNotFound`` once the backend no longer knows the sandbox; the shared layer
        tells a lifetime that ran out from an early deletion. Any other error means the backend
        couldn't say.
        """
        ...


class SandboxTermination(BaseModel):
    """One sandbox that stopped before its caller released it, and why, as far as its backend could tell."""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(description="The sandbox backend, such as `opensandbox`.")
    sandbox_id: str = Field(description="The backend's ID for the sandbox.")
    role: str | None = Field(
        default=None,
        description="What the sandbox was for, as its caller names it; Harbor's are `agent` and `verifier`.",
    )
    cause: TerminationCause = Field(description="Why it stopped; `unknown` when the backend couldn't say why.")
    detail: str = Field(description="One line for people: the cause and the backend's own words for it.")
    evidence: dict[str, Any] = Field(
        default_factory=dict, description="What the backend reported, in its own terms, such as its state and events."
    )
    observed_at: datetime = Field(description="When the sandbox's state was read, in UTC.")


def diagnose_gone(*, ends_at: datetime | None, now: datetime) -> TerminationDiagnosis:
    """The diagnosis for a sandbox the backend no longer knows: its lifetime ran out, or something deleted it.

    ``ends_at`` is the latest the sandbox could have lived; a backend without its own lifetime
    still has one, enforced by a reaper.
    """
    if ends_at is not None and now >= ends_at:
        return TerminationDiagnosis(
            TerminationCause.LIFETIME_REACHED, f"the sandbox reached the end of its lifetime at {ends_at.isoformat()}"
        )
    return TerminationDiagnosis(TerminationCause.DELETED, "the sandbox was deleted before its caller was done with it")


async def check_termination(
    diagnostics: TerminationDiagnostics,
    sandbox_id: str,
    *,
    ends_at: datetime | None,
    role: str | None,
    budget_sec: float,
    logger: logging.Logger,
    wait_for_stop: bool = False,
    poll_interval_sec: float = 1.0,
) -> SandboxTermination | None:
    """Whether ``sandbox_id`` already stopped, and why; ``None`` while it is alive or when the backend can't say.

    With ``wait_for_stop``, for a caller that saw the sandbox die (a command's output cut off),
    an alive answer is asked again every ``poll_interval_sec``: a backend's status lags a dead
    sandbox by seconds, and the caller releases it as soon as the error reaches it. A sandbox
    still alive at the end is recorded as nothing, since a dropped connection cuts output off too.

    Never raises and gives up after ``budget_sec``, so a caller releasing the sandbox always goes on.
    """
    try:
        async with asyncio.timeout(budget_sec):
            while (diagnosis := await diagnostics.diagnose_termination(sandbox_id)) is None and wait_for_stop:
                await asyncio.sleep(poll_interval_sec)
    except SandboxNotFound:
        diagnosis = diagnose_gone(ends_at=ends_at, now=datetime.now(UTC))
    except TimeoutError:
        logger.warning(
            "Sandbox %s was not reported stopped within %gs; recording no cause", sandbox_id, budget_sec, exc_info=True
        )
        return None
    except Exception:
        logger.warning("Could not check whether sandbox %s stopped early", sandbox_id, exc_info=True)
        return None
    if diagnosis is None:
        return None
    observed_at = datetime.now(UTC)
    return SandboxTermination(
        backend=diagnostics.backend,
        sandbox_id=sandbox_id,
        role=role,
        cause=diagnosis.cause,
        detail=diagnosis.detail,
        evidence=diagnosis.evidence,
        observed_at=observed_at,
    )


def termination_filename(sandbox_id: str) -> str:
    """Name of one sandbox's termination record."""
    safe = "".join(char if char.isalnum() or char in "-._" else "_" for char in sandbox_id)
    return f"{TERMINATION_FILENAME_PREFIX}-{safe}.json"


def write_termination(directory: Path, termination: SandboxTermination) -> Path:
    """Write ``termination`` into ``directory`` in one rename, so a reader never sees half a record."""
    path = directory / termination_filename(termination.sandbox_id)
    temporary = path.with_name(f"{path.name}.tmp")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(termination.model_dump_json(indent=2))
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return path
