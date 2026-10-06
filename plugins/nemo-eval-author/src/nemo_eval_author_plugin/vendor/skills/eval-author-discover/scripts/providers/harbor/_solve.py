# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Harbor tasks end to end with Harbor's oracle agent.

The oracle agent runs a task's reference ``solution/solve.sh`` in the task's own
environment, then Harbor's verifier grades it. A task counts as run when the trial
finishes without an error and the verifier writes a reward. The reward's value is
secondary: Solve asks whether the eval machinery works, not whether the reference
solution is right.

Standard library only. This drives the ``harbor`` CLI the user would run, rather
than Harbor's Python API, so it proves the same command the report recommends.
Job output goes to a caller-owned directory outside the repository. Harbor never
auto-grants host-environment access here: ``stdin`` is closed and ``-y`` is not
passed, so a task that asks for access fails with Harbor's own message.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_TIMEOUT_SEC = 1800
_BRIEF_CHARS = 300


@dataclass(frozen=True)
class TaskRun:
    """One oracle trial: ``rewards`` is what the verifier wrote, ``error`` why the trial did not finish."""

    task: Path
    rewards: dict[str, float] | None
    error: str | None

    @property
    def completed(self) -> bool:
        return self.error is None

    @property
    def brief_error(self) -> str | None:
        """The telling line of ``error``: Harbor's errors can carry a whole Docker build log."""
        if self.error is None:
            return None
        lines = [line.strip() for line in self.error.splitlines() if line.strip()]
        telling = [line for line in lines if "error" in line.lower() or "failed" in line.lower()]
        brief = telling[-1] if len(lines) > 1 and telling else lines[0]
        return brief if len(brief) <= _BRIEF_CHARS else brief[: _BRIEF_CHARS - 3] + "..."


def has_solution(task_dir: Path) -> bool:
    """Return whether the oracle agent has a reference solution to run."""
    return (task_dir / "solution" / "solve.sh").is_file()


def command(repo_root: Path, task: str) -> str:
    """Return the command that reproduces one oracle trial."""
    return "cd {} && harbor run -p {} -a oracle".format(repo_root, task)


def run_oracle(harbor: str, task_dir: Path, repo_root: Path, jobs_root: Path) -> TaskRun:
    """Run one task with the oracle agent and read the trial Harbor recorded."""
    # Tasks in different datasets share names, so each run gets its own jobs directory.
    jobs_dir = Path(tempfile.mkdtemp(dir=jobs_root))
    job_name = "solve"
    try:
        completed = subprocess.run(
            [harbor, "run", "-p", str(task_dir), "-a", "oracle", "-o", str(jobs_dir), "--job-name", job_name, "-q"],
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TaskRun(task_dir, None, "Harbor did not finish within {} minutes.".format(_TIMEOUT_SEC // 60))
    except OSError as exc:
        return TaskRun(task_dir, None, "Cannot start Harbor: {}".format(exc))

    trials = sorted((jobs_dir / job_name).glob("*/result.json"))
    if not trials:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        return TaskRun(
            task_dir,
            None,
            "Harbor exited with code {} without recording a trial: {}".format(
                completed.returncode, detail[-1] if detail else "no output"
            ),
        )
    try:
        trial = json.loads(trials[0].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return TaskRun(task_dir, None, "Cannot read Harbor's trial result: {}".format(exc))

    rewards = (trial.get("verifier_result") or {}).get("rewards")
    exception = trial.get("exception_info")
    if exception:
        error = "{}: {}".format(exception.get("exception_type"), exception.get("exception_message"))
    elif not rewards:
        error = "The verifier finished without writing a reward."
    else:
        error = None
    return TaskRun(task_dir, rewards, error)
