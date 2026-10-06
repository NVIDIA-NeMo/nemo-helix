# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a Gym manifest workload's verifier fixture with NeMo Gym's own CLI.

``gym env test <name> --kind <kind> --json`` scores the workload's recorded
verifier cases offline: no model, no servers, no containers. It is the closest
thing Gym has to Harbor's oracle run, so it proves the resources server installs
and its verifier grades.

Standard library only. Gym builds the resources server's venv where it always
does, ``resources_servers/<name>/.venv``, and offers no way to put it elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from providers.gym._judge import GymRun, run_gym


def command(repo_root: Path, name: str, kind: str) -> str:
    """Return the command that reproduces one fixture run."""
    return "cd {} && gym env test {} --kind {}".format(repo_root, name, kind)


def run_fixture(gym: str, name: str, kind: str, repo_root: Path) -> GymRun:
    """Score one workload's verifier fixture; the report's ``cases`` holds each case Gym scored."""
    return run_gym(gym, ["env", "test", name, "--kind", kind, "--json"], repo_root)


def cases(run: GymRun) -> list[dict]:
    """Return the verifier cases a fixture run scored."""
    found = (run.report or {}).get("cases")
    return found if isinstance(found, list) else []
