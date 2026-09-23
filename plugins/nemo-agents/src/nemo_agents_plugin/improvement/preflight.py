# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight checks — fail fast with actionable errors before slow operations.

Each check raises ``PreflightError`` with a clear remediation message. Callers
should run preflights at the start of evaluate-suite / optimize-skills so users
don't wait for a docker build before learning ``harbor`` isn't installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class PreflightError(RuntimeError):
    """A preflight check failed. Message contains remediation guidance."""


def _which(name: str) -> str | None:
    return shutil.which(name)


def check_docker() -> None:
    """Verify ``docker`` CLI exists and the daemon is reachable."""
    if not _which("docker"):
        raise PreflightError(
            "'docker' not found on PATH. Install Docker Desktop / docker-ce "
            "(https://docs.docker.com/get-docker/) and ensure the daemon is running."
        )
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise PreflightError("'docker info' timed out — is the docker daemon running?") from exc
    if proc.returncode != 0:
        msg = proc.stderr.strip() or "docker daemon unreachable"
        raise PreflightError(
            f"'docker info' failed ({proc.returncode}): {msg}\nStart Docker Desktop or run: sudo systemctl start docker"
        )


def check_harbor() -> None:
    """Verify the ``harbor`` CLI is available."""
    if not _which("harbor"):
        raise PreflightError(
            "'harbor' not found on PATH. Install Harbor (see harbor docs) before running the Harbor eval runner."
        )


def check_nat_runner(project_root: Path) -> Path:
    """Verify the NAT runner script is discoverable and return its path."""
    import os

    env = os.environ.get("NHX_AGENTS_NAT_RUNNER")
    if env:
        p = Path(env)
        if not p.exists():
            raise PreflightError(
                f"NHX_AGENTS_NAT_RUNNER points to {p} but the file does not exist. "
                "Set it to the path of nat_runner.py (or unset it to use the default)."
            )
        return p
    p = project_root / "tests" / "agentic-use" / "nat_runner.py"
    if not p.exists():
        raise PreflightError(
            f"NAT runner not found at {p}. Either:\n"
            "  - set NHX_AGENTS_NAT_RUNNER=<path-to-nat_runner.py>, or\n"
            "  - place a NAT runner at tests/agentic-use/nat_runner.py, or\n"
            "  - use --runner harbor if your evals have task.toml files."
        )
    return p


def check_evals_dir(evals_dir: Path) -> None:
    """Verify the evals directory exists and contains at least one task."""
    if not evals_dir.exists():
        raise PreflightError(f"--evals path does not exist: {evals_dir}")
    if not evals_dir.is_dir():
        raise PreflightError(f"--evals must be a directory, got file: {evals_dir}")

    # Look for any task subdirectory with a marker file. Skip unreadable
    # children (e.g. systemd-private dirs under /tmp) rather than crash —
    # they're inherently not usable tasks.
    has_any = False
    try:
        children = list(evals_dir.iterdir())
    except OSError as exc:
        raise PreflightError(f"Cannot read --evals directory {evals_dir}: {exc}") from exc
    for child in children:
        try:
            if not child.is_dir():
                continue
            if (child / "task.toml").exists() or (child / "workflow.yml").exists():
                has_any = True
                break
        except OSError:
            continue
    if not has_any:
        raise PreflightError(
            f"No eval tasks found in {evals_dir}. Each subdirectory must contain "
            "either task.toml (Harbor) or workflow.yml (NAT) plus instruction.md "
            "and tests/test_outputs.py."
        )


def check_dockerfile(agent_root: Path, dockerfile_name: str = "Dockerfile.agentic-base") -> None:
    """Verify the agentic-base Dockerfile is present in the agent root."""
    p = agent_root / dockerfile_name
    if not p.exists():
        raise PreflightError(
            f"{dockerfile_name} not found at {p}. The Harbor runner expects this file "
            f"in the agent root. Either:\n"
            f"  - add {dockerfile_name} to your agent repo\n"
            f"  - or set --skip-build if you've pre-built the nhx-agentic-base:latest image."
        )


def check_skills_path(agent_root: Path, skills_path: str) -> None:
    """Verify the skills directory exists inside the agent root."""
    p = agent_root / skills_path
    if not p.exists():
        raise PreflightError(
            f"--skills-path resolves to {p} which does not exist. The optimize-skills "
            f"strategy needs to write skill files there. Either:\n"
            f"  - create the directory: mkdir -p {p}\n"
            f"  - or pass --skills-path pointing at an existing skills directory."
        )
    if not p.is_dir():
        raise PreflightError(f"--skills-path resolves to {p} which is a file, not a directory.")


def check_evals_inside_agent(agent_root: Path, evals_dir: Path) -> None:
    """Verify evals_dir is inside agent_root (loop invariant)."""
    try:
        evals_dir.resolve().relative_to(agent_root.resolve())
    except ValueError as exc:
        raise PreflightError(
            f"--evals ({evals_dir}) must be inside --agent ({agent_root}) for v0. "
            "The optimize-skills loop creates a worktree from agent_root and re-runs "
            "the evals from inside it; this requires the evals to live in the agent's repo."
        ) from exc


def check_forge_tooling() -> None:
    """Verify a forge CLI (``gh`` or ``glab``) is available for MR/PR creation.

    Only checks PATH presence, not auth state. Callers should gate this on
    whatever opt-in actually triggers MR/PR creation (e.g. ``open_pr=True``)
    so users who don't need it aren't forced to install one.
    """
    if not _which("gh") and not _which("glab"):
        raise PreflightError(
            "open_pr=True requires either 'gh' (GitHub) or 'glab' (GitLab) "
            "on PATH and authenticated. Install one and re-run, or unset "
            "open_pr to keep the branch local at the end of the loop."
        )


def check_anthropic_api() -> None:
    """Verify Anthropic API credentials are present in the environment.

    Used by callers that hit the Anthropic API directly — the LLM analyzer
    in optimize-skills and analyze-batch. The Claude CLI invoked by the
    loop's coding-agent step uses OAuth and is checked separately by
    ``ClaudeCodingAgent.preflight()``.

    Only checks for *presence* of credentials, not validity: an actual API
    call would slow down every invocation, cost tokens, and conflate
    "credential bad" with "API momentarily unreachable".
    """
    import os

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        raise PreflightError(
            "ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) is not set. The LLM "
            "analyzer needs it to generate gap-analysis hypotheses. Either:\n"
            "  - export ANTHROPIC_API_KEY=<your-key>, or\n"
            "  - if using a proxy: export ANTHROPIC_API_KEY=<proxy-key> "
            "ANTHROPIC_BASE_URL=<proxy-url>"
        )
