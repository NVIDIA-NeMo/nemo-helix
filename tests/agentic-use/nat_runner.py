#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT-based task runner for agentic-use evals.

Replaces ``harbor run`` as the orchestrator for the NeMo Helix agentic benchmark tasks.
The runner mirrors Harbor's three-phase execution model:

  1. BUILD  – Build the Docker image from each task's ``environment/Dockerfile``
               (which inherits from ``nhx-agentic-base:latest``).
  2. AGENT  – Run the task instruction via one of several backends:
               - ``aut`` (default): invoke a deployed platform agent-under-test
                 using ``nemo agents invoke``.
               - ``workflow``: run ``nat run`` with the task-local workflow.
               - ``claude-code``: run Claude Code directly against instruction.md.
               - ``codex``: run Codex CLI headlessly against instruction.md.
               - ``cursor-agent``: run Cursor Agent headlessly against instruction.md.
  3. VERIFY – Run pytest against ``tests/test_outputs.py`` inside the container
               and write a ``reward.txt`` (1 = pass, 0 = fail) to the output dir.

Usage
-----
    # Build the base image once:
    docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .

    # Run a single task:
    python tests/agentic-use/nat_runner.py workspace-basic-mcp

    # Run multiple tasks (sequential):
    python tests/agentic-use/nat_runner.py workspace-basic-mcp secrets-crud-cli

    # Run all tasks matching a glob:
    python tests/agentic-use/nat_runner.py "*-easy"

    # Override the model used by the selected agent backend:
    NAT_AGENT_MODEL=meta/llama-3.1-70b-instruct \\
    python tests/agentic-use/nat_runner.py workspace-basic-mcp

Required environment variables
-------------------------------
    NVIDIA_API_KEY      API key for NVIDIA NIM inference (https://build.nvidia.com).
                        Used by seed_providers.py to create platform secrets.
    ANTHROPIC_API_KEY   Anthropic-compatible API key.
                        Required for ``claude-code`` backend.
    OPENAI_API_KEY      OpenAI API key.
                        Used by ``codex`` backend. If unset, pass
                        ``--codex-auth-json`` explicitly.
    CURSOR_API_KEY      Cursor API key.
                        Required for ``cursor-agent`` backend.

Optional environment variables
-------------------------------
    INFERENCE_NVIDIA_API_KEY  LiteLLM virtual key for inference-api.nvidia.com.
                     Used by seed_providers.py to create the inference
                     provider secret. If unset, the nvidia-inference-api
                     provider is skipped during seeding.
    NHX_BASE_URL     URL of the NeMo Helix API inside the container.
                     Defaults to http://localhost:8080.
    NAT_AGENT_MODEL  Optional candidate-agent model for any backend.
    ANTHROPIC_BASE_URL  Anthropic-compatible endpoint for ``claude-code``.
                     Defaults to https://inference-api.nvidia.com.
    NAT_RUNNER_JOBS_DIR  Directory for run artifacts (logs, reward files).
                     Defaults to ./nat-jobs/ in the repo root.
    NAT_TIMEOUT      Timeout in seconds for the agent phase. Defaults to 600.
    DOCKER_EXTRA_ARGS  Additional arguments passed to ``docker run``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
import stat
import subprocess
import sys
import textwrap
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = Path(__file__).resolve().parent
SHARED_DIR = TASKS_DIR / "shared"
CODEX_AGENT_SCRIPT_TEMPLATE_PATH = TASKS_DIR / "scripts" / "codex_agent_runner.sh"
NAT_TRACE_EXPORT_SCRIPT_CONTAINER_PATH = "/app/tests/agentic-use/scripts/nat_trace_export.py"

DEFAULT_TIMEOUT = int(os.environ.get("NAT_TIMEOUT", "600"))
DEFAULT_JOBS_DIR = REPO_ROOT / "nat-jobs"
DEFAULT_LOCAL_NHX_BASE_URL = "http://localhost:8080"
DEFAULT_AUT_AGENT_CONFIG: str | None = None
FILES_STORAGE_CONFIG = '{"type":"local","path":"/data/files_storage"}'
PLATFORM_CONFIG_PATH = "/app/packages/nhx_platform/config/local.yaml"
DOCKER_SOCKET_HOST_PATH = Path("/var/run/docker.sock")
DOCKER_SOCKET_CONTAINER_PATH = "/var/run/docker.sock"
PLACEHOLDER_SECRET_VALUES = {"null", "none"}
CONTAINER_WRITABLE_DIR_MODE = 0o777


def _normalize_secret(value: str | None) -> str:
    """Normalize optional secret env values before validation or injection."""
    normalized = (value or "").strip()
    if normalized.lower() in PLACEHOLDER_SECRET_VALUES:
        return ""
    return normalized


def _secret_from_env(name: str) -> str:
    return _normalize_secret(os.environ.get(name))


def _ensure_container_writable_dir(path: Path) -> None:
    """Create a bind-mount directory writable by container users with mismatched UID/GID."""
    path.mkdir(parents=True, exist_ok=True)
    current_mode = stat.S_IMODE(path.stat().st_mode)
    desired_mode = current_mode | CONTAINER_WRITABLE_DIR_MODE
    if desired_mode != current_mode:
        path.chmod(desired_mode)


_CANDIDATE_PARAM_STRING_KEYS = frozenset(
    {
        "permission_mode",
        "intelligence",
        "speed",
        "sandbox",
        "mode",
    }
)
_CANDIDATE_PARAM_NUMBER_KEYS = frozenset({"max_budget_usd"})
_CANDIDATE_PARAM_OBJECT_KEYS = frozenset({"config"})
_CANDIDATE_PARAM_KEYS = _CANDIDATE_PARAM_STRING_KEYS | _CANDIDATE_PARAM_NUMBER_KEYS | _CANDIDATE_PARAM_OBJECT_KEYS


def _coerce_candidate_number_param(key: str, value: object) -> int | float:
    if isinstance(value, bool):
        raise ValueError(f"--candidate-params key {key!r} must be numeric")
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"--candidate-params key {key!r} must be numeric") from exc
    raise ValueError(f"--candidate-params key {key!r} must be numeric")


def _parse_candidate_params(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--candidate-params must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--candidate-params must be a JSON object")
    unknown_keys = sorted(set(parsed) - _CANDIDATE_PARAM_KEYS)
    if unknown_keys:
        raise ValueError(f"--candidate-params includes unsupported key(s): {', '.join(unknown_keys)}")
    normalized: dict[str, Any] = {}
    for key, raw_value in parsed.items():
        if key in _CANDIDATE_PARAM_STRING_KEYS:
            if not isinstance(raw_value, str):
                raise ValueError(f"--candidate-params key {key!r} must be a string")
            normalized[key] = raw_value
        elif key in _CANDIDATE_PARAM_NUMBER_KEYS:
            normalized[key] = _coerce_candidate_number_param(key, raw_value)
        elif key in _CANDIDATE_PARAM_OBJECT_KEYS:
            if not isinstance(raw_value, dict):
                raise ValueError(f"--candidate-params key {key!r} must be a JSON object")
            normalized[key] = dict(raw_value)
    return normalized


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, check: bool = True, capture: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Thin wrapper around subprocess.run with consistent logging."""
    print(f"[nat_runner] $ {' '.join(_redact_cmd_for_logging(cmd))}")
    result = subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        **kwargs,
    )
    return result


def _redact_cmd_for_logging(cmd: list[str]) -> list[str]:
    """Redact secret values in command logs.

    This primarily protects ``docker run -e KEY=VALUE`` arguments from leaking
    API keys/tokens to stdout while preserving the command shape for debugging.
    """
    redacted: list[str] = []
    sensitive_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    for token in cmd:
        if "=" not in token:
            redacted.append(token)
            continue
        left, right = token.split("=", 1)
        # Handle "-e NAME=value" style tokens (left can include the env key).
        env_key = left.split()[-1] if left else left
        if any(marker in env_key.upper() for marker in sensitive_markers):
            redacted.append(f"{left}=***REDACTED***")
        else:
            redacted.append(f"{left}={right}")
    return redacted


def build_task_image(task_dir: Path, tag: str) -> None:
    """Build a task-specific Docker image from environment/Dockerfile."""
    env_dockerfile = task_dir / "environment" / "Dockerfile"
    if not env_dockerfile.exists():
        raise FileNotFoundError(f"No environment/Dockerfile found in {task_dir}")
    _run(
        [
            "docker",
            "build",
            "-f",
            str(env_dockerfile),
            "-t",
            tag,
            str(env_dockerfile.parent),
        ]
    )


def _docker_run(
    image: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    mounts: list[tuple[str, str]] | None = None,
    workdir: str | None = None,
    remove: bool = True,
    timeout: int | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a command inside a Docker container."""
    cmd = ["docker", "run"]
    if remove:
        cmd.append("--rm")
    if workdir:
        cmd += ["-w", workdir]

    # Pass environment variables
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]

    # Bind mounts
    for host_path, container_path in mounts or []:
        cmd += ["-v", f"{host_path}:{container_path}"]

    # Extra docker args from environment
    extra = (extra_args or []) + (os.environ.get("DOCKER_EXTRA_ARGS", "").split() or [])
    cmd += extra

    cmd.append(image)
    cmd += command

    kwargs: dict = {}
    if timeout:
        kwargs["timeout"] = timeout

    return _run(cmd, check=False, **kwargs)


def _docker_image_exists(tag: str) -> bool:
    """Return True when a Docker image tag exists locally."""
    result = _run(["docker", "image", "inspect", tag], check=False, capture=True)
    return result.returncode == 0


def _capture_repo_provenance() -> dict[str, Any]:
    """Best-effort capture of repository provenance for a benchmark run.

    Returned keys (any may be ``None`` when git is unavailable or we are not
    inside a worktree):

    - ``commit_sha``   full HEAD sha
    - ``commit_short`` first 12 chars of ``commit_sha``
    - ``commit_dirty`` ``True`` when the working tree has uncommitted changes
                       relative to HEAD; ``False`` otherwise
    - ``branch``       symbolic branch name (``None`` when detached)
    - ``remote_url``   ``remote.origin.url`` if configured

    Provenance is captured once per ``main()`` invocation and stamped onto
    every ``result.json`` so candidate-vs-baseline comparisons can verify the
    runs were produced from the same source tree.
    """
    payload: dict[str, Any] = {
        "commit_sha": None,
        "commit_short": None,
        "commit_dirty": None,
        "branch": None,
        "remote_url": None,
    }
    git_prefix = ["git", "-C", str(REPO_ROOT)]
    try:
        sha = subprocess.run(
            git_prefix + ["rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if sha:
            payload["commit_sha"] = sha
            payload["commit_short"] = sha[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return payload

    try:
        status = subprocess.run(
            git_prefix + ["status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        payload["commit_dirty"] = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        branch = subprocess.run(
            git_prefix + ["rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if branch and branch != "HEAD":
            payload["branch"] = branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        remote = subprocess.run(
            git_prefix + ["config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if remote:
            payload["remote_url"] = remote
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return payload


def _capture_image_digest(image_tag: str) -> str | None:
    """Return the local docker image Id (sha256:...) for ``image_tag``, or None.

    This is recorded in provenance so a baseline vs candidate comparison can
    detect a stealth rebuild of the agentic-base image between the two runs.
    """
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image_tag],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Pinned-commit support (Layer 2)
# ---------------------------------------------------------------------------
#
# When the runner is invoked with ``--commit <ref>``, the source tree used for
# the run (task definitions, verifiers, AUT config, framework code) is pinned
# to that commit via a git worktree. The ``nhx-agentic-base`` image is rebuilt
# from that worktree (tagged both ``:<short-sha>`` and ``:latest`` so per-task
# Dockerfiles' ``FROM nhx-agentic-base:latest`` line picks up the pinned base)
# and then ``nat_runner.py`` re-execs from the pinned worktree path so that
# ``Path(__file__).parents[2]`` resolves to the pinned tree for the rest of
# the run.

_PINNED_GUARD_ENV = "_NAT_RUNNER_PINNED_SHA"
_PINNED_REF_ENV = "_NAT_RUNNER_PINNED_REF"


def _git_resolve(ref: str, *, cwd: Path) -> str:
    """Resolve ``ref`` to a full commit sha within ``cwd``; raise on failure."""
    proc = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Cannot resolve commit ref {ref!r} in {cwd}: {proc.stderr.strip() or proc.stdout.strip()}")
    sha = proc.stdout.strip()
    if not sha:
        raise RuntimeError(f"git rev-parse returned empty sha for {ref!r}")
    return sha


def _ensure_pinned_worktree(sha: str, worktree_dir: Path, *, repo_root: Path) -> None:
    """Make sure ``worktree_dir`` exists and is checked out to ``sha`` (detached).

    Idempotent: a fresh worktree is created when missing; an existing worktree
    at the same sha is reused; an existing worktree at a different sha raises.
    """
    if worktree_dir.exists():
        try:
            existing = _git_resolve("HEAD", cwd=worktree_dir)
        except RuntimeError as e:
            raise RuntimeError(f"Worktree directory {worktree_dir} exists but is not a usable git worktree: {e}") from e
        if existing != sha:
            raise RuntimeError(
                f"Worktree {worktree_dir} is at {existing[:12]}, expected {sha[:12]}. "
                f"Remove it or pass a different --worktree-dir."
            )
        return

    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(worktree_dir), sha],
    )


def _build_pinned_agentic_base(worktree_dir: Path, short_sha: str) -> str:
    """Build ``nhx-agentic-base:<short_sha>`` from the pinned worktree.

    Idempotent: if the image already exists locally we skip the (expensive)
    rebuild and just retag ``:latest`` so per-task Dockerfiles inherit it.
    """
    pinned_tag = f"nhx-agentic-base:{short_sha}"
    dockerfile = worktree_dir / "Dockerfile.agentic-base"
    if not dockerfile.exists():
        raise FileNotFoundError(f"Pinned worktree is missing {dockerfile}; cannot build agentic base image.")

    if not _docker_image_exists(pinned_tag):
        print(f"[nat_runner] Building pinned agentic base image {pinned_tag} from {worktree_dir} ...")
        _run(
            [
                "docker",
                "build",
                "-f",
                str(dockerfile),
                "-t",
                pinned_tag,
                "-t",
                "nhx-agentic-base:latest",
                str(worktree_dir),
            ]
        )
    else:
        print(f"[nat_runner] Reusing existing pinned agentic base image {pinned_tag}.")
        # Make sure :latest still points at the pinned digest so task images
        # (FROM nhx-agentic-base:latest) inherit from the pinned base.
        _run(["docker", "tag", pinned_tag, "nhx-agentic-base:latest"], check=False)
    return pinned_tag


def _setup_pinned_run(
    commit_ref: str,
    worktree_dir: Path | None,
    *,
    repo_root: Path,
) -> tuple[Path, str, str, str]:
    """Resolve ``commit_ref``, ensure a worktree exists, build the pinned image.

    Returns ``(worktree_path, sha, short_sha, pinned_image_tag)``.
    """
    sha = _git_resolve(commit_ref, cwd=repo_root)
    short_sha = sha[:12]
    target = (worktree_dir or (repo_root / ".nat-worktrees" / short_sha)).resolve()
    _ensure_pinned_worktree(sha, target, repo_root=repo_root)
    pinned_tag = _build_pinned_agentic_base(target, short_sha)
    return target, sha, short_sha, pinned_tag


def _strip_pinned_args(argv: list[str]) -> list[str]:
    """Return ``argv[1:]`` with ``--commit`` / ``--worktree-dir`` removed.

    Handles both ``--flag value`` and ``--flag=value`` forms. Used when the
    runner re-execs from the pinned worktree so the second invocation sees a
    "normal" non-pinned argv.
    """
    out: list[str] = []
    skip_next = False
    drop_flags = {"--commit", "--worktree-dir"}
    for token in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in drop_flags:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in drop_flags):
            continue
        out.append(token)
    return out


def _task_agent_timeout(task_dir: Path) -> int | None:
    """Return per-task agent timeout from task.toml when present."""
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return None
    try:
        data = tomllib.loads(task_toml.read_text())
    except Exception:
        return None
    agent = data.get("agent")
    if not isinstance(agent, dict):
        return None
    timeout_value = agent.get("timeout_sec")
    if isinstance(timeout_value, (int, float)) and timeout_value > 0:
        return int(timeout_value)
    return None


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def _build_workflow_agent_cmd(workflow_container: str, instruction_container: str) -> list[str]:
    """Build the ``bash -c`` command that runs the ``nat run`` workflow backend."""
    return [
        "bash",
        "-c",
        textwrap.dedent(f"""\
            /app/.venv/bin/nat run \\
              --config_file {workflow_container} \\
              --input "$(cat {instruction_container})" \\
              2>&1 | tee /tmp/nat_agent.log
            EXIT=${{PIPESTATUS[0]}}
            cp /tmp/nat_agent.log /logs/agent/nat_agent.log 2>/dev/null || true
            if [ -f /logs/agent/intermediate_steps.jsonl ]; then
              /app/.venv/bin/python {NAT_TRACE_EXPORT_SCRIPT_CONTAINER_PATH} convert-jsonl \\
                --input /logs/agent/intermediate_steps.jsonl \\
                --output /logs/agent/trajectory.json \\
                >> /tmp/nat_agent.log 2>&1
              CONVERT_EXIT=$?
              cp /tmp/nat_agent.log /logs/agent/nat_agent.log 2>/dev/null || true
              if [ $EXIT -eq 0 ] && [ $CONVERT_EXIT -ne 0 ]; then
                exit $CONVERT_EXIT
              fi
            elif [ $EXIT -eq 0 ]; then
              echo "NAT telemetry exporter did not create /logs/agent/intermediate_steps.jsonl" | tee -a /tmp/nat_agent.log
              cp /tmp/nat_agent.log /logs/agent/nat_agent.log 2>/dev/null || true
              exit 1
            fi
            exit $EXIT
        """),
    ]


def _build_aut_agent_cmd(instruction_container: str) -> list[str]:
    """Build the ``bash -c`` command that drives the AUT (deep-agent) backend.

    The script runs entirely inside the task container and is responsible for:

    1. Resolving env-var placeholders inside the AUT yaml config when present.
    2. Ensuring the AUT entity matches the resolved config (recreating it
       when a config is provided, creating it when missing otherwise).
    3. Optionally seeding inference providers + secrets so AUT configs that
       resolve models via the platform's inference gateway can run inside an
       ephemeral test container with no pre-existing entities.
    4. Deploying the AUT, waiting for ``/health`` readiness, and invoking the
       task instruction through an artifact-capture helper that prefers NAT's
       ``/generate/atif`` endpoint and falls back to legacy generate routes.
    5. Collecting diagnostics (deployment list/get, NeMo Helix API logs, NAT
       subprocess logs) into ``/logs/agent/`` on failure.

    Provider seeding is handled by ``seed_providers.py`` via a declarative
    ``providers.yaml`` manifest — see :func:`run_agent_phase`'s docstring.
    """
    return [
        "bash",
        "-c",
        textwrap.dedent(f"""\
            set -euo pipefail
            EFFECTIVE_AUT_AGENT_CONFIG="${{AUT_AGENT_CONFIG}}"
            if [ -n "${{AUT_AGENT_CONFIG}}" ] && [ -n "${{NVIDIA_API_KEY:-}}" -o -n "${{ANTHROPIC_API_KEY:-}}" ]; then
              /app/.venv/bin/python -c "from pathlib import Path; import os; text = Path(os.environ['AUT_AGENT_CONFIG']).read_text(); text = text.replace('\\${{NVIDIA_API_KEY}}', os.environ.get('NVIDIA_API_KEY', '')); text = text.replace('\\${{ANTHROPIC_API_KEY}}', os.environ.get('ANTHROPIC_API_KEY', '')); Path('/tmp/aut_agent.resolved.yml').write_text(text)"
              EFFECTIVE_AUT_AGENT_CONFIG="/tmp/aut_agent.resolved.yml"
            fi
            # Ensure AUT entity matches the resolved config. When a config is
            # provided, recreate an existing AUT so --agent-model rewrites and
            # config edits are applied instead of reusing stale entity state.
            if /app/.venv/bin/nemo agents get "${{AUT_AGENT_NAME}}" >/tmp/aut_get_before.log 2>&1; then
              if [ -n "${{EFFECTIVE_AUT_AGENT_CONFIG}}" ]; then
                echo "AUT '${{AUT_AGENT_NAME}}' already exists; recreating from AUT_AGENT_CONFIG."
                /app/.venv/bin/nemo agents undeploy --agent "${{AUT_AGENT_NAME}}" >/tmp/aut_undeploy_before_recreate.log 2>&1 || true
                /app/.venv/bin/nemo agents delete "${{AUT_AGENT_NAME}}" >/tmp/aut_delete_before_recreate.log 2>&1 || true
                /app/.venv/bin/nemo agents create --name "${{AUT_AGENT_NAME}}" --agent-config "${{EFFECTIVE_AUT_AGENT_CONFIG}}" >/tmp/aut_create.log 2>&1
              else
                echo "AUT '${{AUT_AGENT_NAME}}' already exists."
              fi
            else
              if [ -z "${{EFFECTIVE_AUT_AGENT_CONFIG}}" ]; then
                echo "AUT agent '${{AUT_AGENT_NAME}}' not found and no AUT_AGENT_CONFIG was provided." >&2
                echo "Set --aut-agent-config or AUT_AGENT_CONFIG so the runner can create the agent." >&2
                echo "Expected resolved config path in container env: '${{AUT_AGENT_CONFIG:-<unset>}}'" >&2
                cp /tmp/aut_get_before.log /logs/agent/aut_get_before.log 2>/dev/null || true
                exit 1
              fi
              /app/.venv/bin/nemo agents create --name "${{AUT_AGENT_NAME}}" --agent-config "${{EFFECTIVE_AUT_AGENT_CONFIG}}" >/tmp/aut_create.log 2>&1
            fi
            # Seed inference providers via declarative manifest.
            if [ "${{AUT_SEED_PROVIDERS:-1}}" = "1" ]; then
              /app/.venv/bin/python /app/tests/agentic-use/seed_providers.py \
                --manifest /app/tests/agentic-use/providers.yaml \
                --base-url "${{NHX_BASE_URL:-http://localhost:8080}}" \
                2>&1 | tee /tmp/aut_provider_seed.log
            fi
            collect_aut_diagnostics() {{
              local restore_errexit=0
              case "$-" in
                *e*) restore_errexit=1 ;;
              esac
              set +e
              /app/.venv/bin/nemo agents deployments list >/tmp/aut_deployments.list.json 2>&1
              cp /tmp/aut_deployments.list.json /logs/agent/aut_deployments.list.json 2>/dev/null || true
              dep_name=$(
                /app/.venv/bin/python -c "import json,sys; data=json.load(sys.stdin).get('data', []); match=next((d.get('name') for d in data if d.get('agent') == '${{AUT_AGENT_NAME}}'), ''); print(match)" </tmp/aut_deployments.list.json 2>/dev/null
              )
              if [ -n "$dep_name" ]; then
                /app/.venv/bin/nemo agents deployments get "$dep_name" >/tmp/aut_deployment.get.json 2>&1
                cp /tmp/aut_deployment.get.json /logs/agent/aut_deployment.get.json 2>/dev/null || true
              fi
              cp /tmp/aut_create.log /logs/agent/aut_create.log 2>/dev/null || true
              cp /tmp/aut_get_before.log /logs/agent/aut_get_before.log 2>/dev/null || true
              cp /tmp/aut_provider_seed.log /logs/agent/aut_provider_seed.log 2>/dev/null || true
              cp /tmp/aut_undeploy.log /logs/agent/aut_undeploy.log 2>/dev/null || true
              cp /tmp/aut_undeploy_before_recreate.log /logs/agent/aut_undeploy_before_recreate.log 2>/dev/null || true
              cp /tmp/aut_delete_before_recreate.log /logs/agent/aut_delete_before_recreate.log 2>/dev/null || true
              cp /tmp/nhx-api.log /logs/agent/nhx-api.log 2>/dev/null || true
              # Collect agent subprocess logs.  The runtime artifact layout
              # moved out of the plugin source tree (<plugin>/.tmp/system/)
              # into nhx_user_data_dir()/agents/system/, with deterministic
              # per-deployment filenames.  Use `nemo agents logs --path` so
              # this script doesn't have to keep its own copy of the layout
              # convention.  ``|| true`` on the pipeline because we run under
              # ``set -euo pipefail`` and ``nemo agents logs`` exits 1 if no
              # deployment is found — a normal post-error state during
              # diagnostics that must not abort cleanup.
              mkdir -p /logs/agent/nat_subprocess_logs 2>/dev/null || true
              if [ -d /logs/agent/nat_subprocess_logs ]; then
                ( /app/.venv/bin/nemo agents logs --agent "${{AUT_AGENT_NAME}}" --path 2>/dev/null | while read -r aut_log_path; do
                  if [ -n "$aut_log_path" ] && [ -f "$aut_log_path" ]; then
                    cp "$aut_log_path" /logs/agent/nat_subprocess_logs/ 2>/dev/null || true
                  fi
                done ) || true
              fi
              if [ "$restore_errexit" -eq 1 ]; then
                set -e
              fi
              return 0
            }}
            cleanup() {{
              cleanup_rc=$?
              set +e
              collect_aut_diagnostics
              /app/.venv/bin/nemo agents undeploy --agent "${{AUT_AGENT_NAME}}" >/tmp/aut_undeploy_after.log 2>&1 || true
              cp /tmp/aut_undeploy_after.log /logs/agent/aut_undeploy_after.log 2>/dev/null || true
              cp /tmp/nat_agent.log /logs/agent/nat_agent.log 2>/dev/null || true
              exit "$cleanup_rc"
            }}
            trap cleanup EXIT
            # Ensure a clean deployment state before invocation.
            /app/.venv/bin/nemo agents undeploy --agent "${{AUT_AGENT_NAME}}" >/tmp/aut_undeploy.log 2>&1 || true
            /app/.venv/bin/nemo agents deploy --agent "${{AUT_AGENT_NAME}}"
            /app/.venv/bin/nemo agents deployments wait --agent "${{AUT_AGENT_NAME}}"
            # Deployment status can become "running" before the NAT server
            # is fully ready to accept HTTP traffic. Enforce /health readiness.
            dep_endpoint=$(
              /app/.venv/bin/nemo agents deployments list 2>/dev/null | /app/.venv/bin/python -c "import json,sys; data=json.load(sys.stdin).get('data', []); match=next((d.get('endpoint') for d in data if d.get('agent') == '${{AUT_AGENT_NAME}}' and d.get('status') == 'running' and d.get('endpoint')), ''); print(match)"
            )
            if [ -z "$dep_endpoint" ]; then
              echo "No running AUT deployment endpoint found for agent '${{AUT_AGENT_NAME}}'." >&2
              exit 1
            fi
            health_wait="${{AUT_HEALTH_WAIT_SECONDS:-60}}"
            aut_healthy=0
            for i in $(seq 1 "$health_wait"); do
              if curl -sf "$dep_endpoint/health" >/dev/null 2>&1; then
                aut_healthy=1
                break
              fi
              sleep 1
            done
            if [ "$aut_healthy" -ne 1 ]; then
              echo "AUT deployment did not become healthy within $health_wait seconds: $dep_endpoint" >&2
              exit 1
            fi
            set +e
            /app/.venv/bin/python {NAT_TRACE_EXPORT_SCRIPT_CONTAINER_PATH} invoke-aut \
              --endpoint "$dep_endpoint" \
              --instruction {instruction_container} \
              --output-dir /logs/agent \
              --timeout "${{AUT_INVOKE_HTTP_TIMEOUT:-600}}" \
              2>&1 | tee /tmp/nat_agent.log
            rc=${{PIPESTATUS[0]}}
            set -e
            if [ $rc -ne 0 ]; then
              collect_aut_diagnostics
            fi
            exit $rc
        """),
    ]


def _build_claude_code_agent_cmd(instruction_container: str, agent_params: dict[str, Any]) -> list[str]:
    """Build the ``bash -c`` command that runs the claude-code backend.

    ``--output-format json`` wraps the agent's final response in a single
    JSON object that includes a cumulative ``usage`` block (input/output
    plus cache_creation/cache_read tokens), ``num_turns``, ``duration_ms``,
    and ``total_cost_usd``. :func:`_extract_usage_metrics` parses that
    envelope and surfaces the same token shape we record for AUT, so the
    dual-baseline comparison is apples-to-apples.
    """
    extra_args = _claude_code_shell_args(agent_params)
    return [
        "bash",
        "-c",
        textwrap.dedent(f"""\
            set -euo pipefail
            model_args=()
            if [ -n "${{AGENT_MODEL:-}}" ]; then
              model_args=(--model "${{AGENT_MODEL}}")
            fi
            extra_args=({extra_args})
            export CLAUDE_CONFIG_DIR=/logs/agent/sessions
            mkdir -p "$CLAUDE_CONFIG_DIR"
            set +e
            claude -p "$(cat {instruction_container})" \
              "${{model_args[@]}}" \
              "${{extra_args[@]}}" \
              --output-format json \
              > /tmp/nat_agent.log 2> /tmp/nat_agent.stderr
            rc=$?
            set -e
            cp /tmp/nat_agent.log /logs/agent/nat_agent.log 2>/dev/null || true
            cp /tmp/nat_agent.stderr /logs/agent/nat_agent.stderr 2>/dev/null || true
            if [ $rc -eq 0 ]; then
              /app/.venv/bin/python {NAT_TRACE_EXPORT_SCRIPT_CONTAINER_PATH} convert-claude-session \
                --projects-dir /logs/agent/sessions/projects \
                --output /logs/agent/trajectory.json \
                --instruction {instruction_container} \
                --final-message /logs/agent/nat_agent.log \
                >> /tmp/nat_agent.log 2>&1
              rc=$?
              cp /tmp/nat_agent.log /logs/agent/nat_agent.log 2>/dev/null || true
            fi
            exit $rc
        """),
    ]


def _claude_code_shell_args(agent_params: dict[str, Any]) -> str:
    args: list[str] = []
    permission_mode = agent_params.get("permission_mode")
    if isinstance(permission_mode, str):
        args.extend(["--permission-mode", permission_mode])
    max_budget_usd = agent_params.get("max_budget_usd")
    if isinstance(max_budget_usd, int | float):
        args.extend(["--max-budget-usd", str(max_budget_usd)])
    return " ".join(shlex.quote(arg) for arg in args)


def _codex_config_shell_args(agent_params: dict[str, Any]) -> str:
    config: dict[str, Any] = {}
    intelligence = agent_params.get("intelligence")
    if isinstance(intelligence, str):
        config["model_reasoning_effort"] = intelligence
    speed = agent_params.get("speed")
    if speed == "fast":
        config["service_tier"] = "fast"
    explicit_config = agent_params.get("config")
    if isinstance(explicit_config, dict):
        config.update(explicit_config)
    args: list[str] = []
    for key, value in sorted(config.items()):
        args.append("-c")
        args.append(f"{key}={json.dumps(value)}")
    return " ".join(shlex.quote(arg) for arg in args)


def _build_codex_agent_cmd(instruction_container: str, agent_params: dict[str, Any]) -> list[str]:
    """Build the command that runs Codex CLI headlessly inside the task container."""
    return ["bash", "-c", _codex_agent_script(instruction_container, agent_params)]


def _codex_agent_script(instruction_container: str, agent_params: dict[str, Any]) -> str:
    config_args = _codex_config_shell_args(agent_params)
    return (
        CODEX_AGENT_SCRIPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("@@INSTRUCTION_CONTAINER@@", shlex.quote(instruction_container))
        .replace("@@CODEX_CONFIG_ARGS@@", config_args)
    )


def _build_cursor_agent_cmd(instruction_container: str, agent_params: dict[str, Any]) -> list[str]:
    """Build the command that runs Cursor Agent headlessly inside the task container."""
    extra_args = _cursor_agent_shell_args(agent_params)
    return [
        "bash",
        "-c",
        textwrap.dedent(f"""\
            set -euo pipefail
            model_args=()
            if [ -n "${{AGENT_MODEL:-}}" ]; then
              model_args=(--model "${{AGENT_MODEL}}")
            fi
            extra_args=({extra_args})
            set +e
            cursor-agent \
              --print \
              --output-format stream-json \
              --trust \
              --workspace /app \
              "${{model_args[@]}}" \
              "${{extra_args[@]}}" \
              "$(cat {instruction_container})" \
              > /tmp/nat_agent.log 2> /tmp/nat_agent.stderr
            rc=$?
            set -e
            cp /tmp/nat_agent.log /logs/agent/nat_agent.log 2>/dev/null || true
            cp /tmp/nat_agent.stderr /logs/agent/nat_agent.stderr 2>/dev/null || true
            if [ $rc -eq 0 ]; then
              /app/.venv/bin/python {NAT_TRACE_EXPORT_SCRIPT_CONTAINER_PATH} convert-cursor-jsonl \
                --input /logs/agent/nat_agent.log \
                --output /logs/agent/trajectory.json \
                --instruction {instruction_container} \
                --final-message /logs/agent/final_message.json \
                >> /tmp/nat_agent.log 2>&1
              rc=$?
              cp /tmp/nat_agent.log /logs/agent/nat_agent.log 2>/dev/null || true
            fi
            exit $rc
        """),
    ]


def _cursor_agent_shell_args(agent_params: dict[str, Any]) -> str:
    args: list[str] = []
    if "sandbox" in agent_params:
        sandbox = agent_params.get("sandbox")
        args.extend(["--sandbox", sandbox if isinstance(sandbox, str) else "disabled"])
    mode = agent_params.get("mode")
    if isinstance(mode, str):
        args.extend(["--mode", mode])
    return " ".join(shlex.quote(arg) for arg in args)


def run_agent_phase(
    task_dir: Path,
    image: str,
    output_dir: Path,
    *,
    nvidia_api_key: str,
    anthropic_api_key: str,
    anthropic_base_url: str,
    nhx_base_url: str,
    agent_model: str | None,
    agent_params: dict[str, Any],
    codex_auth_json: Path | None,
    timeout: int,
    agent_backend: str,
    aut_agent_name: str,
    aut_agent_config: str | None,
    aut_seed_providers: bool,
    state_dir: Path,
    workspace_dir: Path,
) -> bool:
    """Run the agent phase inside the container.

    Backends:
      - ``aut``: uses ``nemo agents invoke`` against a deployed AUT agent.
      - ``workflow``: uses ``nat run`` against task-local workflow.yml.
      - ``claude-code``: invokes Claude Code against task-local instruction.md.
      - ``codex``: invokes Codex CLI against task-local instruction.md.
      - ``cursor-agent``: invokes Cursor Agent against task-local instruction.md.

    Env / provider / secret matrix
    ------------------------------
    The agent container receives a curated set of env vars (passed via
    ``--env`` on ``docker run``). The matrix below documents what each one
    is for, which backend reads it, and how it lines up with platform
    inference providers on the seed phase.

    - ``NVIDIA_API_KEY``           NGC NVAPI key (``nvapi-…``). Used by
                                   ``seed_providers.py`` to create platform
                                   secrets for the ``nvidia-build`` provider.
    - ``ANTHROPIC_API_KEY``        Direct Anthropic-compatible key. Used by
                                   the ``claude-code`` backend.
    - ``OPENAI_API_KEY``           OpenAI API key. Used by the ``codex``
                                   backend.
    - ``CURSOR_API_KEY``           Cursor API key. Used by the
                                   ``cursor-agent`` backend.
    - ``INFERENCE_NVIDIA_API_KEY`` LiteLLM virtual key (``sk-…``). Used by
                                   ``seed_providers.py`` to create the
                                   ``nvidia-inference-api`` provider secret.
    - ``AUT_AGENT_NAME``           Which deep-agent to instantiate
                                   (required; no default).
    - ``AUT_AGENT_CONFIG``         Container path to the AUT yaml config.
                                   Empty string means "agent already exists,
                                   don't try to create it from config".
    - ``AUT_SEED_PROVIDERS``       ``"1"`` to run the inference-provider
                                   seed phase inside the agent container,
                                   ``"0"`` to assume the providers were
                                   pre-seeded by the harness.
    - ``AUT_HEALTH_WAIT_SECONDS``  Seconds to wait for the deployed AUT's
                                   ``/health`` to return 200 after deploy.

    The AUT seed phase (when ``AUT_SEED_PROVIDERS=1``) is handled by
    ``seed_providers.py``, which reads a declarative ``providers.yaml``
    manifest and creates secrets/providers via the NeMo SDK.
    """
    instruction_path = task_dir / "instruction.md"

    if not instruction_path.exists():
        raise FileNotFoundError(f"instruction.md not found in {task_dir}")
    workflow_path = task_dir / "workflow.yml"
    if agent_backend == "workflow" and not workflow_path.exists():
        raise FileNotFoundError(f"workflow.yml not found in {task_dir}")

    instruction = instruction_path.read_text()
    agent_log_dir = output_dir / "agent"
    _ensure_container_writable_dir(agent_log_dir)
    _ensure_container_writable_dir(workspace_dir)
    _ensure_container_writable_dir(state_dir)

    # Build the environment passed into the container
    env: dict[str, str] = {
        "NHX_BASE_URL": nhx_base_url,
        "AGENTIC_USE_WORKSPACE_DIR": "/app/workspace",
        "DATABASE_DIALECT": "sqlite",
        "DATABASE_PATH": "/data/nhx-platform.db",
        "NHX_FILES_DEFAULT_STORAGE_CONFIG": FILES_STORAGE_CONFIG,
        "NHX_CONFIG_FILE_PATH": PLATFORM_CONFIG_PATH,
        "NEMO_AGENTS_GATEWAY_READ_TIMEOUT": str(timeout),
        "NEMO_AGENTS_INVOKE_TIMEOUT": str(timeout),
        "AUT_INVOKE_HTTP_TIMEOUT": str(timeout),
    }
    if DOCKER_SOCKET_HOST_PATH.exists():
        env["DOCKER_HOST"] = f"unix://{DOCKER_SOCKET_CONTAINER_PATH}"
    if agent_params:
        env["NAT_CANDIDATE_PARAMS"] = json.dumps(agent_params, sort_keys=True)
    nvidia_api_key = _normalize_secret(nvidia_api_key)
    anthropic_api_key = _normalize_secret(anthropic_api_key)
    if agent_backend in {"aut", "workflow"} and nvidia_api_key:
        env["NVIDIA_API_KEY"] = nvidia_api_key
    if agent_backend in {"aut", "claude-code"} and anthropic_api_key:
        env["ANTHROPIC_API_KEY"] = anthropic_api_key
    openai_api_key = _secret_from_env("OPENAI_API_KEY")
    if agent_backend == "codex" and openai_api_key:
        env["OPENAI_API_KEY"] = openai_api_key
    cursor_api_key = _secret_from_env("CURSOR_API_KEY")
    if agent_backend == "cursor-agent" and cursor_api_key:
        env["CURSOR_API_KEY"] = cursor_api_key
    # INFERENCE_NVIDIA_API_KEY is consumed by seed_providers.py inside the
    # container to create the inference-api.nvidia.com provider secret. If
    # unset, seed_providers.py skips that provider with a warning (see
    # providers.yaml entry with `from_env: INFERENCE_NVIDIA_API_KEY`).
    # Fall back to NVIDIA_INFERENCE_API_KEY for developers whose shell exports
    # the key under that name (e.g. via zshrc).
    inference_nvidia_api_key = _secret_from_env("INFERENCE_NVIDIA_API_KEY") or _secret_from_env(
        "NVIDIA_INFERENCE_API_KEY"
    )
    if agent_backend == "aut" and aut_seed_providers and inference_nvidia_api_key:
        env["INFERENCE_NVIDIA_API_KEY"] = inference_nvidia_api_key
    if agent_model:
        if agent_backend in {"aut", "workflow"}:
            env["NAT_MODEL"] = agent_model
        elif agent_backend in {"claude-code", "codex", "cursor-agent"}:
            env["AGENT_MODEL"] = agent_model
    aut_config_host: Path | None = None
    aut_config_container = "/tmp/aut_agent.yml"
    if agent_backend == "aut":
        env["AUT_AGENT_NAME"] = aut_agent_name
        env["AUT_SEED_PROVIDERS"] = "1" if aut_seed_providers else "0"
        env["AUT_HEALTH_WAIT_SECONDS"] = os.environ.get("NAT_AUT_HEALTH_WAIT_SECONDS", "60")
        if aut_agent_config:
            aut_config_path = Path(aut_agent_config)
            if not aut_config_path.is_absolute():
                aut_config_path = (REPO_ROOT / aut_config_path).resolve()
            if not aut_config_path.exists():
                raise FileNotFoundError(f"AUT config not found: {aut_config_path}")
            aut_config_host = _prepare_aut_config_for_runtime(
                aut_config_path,
                agent_log_dir,
                nat_model=agent_model,
                nhx_base_url=nhx_base_url,
            )
            env["AUT_AGENT_CONFIG"] = aut_config_container
        else:
            env["AUT_AGENT_CONFIG"] = ""

    # Write the instruction to a temp file that we bind-mount into the container
    # so that the entrypoint can pipe it to ``nat run``.
    instruction_host = agent_log_dir / "instruction.md"
    instruction_host.write_text(instruction)

    # Container-side paths
    instruction_container = "/tmp/nat_instruction.md"
    workflow_host: Path | None = None
    workflow_container: str | None = None
    if agent_backend == "workflow":
        workflow_host = _prepare_workflow_for_runtime(
            workflow_path,
            agent_log_dir,
            nhx_base_url,
            nat_model=agent_model,
        )
        workflow_container = "/tmp/nat_workflow.yml"
        agent_cmd = _build_workflow_agent_cmd(workflow_container, instruction_container)
    elif agent_backend == "aut":
        agent_cmd = _build_aut_agent_cmd(instruction_container)
    elif agent_backend == "claude-code":
        env["ANTHROPIC_BASE_URL"] = anthropic_base_url
        agent_cmd = _build_claude_code_agent_cmd(instruction_container, agent_params)
    elif agent_backend == "codex":
        agent_cmd = _build_codex_agent_cmd(instruction_container, agent_params)
    elif agent_backend == "cursor-agent":
        agent_cmd = _build_cursor_agent_cmd(instruction_container, agent_params)
    else:
        raise ValueError(f"Unsupported agent backend: {agent_backend}")

    print(f"\n[nat_runner] === AGENT PHASE: {task_dir.name} ===")
    mounts = [
        (str(instruction_host), instruction_container),
        (str(agent_log_dir), "/logs/agent"),
        (str(workspace_dir), "/app/workspace"),
    ]
    if agent_backend == "codex" and not openai_api_key:
        if codex_auth_json is not None:
            mounts.append((str(codex_auth_json), "/tmp/codex_host_auth.json:ro"))
    if workflow_host is not None and workflow_container is not None:
        mounts.append((str(workflow_host), workflow_container))
    if aut_config_host is not None:
        mounts.append((str(aut_config_host), aut_config_container))
    if DOCKER_SOCKET_HOST_PATH.exists():
        mounts.append((str(DOCKER_SOCKET_HOST_PATH), DOCKER_SOCKET_CONTAINER_PATH))

    result = _docker_run(
        image,
        agent_cmd,
        env=env,
        mounts=mounts + [(str(state_dir), "/data")],
        timeout=timeout + 120,  # extra buffer for NeMo Helix startup
    )

    # Save agent log. ``_docker_run`` does not capture stdout (logs stream to
    # the controlling terminal in real time), so ``result.stdout`` is normally
    # empty. The agent script inside the container copies ``/tmp/nat_agent.log``
    # to ``/logs/agent/nat_agent.log`` (which is bind-mounted to
    # ``agent_log_dir`` on the host), so the file is what we read for
    # post-hoc inspection.
    agent_log_file = agent_log_dir / "nat_agent.log"
    if result.stdout:
        agent_log_file.write_text(result.stdout)

    agent_log_text = agent_log_file.read_text() if agent_log_file.exists() else (result.stdout or "")
    success = result.returncode == 0
    if success and agent_backend == "aut" and agent_log_text and _agent_log_has_workflow_error(agent_log_text):
        print("[nat_runner] Agent phase returned workflow_error payload; marking AGENT phase failed")
        success = False
    print(f"[nat_runner] Agent phase {'PASSED' if success else 'FAILED'} (exit={result.returncode})")
    return success


def _prepare_workflow_for_runtime(
    workflow_path: Path,
    output_dir: Path,
    nhx_base_url: str,
    *,
    nat_model: str | None = None,
) -> Path:
    """Prepare a NAT workflow file compatible with current NAT schema.

    NAT >=1.6 expects function-group types such as ``mcp_client`` under
    ``function_groups`` rather than ``functions``. To remain backward-compatible
    with existing task files, we rewrite that shape at runtime.
    """
    text = workflow_path.read_text()
    # Ensure MCP server URL follows runner's effective base URL.
    text = text.replace("http://localhost:8080", nhx_base_url)
    if nat_model:
        text = text.replace(
            "model_name: nvidia/llama-3.1-nemotron-70b-instruct",
            f"model_name: {nat_model}",
            1,
        )

    if "_type: mcp_client" in text or "_type: per_user_mcp_client" in text:
        # Backward-compat shim for legacy task configs:
        # NAT >=1.6 expects MCP client definitions under `function_groups`.
        if "\nfunction_groups:\n" not in text and "\nfunctions:\n" in text:
            text = text.replace("\nfunctions:\n", "\nfunction_groups:\n", 1)

    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError(f"Workflow config must be a mapping: {workflow_path}")
    general = config.setdefault("general", {})
    if not isinstance(general, dict):
        raise ValueError(f"Workflow general config must be a mapping: {workflow_path}")
    telemetry = general.setdefault("telemetry", {})
    if not isinstance(telemetry, dict):
        raise ValueError(f"Workflow telemetry config must be a mapping: {workflow_path}")
    tracing = telemetry.setdefault("tracing", {})
    if not isinstance(tracing, dict):
        raise ValueError(f"Workflow telemetry tracing config must be a mapping: {workflow_path}")
    tracing["agentic_use_file_trace"] = {
        "_type": "file",
        "output_path": "/logs/agent/intermediate_steps.jsonl",
        "project": "agentic-use",
        "mode": "overwrite",
        "cleanup_on_init": True,
    }

    text = yaml.dump(config, default_flow_style=False, sort_keys=False)
    rewritten = output_dir / "workflow.runtime.yml"
    rewritten.write_text(text)
    return rewritten


def _prepare_aut_config_for_runtime(
    config_path: Path,
    output_dir: Path,
    *,
    nat_model: str | None = None,
    nhx_base_url: str = DEFAULT_LOCAL_NHX_BASE_URL,
    workspace: str = "default",
) -> Path:
    """Prepare AUT config for IGW-routed container runtime.

    All AUT LLM traffic routes through the Inference Gateway.  The config is
    rewritten so that ``base_url`` points at the IGW OpenAI-compatible endpoint
    and ``api_key`` is set to ``not-used`` (IGW retrieves upstream credentials
    from the secrets service).  ``${NEMO_DEFAULT_MODEL}`` is resolved from the
    host CLI context before the config is mounted into the task container.
    ``model_name`` stays in entity form (dashes, e.g.
    ``aws-anthropic-claude-opus-4-5``) because IGW resolves entity names to the
    served model name internally.

    Uses the same :func:`nemo_agents_plugin.utils.inject_gateway_url` function
    that production ``nemo agents deployments create`` calls, ensuring parity
    between benchmark runs and deployed agents.
    """
    from nemo_agents_plugin.utils import inject_default_model, inject_gateway_url

    config = yaml.safe_load(config_path.read_text())

    if nat_model:
        for llm_cfg in config.get("llms", {}).values():
            if isinstance(llm_cfg, dict) and llm_cfg.get("_type") in ("openai", "nim"):
                llm_cfg["model_name"] = nat_model
                break

    config = inject_default_model(config)
    config = inject_gateway_url(config, workspace, base_url=nhx_base_url)

    rewritten = output_dir / "aut.runtime.yml"
    with rewritten.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return rewritten


class TokenMetrics(TypedDict):
    """Token usage metrics returned by :func:`_extract_usage_metrics`.

    Naming convention boundary: the extractor's internal accumulator and
    helpers use Anthropic-style ``input_tokens`` / ``output_tokens`` (the
    raw JSON keys from claude-code and AUT). The output schema converts
    those to ``prompt_tokens`` / ``completion_tokens`` (the terms used
    downstream by the gate script and eval-out summaries). The four cache
    buckets and ``n_assistant_messages`` keep their accumulator names —
    they're already aligned with the output schema.
    """

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cache_creation_tokens: int | None
    cache_read_tokens: int | None
    n_assistant_messages: int | None
    cost_usd: float | None
    num_turns: int | None
    duration_ms: float | None


def _is_nonnegative_int(value: object) -> bool:
    """Token-budget gate predicate for a parsed usage envelope."""
    return isinstance(value, int) and value >= 0


def _iter_agent_log_json_payloads(agent_log: str) -> list[dict[str, Any]]:
    """Return JSON dict payloads embedded in an agent log, newest-first after the full log."""
    candidates = [agent_log.strip()]
    lines = [ln.strip() for ln in agent_log.splitlines() if ln.strip()]
    if lines:
        candidates.append(lines[-1])
        candidates.extend(reversed(lines))

    payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


def _agent_log_has_workflow_error(agent_log: str) -> bool:
    """Detect AUT workflow errors returned as successful HTTP JSON payloads."""
    for payload in _iter_agent_log_json_payloads(agent_log):
        if payload.get("code") == "workflow_error":
            return True
    return False


def _extract_usage_metrics(agent_log: str) -> TokenMetrics:
    """Extract token usage metrics from an agent log.

    Aggregates across **all** assistant turns when the payload exposes a
    ``messages[]`` array (the AUT shape from ``nemo agents invoke``). Falls
    back to a flat top-level ``usage`` block (the claude-code shape from
    ``claude -p --output-format json``) when no per-message breakdown is
    available.

    Token buckets follow Anthropic's prompt-caching shape so AUT and
    claude-code produce comparable numbers:

    - ``prompt_tokens``         uncached input tokens (full input price)
    - ``completion_tokens``     assistant generation
    - ``cache_creation_tokens`` written to cache (~125% input price)
    - ``cache_read_tokens``     read from cache (~10% input price)
    - ``total_tokens``          sum of the four buckets — the full token
                                throughput through the model, regardless of
                                cache discount

    For claude-code's JSON output, also surfaces ``cost_usd``, ``num_turns``,
    and ``duration_ms`` when present (None for AUT).

    See :class:`TokenMetrics` for the naming convention boundary between the
    accumulator's Anthropic-style ``input_tokens`` / ``output_tokens`` locals
    and the output schema's ``prompt_tokens`` / ``completion_tokens`` keys.
    """
    zero: TokenMetrics = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cache_creation_tokens": None,
        "cache_read_tokens": None,
        "n_assistant_messages": None,
        "cost_usd": None,
        "num_turns": None,
        "duration_ms": None,
    }
    if not agent_log.strip():
        return zero

    def _first_int(usage_obj: dict[str, Any], keys: tuple[str, ...]) -> tuple[int | None, bool]:
        for key in keys:
            value = usage_obj.get(key)
            if isinstance(value, int):
                return value, True
        return None, False

    def _bucket_from_usage(usage_obj: dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None, bool]:
        """Return (input, output, cache_creation, cache_read, has_known_key)."""
        input_tokens, has_input = _first_int(usage_obj, ("input_tokens", "prompt_tokens", "inputTokens"))
        output_tokens, has_output = _first_int(usage_obj, ("output_tokens", "completion_tokens", "outputTokens"))
        # Anthropic raw shape (claude-code --output-format json).
        cache_creation_tokens, has_cache_creation = _first_int(
            usage_obj,
            ("cache_creation_input_tokens", "cacheWriteTokens"),
        )
        cache_read_tokens, has_cache_read = _first_int(
            usage_obj,
            ("cache_read_input_tokens", "cacheReadTokens", "cached_input_tokens"),
        )
        # LangChain normalised shape (AUT via langchain_anthropic / LiteLLM).
        details = usage_obj.get("input_token_details")
        if isinstance(details, dict):
            if not has_cache_creation:
                cache_creation_tokens, has_cache_creation = _first_int(details, ("cache_creation",))
            if not has_cache_read:
                cache_read_tokens, has_cache_read = _first_int(details, ("cache_read",))
        # OpenAI-style usage reports surface total input tokens plus a
        # cached-input subset under ``cached_input_tokens``. Convert that to
        # the benchmark's bucketed shape:
        # prompt_tokens=uncached input, cache_read_tokens=cached input.
        if (
            "cached_input_tokens" in usage_obj
            and has_input
            and has_cache_read
            and input_tokens is not None
            and cache_read_tokens is not None
        ):
            input_tokens = max(input_tokens - cache_read_tokens, 0)
        has_known_key = has_input or has_output or has_cache_creation or has_cache_read
        return input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, has_known_key

    def _looks_usage_bearing(d: dict[str, Any]) -> bool:
        """Heuristic: does this dict contain something we can extract usage from?"""
        if "messages" in d:
            return True
        for key in ("usage", "usage_metadata", "response_metadata", "data", "metrics"):
            if isinstance(d.get(key), dict):
                return True
        return False

    # Some agent log emit a trailing diagnostic JSON line that parses cleanly
    # but contains no usage data, ahead of the actual usage-bearing payload.
    # Prefer the first parsed dict that looks usage-bearing; fall back to any
    # parsed dict so behavior on legacy logs is unchanged.
    payload: dict[str, Any] | None = None
    fallback_payload: dict[str, Any] | None = None
    for parsed in _iter_agent_log_json_payloads(agent_log):
        if _looks_usage_bearing(parsed):
            payload = parsed
            break
        if fallback_payload is None:
            fallback_payload = parsed
    if payload is None:
        payload = fallback_payload
    if not payload:
        return zero

    # Walk both the top-level payload and any nested ``data`` wrapper.
    payload_candidates: list[dict[str, Any]] = [payload]
    nested_data = payload.get("data")
    if isinstance(nested_data, dict):
        payload_candidates.append(nested_data)

    sums = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "n_assistant_messages": 0,
    }
    bucket_presence = {
        "input_tokens": False,
        "output_tokens": False,
        "cache_creation_tokens": False,
        "cache_read_tokens": False,
    }
    has_data = False

    # Path 1 (preferred when present): walk every message in messages[] and
    # accumulate. AUT's ``nemo agents invoke`` payload contains one entry per
    # turn; LangChain attaches ``usage_metadata`` to every assistant message.
    for candidate_payload in payload_candidates:
        msgs = candidate_payload.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            usage_obj: dict[str, Any] | None = None
            for key in ("usage_metadata", "usage"):
                value = msg.get(key)
                if isinstance(value, dict) and value:
                    usage_obj = value
                    break
            if usage_obj is None:
                response_metadata = msg.get("response_metadata")
                if isinstance(response_metadata, dict):
                    token_usage = response_metadata.get("token_usage")
                    if isinstance(token_usage, dict) and token_usage:
                        usage_obj = token_usage
            if not usage_obj:
                continue
            input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, has_known_key = _bucket_from_usage(
                usage_obj
            )
            if has_known_key:
                if input_tokens is not None:
                    sums["input_tokens"] += input_tokens
                    bucket_presence["input_tokens"] = True
                if output_tokens is not None:
                    sums["output_tokens"] += output_tokens
                    bucket_presence["output_tokens"] = True
                if cache_creation_tokens is not None:
                    sums["cache_creation_tokens"] += cache_creation_tokens
                    bucket_presence["cache_creation_tokens"] = True
                if cache_read_tokens is not None:
                    sums["cache_read_tokens"] += cache_read_tokens
                    bucket_presence["cache_read_tokens"] = True
                sums["n_assistant_messages"] += 1
                has_data = True
            else:
                # Non-empty usage payload that yielded zero across all four
                # token buckets is almost always schema drift (Anthropic field
                # rename, LangChain wrapper change). Warn loudly so future
                # extraction bugs don't silently zero out token accounting
                # the way the messages[-1] undercount did.
                print(
                    f"[nat_runner] WARN: usage payload yielded zero across all "
                    f"token buckets; ignoring (likely schema drift). "
                    f"keys={sorted(usage_obj.keys())[:8]} "
                    f"sample={repr(usage_obj)[:300]}"
                )
        if has_data:
            break

    # Path 2 (fallback): flat top-level ``usage``/``usage_metadata``. This is
    # the claude-code ``--output-format json`` shape, which already aggregates
    # across all turns of the inner agent loop.
    if not has_data:
        for candidate_payload in payload_candidates:
            for key in ("usage", "usage_metadata"):
                usage_obj = candidate_payload.get(key)
                if isinstance(usage_obj, dict) and usage_obj:
                    (
                        input_tokens,
                        output_tokens,
                        cache_creation_tokens,
                        cache_read_tokens,
                        has_known_key,
                    ) = _bucket_from_usage(usage_obj)
                    if has_known_key:
                        if input_tokens is not None:
                            sums["input_tokens"] = input_tokens
                            bucket_presence["input_tokens"] = True
                        if output_tokens is not None:
                            sums["output_tokens"] = output_tokens
                            bucket_presence["output_tokens"] = True
                        if cache_creation_tokens is not None:
                            sums["cache_creation_tokens"] = cache_creation_tokens
                            bucket_presence["cache_creation_tokens"] = True
                        if cache_read_tokens is not None:
                            sums["cache_read_tokens"] = cache_read_tokens
                            bucket_presence["cache_read_tokens"] = True
                        # ``num_turns`` (claude-code) is the natural N here.
                        num_turns_value = candidate_payload.get("num_turns")
                        sums["n_assistant_messages"] = num_turns_value if isinstance(num_turns_value, int) else 1
                        has_data = True
                        break
            if has_data:
                break

    if not has_data:
        # We parsed JSON successfully but found no usable token usage in any
        # known shape. Emit enough fingerprint to track down the offending
        # payload from logs.
        print(
            f"[nat_runner] WARN: parsed agent-log JSON but found no "
            f"usable token usage; returning all-None metrics. "
            f"top_keys={sorted(payload.keys())[:8]}"
        )
        return zero

    total_components = [
        sums["input_tokens"] if bucket_presence["input_tokens"] else None,
        sums["output_tokens"] if bucket_presence["output_tokens"] else None,
        sums["cache_creation_tokens"] if bucket_presence["cache_creation_tokens"] else None,
        sums["cache_read_tokens"] if bucket_presence["cache_read_tokens"] else None,
    ]
    total = sum(component for component in total_components if component is not None)
    out: TokenMetrics = {
        "prompt_tokens": sums["input_tokens"] if bucket_presence["input_tokens"] else None,
        "completion_tokens": sums["output_tokens"] if bucket_presence["output_tokens"] else None,
        "total_tokens": total if any(component is not None for component in total_components) else None,
        "cache_creation_tokens": sums["cache_creation_tokens"] if bucket_presence["cache_creation_tokens"] else None,
        "cache_read_tokens": sums["cache_read_tokens"] if bucket_presence["cache_read_tokens"] else None,
        "n_assistant_messages": sums["n_assistant_messages"],
        "cost_usd": None,
        "num_turns": None,
        "duration_ms": None,
    }

    # Optional claude-code-only signals when present in the JSON envelope.
    for candidate_payload in payload_candidates:
        if not isinstance(candidate_payload, dict):
            continue
        cost = candidate_payload.get("total_cost_usd")
        if isinstance(cost, int | float) and out["cost_usd"] is None:
            out["cost_usd"] = float(cost)
        nt = candidate_payload.get("num_turns")
        if isinstance(nt, int) and out["num_turns"] is None:
            out["num_turns"] = nt
        dm = candidate_payload.get("duration_ms")
        if isinstance(dm, int | float) and out["duration_ms"] is None:
            out["duration_ms"] = float(dm)
    return out


def run_verify_phase(
    task_dir: Path,
    image: str,
    output_dir: Path,
    *,
    nhx_base_url: str,
    smoke_workspace: str | None = None,
    state_dir: Path,
    workspace_dir: Path,
    agent_backend: str,
    agent_model: str,
) -> tuple[bool, str]:
    """Run pytest on test_outputs.py inside the container.

    Returns (passed: bool, stdout: str).
    """
    tests_dir = task_dir / "tests"
    test_file = tests_dir / "test_outputs.py"
    if not test_file.exists():
        print(f"[nat_runner] ERROR: No test_outputs.py in {tests_dir}; verification failed")
        return False, ""

    agent_log_dir = output_dir / "agent"
    verifier_log_dir = output_dir / "verifier"
    _ensure_container_writable_dir(agent_log_dir)
    _ensure_container_writable_dir(verifier_log_dir)
    _ensure_container_writable_dir(workspace_dir)
    _ensure_container_writable_dir(state_dir)

    smoke_seed_cmd = ""
    smoke_cleanup_cmd = ""
    if smoke_workspace:
        smoke_seed_cmd = textwrap.dedent("""\
            # Optional smoke helper: seed a workspace expected by workspace-basic tests.
            /app/.venv/bin/nemo workspaces create "${SMOKE_WORKSPACE}" \
              --description "Seeded by nat_runner smoke mode" >/dev/null 2>&1 || true
        """)
        smoke_cleanup_cmd = textwrap.dedent("""\
            # Always attempt cleanup so smoke runs don't leave residue.
            /app/.venv/bin/nemo workspaces delete "${SMOKE_WORKSPACE}" >/dev/null 2>&1 || true
        """)

    verify_cmd = [
        "bash",
        "-c",
        textwrap.dedent(f"""\
            export PYTHONPATH="/app/tests/agentic-use/shared:/app/packages/nemo_evaluator_sdk/src:${{PYTHONPATH}}"
            export NAT_AGENT=1
            {smoke_seed_cmd}
            /app/.venv/bin/python -m pytest /tests/test_outputs.py -rA -v 2>&1 | tee /logs/verifier/test-stdout.txt
            EXIT=${{PIPESTATUS[0]}}
            {smoke_cleanup_cmd}
            if [ $EXIT -eq 0 ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
            exit $EXIT
        """),
    ]

    print(f"\n[nat_runner] === VERIFY PHASE: {task_dir.name} ===")
    result = _docker_run(
        image,
        verify_cmd,
        env={
            "NHX_BASE_URL": nhx_base_url,
            "NAT_AGENT": "1",
            "NAT_AGENT_BACKEND": agent_backend,
            "NAT_AGENT_MODEL": agent_model,
            "AGENTIC_USE_TASK_DIR": "/task",
            "AGENTIC_USE_WORKSPACE_DIR": "/app/workspace",
            "SMOKE_WORKSPACE": smoke_workspace or "",
            "DATABASE_DIALECT": "sqlite",
            "DATABASE_PATH": "/data/nhx-platform.db",
            "NHX_FILES_DEFAULT_STORAGE_CONFIG": FILES_STORAGE_CONFIG,
            "NHX_CONFIG_FILE_PATH": PLATFORM_CONFIG_PATH,
            **({"DOCKER_HOST": f"unix://{DOCKER_SOCKET_CONTAINER_PATH}"} if DOCKER_SOCKET_HOST_PATH.exists() else {}),
        },
        mounts=[
            (str(tests_dir), "/tests"),
            (str(task_dir), "/task"),
            (str(workspace_dir), "/app/workspace"),
            (str(SHARED_DIR), "/app/tests/agentic-use/shared:ro"),
            (str(REPO_ROOT / "packages" / "nemo_evaluator_sdk" / "src"), "/app/packages/nemo_evaluator_sdk/src:ro"),
            (str(agent_log_dir), "/logs/agent"),
            (str(verifier_log_dir), "/logs/verifier"),
            # Persist service/database state across AGENT and VERIFY containers.
            # This restores Harbor-like state continuity while keeping phase isolation.
            (str(state_dir), "/data"),
            *(
                [(str(DOCKER_SOCKET_HOST_PATH), DOCKER_SOCKET_CONTAINER_PATH)]
                if DOCKER_SOCKET_HOST_PATH.exists()
                else []
            ),
        ],
    )

    stdout = result.stdout or ""
    if (verifier_log_dir / "test-stdout.txt").exists():
        stdout = (verifier_log_dir / "test-stdout.txt").read_text()

    # Write reward.txt to output dir (may already have been written by the verify_cmd)
    reward_file = verifier_log_dir / "reward.txt"
    passed = result.returncode == 0
    if not reward_file.exists():
        reward_file.write_text("1\n" if passed else "0\n")

    print(f"[nat_runner] Verify phase {'PASSED' if passed else 'FAILED'} (exit={result.returncode})")
    return passed, stdout


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


def run_task(
    task_name: str,
    *,
    jobs_dir: Path,
    nvidia_api_key: str,
    anthropic_api_key: str,
    anthropic_base_url: str,
    nhx_base_url: str,
    agent_model: str | None,
    agent_params: dict[str, Any] | None,
    codex_auth_json: Path | None,
    agent_timeout: int,
    skip_build: bool,
    build_only: bool,
    skip_agent: bool,
    agent_backend: str,
    aut_agent_name: str,
    aut_agent_config: str | None,
    aut_seed_providers: bool,
    smoke_workspace: str | None,
    candidate_id: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict:
    """Run a single agentic-use eval task end to end."""
    task_start_monotonic = time.monotonic()
    task_dir = TASKS_DIR / task_name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = jobs_dir / f"{ts}-{task_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    _ensure_container_writable_dir(state_dir)
    workspace_dir = output_dir / "workspace"
    _ensure_container_writable_dir(workspace_dir)

    image_tag = f"nhx-nat-{task_name}:latest"

    result: dict = {
        "task": task_name,
        "timestamp": ts,
        "output_dir": str(output_dir),
        "image": image_tag,
        "agent_backend": agent_backend,
        "agent_model": _agent_model_for_backend(agent_backend=agent_backend, agent_model=agent_model),
        "candidate_params": agent_params or {},
        "candidate_id": candidate_id,
        "build": None,
        "agent": None,
        "verify": None,
        "passed": None,
        "reward": None,
        "runtime_sec": None,
        "metrics": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cache_creation_tokens": None,
            "cache_read_tokens": None,
            "n_assistant_messages": None,
            "cost_usd": None,
            "num_turns": None,
            "duration_ms": None,
            "token_metrics_status": "unavailable",
            "token_metrics_note": None,
        },
        "verifier_scores": None,
        "provenance": dict(provenance) if provenance else None,
    }
    task_timeout = _task_agent_timeout(task_dir)
    effective_agent_timeout = max(agent_timeout, task_timeout or 0)
    if task_timeout and task_timeout > agent_timeout:
        print(
            "[nat_runner] Using task timeout override: "
            f"{task_timeout}s from {task_dir / 'task.toml'} "
            f"(requested --timeout was {agent_timeout}s)"
        )

    # ------------------------------------------------------------------
    # 1. BUILD
    # ------------------------------------------------------------------
    if not skip_build:
        print(f"\n[nat_runner] === BUILD PHASE: {task_name} ===")
        try:
            build_task_image(task_dir, image_tag)
            result["build"] = "ok"
        except Exception as e:
            print(f"[nat_runner] Build FAILED: {e}")
            result["build"] = f"error: {e}"
            result["passed"] = False
            result["reward"] = 0
            result["runtime_sec"] = round(time.monotonic() - task_start_monotonic, 3)
            _write_result(output_dir, result)
            return result
    else:
        if _docker_image_exists(image_tag):
            result["build"] = "skipped"
        else:
            msg = (
                f"--skip-build requested but task image {image_tag!r} is not available locally. "
                "Run without --skip-build to build the task image first."
            )
            print(f"[nat_runner] Build FAILED: {msg}")
            result["build"] = f"error: {msg}"
            result["passed"] = False
            result["reward"] = 0
            result["runtime_sec"] = round(time.monotonic() - task_start_monotonic, 3)
            _write_result(output_dir, result)
            return result

    if build_only:
        result["agent"] = "skipped"
        result["verify"] = "skipped"
        result["passed"] = None
        result["reward"] = None
        result["runtime_sec"] = round(time.monotonic() - task_start_monotonic, 3)
        _write_result(output_dir, result)
        return result

    # ------------------------------------------------------------------
    # 2. AGENT
    # ------------------------------------------------------------------
    if not skip_agent:
        try:
            agent_ok = run_agent_phase(
                task_dir,
                image_tag,
                output_dir,
                nvidia_api_key=nvidia_api_key,
                anthropic_api_key=anthropic_api_key,
                anthropic_base_url=anthropic_base_url,
                nhx_base_url=nhx_base_url,
                agent_model=agent_model,
                agent_params=agent_params or {},
                codex_auth_json=codex_auth_json,
                timeout=effective_agent_timeout,
                agent_backend=agent_backend,
                aut_agent_name=aut_agent_name,
                aut_agent_config=aut_agent_config,
                aut_seed_providers=aut_seed_providers,
                state_dir=state_dir,
                workspace_dir=workspace_dir,
            )
            result["agent"] = "ok" if agent_ok else "failed"
            agent_log_file = output_dir / "agent" / "nat_agent.log"
            if agent_log_file.exists():
                usage_metrics = _extract_usage_metrics(agent_log_file.read_text())
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cache_creation_tokens",
                    "cache_read_tokens",
                    "n_assistant_messages",
                    "cost_usd",
                    "num_turns",
                    "duration_ms",
                ):
                    result["metrics"][key] = usage_metrics.get(key)
                has_tokens = any(
                    _is_nonnegative_int(result["metrics"].get(k))
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
                if has_tokens:
                    result["metrics"]["token_metrics_status"] = "available"
                else:
                    result["metrics"]["token_metrics_note"] = "Backend did not emit token usage in agent log payload."
            if agent_backend == "claude-code" and result["metrics"]["token_metrics_status"] != "available":
                result["metrics"]["token_metrics_note"] = (
                    "claude-code backend did not emit a parseable JSON usage envelope; "
                    "ensure the agent invocation runs with --output-format json."
                )
            if not agent_ok:
                result["passed"] = False
                result["reward"] = 0
                result["runtime_sec"] = round(time.monotonic() - task_start_monotonic, 3)
                _write_result(output_dir, result)
                return result
        except Exception as e:
            print(f"[nat_runner] Agent phase error: {e}")
            result["agent"] = f"error: {e}"
            result["passed"] = False
            result["reward"] = 0
            result["runtime_sec"] = round(time.monotonic() - task_start_monotonic, 3)
            _write_result(output_dir, result)
            return result
    else:
        result["agent"] = "skipped"

    # ------------------------------------------------------------------
    # 3. VERIFY
    # ------------------------------------------------------------------
    try:
        verify_ok, stdout = run_verify_phase(
            task_dir,
            image_tag,
            output_dir,
            nhx_base_url=nhx_base_url,
            smoke_workspace=smoke_workspace,
            state_dir=state_dir,
            workspace_dir=workspace_dir,
            agent_backend=agent_backend,
            agent_model=_agent_model_for_backend(
                agent_backend=agent_backend,
                agent_model=agent_model,
            ),
        )
        result["verify"] = "ok" if verify_ok else "failed"
        result["passed"] = verify_ok
        result["reward"] = 1 if verify_ok else 0
        result["verifier_scores"] = _read_json_object(output_dir / "verifier" / "evaluator_scores.json")
    except Exception as e:
        print(f"[nat_runner] Verify phase error: {e}")
        result["verify"] = f"error: {e}"
        result["passed"] = False
        result["reward"] = 0

    result["runtime_sec"] = round(time.monotonic() - task_start_monotonic, 3)
    _write_result(output_dir, result)
    return result


def _agent_model_for_backend(
    *,
    agent_backend: str,
    agent_model: str | None,
) -> str:
    if agent_backend in {"claude-code", "codex", "cursor-agent"}:
        return agent_model or "default"
    return agent_model or "unknown"


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[nat_runner] WARN: could not read JSON object from {path}: {exc}")
        return None
    return payload if isinstance(payload, dict) else None


def _agent_model_from_env(agent_backend: str) -> str | None:
    """Resolve the candidate-agent model from env, including legacy names."""
    unified = os.environ.get("NAT_AGENT_MODEL") or os.environ.get("AGENT_MODEL")
    if unified:
        return unified
    if agent_backend in {"aut", "workflow"}:
        return os.environ.get("NAT_MODEL")
    if agent_backend == "claude-code":
        return os.environ.get("CLAUDE_MODEL")
    if agent_backend == "codex":
        return os.environ.get("CODEX_MODEL")
    if agent_backend == "cursor-agent":
        return os.environ.get("CURSOR_MODEL")
    return None


def _write_result(output_dir: Path, result: dict) -> None:
    """Persist the result JSON to the output directory."""
    if result.get("runtime_sec") is None:
        result["runtime_sec"] = None
    result_file = output_dir / "result.json"
    result_file.write_text(json.dumps(result, indent=2))
    reward = result.get("reward")
    print(f"\n[nat_runner] Result: reward={reward}  ({output_dir})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_tasks(patterns: list[str]) -> list[str]:
    """Expand task name patterns (fnmatch globs) to concrete task names."""
    all_tasks = [
        d.name
        for d in TASKS_DIR.iterdir()
        if d.is_dir() and (d / "instruction.md").exists() and d.name != "example-test-template"
    ]
    if not patterns:
        return sorted(all_tasks)

    resolved: list[str] = []
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            matches = [t for t in all_tasks if fnmatch.fnmatch(t, pattern)]
            resolved.extend(sorted(matches))
        elif pattern in all_tasks:
            resolved.append(pattern)
        else:
            raise ValueError(f"Unknown task: {pattern!r}. Available: {sorted(all_tasks)}")
    return resolved


def _read_manifest(manifest_path: Path) -> list[str]:
    if not manifest_path.is_absolute():
        manifest_path = (TASKS_DIR / manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Task manifest not found: {manifest_path}")

    patterns: list[str] = []
    for raw_line in manifest_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NAT-based runner for NeMo Helix agentic-use eval tasks (replaces harbor run)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Run one task
              python tests/agentic-use/nat_runner.py workspace-basic-mcp

              # Run all easy tasks
              python tests/agentic-use/nat_runner.py "*-easy"

              # Run all tasks
              python tests/agentic-use/nat_runner.py --all

              # Skip build only if task image already exists locally
              python tests/agentic-use/nat_runner.py --skip-build workspace-basic-mcp

              # Skip both build and agent (only run verifier against an already-running NeMo Helix)
              python tests/agentic-use/nat_runner.py --skip-build --skip-agent workspace-basic-mcp
        """),
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        metavar="TASK_OR_GLOB",
        help="Task name(s) or glob patterns (e.g. 'workspace-basic-mcp', '*-easy')",
    )
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Task manifest file with one task/glob per line. Relative paths resolve from tests/agentic-use/.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip Docker build phase and use existing prebuilt task image "
        "(nhx-nat-<task>:latest). Fails if task image is missing.",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Skip agent phase; only run the verifier (useful for debugging tests)",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build selected task images and skip agent/verifier phases.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=DEFAULT_JOBS_DIR,
        help=f"Directory to store run artifacts (default: {DEFAULT_JOBS_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Agent timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--nhx-base-url",
        default=DEFAULT_LOCAL_NHX_BASE_URL,
        help="NeMo Helix API base URL inside the container (default: http://localhost:8080).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tasks and exit",
    )
    parser.add_argument(
        "--agent-backend",
        choices=["aut", "workflow", "claude-code", "codex", "cursor-agent"],
        default=os.environ.get("NAT_AGENT_BACKEND", "aut"),
        help="Agent backend for AGENT phase: 'aut' uses nemo agents invoke (default), "
        "'workflow' uses task-local nat run workflow.yml, "
        "'claude-code' runs Claude Code in the task container, "
        "'codex' runs Codex CLI in the task container, "
        "'cursor-agent' runs Cursor Agent in the task container",
    )
    parser.add_argument(
        "--anthropic-base-url",
        default=os.environ.get("ANTHROPIC_BASE_URL", "https://inference-api.nvidia.com"),
        help="Anthropic-compatible endpoint for --agent-backend=claude-code "
        "(default: https://inference-api.nvidia.com)",
    )
    parser.add_argument(
        "--agent-model",
        default=None,
        help="Candidate-agent model for the selected backend "
        "(default: NAT_AGENT_MODEL/AGENT_MODEL or the backend/task default)",
    )
    parser.add_argument(
        "--model",
        dest="agent_model",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--codex-auth-json",
        type=Path,
        default=None,
        help="Explicit path to a Codex auth.json file to mount read-only for "
        "--agent-backend=codex when OPENAI_API_KEY is unset.",
    )
    parser.add_argument(
        "--candidate-id",
        default=os.environ.get("NAT_CANDIDATE_ID"),
        help="Optional stable candidate id to record in result.json for matrix benchmark aggregation.",
    )
    parser.add_argument(
        "--candidate-params",
        default=os.environ.get("NAT_CANDIDATE_PARAMS"),
        help="Optional JSON object of candidate-specific params to record and pass to backend command construction.",
    )
    parser.add_argument(
        "--aut-agent-name",
        default=os.environ.get("AUT_AGENT_NAME"),
        help="Name of platform agent-under-test when --agent-backend=aut (required for AUT mode).",
    )
    parser.add_argument(
        "--aut-agent-config",
        default=os.environ.get("AUT_AGENT_CONFIG", DEFAULT_AUT_AGENT_CONFIG),
        help="Agent config path used to create AUT when it does not already exist (required for AUT mode).",
    )
    parser.add_argument(
        "--aut-seed-providers",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("NAT_AUT_SEED_PROVIDERS", "1").lower() not in {"0", "false", "no"},
        help="Seed ngc-api-key and NVIDIA inference providers before AUT deploy "
        "(default: enabled; disable with --no-aut-seed-providers or NAT_AUT_SEED_PROVIDERS=0).",
    )
    parser.add_argument(
        "--smoke-workspace",
        default=os.environ.get("NAT_SMOKE_WORKSPACE"),
        help="Optional workspace name to create before verifier and delete after verifier. "
        "Useful for framework smoke runs (example: harbor-test-workspace).",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        default=os.environ.get("NAT_ALLOW_DIRTY", "").lower() in {"1", "true", "yes"},
        help="Allow runs against a dirty working tree. By default the runner refuses "
        "to start when uncommitted changes are present so that captured provenance "
        "(commit_sha) is reproducible.",
    )
    parser.add_argument(
        "--commit",
        default=os.environ.get("NAT_COMMIT"),
        help="Pin the run to a specific commit (any ref understood by `git rev-parse` "
        "is accepted: full sha, short sha, branch, tag). When given, the runner sets up "
        "(or reuses) a git worktree at that commit, rebuilds the nhx-agentic-base image "
        "from it, and re-execs itself from the pinned worktree so all source files "
        "(task definitions, verifiers, framework code) come from that commit.",
    )
    parser.add_argument(
        "--worktree-dir",
        type=Path,
        default=os.environ.get("NAT_WORKTREE_DIR"),
        help="Override the location of the pinned worktree when --commit is set "
        "(default: <repo>/.nat-worktrees/<short-sha>). Idempotent: an existing "
        "worktree at the requested sha is reused; mismatched shas error out.",
    )
    args = parser.parse_args()
    if args.agent_model is None:
        args.agent_model = _agent_model_from_env(args.agent_backend)

    # Resolve --jobs-dir to an absolute path. Docker treats any string before
    # the colon in `-v host:container` that does not start with '/' as a
    # *named volume* (whose name forbids slashes), so a relative value like
    # `--jobs-dir nat-jobs` becomes `nat-jobs/<ts>-<task>/agent:/logs/agent`
    # and Docker rejects it with `includes invalid characters for a local
    # volume name`. Resolving here keeps both CLI and notebook callers safe.
    args.jobs_dir = args.jobs_dir.expanduser().resolve()

    # Pinned-commit re-exec gate. When the operator passed --commit (and we
    # have not already re-exec'd ourselves), set up the worktree, build the
    # pinned agentic-base image, and hand off to the worktree's nat_runner.py
    # with --commit/--worktree-dir stripped from argv so the second invocation
    # is a vanilla unpinned run reading source from the pinned tree.
    if args.commit and not os.environ.get(_PINNED_GUARD_ENV):
        try:
            worktree, sha, short_sha, pinned_tag = _setup_pinned_run(
                args.commit, args.worktree_dir, repo_root=REPO_ROOT
            )
        except (RuntimeError, FileNotFoundError) as e:
            print(f"ERROR: failed to set up pinned run: {e}", file=sys.stderr)
            return 2
        target_script = worktree / "tests" / "agentic-use" / "nat_runner.py"
        if not target_script.exists():
            print(
                f"ERROR: pinned worktree {worktree} does not contain {target_script}; "
                "the requested commit predates the agentic-use framework.",
                file=sys.stderr,
            )
            return 2
        new_argv = [sys.executable, str(target_script), *_strip_pinned_args(sys.argv)]
        env = os.environ.copy()
        env[_PINNED_GUARD_ENV] = sha
        env[_PINNED_REF_ENV] = args.commit
        env["NAT_PINNED_IMAGE_TAG"] = pinned_tag
        print(
            f"[nat_runner] Pinning run to commit {short_sha} via worktree {worktree};"
            f" pinned image={pinned_tag}; re-exec'ing from pinned tree."
        )
        os.execvpe(sys.executable, new_argv, env)
        # os.execvpe replaces this process; the lines below are unreachable
        # unless exec failed (in which case execvpe raises OSError).

    if args.list:
        all_tasks = sorted(
            [
                d.name
                for d in TASKS_DIR.iterdir()
                if d.is_dir() and (d / "instruction.md").exists() and d.name != "example-test-template"
            ]
        )
        print("\n".join(all_tasks))
        return 0

    # Argument-shape validation runs before environment validation so a
    # contradictory flag combination (e.g. `--all some-task`) doesn't get
    # masked by a missing-API-key error.
    if args.all and args.manifest is not None:
        print("ERROR: --all and --manifest are mutually exclusive.", file=sys.stderr)
        return 1
    if args.all and args.tasks:
        print(
            "ERROR: --all and TASK_OR_GLOB args are mutually exclusive; remove one.",
            file=sys.stderr,
        )
        return 1
    if args.manifest is not None and args.tasks:
        print("ERROR: Provide either TASK_OR_GLOB args or --manifest, not both.", file=sys.stderr)
        return 1
    try:
        candidate_params = _parse_candidate_params(args.candidate_params)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Validate API keys early
    nvidia_api_key = _secret_from_env("NVIDIA_API_KEY")
    anthropic_api_key = _secret_from_env("ANTHROPIC_API_KEY") or nvidia_api_key
    openai_api_key = _secret_from_env("OPENAI_API_KEY")
    cursor_api_key = _secret_from_env("CURSOR_API_KEY")
    if not args.skip_agent and not args.build_only:
        if args.agent_backend in {"aut", "workflow"} and not nvidia_api_key:
            print("ERROR: NVIDIA_API_KEY environment variable is required for this backend.", file=sys.stderr)
            print("  export NVIDIA_API_KEY=<your-key>  # from https://build.nvidia.com", file=sys.stderr)
            return 1
        if args.agent_backend == "claude-code" and not anthropic_api_key:
            print(
                "ERROR: ANTHROPIC_API_KEY (or NVIDIA_API_KEY fallback) is required for claude-code backend.",
                file=sys.stderr,
            )
            return 1
        if args.agent_backend == "codex" and not openai_api_key and args.codex_auth_json is None:
            print(
                "ERROR: OPENAI_API_KEY or --codex-auth-json is required for codex backend.",
                file=sys.stderr,
            )
            return 1
        if args.agent_backend == "codex" and not openai_api_key and args.codex_auth_json is not None:
            args.codex_auth_json = args.codex_auth_json.expanduser().resolve()
            if not args.codex_auth_json.exists():
                print(f"ERROR: Codex auth file does not exist: {args.codex_auth_json}", file=sys.stderr)
                return 1
        if args.agent_backend == "cursor-agent" and not cursor_api_key:
            print("ERROR: CURSOR_API_KEY environment variable is required for cursor-agent backend.", file=sys.stderr)
            return 1
        if args.agent_backend == "aut":
            if not args.aut_agent_name:
                print(
                    "ERROR: --agent-backend=aut requires --aut-agent-name (or AUT_AGENT_NAME env var).",
                    file=sys.stderr,
                )
                return 1
            if not args.aut_agent_config:
                print(
                    "ERROR: --agent-backend=aut requires --aut-agent-config "
                    "(or AUT_AGENT_CONFIG env var) so missing agents can be created.",
                    file=sys.stderr,
                )
                return 1
            resolved_aut_config = Path(args.aut_agent_config)
            if not resolved_aut_config.is_absolute():
                resolved_aut_config = (REPO_ROOT / resolved_aut_config).resolve()
            if not resolved_aut_config.exists():
                print(
                    "ERROR: AUT config path does not exist: "
                    f"{resolved_aut_config} (from --aut-agent-config={args.aut_agent_config!r})",
                    file=sys.stderr,
                )
                return 1

    # Resolve task list
    patterns: list[str]
    if args.manifest is not None:
        try:
            patterns = _read_manifest(args.manifest)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    else:
        patterns = args.tasks
    try:
        tasks = resolve_tasks([] if args.all else patterns)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not tasks:
        print("ERROR: No tasks specified. Use --all to run all tasks or provide task names.", file=sys.stderr)
        return 1

    print(f"[nat_runner] Running {len(tasks)} task(s): {', '.join(tasks)}")
    backend_suffix = ""
    if args.agent_backend == "aut":
        backend_suffix = (
            f" (agent={args.aut_agent_name}, model={args.agent_model or 'default'}, config={args.aut_agent_config})"
        )
    elif args.agent_backend == "workflow":
        backend_suffix = f" (model={args.agent_model or 'default'})"
    elif args.agent_backend == "claude-code":
        backend_suffix = f" (model={args.agent_model or 'default'}, base_url={args.anthropic_base_url})"
    elif args.agent_backend in {"codex", "cursor-agent"}:
        backend_suffix = f" (model={args.agent_model or 'default'})"
    print(f"[nat_runner] Agent backend: {args.agent_backend}{backend_suffix}")
    env_nhx_base_url = os.environ.get("NHX_BASE_URL")
    if env_nhx_base_url and env_nhx_base_url != args.nhx_base_url:
        print(
            "[nat_runner] NOTE: Ignoring NHX_BASE_URL from environment "
            f"({env_nhx_base_url!r}); using --nhx-base-url={args.nhx_base_url!r}."
        )
    print(f"[nat_runner] Effective NeMo Helix base URL: {args.nhx_base_url}")
    if args.agent_backend == "aut":
        print(f"[nat_runner] Effective AUT config: {args.aut_agent_config}")
    if args.smoke_workspace:
        print(f"[nat_runner] Smoke workspace lifecycle enabled: {args.smoke_workspace}")

    # Capture run-wide provenance once, before the task loop, so every task
    # records the same commit/image fingerprint even if the working tree is
    # rebuilt mid-batch.
    provenance = _capture_repo_provenance()
    provenance["routing_mode"] = "igw"
    provenance["agentic_base_image_digest"] = _capture_image_digest("nhx-agentic-base:latest")
    pinned_sha = os.environ.get(_PINNED_GUARD_ENV)
    pinned_ref = os.environ.get(_PINNED_REF_ENV)
    pinned_image_tag = os.environ.get("NAT_PINNED_IMAGE_TAG")
    if pinned_sha:
        provenance["pinned"] = True
        provenance["pinned_to_commit"] = pinned_ref or pinned_sha
        provenance["pinned_image_tag"] = pinned_image_tag
        # When pinned, dirty=True is impossible by construction (worktree is
        # detached at the resolved sha) so don't fail the guard.
        provenance["commit_dirty"] = False
    else:
        provenance["pinned"] = False
        provenance["pinned_to_commit"] = None
        provenance["pinned_image_tag"] = None
        if provenance.get("commit_dirty") and not args.allow_dirty:
            print(
                "ERROR: working tree is dirty (uncommitted changes detected). "
                "Re-run with --allow-dirty to override (NAT_ALLOW_DIRTY=1 also works), "
                "or commit your changes for reproducible provenance.",
                file=sys.stderr,
            )
            return 2
    pinned_tail = f" pinned={provenance['pinned_to_commit']}" if provenance["pinned"] else ""
    print(
        f"[nat_runner] Provenance: commit={provenance.get('commit_short') or 'unknown'}"
        f" branch={provenance.get('branch') or 'detached'}"
        f" dirty={bool(provenance.get('commit_dirty'))}"
        f" agentic_base={(provenance.get('agentic_base_image_digest') or 'n/a')[:24]}"
        f"{pinned_tail}"
    )

    # Run tasks
    results = []
    failed = []
    for task_name in tasks:
        try:
            result = run_task(
                task_name,
                jobs_dir=args.jobs_dir,
                nvidia_api_key=nvidia_api_key,
                anthropic_api_key=anthropic_api_key,
                anthropic_base_url=args.anthropic_base_url,
                nhx_base_url=args.nhx_base_url,
                agent_model=args.agent_model,
                agent_params=candidate_params,
                codex_auth_json=args.codex_auth_json,
                agent_timeout=args.timeout,
                skip_build=args.skip_build,
                build_only=args.build_only,
                skip_agent=args.skip_agent,
                agent_backend=args.agent_backend,
                aut_agent_name=args.aut_agent_name,
                aut_agent_config=args.aut_agent_config,
                aut_seed_providers=args.aut_seed_providers,
                smoke_workspace=args.smoke_workspace,
                candidate_id=args.candidate_id,
                provenance=provenance,
            )
            results.append(result)
            if args.build_only:
                if result.get("build") not in {"ok", "skipped"}:
                    failed.append(task_name)
            elif result.get("reward") != 1:
                failed.append(task_name)
        except Exception as e:
            print(f"[nat_runner] ERROR running task {task_name}: {e}", file=sys.stderr)
            failed.append(task_name)

    # Summary
    total = len(results)
    if args.build_only:
        build_ok = sum(1 for r in results if r.get("build") == "ok")
        build_skipped = sum(1 for r in results if r.get("build") == "skipped")
        build_failed = len(failed)
        print(f"\n{'=' * 60}")
        print(f"[nat_runner] BUILD SUMMARY: {build_ok + build_skipped}/{total} task images ready")
        print(f"[nat_runner] built: {build_ok}  skipped: {build_skipped}  failed: {build_failed}")
        if failed:
            print(f"[nat_runner] FAILED builds: {', '.join(failed)}")
        print(f"{'=' * 60}")
        return 0 if not failed else 1

    passed = sum(1 for r in results if r.get("reward") == 1)
    token_values = [
        r.get("metrics", {}).get("total_tokens")
        for r in results
        if isinstance(r.get("metrics", {}).get("total_tokens"), int)
    ]
    avg_total_tokens = (sum(token_values) / len(token_values)) if token_values else None
    print(f"\n{'=' * 60}")
    print(f"[nat_runner] SUMMARY: {passed}/{total} tasks passed")
    print(f"[nat_runner] pass_rate: {passed / total:.3f}" if total else "[nat_runner] pass_rate: n/a")
    if avg_total_tokens is not None:
        print(f"[nat_runner] avg_total_tokens: {avg_total_tokens:.2f}")
    else:
        print("[nat_runner] avg_total_tokens: unavailable (no usage payloads)")
    if failed:
        print(f"[nat_runner] FAILED tasks: {', '.join(failed)}")
    print(f"{'=' * 60}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
