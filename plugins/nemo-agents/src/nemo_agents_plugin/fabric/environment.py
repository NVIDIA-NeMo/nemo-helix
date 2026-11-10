# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Platform-owned environment paths before Fabric runtime startup."""

import logging
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.adapters import uses_default_deepagents_backend

logger = logging.getLogger(__name__)

#: Prefix for the directory an agent tree is copied to when its mounted location
#: cannot be written to.
RUNTIME_BASE_DIR_PREFIX = "nemo-agent-runtime"
DEEPAGENTS_WORKSPACE_DIR = ".nemo-deepagents"
_WORKSPACE_DEPENDENCY_DIRS = {".venv", "node_modules", "__pycache__"}


@dataclass(frozen=True, slots=True)
class RuntimeBaseDir:
    """An agent root the runtime can write to, and whether this process created it."""

    path: Path
    staged: bool


def resolve_runtime_base_dir(config_path: Path) -> RuntimeBaseDir:
    """Return an agent root the runtime can write to, copying the tree out if it cannot.

    Fabric resolves everything against the agent root — skills and prompts to
    read, but also the workspace and artifacts to write. In k8s the agent config
    arrives on a read-only ConfigMap ``subPath`` whose parent kubelet creates as
    root, so the runtime user can read its agent but cannot create anything
    beside it. Docker has no such mount and is unaffected.

    A staged copy belongs to one process lifecycle: it goes to a fresh directory
    and :func:`release_runtime_base_dir` removes it at shutdown, so a later
    runtime can never read skills or prompts left behind by an earlier one. The
    tree is bounded by the spec-fileset staging limits, so copying it is cheap
    next to starting a runtime.
    """
    source = config_path.parent
    if os.access(source, os.W_OK):
        return RuntimeBaseDir(path=source, staged=False)

    target = Path(tempfile.mkdtemp(prefix=f"{RUNTIME_BASE_DIR_PREFIX}-"))
    logger.info("Agent root %s is not writable; staging it to %s for the runtime.", source, target)
    shutil.copytree(source, target, dirs_exist_ok=True)
    # copytree carries the source's mode across, so the copy of a read-only mount
    # is read-only too — restore write access on the directories we just made.
    for directory in (target, *(path for path in target.rglob("*") if path.is_dir())):
        directory.chmod(directory.stat().st_mode | stat.S_IRWXU)
    return RuntimeBaseDir(path=target, staged=True)


def release_runtime_base_dir(base_dir: RuntimeBaseDir) -> None:
    """Remove a staged agent root. A root used in place is left alone."""
    if not base_dir.staged:
        return
    shutil.rmtree(base_dir.path, ignore_errors=True)
    logger.info("Removed staged agent root %s.", base_dir.path)


def ensure_local_workspace_dir(agent_config: AgentConfig, base_dir: Path) -> None:
    """Create the configured local workspace relative to the agent base directory."""
    if agent_config.environment.provider != "local":
        return

    configured_workspace = Path(agent_config.environment.workspace)
    if configured_workspace.is_absolute():
        raise ValueError("Local workspace path must be relative to the agent base directory.")

    resolved_base_dir = base_dir.resolve()
    workspace = (resolved_base_dir / configured_workspace).resolve()
    if not workspace.is_relative_to(resolved_base_dir):
        raise ValueError("Local workspace path must remain within the agent base directory.")

    workspace.mkdir(parents=True, exist_ok=True)


def prepare_deepagents_workspace_baseline(agent_config: AgentConfig, base_dir: Path) -> Path | None:
    """Save the initial workspace inputs before serving conversations.

    Initial capture assumes a fresh deployment workspace. Keep this baseline
    across process restarts; recapturing the live workspace could include session
    output. Replacing the deployment workspace also replaces its baseline.
    """
    harness = agent_config.harnesses.get(agent_config.default_harness or "")
    if harness is None or not uses_default_deepagents_backend(harness):
        return None
    if agent_config.environment.provider != "local" or not agent_config.environment.workspace:
        return None

    ensure_local_workspace_dir(agent_config, base_dir)
    workspace = (base_dir / agent_config.environment.workspace).resolve()
    storage = workspace / DEEPAGENTS_WORKSPACE_DIR
    baseline = storage / "inputs"
    if storage.is_symlink() or baseline.is_symlink():
        raise ValueError("DeepAgents workspace baseline directories must not be symlinks.")
    if baseline.is_dir():
        return baseline
    storage.mkdir(exist_ok=True)

    excluded = {storage, *(workspace / name for name in ("large_tool_results", "conversation_history", ".fabric"))}
    for configured_path in (
        agent_config.environment.artifacts,
        agent_config.runtime.artifacts,
        agent_config.telemetry.output_dir,
    ):
        if configured_path:
            output = (base_dir / configured_path).resolve()
            if output != workspace and output.is_relative_to(workspace):
                excluded.add(output)

    # Publish only a complete copy; a failed startup must not leave a reusable partial baseline.
    with tempfile.TemporaryDirectory(prefix=".inputs-", dir=storage) as temporary:
        staged = Path(temporary) / "inputs"
        _copy_workspace_inputs(workspace, staged, excluded)
        staged.rename(baseline)
    return baseline


def _copy_workspace_inputs(source: Path, target: Path, excluded: set[Path]) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            path = Path(directory) / name
            resolved = path.resolve()
            if (
                not resolved.is_relative_to(source)
                or _WORKSPACE_DEPENDENCY_DIRS.intersection(resolved.relative_to(source).parts)
                or any(resolved.is_relative_to(root) for root in excluded)
                or (path.is_dir() and (path / "pyvenv.cfg").is_file())
            ):
                ignored.add(name)
        return ignored

    shutil.copytree(source, target, symlinks=True, ignore=ignore)
    # Internal links must resolve inside each copied workspace, including originally absolute links.
    for link in target.rglob("*"):
        if link.is_symlink():
            original = source / link.relative_to(target)
            destination = target / original.resolve().relative_to(source)
            link.unlink()
            link.symlink_to(os.path.relpath(destination, link.parent))
