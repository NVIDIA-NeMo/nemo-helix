# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Platform-owned environment paths before Fabric runtime startup."""

import logging
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.adapters import uses_default_deepagents_backend
from nemo_fabric import FabricConfig

logger = logging.getLogger(__name__)

#: Prefix for the directory an agent tree is copied to when its mounted location
#: cannot be written to.
RUNTIME_BASE_DIR_PREFIX = "nemo-agent-runtime"
DEEPAGENTS_WORKSPACE_DIR = ".nemo-deepagents"
DEEPAGENTS_SKILLS_DIR = ".agents/skills"
_WORKSPACE_DEPENDENCY_DIRS = {".venv", "node_modules", "__pycache__"}
_MCP_COMMAND_ENV_VAR_RE = re.compile(r"\$(\w+|\{[^}]*\})", re.ASCII)
_MCP_STDIO_LAUNCHER = "import os, sys; os.chdir(sys.argv[1]); os.execvpe(sys.argv[2], sys.argv[2:], os.environ)"


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
        _stage_deepagents_skills(agent_config, base_dir, workspace, staged, excluded)
        staged.rename(baseline)
    return baseline


def create_deepagents_workspace(baseline: Path) -> Path:
    """Copy the saved inputs into a private workspace for one runtime."""
    sessions = baseline.parent / "sessions"
    if sessions.is_symlink():
        raise ValueError("DeepAgents session storage must not be a symlink.")
    sessions.mkdir(exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="session-", dir=sessions))
    try:
        # Baseline links are already relative to the copied tree.
        shutil.copytree(baseline, workspace, symlinks=True, dirs_exist_ok=True)
    except BaseException:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    return workspace


def discover_skill_directories(directory: Path) -> list[Path]:
    """Resolve one skill bundle or the bundles immediately inside a collection."""
    if not directory.is_dir():
        raise ValueError(f"Skill path {directory} must reference an existing directory.")
    if (directory / "SKILL.md").is_file():
        return [directory.resolve()]
    skills = [child.resolve() for child in sorted(directory.iterdir()) if (child / "SKILL.md").is_file()]
    if not skills:
        raise ValueError(f"Skill path {directory} does not contain SKILL.md or any immediate child skill bundles.")
    return skills


def _stage_deepagents_skills(
    config: AgentConfig, base_dir: Path, workspace: Path, target: Path, excluded: set[Path]
) -> None:
    if config.skills is None:
        return
    sources: dict[str, Path] = {}
    for configured_path in config.skills.paths:
        for source in discover_skill_directories(base_dir / Path(configured_path).expanduser()):
            previous = sources.setdefault(source.name, source)
            if previous != source:
                raise ValueError(f"Skill directories {previous} and {source} both stage as {source.name!r}.")

    for name, source in sources.items():
        relative = Path(DEEPAGENTS_SKILLS_DIR) / name
        destination = target / relative
        if destination.exists() or destination.is_symlink():
            if source != (workspace / relative).resolve():
                raise ValueError(f"Cannot stage skill {source}: workspace path {relative} already exists.")
            # Replace only this bundle's workspace copy so linked skill assets are included.
            if destination.is_symlink():
                destination.unlink()
            else:
                shutil.rmtree(destination)
        _copy_workspace_inputs(source, destination, excluded, follow_symlinks=True)


def deepagents_workspace_config(config: AgentConfig, *, base_dir: Path, workspace_dir: Path) -> AgentConfig:
    """Derive a runtime config for a workspace populated from the DeepAgents baseline."""
    runtime_config = config.model_copy(deep=True)
    runtime_config.environment.workspace = workspace_dir.resolve().relative_to(base_dir.resolve()).as_posix()
    if runtime_config.skills is not None and runtime_config.skills.paths:
        runtime_config.skills.paths = [f"/{DEEPAGENTS_SKILLS_DIR}"]
    return runtime_config


def preserve_deepagents_mcp_cwd(config: FabricConfig, *, workspace: Path) -> None:
    """Keep local MCP processes at their original cwd when file tools move to a private workspace."""
    if config.mcp is None:
        return
    runtime_env = {**os.environ, **(config.environment.env if config.environment is not None else {})}
    for server in config.mcp.servers.values():
        if server.transport != "stdio":
            continue
        # Match the adapter's expandvars/strip before moving the command into args.
        command = _MCP_COMMAND_ENV_VAR_RE.sub(
            lambda match: runtime_env.get(match.group(1).removeprefix("{").removesuffix("}"), match.group(0)),
            server.url,
        ).strip()
        # Isolate only the launcher from Python startup hooks; exec retains the server's env.
        server.args = ["-I", "-S", "-c", _MCP_STDIO_LAUNCHER, str(workspace), command, *server.args]
        server.url = sys.executable


def _copy_workspace_inputs(source: Path, target: Path, excluded: set[Path], *, follow_symlinks: bool = False) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            path = Path(directory) / name
            resolved = path.resolve()
            relative = resolved.relative_to(source) if resolved.is_relative_to(source) else path.relative_to(source)
            if (
                (not follow_symlinks and not resolved.is_relative_to(source))
                or _WORKSPACE_DEPENDENCY_DIRS.intersection(relative.parts)
                or any(resolved.is_relative_to(root) for root in excluded)
                or (path.is_dir() and (path / "pyvenv.cfg").is_file())
            ):
                ignored.add(name)
        return ignored

    shutil.copytree(source, target, symlinks=not follow_symlinks, ignore=ignore)
    if follow_symlinks:
        return  # Configured skill bundles get independent copies of linked assets, too.
    # Internal links must resolve inside each copied workspace, including originally absolute links.
    for link in target.rglob("*"):
        if link.is_symlink():
            original = source / link.relative_to(target)
            destination = target / original.resolve().relative_to(source)
            link.unlink()
            link.symlink_to(os.path.relpath(destination, link.parent))
