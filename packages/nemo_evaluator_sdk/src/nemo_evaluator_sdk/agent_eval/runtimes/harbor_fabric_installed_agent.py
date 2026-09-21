# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Harbor *installed* agent that runs a NeMo Fabric harness on any task image.

``nemo_fabric.integrations.harbor:FabricAgent`` installs Fabric with ``python3 -m venv`` + pip, so it
only runs on task images that already ship a new enough CPython -- most Harbor tasks do not. This
agent keeps Fabric's spec/result protocol and replaces only the install step: it provisions curl
through whatever package manager the image has, installs ``uv``, and builds the Fabric virtualenv
from a uv-managed interpreter. Nothing is required of the task's Dockerfile.

Select it with ``agent_import_path`` and configure it with ``agent_kwargs``::

    HarborRuntimeConfig(
        agent_import_path="nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_installed_agent:FabricInstalledAgent",
        agent_kwargs={
            "fabric_adapter_id": "nvidia.fabric.langchain.deepagents",
            "fabric_package": "nemo-fabric[deepagents,relay]==0.3.0b1",
        },
        agent_model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
        agent_env_from_host=["NVIDIA_API_KEY"],
    )

It accepts every ``FabricAgent`` keyword plus :class:`NemoFabricAgent`'s ``fabric_model_api_key_env``
and this class's ``fabric_python_version``. ``fabric_package`` is required -- the harness extra to
install cannot be derived from the adapter id.

Two things the task image must still provide: a supported package manager (apt-get, dnf, yum, or
apk) when it has no curl, and glibc. ``nemo-fabric-runtime`` publishes no musllinux wheels, so
Alpine-based tasks fail at the final ``uv pip install`` with an unsatisfiable resolution -- verified
against 0.3.0b1, and not something this agent can work around.
"""

from __future__ import annotations

import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_fabric_agent import NemoFabricAgent

#: Interpreter uv provisions for the Fabric virtualenv. Fabric's Harbor extra needs >= 3.12.
DEFAULT_FABRIC_PYTHON_VERSION = "3.12"
#: Turn budget applied when the caller names none.
#:
#: Fabric defaults ``max_turns`` and ``timeout_seconds`` to ``None``, which leaves the harness
#: unbounded. On a task it cannot finish, the loop then runs until Harbor kills the agent phase, and
#: a killed phase produces no ``RunResult`` at all -- no trajectory, no error, nothing to debug, just
#: an ``AgentTimeoutError``. Measured on `terminal-bench-sample`: six of ten trials died that way.
#: A ceiling cannot be derived from the task, so this is a deliberately generous guess whose only job
#: is to make the harness stop on its own terms. Pass ``fabric_max_turns`` to size it properly, or
#: ``fabric_max_turns=None`` for Fabric's unbounded behaviour.
DEFAULT_FABRIC_MAX_TURNS = 50
_UV_INSTALLER_URL = "https://astral.sh/uv/install.sh"
#: Proxy and TLS settings ``FabricAgent`` forwards to its install step, mirrored for this one.
_INSTALL_ENV_NAMES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "PIP_CERT",
        "PIP_CLIENT_CERT",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        "PIP_INDEX_URL",
        "PIP_TRUSTED_HOST",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


class FabricInstalledAgent(BaseInstalledAgent):
    """A Harbor installed agent that delegates the run to a wrapped :class:`NemoFabricAgent`.

    Harbor's install scaffolding and Fabric's run protocol are combined by composition rather than
    by inheriting both. Subclassing ``BaseInstalledAgent`` *and* ``NemoFabricAgent`` also works at
    runtime, but it inherits two incompatible declarations of ``options_model`` -- ``BaseAgent``
    declares ``type[AgentOptions] | None`` and ``BaseInstalledAgent`` narrows it to
    ``InstalledAgentOptions`` -- which nothing can reconcile under a type checker that compares
    overrides invariantly.

    Composition also states the intent directly: this class owns installation, so
    ``FabricAgent.setup()`` -- the ``python3 -m venv`` step that cannot run on a bare image -- is
    never called, rather than being shadowed by method resolution order.

    Its cost is that Harbor assigns some agent state *after* construction, which then has to reach
    the wrapped agent; :attr:`_HARBOR_ASSIGNED_ATTRIBUTES` names those fields and :meth:`run` copies
    them across. Properties would forward them more automatically but cannot be typed: ``BaseAgent``
    declares ``session_id`` and ``context_id`` as plain attributes, and overriding an attribute with
    a property is itself an incompatible override.
    """

    SUPPORTS_ATIF = True

    #: Declared, never assigned here: Harbor attaches the resolved skill list at dispatch time and
    #: ``BaseAgent`` does not declare it, so this is the only place it is typed. Leaving it unset
    #: keeps the ``hasattr`` guard in :meth:`_copy_harbor_assigned_state` meaningful.
    skills: list[str]

    def __init__(
        self,
        logs_dir: Path,
        prompt_template_path: Path | str | None = None,
        version: str | None = None,
        extra_env: dict[str, str] | None = None,
        *,
        fabric_python_version: str | float | int = DEFAULT_FABRIC_PYTHON_VERSION,
        model_name: str | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        **fabric_kwargs: Any,
    ) -> None:
        # YAML reads an unquoted `3.10` as the float 3.1, which would silently install the wrong
        # interpreter; the intent is unrecoverable here, so reject it.
        if not isinstance(fabric_python_version, str):
            raise ValueError(
                f"fabric_python_version must be a string: quote it as {str(fabric_python_version)!r} "
                "so a version like 3.10 survives YAML parsing"
            )
        if not fabric_python_version.strip():
            raise ValueError("fabric_python_version must not be empty")
        self.fabric_python_version = fabric_python_version
        super().__init__(
            logs_dir,
            prompt_template_path,
            version,
            extra_env,
            model_name=model_name,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
        )
        # An explicit `fabric_max_turns=None` still means unbounded; only absence takes the default.
        fabric_kwargs.setdefault("fabric_max_turns", DEFAULT_FABRIC_MAX_TURNS)
        # `**fabric_kwargs` rather than an enumerated signature, so every FabricAgent constructor
        # argument stays reachable without this class tracking upstream's parameter list.
        self.fabric = NemoFabricAgent(
            logs_dir=logs_dir,
            extra_env=extra_env,
            model_name=model_name,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
            **fabric_kwargs,
        )
        if not self.fabric.fabric_package:
            raise ValueError(
                "fabric_package is required: name the Fabric distribution and harness extra to "
                'install into the task container, e.g. "nemo-fabric[deepagents,relay]==0.3.0b1"'
            )

    @staticmethod
    def name() -> str:
        return "nemo-fabric-installed"

    # -- State Harbor assigns after construction, copied to the wrapped agent ---------------------

    #: Attributes Harbor sets on the agent object it built, which the wrapped agent has to see.
    #: ``session_id``/``context_id`` come from ``trial.py`` and ``regrade.py``; Fabric reads both
    #: into its run context as ``harbor_session_id``/``harbor_context_id``. ``skills`` comes from
    #: ``trial.py``, ``job.py``, and ``job_plan.py`` and is not declared on ``BaseAgent`` at all --
    #: Harbor attaches it dynamically -- so nothing but this list records that it must be copied.
    _HARBOR_ASSIGNED_ATTRIBUTES = ("session_id", "context_id", "skills")

    def _copy_harbor_assigned_state(self) -> None:
        for attribute in self._HARBOR_ASSIGNED_ATTRIBUTES:
            if hasattr(self, attribute):
                setattr(self.fabric, attribute, getattr(self, attribute))

    # -- Install ---------------------------------------------------------------------------------

    def get_version_command(self) -> str | None:
        return (
            f"{shlex.quote(self._venv_python)} -c "
            "\"import importlib.metadata; print(importlib.metadata.version('nemo-fabric-runtime'))\""
        )

    async def install(self, environment: BaseEnvironment) -> None:
        """Provision curl, uv, a uv-managed interpreter, and the Fabric virtualenv."""
        await self._ensure_curl(environment)
        await self._prepare_directories(environment)
        bundle = self.fabric.fabric_config_bundle
        if bundle is not None:
            await environment.upload_dir(bundle, self.fabric.fabric_config_target)
        await self.exec_as_agent(
            environment,
            command=self._install_fabric_command(),
            env={name: value for name, value in self.extra_env.items() if name in _INSTALL_ENV_NAMES},
            timeout_sec=self.fabric.fabric_timeout_sec,
        )

    async def _ensure_curl(self, environment: BaseEnvironment) -> None:
        """Install curl and CA certificates through whichever package manager the image has.

        Harbor 0.23 offers ``ensure_system_dependencies()`` for this; the repository pins 0.20, so
        the same walk lives here. Both are short-circuited by an image that already has curl, except
        that CA certificates are installed unconditionally: ``ubuntu:24.04`` ships neither, and
        without the certificates the uv installer's HTTPS fetch fails with a bare curl exit code
        that reads like a network outage.
        """
        await self.exec_as_root(
            environment,
            command=(
                "if command -v apt-get >/dev/null 2>&1; then "
                "apt-get update && apt-get install -y curl ca-certificates; "
                "elif command -v apk >/dev/null 2>&1; then apk add --no-cache curl ca-certificates; "
                "elif command -v dnf >/dev/null 2>&1; then dnf install -y curl ca-certificates; "
                "elif command -v yum >/dev/null 2>&1; then yum install -y curl ca-certificates; "
                "elif command -v curl >/dev/null 2>&1; then true; "
                "else echo 'curl is required to install uv, and no supported package manager was found' >&2; "
                "exit 1; fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

    async def _prepare_directories(self, environment: BaseEnvironment) -> None:
        """Create the dirs Fabric writes to and hand them to the agent user.

        ``FabricAgent.setup()`` makes these as the agent user; here they may land on a root-owned
        mount, so create them as root and chown instead.
        """
        paths = ["/logs/agent", self._venv_path]
        if self.fabric.fabric_config_bundle is not None:
            paths.append(self.fabric.fabric_config_target)
        quoted = " ".join(shlex.quote(path) for path in paths)
        user = shlex.quote(str(environment.default_user or "root"))
        await self.exec_as_root(
            environment,
            command=f"mkdir -p {quoted} && chown -R {user}:{user} {quoted}",
        )

    def _install_fabric_command(self) -> str:
        """Install uv, then build the Fabric virtualenv from a uv-managed interpreter.

        The task image's own python is never used: it is frequently absent, and when present it is
        often older than the 3.12 Fabric's Harbor integration needs.
        """
        package = self.fabric.fabric_package
        if not package:  # Rejected in __init__; re-checked so the command builder stays total.
            raise ValueError("fabric_package is required")
        python_version = shlex.quote(self.fabric_python_version)
        return (
            "set -euo pipefail; "
            f"curl -LsSf {shlex.quote(_UV_INSTALLER_URL)} | sh; "
            'if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; fi; '
            f"uv python install {python_version}; "
            f"uv venv {shlex.quote(self._venv_path)} --python {python_version} --clear; "
            f"uv pip install --python {shlex.quote(self._venv_python)} {shlex.quote(package)}"
        )

    @property
    def _venv_path(self) -> str:
        venv = PurePosixPath(self.fabric.fabric_venv_path)
        if not venv.is_absolute() or ".." in venv.parts:
            raise ValueError("fabric_venv_path must be an absolute task-environment path")
        return str(venv)

    @property
    def _venv_python(self) -> str:
        """The interpreter Fabric runs under. Mirrors ``FabricAgent._runner_python``."""
        return str(PurePosixPath(self._venv_path) / "bin" / "python")

    # -- Run -------------------------------------------------------------------------------------

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self._copy_harbor_assigned_state()
        await self.fabric.run(instruction, environment, context)

    def populate_context_post_run(self, context: AgentContext) -> None:
        self.fabric.populate_context_post_run(context)
