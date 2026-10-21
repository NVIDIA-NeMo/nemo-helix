# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a Harbor agent's ``setup()`` inside the sandbox's agent setup egress window.

Harbor builds the agent from :data:`SETUP_EGRESS_AGENT` with the real agent (a built-in name or an
import path) in the ``agent_setup_egress_inner`` kwarg and the setup hosts in ``agent_setup_egress_hosts``.
What it gets back is an instance of a subclass of the real agent whose only change is ``setup()``, so
everything else (name, version, run, context population, error patterns) is the real agent's.

An environment with its own ``agent_setup_egress()`` window (our OpenSandbox environment) opens it.
Any other Harbor environment that can switch to an allowlist after start gets the setup hosts added to
the task's policy through Harbor's ``set_network_policy()`` for the duration of ``setup()``, and on close
the processes setup left holding an outside connection are killed. Elsewhere, setup runs under the task's
own policy.
"""

from __future__ import annotations

import shlex
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from functools import cache
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.acp_registry import is_acp_registry_shorthand, registry_spec_from_agent_name
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.config import AgentConfig
from nemo_evals.harbor.setup_cleanup import KILL_CONNECTION_HOLDERS_SCRIPT, LIST_PROCESSES_SCRIPT, SHELL, SYSTEM_PATH

SETUP_EGRESS_AGENT = "nemo_evals.harbor.agent_setup_egress:AgentSetupEgress"
INNER_AGENT_KWARG = "agent_setup_egress_inner"
SETUP_HOSTS_KWARG = "agent_setup_egress_hosts"


@cache
def _inner_agent_class(inner: str) -> type[BaseAgent]:
    return AgentFactory.get_agent_class_from_config(AgentConfig(name=inner))


class _CommandGuard:
    """Refuses every command on ``environment`` while its setup egress window is closing.

    ``exec`` and ``service_exec`` are wrapped on the instance; the cleanup's own commands go through the
    originals. A close that fails or is cancelled leaves the guard refusing, so task commands (artifact
    collection, the verifier) never run with the setup hosts possibly still reachable.
    """

    def __init__(self, environment: BaseEnvironment) -> None:
        self.closing = False
        self._environment = environment
        self._exec = environment.exec
        self._service_exec = environment.service_exec
        self._shadowed = {
            name: vars(environment)[name] for name in ("exec", "service_exec") if name in vars(environment)
        }
        # Instance attributes shadow the class's methods; Harbor's own service_exec calls self.exec.
        setattr(environment, "exec", self.exec)
        setattr(environment, "service_exec", self.service_exec)

    def _refuse_while_closing(self) -> None:
        if self.closing:
            raise RuntimeError(
                "the agent setup egress window did not close cleanly; refusing to run more commands in this environment"
            )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        self._refuse_while_closing()
        return await self._exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    async def service_exec(
        self,
        command: str,
        *,
        service: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        self._refuse_while_closing()
        return await self._service_exec(command, service=service, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    def release(self) -> None:
        """Puts the environment's own ``exec`` and ``service_exec`` back; only after a clean close."""
        for name in ("exec", "service_exec"):
            if name in self._shadowed:
                setattr(self._environment, name, self._shadowed[name])
            else:
                delattr(self._environment, name)

    async def run_script(self, script: str, **env: str) -> ExecResult:
        return await self._exec(
            f"{SHELL} -c {shlex.quote(script)}", cwd="/", env={"PATH": SYSTEM_PATH, **env}, user="root"
        )

    async def list_processes(self) -> tuple[str, ...]:
        result = await self.run_script(LIST_PROCESSES_SCRIPT)
        if result.return_code != 0:
            raise RuntimeError(f"could not list environment processes (exit {result.return_code}): {result.stderr}")
        return tuple(line.strip() for line in (result.stdout or "").splitlines() if ":" in line)

    async def kill_connection_holders_except(self, keep: tuple[str, ...]) -> None:
        result = await self.run_script(KILL_CONNECTION_HOLDERS_SCRIPT, KEEP=" ".join(keep))
        if result.return_code != 0:
            raise RuntimeError(
                f"could not stop processes left by agent setup (exit {result.return_code}): {result.stderr}"
            )


@asynccontextmanager
async def _network_policy_window(environment: BaseEnvironment, hosts: tuple[str, ...]) -> AsyncGenerator[None, None]:
    """Add ``hosts`` to the environment's network policy for the block, then restore the policy.

    A public baseline is already open, and an environment that can't switch to an allowlist after start
    keeps the task's policy. Harbor's ``set_network_policy()`` rejects hosts the provider can't express.

    Providers keep established connections across a policy change, so closing also kills every process
    started since entry that still holds a connection outside the environment. Until closing has finished,
    the environment refuses every other command (see :class:`_CommandGuard`).
    """
    if not hosts:
        yield
        return
    capabilities = environment.capabilities
    baseline = environment.network_policy
    if baseline.network_mode == NetworkMode.PUBLIC or not (
        capabilities.dynamic_network_policy and capabilities.network_allowlist
    ):
        yield
        return
    allowed = list(dict.fromkeys([*baseline.allowed_hosts, *hosts]))
    guard = _CommandGuard(environment)
    try:
        keep = await guard.list_processes()
    except BaseException:
        guard.release()
        raise
    try:
        await environment.set_network_policy(NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=allowed))
        yield
    finally:
        guard.closing = True
        await environment.set_network_policy(baseline)
        await guard.kill_connection_holders_except(keep)
        guard.closing = False
        guard.release()


@cache
def _with_setup_egress(agent_class: type[BaseAgent], hosts: tuple[str, ...]) -> type[BaseAgent]:
    class SetupEgressAgent(agent_class):  # ty: ignore[unsupported-base]  # a class known only at run time
        async def setup(self, environment: BaseEnvironment) -> None:
            window = getattr(environment, "agent_setup_egress", None)
            if window is None:
                async with _network_policy_window(environment, hosts):
                    await super().setup(environment)
                return
            async with window():
                await super().setup(environment)

    SetupEgressAgent.__name__ = agent_class.__name__
    SetupEgressAgent.__qualname__ = agent_class.__qualname__
    SetupEgressAgent.__module__ = agent_class.__module__
    return SetupEgressAgent


class AgentSetupEgress:
    """Stands in for the real agent class in Harbor's ``AgentConfig.import_path``; never instantiated itself."""

    def __new__(cls, *args: Any, **kwargs: Any) -> BaseAgent:
        inner = kwargs.pop(INNER_AGENT_KWARG)
        hosts = tuple(kwargs.pop(SETUP_HOSTS_KWARG, ()))
        if is_acp_registry_shorthand(inner):
            kwargs["registry_spec"] = registry_spec_from_agent_name(inner)
        return _with_setup_egress(_inner_agent_class(inner), hosts)(*args, **kwargs)

    @classmethod
    def preflight(cls, kwargs: dict[str, Any] | None = None, env: Mapping[str, str] | None = None) -> None:
        """The real agent's preflight, without the wrapper's kwargs."""
        options = dict(kwargs or {})
        inner = options.pop(INNER_AGENT_KWARG)
        options.pop(SETUP_HOSTS_KWARG, None)
        _inner_agent_class(inner).preflight(kwargs=options, env=env)
