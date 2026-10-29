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

A setup that fails on the network (Harbor's :class:`NetworkConnectionError`, or a window whose hosts don't
resolve yet) is retried in a freshly opened window, after a short wait. Harbor's setup timeout covers every
attempt together, so retrying never makes a trial take longer than its setup budget.
"""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from functools import cache
from typing import Any, ClassVar

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.acp_registry import is_acp_registry_shorthand, registry_spec_from_agent_name
from harbor.agents.installed.base import NetworkConnectionError
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.config import AgentConfig
from nemo_evals.harbor.setup_cleanup import KILL_CONNECTION_HOLDERS_SCRIPT, LIST_PROCESSES_SCRIPT, SHELL, SYSTEM_PATH
from nemo_evals.harbor.setup_egress_readiness import SetupEgressNotReadyError

SETUP_EGRESS_AGENT = "nemo_evals.harbor.agent_setup_egress:AgentSetupEgress"
INNER_AGENT_KWARG = "agent_setup_egress_inner"
SETUP_HOSTS_KWARG = "agent_setup_egress_hosts"

SETUP_RETRY_DELAYS_SEC = (5.0, 15.0)
"""The wait before each retry of a setup that failed on the network; one retry per entry."""
NETWORK_SETUP_ERRORS = (NetworkConnectionError, SetupEgressNotReadyError)


@cache
def _inner_agent_class(inner: str) -> type[BaseAgent]:
    return AgentFactory.get_agent_class_from_config(AgentConfig(name=inner))


# Every BaseEnvironment method that runs something inside the environment or moves files in or out of it.
# Providers implement transfers through their own private exec (Daytona, Modal and Beam run tar), so
# guarding exec alone would let a download run. start, stop, attach and set_network_policy stay usable.
GUARDED_METHODS = (
    "exec",
    "service_exec",
    "upload_file",
    "upload_dir",
    "download_file",
    "download_dir",
    "download_dir_with_exclusions",
    "download_dir_filtered",
    "service_download_file",
    "service_download_dir",
    "service_download_dir_with_exclusions",
    "is_dir",
    "is_file",
    "service_is_dir",
    "reset_dirs",
    "ensure_dirs",
    "empty_dirs",
    "run_healthcheck",
)


class _CommandGuard:
    """Refuses every command and file transfer on ``environment`` while its setup egress window is closing.

    The :data:`GUARDED_METHODS` are wrapped on the instance; the cleanup's own commands go through the
    original ``exec``. A close that fails or is cancelled leaves the guard refusing, so task commands
    (artifact collection, log downloads, the verifier) never run with the setup hosts possibly still reachable.
    """

    def __init__(self, environment: BaseEnvironment) -> None:
        self.closing = False
        self._environment = environment
        self._exec = environment.exec
        self._shadowed = {name: vars(environment)[name] for name in GUARDED_METHODS if name in vars(environment)}
        # Instance attributes shadow the class's methods, including where Harbor calls them on self.
        for name in GUARDED_METHODS:
            setattr(environment, name, self._refusing(getattr(environment, name)))

    def _refusing(self, method: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def refusing(*args: Any, **kwargs: Any) -> Any:
            if self.closing:
                raise RuntimeError(
                    "the agent setup egress window did not close cleanly; refusing to run more commands in this "
                    "environment"
                )
            return await method(*args, **kwargs)

        return refusing

    def release(self) -> None:
        """Puts the environment's own methods back; only after a clean close."""
        for name in GUARDED_METHODS:
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

    A public baseline is already open, and an environment that can't switch to an allowlist after start,
    or that runs compose services, keeps the task's policy. A policy the provider can't express (a wildcard
    host it doesn't support) raises before anything changes.

    Providers keep established connections across a policy change, so closing also kills every process
    started since entry that still holds a connection outside the environment. Until closing has finished,
    the environment refuses every other command and transfer (see :class:`_CommandGuard`), as it does for
    good once opening fails partway.
    """
    if not hosts:
        yield
        return
    capabilities = environment.capabilities
    baseline = environment.network_policy
    if (
        baseline.network_mode == NetworkMode.PUBLIC
        or not (capabilities.dynamic_network_policy and capabilities.network_allowlist)
        or _uses_compose(environment)
    ):
        yield
        return
    opened = NetworkPolicy(
        network_mode=NetworkMode.ALLOWLIST, allowed_hosts=list(dict.fromkeys([*baseline.allowed_hosts, *hosts]))
    )
    environment.validate_network_policy_support(opened)
    guard = _CommandGuard(environment)
    try:
        keep = await guard.list_processes()
    except BaseException:
        guard.release()
        raise
    try:
        await environment.set_network_policy(opened)
    except BaseException:
        # The provider may have applied part of the policy, but Harbor still caches the baseline, so
        # restoring it would be a no-op. The policy is unknown: refuse everything from here on.
        guard.closing = True
        raise
    try:
        yield
    finally:
        guard.closing = True
        await environment.set_network_policy(baseline)
        await guard.kill_connection_holders_except(keep)
        guard.closing = False
        guard.release()


def _uses_compose(environment: BaseEnvironment) -> bool:
    """Whether the environment runs compose services, which share its network but not its processes.

    The cleanup only sees the main service's processes, so a compose environment keeps the task's policy.
    Every Harbor provider that supports compose reports per instance whether it uses it.
    """
    return environment.capabilities.docker_compose and environment._uses_compose


@cache
def _with_setup_egress(agent_class: type[BaseAgent], hosts: tuple[str, ...]) -> type[BaseAgent]:
    class SetupEgressAgent(agent_class):  # ty: ignore[unsupported-base]  # a class known only at run time
        async def setup(self, environment: BaseEnvironment) -> None:
            inner_setup = super().setup
            for attempt, delay in enumerate((*SETUP_RETRY_DELAYS_SEC, None), start=1):
                try:
                    await _setup_in_window(inner_setup, environment, hosts)
                    return
                except NETWORK_SETUP_ERRORS as exc:
                    if delay is None:
                        raise
                    self.logger.warning(
                        "Agent setup failed on the network (attempt %d): %s; retrying in %.0f s",
                        attempt,
                        str(exc).splitlines()[0],
                        delay,
                    )
                    await asyncio.sleep(delay)

    SetupEgressAgent.__name__ = agent_class.__name__
    SetupEgressAgent.__qualname__ = agent_class.__qualname__
    SetupEgressAgent.__module__ = agent_class.__module__
    return SetupEgressAgent


async def _setup_in_window(
    setup: Callable[[BaseEnvironment], Awaitable[None]], environment: BaseEnvironment, hosts: tuple[str, ...]
) -> None:
    window = getattr(environment, "agent_setup_egress", None)
    if window is None:
        async with _network_policy_window(environment, hosts):
            await setup(environment)
        return
    async with window():
        await setup(environment)


def setup_timeout_multiplier(inner: str, multiplier: float) -> float:
    """The setup timeout multiplier that gives ``inner`` its own setup timeout through the wrapper.

    Harbor 0.24 reads ``DEFAULT_SETUP_TIMEOUT_SEC`` from the class it imports, which is :class:`AgentSetupEgress`
    and can't know the inner agent. Scaling the multiplier keeps the inner agent's timeout (1200 s for
    hermes) times the caller's ``multiplier``. Harbor 0.23 has no per-agent default and uses 360 s for
    every agent, so there the multiplier is left as it is.
    """
    default = AgentSetupEgress.DEFAULT_SETUP_TIMEOUT_SEC
    inner_timeout = getattr(_inner_agent_class(inner), "DEFAULT_SETUP_TIMEOUT_SEC", default)
    return multiplier * inner_timeout / default


class AgentSetupEgress:
    """Stands in for the real agent class in Harbor's ``AgentConfig.import_path``; never instantiated itself."""

    DEFAULT_SETUP_TIMEOUT_SEC: ClassVar[float] = 360.0
    """Harbor's own default; :func:`setup_timeout_multiplier` scales it to the inner agent's."""

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
