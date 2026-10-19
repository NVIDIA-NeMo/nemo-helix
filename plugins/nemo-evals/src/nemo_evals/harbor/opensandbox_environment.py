# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox environment task Jobs run trials in: scaled-evals' environment plus an agent channel.

A no-network task (``allow_internet = false``) normally gets a deny-all sandbox. An agent that runs
inside the sandbox (codex, hermes, ...) then can't reach its model. ``agent_channel_hosts`` keeps
those hosts (the inference gateway) reachable for such tasks; the internet stays closed. Imported
only by the dispatcher image, which carries the ``scaled-evals`` group.

``agent_setup_hosts`` lets an agent that installs itself from the network (apt, npm, PyPI) run on a
no-network task. The dispatcher rewrites such a task so its sandbox starts on an allowlist of those
hosts and its agent and verifier phases are no-network; Harbor then calls ``set_network_policy``
between ``agent.setup()`` and ``agent.run()``. Each switch removes and adds egress rules on the live
sandbox, kills every process started since the sandbox came up (an install can leave a daemon
holding a connection, and the sidecar keeps established connections across a policy change), and
reads the policy back.
"""

from __future__ import annotations

import shlex
from typing import Any, override

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode, NetworkPolicy
from nhx_sandbox.egress import EgressAllowlist, EgressPolicy, EgressRule, build_egress_policy
from nhx_sandbox.opensandbox_policy import EgressVerificationError, canonical_egress_target, verify_applied_egress
from scaled_evals.harbor_opensandbox_environment import NemoOpenSandboxEnvironment

# A process's identity is "<pid>:<start time>", so a PID reused after startup isn't mistaken for the
# process that had it. Fields are read after the last ")", because a command name may contain spaces.
# POSIX sh builtins only: task images range from busybox to full distros, and a builtin starts no
# process the kill loop could hit.
_PROCESS_IDENTITY_FUNCTIONS = """\
ident() {
  id=
  read -r line < "/proc/$1/stat" 2>/dev/null || return 1
  ident_pid=$1
  set -- ${line##*) }
  id="$ident_pid:${20}"
}
"""

# Prints one identity per running process, except this command's own shell.
LIST_PROCESSES_SCRIPT = (
    _PROCESS_IDENTITY_FUNCTIONS
    + """\
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  [ "$pid" = "$$" ] && continue
  ident "$pid" && echo "$id"
done
"""
)

# Kills every process whose identity isn't in $KEEP, except this command's shell and its ancestors
# (the execd chain that runs it).
KILL_NEW_PROCESSES_SCRIPT = (
    _PROCESS_IDENTITY_FUNCTIONS
    + """\
ancestors=" "
p=$$
while [ -n "$p" ] && [ "$p" != 0 ]; do
  ancestors="$ancestors$p "
  read -r line < "/proc/$p/stat" 2>/dev/null || break
  set -- ${line##*) }
  p=$2
done
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  case "$ancestors" in *" $pid "*) continue ;; esac
  ident "$pid" || continue
  case " $KEEP " in *" $id "*) continue ;; esac
  kill -9 "$pid" 2>/dev/null
done
exit 0
"""
)


def network_rules(rules: list[EgressRule]) -> list[Any]:
    """The SDK's rule objects; imported here because only the dispatcher image installs the SDK."""
    from opensandbox.models.sandboxes import NetworkRule

    return [NetworkRule(action=rule.action, target=rule.target) for rule in rules]


class AgentChannelOpenSandboxEnvironment(NemoOpenSandboxEnvironment):
    """``NemoOpenSandboxEnvironment`` whose no-network sandboxes may still reach ``agent_channel_hosts``.

    With ``agent_setup_hosts``, also switches egress at runtime between agent setup and agent run.
    """

    def __init__(
        self,
        *args: Any,
        agent_channel_hosts: list[str] | None = None,
        agent_setup_hosts: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # BaseEnvironment.__init__ may validate against the policy, so set the channel first.
        self._agent_channel_hosts = tuple(agent_channel_hosts or ())
        self._agent_setup_hosts = tuple(agent_setup_hosts or ())
        self._agent_setup_targets = frozenset(canonical_egress_target(host) for host in self._agent_setup_hosts)
        self._startup_pids: tuple[str, ...] | None = None
        super().__init__(*args, **kwargs)

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        """The parent's, plus runtime policy switching when there is a setup allowlist to switch away from."""
        capabilities = super().capabilities
        if not self._agent_setup_hosts:
            return capabilities
        return capabilities.model_copy(update={"dynamic_network_policy": True})

    def _is_setup_policy(self, network_policy: NetworkPolicy) -> bool:
        """Whether ``network_policy`` is the setup allowlist the dispatcher wrote into the task."""
        return (
            bool(self._agent_setup_targets)
            and network_policy.network_mode == NetworkMode.ALLOWLIST
            and {canonical_egress_target(host) for host in network_policy.allowed_hosts} == self._agent_setup_targets
        )

    @override
    def validate_network_policy_support(self, network_policy: NetworkPolicy | None = None) -> None:
        """Accept the setup allowlist, which is operator config rather than a task's request."""
        network_policy = network_policy or self._network_policy
        if self._is_setup_policy(network_policy):
            # Harbor's own capability checks still apply; the trusted-allowlist check does not.
            super(NemoOpenSandboxEnvironment, self).validate_network_policy_support(network_policy)
            return
        super().validate_network_policy_support(network_policy)

    @override
    def egress_policy(self) -> EgressPolicy:
        return self._egress_policy_for(self._network_policy)

    def _egress_policy_for(self, network_policy: NetworkPolicy) -> EgressPolicy:
        """No-network: the agent channel only (empty channel: deny all). Setup: setup hosts plus the channel."""
        if self._is_setup_policy(network_policy):
            targets = (*self._agent_setup_hosts, *self._agent_channel_hosts)
        elif network_policy.network_mode == NetworkMode.NO_NETWORK:
            if not self._agent_channel_hosts:
                return EgressPolicy()
            targets = self._agent_channel_hosts
        elif network_policy == self._network_policy:
            return super().egress_policy()
        else:
            # Only the dispatcher's setup-then-no-network rewrite switches policy at runtime.
            raise ValueError(f"cannot switch this sandbox to network policy {network_policy!r}")
        return build_egress_policy(EgressAllowlist(targets=targets, resolver_addresses=self._resolver_addresses))

    @override
    async def start(self, force_build: bool) -> None:
        """Start the sandbox, then remember its processes so a policy switch can kill only newer ones."""
        await super().start(force_build)
        if self._agent_setup_hosts:
            self._startup_pids = await self._list_pids()

    async def _list_pids(self) -> tuple[str, ...]:
        result = await self.exec(f"sh -c {shlex.quote(LIST_PROCESSES_SCRIPT)}", cwd="/", user="root")
        if result.return_code != 0:
            raise RuntimeError(f"could not list sandbox processes (exit {result.return_code}): {result.stderr}")
        return tuple(line.strip() for line in (result.stdout or "").splitlines() if ":" in line)

    async def _kill_processes_since_start(self) -> None:
        if self._startup_pids is None:
            raise RuntimeError("the sandbox's startup processes were never recorded")
        keep = " ".join(self._startup_pids)
        result = await self.exec(
            f"KEEP={shlex.quote(keep)} sh -c {shlex.quote(KILL_NEW_PROCESSES_SCRIPT)}", cwd="/", user="root"
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"could not stop processes left by agent setup (exit {result.return_code}): {result.stderr}"
            )

    @override
    async def _apply_network_policy(self, network_policy: NetworkPolicy) -> None:
        """Move the live sandbox from the current egress policy to ``network_policy``'s.

        Rules leave before leftover processes are killed, so nothing can reconnect in between; rules
        arrive after, so nothing the previous phase started can use them.
        """
        sandbox = self._require_sandbox()
        current = self.egress_policy()
        wanted = self._egress_policy_for(network_policy)
        wanted_targets = {rule.target for rule in wanted.rules}
        removed = sorted({rule.target for rule in current.rules} - wanted_targets)
        added = [rule for rule in wanted.rules if rule not in current.rules]

        if removed:
            await sandbox.delete_egress_rules(removed)
        await self._kill_processes_since_start()
        if added:
            await sandbox.patch_egress_rules(network_rules(added))

        label = f"OpenSandbox sandbox for session {self.session_id} ({network_policy.network_mode.value})"
        await verify_applied_egress(sandbox, wanted, mode=self._egress_verification, label=label, require_readback=True)
        await self._verify_no_longer_allowed(sandbox, removed, label)
        self.logger.info("Switched %s: removed %d egress rule(s), added %d", label, len(removed), len(added))

    @staticmethod
    async def _verify_no_longer_allowed(sandbox: Any, removed: list[str], label: str) -> None:
        """``verify_applied_egress`` checks rules that must be present; this checks the ones that must be gone."""
        gone = {canonical_egress_target(target) for target in removed}
        applied = await sandbox.get_egress_policy()
        still_allowed = sorted(
            str(rule.target)
            for rule in (getattr(applied, "egress", None) or ())
            if getattr(rule, "action", None) == "allow" and canonical_egress_target(str(rule.target)) in gone
        )
        if still_allowed:
            raise EgressVerificationError(f"{label} still allows {still_allowed} after the policy switch")
