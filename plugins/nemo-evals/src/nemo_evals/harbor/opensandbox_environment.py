# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox environment task Jobs run trials in: nhx-sandbox's environment plus an agent channel.

A no-network task (``allow_internet = false``) normally gets a deny-all sandbox. An agent that runs
inside the sandbox (codex, hermes, ...) then can't reach its model. ``agent_channel_hosts`` keeps
those hosts (the inference gateway) reachable for such tasks; the internet stays closed.

It also sizes each sandbox as submit checked it: ``agent_resource_overrides`` in the agent sandbox only,
then task.toml, then ``sandbox_resource_defaults``.

``agent_setup_hosts`` are reachable only inside :meth:`AgentChannelOpenSandboxEnvironment.agent_setup_egress`,
which :mod:`nemo_evals.harbor.agent_setup_egress` wraps around the agent's ``setup()``. The task's own
network policy is never widened, so nothing a task declares can reach those hosts.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Literal, override

from harbor.environments.base import ExecResult
from harbor.models.task.config import NetworkMode
from nemo_evals.harbor.remote.resources import effective_resources
from nemo_evals.harbor.setup_cleanup import KILL_CONNECTION_HOLDERS_SCRIPT, LIST_PROCESSES_SCRIPT, SHELL, SYSTEM_PATH
from nhx_sandbox.egress import EgressAllowlist, EgressPolicy, EgressRule, build_egress_policy
from nhx_sandbox.harbor_opensandbox_environment import NemoOpenSandboxEnvironment, sandbox_role
from nhx_sandbox.opensandbox_policy import EgressVerificationError, canonical_egress_target, verify_applied_egress

logger = logging.getLogger(__name__)

GATEWAY_CA = "/etc/nemo-helix-tls/gateway-ca.pem"
CA_BUNDLE = "/etc/nemo-helix-tls/ca-bundle.pem"
# The task's own bundle, then Debian/Ubuntu/Alpine, RHEL/Fedora, SUSE, the OpenSSL default some minimal images
# keep, and last Python's certifi, the bundle Python uses when SSL_CERT_FILE is unset.
TRUST_GATEWAY_CA_SCRIPT = """\
set -e
system=
certifi=$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || true)
for candidate in "$TASK_BUNDLE" /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt \
/etc/ssl/ca-bundle.pem /etc/ssl/cert.pem "$certifi"; do
  if [ -s "$candidate" ]; then system=$candidate; break; fi
done
{ if [ -n "$system" ]; then cat "$system"; echo; fi; cat "$GATEWAY_CA"; } > "$CA_BUNDLE"
chmod 755 "${CA_BUNDLE%/*}"
chmod 644 "$GATEWAY_CA" "$CA_BUNDLE"
printf %s "$system"
"""


def network_rules(rules: list[EgressRule]) -> list[Any]:
    """The SDK's rule objects; imported here because only the dispatcher image installs the SDK."""
    from opensandbox.models.sandboxes import NetworkRule  # ty: ignore[unresolved-import]  # Harbor runner venvs only

    return [NetworkRule(action=rule.action, target=rule.target) for rule in rules]


class AgentChannelOpenSandboxEnvironment(NemoOpenSandboxEnvironment):
    """``NemoOpenSandboxEnvironment`` whose no-network sandboxes may still reach ``agent_channel_hosts``."""

    _setup_egress_state: Literal["closed", "open", "closing"] = "closed"

    def __init__(
        self,
        *args: Any,
        agent_channel_hosts: list[str] | None = None,
        agent_resource_overrides: dict[str, int] | None = None,
        sandbox_resource_defaults: dict[str, int] | None = None,
        agent_setup_hosts: list[str] | None = None,
        gateway_ca_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        # BaseEnvironment.__init__ may validate against the policy, so set the channel first.
        self._agent_channel_hosts = tuple(agent_channel_hosts or ())
        self._agent_setup_hosts = tuple(agent_setup_hosts or ())
        self._gateway_ca_file = gateway_ca_file
        if agent_resource_overrides or sandbox_resource_defaults:
            # Harbor's factory passes these as keywords. A copy: the task's own config must keep task.toml's
            # values, since an inherited separate verifier's config is a copy of it, made after this one.
            role = sandbox_role(kwargs["session_id"], kwargs["trial_paths"].trial_dir.name)
            overrides = (agent_resource_overrides or {}) if role == "agent" else {}
            kwargs["task_env_config"], defaulted = effective_resources(
                kwargs["task_env_config"], overrides=overrides, defaults=sandbox_resource_defaults or {}
            )
            if defaulted:
                config = kwargs["task_env_config"]
                logger.warning(
                    "The %s sandbox of %s relies on default resources: %s. Set them in task.toml or the run's "
                    "execution.sandbox.",
                    role,
                    kwargs["session_id"],
                    ", ".join(f"{field}={getattr(config, field)}" for field in defaulted),
                )
        super().__init__(*args, **kwargs)

    @override
    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        if self._gateway_ca_file is not None:
            await self._trust_gateway_ca(self._gateway_ca_file)

    async def _trust_gateway_ca(self, ca_file: str) -> None:
        """Make every later command trust ``ca_file`` alongside the roots it trusted already."""
        await self.upload_file(ca_file, GATEWAY_CA)
        result = await self._exec_cleanup(
            TRUST_GATEWAY_CA_SCRIPT,
            GATEWAY_CA=GATEWAY_CA,
            CA_BUNDLE=CA_BUNDLE,
            TASK_BUNDLE=self._persistent_env.get("SSL_CERT_FILE", ""),
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"Could not install the gateway CA in {self.session_id}'s sandbox (exit {result.return_code}): "
                f"{(result.stderr or result.stdout or '').strip()[-2000:]}"
            )
        if not (result.stdout or "").strip():
            self.logger.warning(
                "%s's sandbox image has no CA bundle, so it trusts only the gateway CA. Install ca-certificates in "
                "the image so it can reach other HTTPS hosts.",
                self.session_id,
            )
        self._persistent_env.update(
            SSL_CERT_FILE=CA_BUNDLE, REQUESTS_CA_BUNDLE=CA_BUNDLE, NODE_EXTRA_CA_CERTS=GATEWAY_CA
        )

    @override
    def egress_policy(self) -> EgressPolicy:
        """No-network tasks get the agent channel only (empty channel: deny all); other modes are unchanged."""
        if self._network_policy.network_mode != NetworkMode.NO_NETWORK or not self._agent_channel_hosts:
            return super().egress_policy()
        return build_egress_policy(
            EgressAllowlist(targets=self._agent_channel_hosts, resolver_addresses=self._resolver_addresses)
        )

    @asynccontextmanager
    async def agent_setup_egress(self) -> AsyncGenerator[None, None]:
        """Let the sandbox reach ``agent_setup_hosts`` for the duration of the block, then close them again.

        The egress sidecar keeps established connections across a rule change, so closing also kills every
        process started since entry that still holds a connection outside the sandbox. Until closing has finished, the sandbox refuses every other command:
        a close that was cancelled or failed must not leave task commands running with the hosts open.
        """
        if not self._agent_setup_hosts:
            yield
            return
        sandbox = self._require_sandbox()
        closed = self.egress_policy()
        allowed = tuple(rule.target for rule in closed.rules if rule.action == "allow")
        opened = build_egress_policy(
            EgressAllowlist(targets=(*allowed, *self._agent_setup_hosts), resolver_addresses=self._resolver_addresses)
        )
        label = f"OpenSandbox sandbox for session {self.session_id}"
        keep = await self._list_processes()
        try:
            await self._switch_egress(sandbox, closed, opened, f"{label} (agent setup)")
            self._setup_egress_state = "open"
            self.logger.info("Opened agent setup egress for %s", label)
            yield
        finally:
            self._setup_egress_state = "closing"
            removed = await self._switch_egress(sandbox, opened, closed, f"{label} (after agent setup)", keep=keep)
            self._setup_egress_state = "closed"
            self.logger.info("Closed agent setup egress for %s (%d rule(s) removed)", label, len(removed))

    async def _switch_egress(
        self,
        sandbox: Any,
        current: EgressPolicy,
        wanted: EgressPolicy,
        label: str,
        *,
        keep: tuple[str, ...] | None = None,
    ) -> list[str]:
        """Apply ``wanted`` over ``current`` on the live sandbox, then read it back; returns the targets removed.

        With ``keep``, every process not in it is killed after the removed rules are gone, so nothing can
        reconnect to them in between.
        """
        added = [rule for rule in wanted.rules if rule not in current.rules]
        removed = sorted({rule.target for rule in current.rules} - {rule.target for rule in wanted.rules})
        if added:
            await sandbox.patch_egress_rules(network_rules(added))
        if removed:
            await sandbox.delete_egress_rules(removed)
        if keep is not None:
            await self._kill_connection_holders_except(keep)
        await verify_applied_egress(sandbox, wanted, mode=self._egress_verification, label=label, require_readback=True)
        await self._verify_no_longer_allowed(sandbox, removed, label)
        return removed

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self._setup_egress_state == "closing":
            raise RuntimeError(
                "the agent setup egress window did not close cleanly; refusing to run more commands in this sandbox"
            )
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    async def _list_processes(self) -> tuple[str, ...]:
        result = await self._exec_cleanup(LIST_PROCESSES_SCRIPT)
        if result.return_code != 0:
            raise RuntimeError(f"could not list sandbox processes (exit {result.return_code}): {result.stderr}")
        return tuple(line.strip() for line in (result.stdout or "").splitlines() if ":" in line)

    async def _kill_connection_holders_except(self, keep: tuple[str, ...]) -> None:
        result = await self._exec_cleanup(KILL_CONNECTION_HOLDERS_SCRIPT, KEEP=" ".join(keep))
        if result.return_code != 0:
            raise RuntimeError(
                f"could not stop processes left by agent setup (exit {result.return_code}): {result.stderr}"
            )

    async def _exec_cleanup(self, script: str, **env: str) -> ExecResult:
        """Runs while every other command is refused, so it goes straight to the parent's ``exec``."""
        return await NemoOpenSandboxEnvironment.exec(
            self, f"{SHELL} -c {shlex.quote(script)}", cwd="/", env={"PATH": SYSTEM_PATH, **env}, user="root"
        )

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
            raise EgressVerificationError(f"{label} still allows {still_allowed} after agent setup")
