# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo's Harbor environment for running scaled-eval trials in OpenSandbox.

Harbor 0.20's ``OpenSandboxEnvironment`` creates the sandbox, waits for it, and runs the trial.
This subclass keeps all of that and changes what the sandbox is allowed to reach and who owns it:

* Egress comes from a trusted allowlist the runner renders into Harbor's environment kwargs, never
  from ``task.toml``. ``no-network`` tasks get deny-all, ``public`` tasks get the trusted allowlist
  (never unrestricted access), and ``allowlist`` tasks must ask for a subset of it or are rejected
  before any sandbox exists.
* The applied policy is read back after create and the sandbox is killed if it does not match, or
  if it cannot be read at all.
* Every create attempt is labelled, so a sandbox the server created but whose handle was lost to a
  failed or retried request can still be found and killed.

Loaded by Harbor through ``environment.import_path``; see ``NEMO_OPENSANDBOX_IMPORT_PATH`` in
``scaled_evals.dispatch.harbor_opensandbox``. This is the interim home until ``nhx-sandbox`` owns
OpenSandbox access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, override
from uuid import uuid4

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.opensandbox import OpenSandboxEnvironment
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from nhx_sandbox.egress import EgressAllowlist, EgressPolicy, build_egress_policy
from nhx_sandbox.opensandbox_policy import (
    EgressVerificationError,
    EgressVerificationMode,
    canonical_egress_target,
    to_opensandbox_policy,
    verify_applied_egress,
)

from scaled_evals.harbor_opensandbox_cleanup import APPLIED_EGRESS_FILENAME

# Sandbox metadata label marking which NeMo component manages the sandbox.
MANAGED_BY_METADATA_KEY = "nemo-managed-by"
# Value of MANAGED_BY_METADATA_KEY on every sandbox this environment creates.
MANAGED_BY_METADATA_VALUE = "scaled-evals"
# Sandbox metadata label shared by every sandbox one create call made, including Harbor's retries.
CREATE_ATTEMPT_METADATA_KEY = "nemo-scaled-evals-create-attempt"


class NemoOpenSandboxEnvironment(OpenSandboxEnvironment):
    """``OpenSandboxEnvironment`` with trusted egress, applied-policy verification, and cleanup.

    Args:
        trusted_allowed_hosts: Destinations trials may reach, as hostnames, ``*.`` wildcard
            hostnames, IP addresses, or CIDRs. Supplied by the runner, not the task.
        resolver_addresses: Nameservers to keep reachable. ``None`` uses the runner's own
            ``/etc/resolv.conf``, which in-cluster is the resolver sandboxes share.
        egress_verification: How strictly to compare the applied policy with the requested one;
            see ``verify_applied_egress``. Readback itself is always required.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args: Any,
        trusted_allowed_hosts: list[str] | None = None,
        resolver_addresses: list[str] | None = None,
        egress_verification: EgressVerificationMode = "default_action",
        **kwargs: Any,
    ) -> None:
        # BaseEnvironment.__init__ validates network policy support, which reads these.
        self._trusted_allowed_hosts = tuple(trusted_allowed_hosts or ())
        self._trusted_targets = frozenset(canonical_egress_target(host) for host in self._trusted_allowed_hosts)
        self._resolver_addresses = tuple(resolver_addresses) if resolver_addresses is not None else None
        self._egress_verification: EgressVerificationMode = egress_verification
        self._expected_egress: EgressPolicy | None = None
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            *args,
            **kwargs,
        )

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        """Advertise every allowlist form ``egress_policy`` can enforce, so Harbor accepts those tasks."""
        return EnvironmentCapabilities(
            gpus=True,
            disable_internet=True,
            mounted=False,
            network_allowlist=True,
            network_allowlist_hostnames=True,
            network_allowlist_wildcard_hostnames=True,
            network_allowlist_ipv4_addresses=True,
            network_allowlist_ipv6_addresses=True,
            network_allowlist_ipv4_cidrs=True,
            network_allowlist_ipv6_cidrs=True,
        )

    @override
    def validate_network_policy_support(self, network_policy: NetworkPolicy | None = None) -> None:
        """Run Harbor's checks, then reject ``allowlist`` tasks that ask for hosts outside the trusted allowlist."""
        super().validate_network_policy_support(network_policy)
        network_policy = network_policy or self._network_policy
        if network_policy.network_mode != NetworkMode.ALLOWLIST:
            return
        outside = sorted(
            host for host in network_policy.allowed_hosts if canonical_egress_target(host) not in self._trusted_targets
        )
        if outside:
            raise ValueError(
                f"Task requests egress to {outside}, which the trusted allowlist for this run does not "
                "include. Tasks may narrow the trusted allowlist but not extend it."
            )

    def egress_policy(self) -> EgressPolicy:
        """Return the egress policy for the current network mode."""
        mode = self._network_policy.network_mode
        if mode == NetworkMode.NO_NETWORK:
            return EgressPolicy()
        targets = (
            tuple(self._network_policy.allowed_hosts) if mode == NetworkMode.ALLOWLIST else self._trusted_allowed_hosts
        )
        return build_egress_policy(EgressAllowlist(targets=targets, resolver_addresses=self._resolver_addresses))

    @override
    def _build_network_policy(self, sdk: dict[str, Any]) -> Any:
        """Send the policy fixed for this create call, so the policy verified afterwards is the one sent."""
        expected = self._expected_egress or self.egress_policy()
        return sdk["NetworkPolicy"].model_validate(to_opensandbox_policy(expected))

    @override
    async def _create_sandbox(self, sdk: dict[str, Any]) -> Any:
        """Create the sandbox through Harbor, kill any orphans from the attempt, then verify its egress.

        The sandbox is killed and the trial fails if the applied policy can't be read or doesn't match.
        """
        # Harbor's _create_sandbox retries transient failures internally. One attempt ID spans all
        # of those retries, so the sweeps below also find sandboxes an earlier retry orphaned.
        attempt_id = uuid4().hex
        caller_metadata = dict(self._metadata)
        self._metadata = {
            **caller_metadata,
            MANAGED_BY_METADATA_KEY: MANAGED_BY_METADATA_VALUE,
            CREATE_ATTEMPT_METADATA_KEY: attempt_id,
        }
        self._expected_egress = self.egress_policy()
        try:
            sandbox = await super()._create_sandbox(sdk)
        except BaseException:
            await self._kill_create_attempt(sdk, attempt_id, keep=None)
            raise
        finally:
            self._metadata = caller_metadata

        await self._kill_create_attempt(sdk, attempt_id, keep=_sandbox_id(sandbox))
        try:
            await verify_applied_egress(
                sandbox,
                self._expected_egress,
                mode=self._egress_verification,
                label=f"OpenSandbox sandbox {_sandbox_id(sandbox)} (session {self.session_id})",
                require_readback=True,
            )
        except BaseException as exc:
            await self._safe_kill(sandbox)
            if isinstance(exc, EgressVerificationError):
                raise RuntimeError(str(exc)) from exc
            raise
        self._record_applied_egress(_sandbox_id(sandbox), self._expected_egress)
        return sandbox

    def _record_applied_egress(self, sandbox_id: str | None, policy: EgressPolicy) -> None:
        """Write the verified policy and its hash into the trial directory for the supervisor."""
        payload = to_opensandbox_policy(policy)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        record = {
            "sandbox_id": sandbox_id,
            "session_id": self.session_id,
            "network_mode": self._network_policy.network_mode.value,
            "policy_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "policy": payload,
        }
        try:
            self.trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)
            (self.trial_paths.trial_dir / APPLIED_EGRESS_FILENAME).write_text(json.dumps(record, indent=2))
        except OSError:
            self.logger.warning("Could not record the applied egress policy", exc_info=True)

    async def _kill_create_attempt(self, sdk: dict[str, Any], attempt_id: str, *, keep: str | None) -> None:
        """Best-effort kill of every sandbox labelled with ``attempt_id`` except ``keep``."""
        try:
            manager = await sdk["SandboxManager"].create(connection_config=self._build_connection_config(sdk))
        except Exception:
            self.logger.warning("Could not reconcile OpenSandbox create attempt %s", attempt_id, exc_info=True)
            return
        try:
            page = 1
            orphans: list[str] = []
            while True:
                result = await manager.list_sandbox_infos(
                    sdk["SandboxFilter"](metadata={CREATE_ATTEMPT_METADATA_KEY: attempt_id}, page=page)
                )
                orphans.extend(info.id for info in result.sandbox_infos if info.id != keep)
                if not result.pagination.has_next_page:
                    break
                page += 1
            for sandbox_id in orphans:
                try:
                    await manager.kill_sandbox(sandbox_id)
                except Exception:
                    self.logger.warning("Could not kill orphaned OpenSandbox sandbox %s", sandbox_id, exc_info=True)
            if orphans:
                self.logger.warning(
                    "Killed OpenSandbox sandboxes left by create attempt %s: %s", attempt_id, ", ".join(orphans)
                )
        except Exception:
            self.logger.warning("Could not reconcile OpenSandbox create attempt %s", attempt_id, exc_info=True)
        finally:
            try:
                await manager.close()
            except Exception:
                pass

    @override
    def _load_opensandbox(self) -> dict[str, Any]:
        """Add the SDK classes ``_kill_create_attempt`` needs to the ones Harbor loads."""
        sdk = super()._load_opensandbox()
        from opensandbox.manager import SandboxManager  # ty: ignore[unresolved-import]  # Harbor 0.20 venv only
        from opensandbox.models.sandboxes import SandboxFilter  # ty: ignore[unresolved-import]

        return {**sdk, "SandboxManager": SandboxManager, "SandboxFilter": SandboxFilter}


def _sandbox_id(sandbox: Any) -> str | None:
    """Return the sandbox's ID from whichever of ``id`` or ``sandbox_id`` the SDK object exposes."""
    return getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None)
