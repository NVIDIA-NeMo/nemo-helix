# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task Job's trial identities, and the call leases that bind each sandbox's address to its trial.

The task Job leases each sandbox's pod IP to the sandbox's trial while the sandbox holds it, so the Inference
Gateway can record the trial's model calls and accept its agent's telemetry into the trial's session and trace.
Leasing is best effort: a lease that can't be made leaves the trial's calls to the read-time join over the
sandbox records in the trial directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import httpx
from nemo_evals.intake.mapping import session_id_for
from nemo_helix_plugin.call_leases.client import AsyncCallLeasesClient
from nemo_helix_plugin.call_leases.types import CreateCallLeaseRequest, EndCallLeaseRequest
from nemo_helix_plugin.client.client import AsyncNemoClient

logger = logging.getLogger(__name__)

LEASE_REQUEST_TIMEOUT_SEC = 10.0


@dataclass(frozen=True)
class TrialIdentity:
    """The Intake session, trace and root span every producer of one trial's spans shares."""

    session_id: str
    trace_id: str
    root_span_id: str

    @classmethod
    def of(cls, run_id: str, trial_name: str) -> TrialIdentity:
        """The identity of trial ``trial_name`` in SDK run ``run_id``: the session publishing already uses."""
        session_id = session_id_for(run_id, trial_name)
        digest = hashlib.sha256(session_id.encode()).hexdigest()
        return cls(session_id=session_id, trace_id=digest[:32], root_span_id=digest[32:48])


class CallLeaseRegistrar:
    """Leases sandbox addresses in the task Job's workspace, through the task Job's own platform client."""

    def __init__(
        self,
        client: AsyncNemoClient,
        *,
        workspace: str,
        evaluation_name: str | None = None,
    ) -> None:
        self._client = client
        self._workspace = workspace
        self._evaluation_name = evaluation_name

    @contextlib.asynccontextmanager
    async def _leases(self) -> AsyncIterator[AsyncCallLeasesClient]:
        # Harbor runs trials on its own event loop, so each call gets a transport bound to the current one.
        event_hooks = {event: list(hooks) for event, hooks in self._client._client.event_hooks.items()}
        async with httpx.AsyncClient(timeout=LEASE_REQUEST_TIMEOUT_SEC, event_hooks=event_hooks) as http_client:
            yield AsyncCallLeasesClient.from_client(self._client.with_http_client(http_client))

    async def lease(
        self,
        *,
        name: str,
        client_address: str,
        identity: TrialIdentity,
        role: Literal["agent", "verifier"],
        started_at: datetime,
    ) -> bool:
        """Lease ``client_address`` to the trial; whether the lease was made."""
        body = CreateCallLeaseRequest(
            name=name,
            client_address=client_address,
            session_id=identity.session_id,
            trace_id=identity.trace_id,
            root_span_id=identity.root_span_id,
            role=role,
            started_at=started_at,
            evaluation_name=self._evaluation_name,
        )
        try:
            async with asyncio.timeout(LEASE_REQUEST_TIMEOUT_SEC), self._leases() as leases:
                (await leases.create_call_lease(workspace=self._workspace, body=body)).data()
        except Exception:
            logger.warning("Could not lease %s to trial %s", client_address, identity.session_id, exc_info=True)
            return False
        return True

    async def end(self, *, name: str, ended_at: datetime) -> None:
        """End lease ``name`` at ``ended_at``."""
        try:
            async with asyncio.timeout(LEASE_REQUEST_TIMEOUT_SEC), self._leases() as leases:
                await leases.end_call_lease(
                    workspace=self._workspace, name=name, body=EndCallLeaseRequest(ended_at=ended_at)
                )
        except Exception:
            logger.warning("Could not end call lease %s", name, exc_info=True)


_REGISTRAR: CallLeaseRegistrar | None = None


@contextlib.contextmanager
def leasing(registrar: CallLeaseRegistrar) -> Iterator[None]:
    """Make ``registrar`` the one this process's sandbox environments lease through, for the block's duration."""
    global _REGISTRAR
    previous, _REGISTRAR = _REGISTRAR, registrar
    try:
        yield
    finally:
        _REGISTRAR = previous


def current_registrar() -> CallLeaseRegistrar | None:
    """The registrar :func:`leasing` installed, or ``None`` when nothing leases."""
    return _REGISTRAR
