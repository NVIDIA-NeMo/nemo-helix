# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The call leases this gateway knows, by leased address.

The lease routes update it as soon as a lease is created or ended here, and a background refresh picks up
leases another replica handled.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from nemo_helix_plugin.call_leases.client import AsyncCallLeasesClient
from nemo_helix_plugin.call_leases.types import CallLease
from nemo_helix_plugin.client.client import AsyncNemoClient

logger = logging.getLogger(__name__)

# Longer than any sandbox lives, so a refresh keeps every lease that can still be ending.
LEASE_LOOKBACK = timedelta(days=2)
_PAGE_SIZE = 200


class CallLeaseCache:
    """Leases by address; :meth:`lookup` finds the one that held an address at a moment."""

    def __init__(self) -> None:
        self._leases: dict[tuple[str, str], CallLease] = {}
        self._by_address: dict[str, list[CallLease]] = defaultdict(list)

    def put(self, lease: CallLease) -> None:
        """Add or replace one lease."""
        self._leases[(lease.workspace or "", lease.name)] = lease
        self._reindex()

    def replace(self, leases: Iterable[CallLease]) -> None:
        """Replace every lease, keeping any this gateway saw more recently than ``leases`` were read."""
        fresh = {(lease.workspace or "", lease.name): lease for lease in leases}
        for key, known in self._leases.items():
            current = fresh.get(key)
            if current is None or (current.ended_at is None and known.ended_at is not None):
                fresh[key] = known
        cutoff = datetime.now(UTC) - LEASE_LOOKBACK
        self._leases = {key: lease for key, lease in fresh.items() if lease.started_at >= cutoff}
        self._reindex()

    def _reindex(self) -> None:
        by_address: dict[str, list[CallLease]] = defaultdict(list)
        for lease in self._leases.values():
            by_address[lease.client_address].append(lease)
        self._by_address = by_address

    def lookup(self, client_address: str | None, at: datetime) -> CallLease | None:
        """The lease that held ``client_address`` at ``at``, if any."""
        if not client_address:
            return None
        holding = [lease for lease in self._by_address.get(client_address, ()) if lease.holds(at)]
        return max(holding, key=lambda lease: lease.started_at, default=None)

    async def refresh(self, client: AsyncNemoClient) -> None:
        """Re-read every lease started within :data:`LEASE_LOOKBACK`, across workspaces."""
        leases = AsyncCallLeasesClient.from_client(client)
        since = (datetime.now(UTC) - LEASE_LOOKBACK).isoformat()
        page = await leases.list_call_leases(
            workspace="-",
            query_params={"page_size": _PAGE_SIZE, "filter": json.dumps({"created_at": {"$gte": since}})},
        )
        self.replace([lease async for lease in page.items()])


async def refresh_lease_cache_task(cache: CallLeaseCache, client: AsyncNemoClient, interval_sec: float) -> None:
    """Refresh ``cache`` every ``interval_sec``, logging and continuing on failures."""
    while True:
        try:
            await cache.refresh(client)
        except Exception:
            logger.warning("Could not refresh the call lease cache", exc_info=True)
        await asyncio.sleep(interval_sec)


_LEASE_CACHE = CallLeaseCache()


def global_lease_cache() -> CallLeaseCache:
    return _LEASE_CACHE
