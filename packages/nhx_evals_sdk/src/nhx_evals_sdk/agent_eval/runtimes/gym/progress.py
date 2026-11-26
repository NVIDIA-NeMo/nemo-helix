# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tell a caller how a Gym run's rollouts are going while it collects them.

A Gym run returns its trials only once collection ends, which can take hours. A runner given a
:class:`RolloutProgress` says how many rollouts it will collect, then hands over each rollout record as it
finishes, so the caller can show progress meanwhile.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

FOLLOW_INTERVAL_S = 5.0
"""How often a colocated run reads the rollouts Gym has written since the last read."""


class RolloutProgress(Protocol):
    """Receives a Gym run's rollout count, then each rollout record as it finishes."""

    async def planned(self, rollouts: int) -> None:
        """The run will collect ``rollouts`` rollouts: tasks times repeats."""
        ...

    async def finished(self, records: Sequence[Mapping[str, Any]]) -> None:
        """``records`` finished since the last call, each as Gym recorded it, failures included."""
        ...

    async def stopped(self) -> None:
        """Collection ended, finished or not: a rollout not reported by now never will be."""
        ...


async def tell(update: Awaitable[None]) -> None:
    """Await a progress callback, logging rather than raising when it fails."""
    try:
        await update
    except Exception:
        logger.warning("Could not report Gym rollout progress", exc_info=True)


class RolloutFileFollower:
    """Hands ``progress`` the records Gym appends to its rollout files, as they are written."""

    def __init__(self, paths: Sequence[Path], progress: RolloutProgress) -> None:
        self._offsets = dict.fromkeys(paths, 0)
        self._progress = progress

    async def follow(self, interval_s: float = FOLLOW_INTERVAL_S) -> None:
        """Report new records every ``interval_s`` until cancelled."""
        while True:
            await asyncio.sleep(interval_s)
            await self.report_new()

    async def report_new(self) -> None:
        """Report the records written since the last read, if any."""
        records = [record for path in self._offsets for record in self._read_new(path)]
        if records:
            await tell(self._progress.finished(records))

    def _read_new(self, path: Path) -> list[dict[str, Any]]:
        try:
            with path.open("rb") as file:
                file.seek(self._offsets[path])
                data = file.read()
        except FileNotFoundError:
            return []
        except OSError:
            logger.warning(
                "Could not read Gym's rollouts from %s for progress; trying again later", path, exc_info=True
            )
            return []

        complete = data[: data.rfind(b"\n") + 1]
        self._offsets[path] += len(complete)
        records = []
        for line in complete.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                logger.debug("Skipping a rollout line in %s that isn't JSON", path)
                continue
            if isinstance(record, dict):
                records.append(record)
        return records
