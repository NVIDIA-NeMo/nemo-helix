# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tell a caller how a run's trials are going while it produces them.

A run returns its trials only once every task has run. A runner or evaluator given a
:class:`TrialProgressListener` says how many trials it will produce, then hands over each trial as it
finishes, so the caller can show progress meanwhile.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Coroutine, Iterable, Sequence
from typing import Any, Protocol, TypeVar

from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial

logger = logging.getLogger(__name__)


class TrialProgressListener(Protocol):
    """Receives a run's planned trial count, then each trial as it finishes."""

    async def planned(self, trials: int, /) -> None:
        """The run will produce ``trials`` trials."""
        ...

    async def finished(self, trials: Sequence[AgentEvalTrial], /) -> None:
        """``trials`` finished since the last call, failed ones included."""
        ...

    async def stopped(self) -> None:
        """The run ended, finished or not: a trial not reported by now never will be."""
        ...


async def tell(update: Awaitable[None]) -> None:
    """Await a progress callback, logging rather than raising when it fails."""
    try:
        await update
    except Exception:
        logger.warning("Could not report trial progress", exc_info=True)


T = TypeVar("T")


async def gather_or_cancel(coroutines: Iterable[Coroutine[Any, Any, T]]) -> list[T]:
    """``asyncio.gather`` that, when one raises, cancels and awaits the rest before re-raising it.

    Plain ``gather`` leaves the others running, so they could report a trial after the run said it stopped
    and outlive the resources the run tears down.
    """
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
