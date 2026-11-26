# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tell a caller how a run's trials are going while it produces them.

A run returns its trials only once every task has run. A runner or evaluator given a
:class:`TrialProgressListener` says how many trials it will produce, then hands over each trial as it
finishes, so the caller can show progress meanwhile.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Sequence
from typing import Protocol

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
