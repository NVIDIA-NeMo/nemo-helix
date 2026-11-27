# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When one trial raises, the run's other trials stop before it reports that it stopped."""

from __future__ import annotations

import asyncio

import pytest
from nhx_evals_sdk.agent_eval.progress import gather_or_cancel


async def test_one_failure_cancels_and_awaits_the_rest_then_raises_it() -> None:
    finished_after_failure: list[str] = []
    cancelled: list[str] = []

    async def fails() -> str:
        raise RuntimeError("task failed")

    async def slow() -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append("slow")
            raise
        finished_after_failure.append("slow")
        return "slow"

    with pytest.raises(RuntimeError, match="task failed"):
        await gather_or_cancel([fails(), slow()])

    assert cancelled == ["slow"]
    assert finished_after_failure == []


async def test_results_keep_their_order_when_nothing_fails() -> None:
    async def value(n: int) -> int:
        await asyncio.sleep(0)
        return n

    assert await gather_or_cancel(value(n) for n in range(3)) == [0, 1, 2]
