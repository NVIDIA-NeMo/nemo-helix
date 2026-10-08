# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from nhx_evals_sdk.resilience.api import _current_scheduler, run_with_resilience, use_resilience_session


@pytest.mark.asyncio
async def test_use_resilience_session_isolates_scheduler_per_concurrent_task():
    async def _run_isolated(endpoint_key: str) -> tuple[int, dict[str, object]]:
        async with use_resilience_session(global_limit=2, endpoint_max_limit=2):
            scheduler = _current_scheduler.get()
            assert scheduler is not None

            async def op() -> str:
                active = _current_scheduler.get()
                assert active is scheduler
                return "ok"

            result = await run_with_resilience(endpoint_key, op, max_attempts=1)
            assert result == "ok"
            return id(scheduler), await scheduler.summary()

    (scheduler_id_a, summary_a), (scheduler_id_b, summary_b) = await asyncio.gather(
        _run_isolated("endpoint-a"),
        _run_isolated("endpoint-b"),
    )

    assert scheduler_id_a != scheduler_id_b
    assert summary_a["operations_started"] == 1
    assert summary_a["operations_completed"] == 1
    assert summary_b["operations_started"] == 1
    assert summary_b["operations_completed"] == 1


@pytest.mark.asyncio
async def test_use_resilience_session_shared_scope_reuses_single_scheduler():
    async with use_resilience_session(global_limit=2, endpoint_max_limit=2):
        scheduler = _current_scheduler.get()
        assert scheduler is not None

        async def _run(endpoint_key: str) -> tuple[int, str]:
            async def op() -> str:
                active = _current_scheduler.get()
                assert active is scheduler
                return "ok"

            result = await run_with_resilience(endpoint_key, op, max_attempts=1)
            return id(_current_scheduler.get()), result

        (scheduler_id_a, result_a), (scheduler_id_b, result_b) = await asyncio.gather(
            _run("endpoint-a"),
            _run("endpoint-b"),
        )

        assert result_a == "ok"
        assert result_b == "ok"
        assert scheduler_id_a == id(scheduler)
        assert scheduler_id_b == id(scheduler)

        summary = await scheduler.summary()
        assert summary["operations_started"] == 2
        assert summary["operations_completed"] == 2
