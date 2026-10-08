# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import httpx
import pytest
from nhx_evals_sdk.resilience.config import ResilienceConfig
from nhx_evals_sdk.resilience.scheduler import (
    ResilienceDeadlineExceededError,
    ResilienceMaxAttemptsExceededError,
    ResilienceQueueFullError,
    ResilienceScheduler,
)


@pytest.mark.asyncio
async def test_scheduler_rejects_when_endpoint_queue_is_full():
    scheduler = ResilienceScheduler(
        ResilienceConfig(
            global_limit=1,
            endpoint_initial_limit=1,
            endpoint_min_limit=1,
            endpoint_max_limit=1,
            global_max_queued=2,
            endpoint_max_queued=1,
        )
    )
    endpoint = "endpoint-a"

    async def slow_operation() -> str:
        await asyncio.sleep(0.2)
        return "ok"

    first = asyncio.create_task(
        scheduler.run_with_resilience(endpoint, slow_operation, max_attempts=1, deadline_at=None)
    )
    await asyncio.sleep(0.01)
    second = asyncio.create_task(
        scheduler.run_with_resilience(endpoint, slow_operation, max_attempts=1, deadline_at=None)
    )
    await asyncio.sleep(0.01)

    with pytest.raises(ResilienceQueueFullError):
        await scheduler.run_with_resilience(endpoint, slow_operation, max_attempts=1, deadline_at=None)

    await first
    await second


@pytest.mark.asyncio
async def test_scheduler_applies_soft_overload_decrease_and_eventual_success():
    scheduler = ResilienceScheduler(
        ResilienceConfig(
            global_limit=4,
            endpoint_initial_limit=4,
            endpoint_min_limit=1,
            endpoint_max_limit=4,
            beta_soft_overload=0.5,
            cooldown_seconds_soft=0.0,
        )
    )
    endpoint = "endpoint-b"
    call_count = 0

    async def flaky_operation() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ReadTimeout("timeout")
        return "ok"

    result = await scheduler.run_with_resilience(endpoint, flaky_operation, max_attempts=2, deadline_at=None)
    controller = await scheduler._get_controller(endpoint)  # noqa: SLF001

    assert result == "ok"
    assert call_count == 2
    assert controller.state.limit <= 2


@pytest.mark.asyncio
async def test_scheduler_raises_deadline_exceeded_when_retry_wait_exceeds_deadline():
    scheduler = ResilienceScheduler(
        ResilienceConfig(
            global_limit=2,
            endpoint_initial_limit=2,
            endpoint_min_limit=1,
            endpoint_max_limit=2,
            backoff_initial_ms=500.0,
            backoff_cap_ms=500.0,
        )
    )
    endpoint = "endpoint-c"

    async def always_overloaded() -> str:
        response = httpx.Response(
            503,
            headers={"Retry-After": "2"},
            request=httpx.Request("GET", "https://example.com"),
        )
        raise httpx.HTTPStatusError("service unavailable", request=response.request, response=response)

    with pytest.raises(ResilienceDeadlineExceededError):
        await scheduler.run_with_resilience(
            endpoint,
            always_overloaded,
            max_attempts=3,
            deadline_at=scheduler.now() + 0.2,
        )


@pytest.mark.asyncio
async def test_scheduler_escalates_repeated_soft_overload_to_hard_behavior():
    scheduler = ResilienceScheduler(
        ResilienceConfig(
            global_limit=4,
            endpoint_initial_limit=4,
            endpoint_min_limit=1,
            endpoint_max_limit=4,
            beta_hard_overload=0.5,
            beta_soft_overload=0.75,
            cooldown_seconds_soft=0.0,
            cooldown_seconds_hard=0.0,
            timeout_soft_overload_escalation_count=2,
            escalation_window_seconds=60.0,
            backoff_initial_ms=0.0,
            backoff_cap_ms=0.0,
        )
    )
    endpoint = "endpoint-d"
    call_count = 0

    async def always_timeout() -> str:
        nonlocal call_count
        call_count += 1
        raise httpx.ReadTimeout("timeout")

    with pytest.raises(ResilienceMaxAttemptsExceededError):
        await scheduler.run_with_resilience(endpoint, always_timeout, max_attempts=3, deadline_at=None)

    controller = await scheduler._get_controller(endpoint)  # noqa: SLF001
    assert call_count == 3
    assert controller.state.overload_soft_count >= 2
    assert controller.state.overload_hard_count >= 1
    assert controller.state.limit == 1


@pytest.mark.asyncio
async def test_scheduler_fast_start_doubles_until_first_failure_then_additive():
    scheduler = ResilienceScheduler(
        ResilienceConfig(
            global_limit=32,
            endpoint_initial_limit=4,
            endpoint_min_limit=1,
            endpoint_max_limit=32,
            success_window=2,
            cooldown_seconds_hard=0.0,
            backoff_initial_ms=0.0,
            backoff_cap_ms=0.0,
        )
    )
    endpoint = "endpoint-fast-start"

    async def ok_operation() -> str:
        return "ok"

    # Two windows of success -> doubles 4 -> 8 -> 16 before any failure.
    for _ in range(4):
        assert await scheduler.run_with_resilience(endpoint, ok_operation, max_attempts=1, deadline_at=None) == "ok"
    controller = await scheduler._get_controller(endpoint)  # noqa: SLF001
    assert controller.state.limit == 16

    # First failure marks the endpoint as no longer in fast-start mode.
    async def overloaded_once() -> str:
        response = httpx.Response(
            503,
            request=httpx.Request("GET", "https://example.com"),
        )
        raise httpx.HTTPStatusError("service unavailable", request=response.request, response=response)

    with pytest.raises(ResilienceMaxAttemptsExceededError):
        await scheduler.run_with_resilience(endpoint, overloaded_once, max_attempts=1, deadline_at=None)

    controller = await scheduler._get_controller(endpoint)  # noqa: SLF001
    assert controller.state.saw_failure is True

    # Next success window should increase additively (+1) instead of doubling.
    current_limit = controller.state.limit
    for _ in range(2):
        assert await scheduler.run_with_resilience(endpoint, ok_operation, max_attempts=1, deadline_at=None) == "ok"

    controller = await scheduler._get_controller(endpoint)  # noqa: SLF001
    assert controller.state.limit == min(controller.state.max_limit, current_limit + 1)


@pytest.mark.asyncio
async def test_scheduler_summary_includes_per_endpoint_counters():
    scheduler = ResilienceScheduler(
        ResilienceConfig(
            global_limit=4,
            endpoint_initial_limit=2,
            endpoint_min_limit=1,
            endpoint_max_limit=8,
            success_window=2,
            cooldown_seconds_hard=0.0,
            backoff_initial_ms=0.0,
            backoff_cap_ms=0.0,
        )
    )
    endpoint = "endpoint-summary"

    async def ok_operation() -> str:
        return "ok"

    # Two successful operations should trigger one additive increase (success_window=2).
    await scheduler.run_with_resilience(endpoint, ok_operation, max_attempts=1, deadline_at=None)
    await scheduler.run_with_resilience(endpoint, ok_operation, max_attempts=1, deadline_at=None)

    summary = await scheduler.summary()
    per_endpoint = summary.get("per_endpoint")
    assert isinstance(per_endpoint, dict)
    endpoint_stats = per_endpoint.get(endpoint)
    assert isinstance(endpoint_stats, dict)
    assert endpoint_stats["operations_started"] == 2
    assert endpoint_stats["operations_completed"] == 2
    assert endpoint_stats["limit_increases"] == 1


@pytest.mark.asyncio
async def test_scheduler_retry_budget_exhaustion_does_not_leak_inflight_or_queue_counters():
    scheduler = ResilienceScheduler(
        ResilienceConfig(
            global_limit=2,
            endpoint_initial_limit=2,
            endpoint_min_limit=1,
            endpoint_max_limit=2,
            retry_budget_burst=0.0,
            retry_budget_tokens_per_sec=0.0,
            backoff_initial_ms=0.0,
            backoff_cap_ms=0.0,
        )
    )
    endpoint = "endpoint-budget"

    async def fail_once_then_retry() -> str:
        raise httpx.ReadTimeout("timeout")

    with pytest.raises(ResilienceQueueFullError, match="Retry budget exhausted"):
        await scheduler.run_with_resilience(endpoint, fail_once_then_retry, max_attempts=2, deadline_at=None)

    controller = await scheduler._get_controller(endpoint)  # noqa: SLF001
    assert controller.state.inflight == 0
    assert controller.state.queued == 0
