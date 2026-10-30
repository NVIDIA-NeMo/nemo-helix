# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``FakePreparer`` follows the real call sequence: keys first, then ready."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from nemo_evals.harbor.remote.preparation import PreparationClients


async def call_preparer(preparer, task, recorded, policy):
    clients = PreparationClients(client=MagicMock(), files=MagicMock())
    return await preparer(
        task,
        recorded,
        policy=policy,
        caller="user:alice",
        clients=clients,
        deadline=datetime.now(timezone.utc),
    )


async def test_step_a_then_ready(fake_preparer, sandbox_policy, harbor_spec):
    task = harbor_spec.tasks[0]
    pending = await call_preparer(fake_preparer, task, None, sandbox_policy)
    assert pending.status == "pending" and set(pending.images) == {"agent"}
    ready = await call_preparer(fake_preparer, task, pending, sandbox_policy)
    assert ready.status == "ready" and ready.environment is not None
    assert ready.environment.agent.image_ref.startswith("registry.invalid/fake/")
    assert fake_preparer.calls == [(task.id, None), (task.id, pending)]


async def test_scripted_outcomes(fake_preparer, sandbox_policy, harbor_spec):
    first, second = harbor_spec.tasks
    fake_preparer.fail = {first.id: "build_failed"}
    fake_preparer.stay_pending = {second.id}
    fake_preparer.with_verifier = {second.id}
    failed = await call_preparer(fake_preparer, first, None, sandbox_policy)
    assert failed.status == "failed" and failed.error is not None and failed.error.code == "build_failed"
    pending = await call_preparer(fake_preparer, second, None, sandbox_policy)
    assert set(pending.images) == {"agent", "verifier"}
    assert await call_preparer(fake_preparer, second, pending, sandbox_policy) == pending


async def test_check_is_scripted_separately(fake_preparer, sandbox_policy, harbor_spec):
    first, second = harbor_spec.tasks
    fake_preparer.reject = {first.id: "gpu_unsupported"}
    clients = PreparationClients(client=MagicMock(), files=MagicMock())
    rejected = (await fake_preparer.check(first, policy=sandbox_policy, clients=clients)).error
    assert rejected is not None and rejected.code == "gpu_unsupported"
    assert (await fake_preparer.check(second, policy=sandbox_policy, clients=clients)).error is None
    assert fake_preparer.checked == [first.id, second.id] and fake_preparer.calls == []


async def test_with_verifier_prepares_verifier_environment(fake_preparer, sandbox_policy, harbor_spec):
    task = harbor_spec.tasks[0]
    fake_preparer.with_verifier = {task.id}
    fake_preparer.ready_at_once = True
    ready = await call_preparer(fake_preparer, task, None, sandbox_policy)
    assert ready.environment is not None and ready.environment.verifier is not None
