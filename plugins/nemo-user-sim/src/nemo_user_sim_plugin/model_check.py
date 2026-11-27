# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether the models a run depends on answer right now, so an outage shows up before a run, not in one."""

import asyncio
from collections.abc import Awaitable, Callable

from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_user_sim_plugin.config import UserSimConfig
from pydantic import BaseModel

# The plugin's own models. The model under test is the agent's or the one a run names, so it isn't here.
ROLES = ("user_model", "judge_model", "summary_model", "evaluator_model")
CHECK_TIMEOUT_SECONDS = 60


class ModelStatus(BaseModel):
    role: str
    model: str
    ok: bool
    error: str | None = None


async def check_models(models: dict[str, str], ask: Callable[[str, str], Awaitable[None]]) -> list[ModelStatus]:
    """Ask each distinct model once and report the result under every role that uses it."""
    first_role = {model: role for role, model in reversed(models.items())}
    outcomes = await asyncio.gather(*(ask(role, model) for model, role in first_role.items()), return_exceptions=True)
    errors = {
        model: str(outcome) or type(outcome).__name__
        for model, outcome in zip(first_role, outcomes, strict=True)
        if isinstance(outcome, BaseException)
    }
    return [
        ModelStatus(role=role, model=model, ok=model not in errors, error=errors.get(model))
        for role, model in models.items()
    ]


async def check_plugin_models(client: AsyncNemoClient, workspace: str) -> list[ModelStatus]:
    settings = UserSimConfig.get()
    gateway = AsyncModelsClient.from_client(client).get_async_openai_client(workspace=workspace)

    async def ask(role: str, model: str) -> None:
        await gateway.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=16,
            timeout=CHECK_TIMEOUT_SECONDS,
            extra_body=settings.model_settings.get(role, {}).get("extra_body"),
        )

    return await check_models({role: getattr(settings, role) for role in ROLES}, ask)
