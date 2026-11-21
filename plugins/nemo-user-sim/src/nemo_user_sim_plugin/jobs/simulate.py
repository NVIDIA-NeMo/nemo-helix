# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``user-sim.simulate``: census-grounded simulated users hold conversations with an agent or model."""

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from nemo_fabric import FabricConfig
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import HelixJobSpec
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_user_sim_plugin import scenarios
from nemo_user_sim_plugin.agents import default_model, resolve_agent
from nemo_user_sim_plugin.config import UserSimConfig
from nemo_user_sim_plugin.engine.assistants import FabricAssistant, ModelAssistant
from nemo_user_sim_plugin.engine.atif import PERSONA_KEYS, row_to_atif
from nemo_user_sim_plugin.engine.conversation import ConversationDeps, run_conversation
from nemo_user_sim_plugin.engine.export import chat_messages
from nemo_user_sim_plugin.engine.roles import RoleAnswerer, RoleModels
from nemo_user_sim_plugin.gateway import GatewayEmbeddings
from nemo_user_sim_plugin.intake import publish_conversation, register_run
from nemo_user_sim_plugin.jobs._common import (
    PREVIEWS_FILESET,
    RUN_ENV,
    TRAJECTORIES,
    fetch_result,
    task_step,
    usersim_models,
)
from nemo_user_sim_plugin.jobs.spec import AgentTarget, EvaluateSpec, RerunSource, SimulateRequest, SimulateSpec
from nemo_user_sim_plugin.personas import require_personas
from pydantic import BaseModel
from usersim.engine.external import materialize_episode_inputs

logger = logging.getLogger(__name__)


class SimulateJob(NemoJob):
    name: ClassVar[str] = "simulate"
    description: ClassVar[str] = "Simulate census-grounded users talking to an agent or model."
    input_spec_schema: ClassVar[type[BaseModel]] = SimulateRequest
    spec_schema: ClassVar[type[BaseModel]] = SimulateSpec
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False

    @classmethod
    async def to_spec(cls, input_spec, *, workspace, entity_client, async_sdk, is_local) -> SimulateSpec:
        request = SimulateRequest.model_validate(input_spec)
        # Checked at submission, not on the spec: stored specs are re-validated whenever runs are listed.
        if request.exclude_personas and request.preview is None:
            raise ValueError("exclude_personas needs the preview that drew those personas.")
        if isinstance(request.target, AgentTarget):
            fabric = await resolve_agent(entity_client, request.target.agent, workspace)
            return SimulateSpec(
                **request.model_dump(),
                assistant_model=default_model(fabric),
                fabric_config=fabric.model_dump(mode="json"),
            )
        return SimulateSpec(**request.model_dump(), assistant_model=request.target.model)

    @classmethod
    async def compile(
        cls, *, workspace, spec, entity_client, job_name, async_sdk, profile=None, options=None
    ) -> HelixJobSpec:
        spec = SimulateSpec.model_validate(spec)
        steps = [
            task_step("simulate", "nemo_user_sim_plugin.tasks.simulate", spec.model_dump(mode="json"), run=job_name)
        ]
        if spec.evaluate:
            steps.append(
                task_step(
                    "evaluate",
                    "nemo_user_sim_plugin.tasks.evaluate",
                    EvaluateSpec(assistant_model=spec.assistant_model, scenarios=spec.scenarios).model_dump(),
                )
            )
        return HelixJobSpec(steps=steps)

    def run(self, config: dict, *, ctx: JobContext, client: NemoClient) -> dict:
        spec = SimulateSpec.model_validate(config)
        work = ctx.storage.persistent
        if spec.scenarios:  # before UserSim first loads its bank
            scenarios.install(
                scenarios.fetch(FilesClient.from_client(client), ctx.workspace, spec.scenarios), work / "scenarios"
            )
        if spec.rerun:  # the same personas and topics as the earlier run, under new conversation IDs
            source = source_rows(client, ctx.workspace, spec.rerun, work)
            rows = rerun_rows(source, spec.rerun.conversations, token=uuid.uuid4().hex[:6])
        elif spec.preview:  # UserSim's draw isn't repeatable, so a previewed run reads the rows it showed
            preview = FilesClient.from_client(client).download_file(
                workspace=ctx.workspace, name=PREVIEWS_FILESET, path=f"{spec.preview}.json"
            )
            rows = preview_rows(json.loads(preview.read()), token=uuid.uuid4().hex[:6])
        else:
            require_personas(spec.locale)
            rows = materialize_episode_inputs(
                locale=spec.locale,
                num_rows=spec.users,
                probe_mix={probe: 1 / len(spec.probes) for probe in spec.probes},
                random_seed=spec.seed,
                models_path=usersim_models(client, ctx.workspace, work, assistant_model=spec.assistant_model),
                max_turns=spec.max_turns,
            )

        def save(name: str, payload: dict) -> None:
            path = work / f"{name}.json"
            path.write_text(json.dumps(payload, default=str))
            ctx.results.save(name, path)

        rows = approved_rows(rows, spec)
        save("inputs", {"rows": rows})  # what a later rerun of this run's conversations reads
        progress = Progress(spec, rows, save=save)
        results = asyncio.run(_converse(rows, spec, ctx.workspace, client, progress, run=os.environ.get(RUN_ENV)))
        finished = [r["row"] for r in results if r["status"] == "ok"]
        if finished:
            pd.DataFrame(finished).to_parquet(work / TRAJECTORIES, index=False)
            ctx.results.save("trajectories", work / TRAJECTORIES)
        totals = progress.summary()["totals"]
        return {"status": "completed" if finished else "failed", **totals}


async def _converse(
    rows: list[dict], spec: SimulateSpec, workspace: str, client: NemoClient, progress: "Progress", *, run: str | None
) -> list[dict]:
    config = UserSimConfig.get()
    async_client = client.to_async()
    gateway = AsyncModelsClient.from_client(async_client).get_async_openai_client(workspace=workspace)
    intake = AsyncIntakeClient.from_client(async_client) if spec.publish_to_intake else None
    roles = RoleAnswerer(RoleModels(config.user_model, config.judge_model, config.summary_model), gateway)
    fabric = FabricConfig.model_validate(spec.fabric_config) if spec.fabric_config else None
    deps = ConversationDeps(
        roles=roles,
        assistant_factory=(
            (lambda _bridge: FabricAssistant(fabric, keep_own_tools=getattr(spec.target, "use_agent_tools", False)))
            if fabric
            else (lambda bridge: ModelAssistant(gateway, spec.assistant_model, bridge))
        ),
        assistant_model=spec.assistant_model,
        embeddings=GatewayEmbeddings(gateway, config.embedding_model),
    )
    agent_name = spec.target.agent if isinstance(spec.target, AgentTarget) else spec.target.model
    limit = asyncio.Semaphore(config.max_concurrent_conversations)
    tagged = (
        intake is not None
        and run is not None
        and await register_run(intake, workspace, run, locale=spec.locale, target=agent_name)
    )

    async def one(index: int, row: dict) -> dict:
        async with limit:
            result = await run_conversation(row, deps)
        if result["status"] == "ok":
            payload = row_to_atif(
                result["row"],
                agent_name=agent_name,
                agent_version="1",
                model_name=spec.assistant_model,
                run=run if tagged else None,
            )
            result["publication"] = await publish_conversation(intake, workspace, payload)
        await progress.record(index, result)
        return result

    await progress.flush()
    results = await asyncio.gather(*(one(index, row) for index, row in enumerate(rows)))
    await progress.flush()
    return results


def source_rows(client: NemoClient, workspace: str, rerun: RerunSource, work: Path) -> list[dict]:
    """The rows an earlier run drew and saved as its ``inputs`` result."""
    try:
        path = fetch_result(client, workspace, rerun.job, "inputs", work / f"{rerun.job}-inputs.json")
    except NotFoundError as exc:
        raise ValueError(f"{rerun.job} started before reruns were possible, so it can't be rerun.") from exc
    return json.loads(path.read_text())["rows"]


def rerun_rows(source: list[dict], conversations: list[str], *, token: str) -> list[dict]:
    """The source rows of ``conversations``, in that order, each under a new trajectory ID.

    UserSim derives a row's ID from its persona and settings, so a rerun keeping it would land on the earlier
    conversation's Intake session. ``token`` is shared by one rerun.
    """
    by_id = {row["trajectory_id"]: row for row in source}
    missing = [conversation for conversation in conversations if conversation not in by_id]
    if missing:
        raise ValueError(f"The run has no conversation {', '.join(missing)}.")
    return [{**by_id[conversation], "trajectory_id": f"{conversation}-r{token}"} for conversation in conversations]


def preview_rows(rows: list[dict], *, token: str) -> list[dict]:
    """A preview's rows under conversation IDs of this run's own.

    Two runs can start from one preview. With the preview's IDs they would publish to the same Intake
    sessions, and a reviewer's feedback on one run would count for the other. ``token`` is shared by one run.
    """
    return [{**row, "trajectory_id": f"{row['trajectory_id']}-p{token}"} for row in rows]


def approved_rows(rows: list[dict], spec: SimulateSpec) -> list[dict]:
    """The materialized rows minus the personas a reviewer removed after previewing this seed's draw."""
    excluded = set(spec.exclude_personas)
    return [row for row in rows if row.get("persona_uuid") not in excluded]


class Progress:
    """Saves ``summary`` and ``transcripts`` as conversations finish, so Studio can show a run while it's going.

    Saves are at most ``interval`` seconds apart; ``flush`` writes whatever is left.
    """

    def __init__(self, spec: SimulateSpec, rows: list[dict], *, save: Callable[[str, dict], None], interval=10.0):
        self._spec = spec
        self._rows = rows
        self._save = save
        self._interval = interval
        self._results: dict[int, dict] = {}
        self._transcripts: dict[str, list[dict]] = {}
        self._saved_at = time.monotonic()
        self._saving = asyncio.Lock()

    async def record(self, index: int, result: dict) -> None:
        self._results[index] = result
        if result["status"] == "ok":
            self._transcripts[self._rows[index]["trajectory_id"]] = chat_messages(
                result["row"]["conversation_messages"]
            )
        if time.monotonic() - self._saved_at >= self._interval and not self._saving.locked():
            try:
                await self.flush()
            except Exception as exc:  # progress is a view; the final flush and the trajectories are the record
                logger.warning("Couldn't save run progress: %s", exc)

    async def flush(self) -> None:
        async with self._saving:
            self._saved_at = time.monotonic()
            await asyncio.to_thread(self._save, "summary", self.summary())
            await asyncio.to_thread(self._save, "transcripts", dict(self._transcripts))

    def summary(self) -> dict[str, Any]:
        conversations = [self._conversation(index, row) for index, row in enumerate(self._rows)]
        count = {status: sum(c["status"] == status for c in conversations) for status in ("ok", "failed", "running")}
        return {
            "target": self._spec.target.model_dump(),
            "assistant_model": self._spec.assistant_model,
            "probes": self._spec.probes,
            "locale": self._spec.locale,
            "totals": {
                "conversations": len(conversations),
                "completed": count["ok"],
                "failed": count["failed"],
                "running": count["running"],
            },
            "conversations": conversations,
        }

    def _conversation(self, index: int, row: dict) -> dict[str, Any]:
        result = self._results.get(index)
        finished = (result or {}).get("row") or {}
        return {
            "trajectory_id": row["trajectory_id"],
            "status": result["status"] if result else "running",
            "error": result["error"] if result else None,
            "turns": finished.get("num_turns"),
            "tool_calls": result["tool_calls"] if result else None,
            # Calls to the agent's own tools, whose results are real, not simulated.
            "own_tool_calls": (result or {}).get("own_tool_calls") or [],
            "publication": ((result or {}).get("publication") or {}).get("status"),
            **{key: row.get(key) for key in PERSONA_KEYS},
        }
