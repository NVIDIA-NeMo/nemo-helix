# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UserSim plugin service: a health route plus the simulate and export job collections."""

import asyncio
import json
import os
import secrets
import string
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from fastapi import APIRouter, Depends, HTTPException
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_helix_plugin.service import NemoService, RouterSpec
from nemo_user_sim_plugin import scenarios
from nemo_user_sim_plugin.agents import AgentResolutionError
from nemo_user_sim_plugin.authz import scope
from nemo_user_sim_plugin.jobs._common import PREVIEWS_FILESET
from nemo_user_sim_plugin.jobs.export import ExportJob
from nemo_user_sim_plugin.jobs.simulate import SimulateJob
from nemo_user_sim_plugin.jobs.spec import SimulateRequest, SimulateSpec
from nemo_user_sim_plugin.model_check import ModelStatus, check_plugin_models
from nemo_user_sim_plugin.personas import MANAGED_ASSETS_ENV, managed_assets_path, persona_card, top_up
from nemo_user_sim_plugin.probes import PROBES, Probe
from nemo_user_sim_plugin.suggestions import probes_for_agent
from nemo_user_sim_plugin.types.endpoints import ProbeSuggestions, SuggestionRequest
from pydantic import BaseModel


class PersonaPreview(BaseModel):
    preview: str
    personas: list[dict[str, Any]]


_NAME_ALPHABET = string.ascii_lowercase + string.digits


def run_name(job: str) -> str:
    """``user-sim.<job>-<8 characters>``, the shape the jobs service gives unnamed jobs.

    Naming here rather than leaving it to the jobs service means ``compile`` knows the name, and the
    simulate step can tag its Intake traces with it.
    """
    return f"user-sim.{job}-{''.join(secrets.choice(_NAME_ALPHABET) for _ in range(8))}"


# Evaluation runs as a simulate job's second step, so it has no collection of its own.
JOBS = (SimulateJob, ExportJob)


async def _pick(spec: SimulateSpec, files: AsyncFilesClient, workspace: str) -> list[dict[str, Any]]:
    """Draw a run's personas in a subprocess, since Data Designer reads its persona folder at import."""
    with tempfile.TemporaryDirectory() as fetched:
        env = {**os.environ, MANAGED_ASSETS_ENV: str(managed_assets_path())}
        if spec.scenarios:
            try:
                found = await scenarios.fetch_async(files, workspace, spec.scenarios)
            except scenarios.ScenarioError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            env[scenarios.FETCHED_ENV] = str(scenarios.save_raw(found, Path(fetched)))
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "nemo_user_sim_plugin.tasks.preview",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        out, err = await process.communicate(spec.model_dump_json().encode())
    if process.returncode != 0:
        reason = err.decode().strip().splitlines()[-1] if err.strip() else "the preview failed"
        raise HTTPException(status_code=422, detail=reason)
    return json.loads(out)


class UserSimService(NemoService):
    name: ClassVar[str] = "user-sim"
    dependencies: ClassVar[list[str]] = ["entities", "jobs"]

    def get_routers(self) -> list[RouterSpec]:
        health = APIRouter()

        @health.get("/healthz")
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def healthz() -> dict[str, object]:
            return {"plugin": self.name, "status": "ok", "jobs": [job.name for job in JOBS]}

        @health.get("/probes")
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def probes() -> list[Probe]:
            """The probes a simulation can run, with what each one tests."""
            return list(PROBES)

        jobs = APIRouter()

        @jobs.post("/probe-suggestions")
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def probe_suggestions(
            workspace: str,
            body: SuggestionRequest,
            client: AsyncNemoClient = Depends(get_nemo_client),
            entities: Any = Depends(get_entity_client),
        ) -> ProbeSuggestions:
            """The probes that fit a registered agent, inferred from its config and ETHOS.md."""
            try:
                return await probes_for_agent(client, entities, workspace, body.agent)
            except AgentResolutionError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        @jobs.get("/model-check")
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def model_check(workspace: str, client: AsyncNemoClient = Depends(get_nemo_client)) -> list[ModelStatus]:
            """Whether each model the plugin runs on answers right now."""
            return await check_plugin_models(client, workspace)

        @jobs.get("/scenario-sets")
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def scenario_sets(
            workspace: str, client: AsyncNemoClient = Depends(get_nemo_client)
        ) -> list[scenarios.ScenarioSet]:
            """The custom topics and tool-use requests uploaded with ``nemo user-sim run --scenarios``."""
            return await scenarios.list_sets(AsyncFilesClient.from_client(client), workspace)

        @jobs.post("/persona-preview")
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def persona_preview(
            workspace: str,
            body: SimulateRequest,
            client: AsyncNemoClient = Depends(get_nemo_client),
            entities: Any = Depends(get_entity_client),
        ) -> PersonaPreview:
            """Pick the personas a run would meet, without running it.

            Takes the same request as a simulate run and resolves the model under test as the run does.
            Submit the request with the returned ``preview``, plus any removed persona UUIDs in
            ``exclude_personas``, to run with the personas shown here. Send that same pair here instead
            to keep the personas left and pick new ones until there are ``users`` again.
            """
            files = AsyncFilesClient.from_client(client)
            kept: list[dict[str, Any]] = []
            if body.preview:
                stored = await files.download_file(
                    workspace=workspace, name=PREVIEWS_FILESET, path=f"{body.preview}.json"
                )
                removed = set(body.exclude_personas)
                kept = [row for row in json.loads(await stored.read()) if row["persona_uuid"] not in removed]
            missing = body.users - len(kept)
            rows = kept
            if missing > 0:
                request = body.model_copy(update={"users": missing, "preview": None, "exclude_personas": []})
                try:
                    spec = await SimulateJob.to_spec(
                        request, workspace=workspace, entity_client=entities, async_sdk=None, is_local=False
                    )
                except AgentResolutionError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                rows = top_up(kept, await _pick(spec, files, workspace), body.exclude_personas)
            # UserSim's draw isn't repeatable, so the run reads these exact rows back.
            preview = f"preview-{secrets.token_hex(6)}"
            await files.create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(name=PREVIEWS_FILESET, purpose=FilesetPurpose.DATASET),
                exist_ok=True,
            )
            content = json.dumps(rows, default=str).encode()
            await files.upload_file(workspace=workspace, name=PREVIEWS_FILESET, path=f"{preview}.json", content=content)
            return PersonaPreview(preview=preview, personas=[persona_card(row) for row in rows])

        for job in JOBS:
            # One ``source`` per job type, as in the Helix Evals plugin: with a shared source, each
            # collection's list returns the others' jobs and fails rendering their specs.
            jobs.include_router(
                add_job_routes(
                    job,
                    service_name=f"user-sim.{job.name}",
                    generate_job_name=partial(run_name, job.name),
                    authz=scope,
                )
            )

        return [
            RouterSpec(router=health, tag="UserSim", description="UserSim plugin health.", prefix="/v1"),
            RouterSpec(
                router=jobs,
                tag="UserSim Jobs",
                description="Simulate conversations, score them and export training data.",
                prefix="/v2/workspaces/{workspace}",
            ),
        ]
