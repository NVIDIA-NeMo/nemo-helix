# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_user_sim_plugin.jobs.spec import UNSUPPORTED_PROBES
from nemo_user_sim_plugin.service import UserSimService, run_name
from nemo_user_sim_plugin.studio import get_studio_spec
from usersim.engine.core.probes import known_probes


def _client() -> TestClient:
    app = FastAPI()
    for spec in UserSimService().get_routers():
        app.include_router(spec.router, prefix=f"/apis/user-sim{spec.prefix}")
    return TestClient(app)


def test_healthz_names_the_jobs():
    body = _client().get("/apis/user-sim/v1/healthz").json()
    assert body == {"plugin": "user-sim", "status": "ok", "jobs": ["simulate", "export"]}


def test_job_routes_are_mounted():
    # Evaluation runs only as a simulate job's second step, so it has no collection of its own.
    jobs = {path.split("/")[7] for path in _client().app.openapi()["paths"] if "/jobs/" in path}
    assert jobs == {"simulate", "export"}


def test_studio_bundle_is_built():
    assert get_studio_spec().bundle_path.is_file()


def test_probe_catalog_covers_every_probe_a_job_accepts():
    probes = _client().get("/apis/user-sim/v1/probes").json()

    assert {probe["name"] for probe in probes} == set(known_probes()) - UNSUPPORTED_PROBES


def test_runs_are_named_before_compile_so_traces_can_carry_the_name():
    names = {run_name("simulate") for _ in range(20)}

    assert len(names) == 20
    assert all(re.fullmatch(r"user-sim\.simulate-[a-z0-9]{8}", name) for name in names)
