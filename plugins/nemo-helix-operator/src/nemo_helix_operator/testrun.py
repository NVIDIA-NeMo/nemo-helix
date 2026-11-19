# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""helix-operator's test routes: run an uploaded checkout's e2e tests against the Helix API in this namespace."""

import asyncio
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from nemo_helix_operator import deploy
from nemo_helix_operator.config import HelixOperatorConfig
from nemo_helix_operator.schemas import BuildSource
from nemo_helix_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.files.client import AsyncFilesClient
from pydantic import BaseModel, ConfigDict, Field

scope = AuthzScope("helix-operator")
TESTS = ["e2e/test_hello_world.py"]

# ponytail: a run is a pytest child of this process, kept only in memory, so an operator restart loses it, and the
# uploaded tests run with this pod's ServiceAccount. Deploy doesn't wait for a run. A Kubernetes Job per run with an
# unprivileged ServiceAccount is the upgrade path.
_runs: dict[str, tuple[int, asyncio.Task[tuple[int, str]]]] = {}


class RunPerms(PermissionSet, namespace="helix-operator.test-runs"):
    CREATE = perm("Run an uploaded checkout's e2e tests against the Helix API")
    READ = perm("Read an e2e test run")


class CreateTestRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: BuildSource = Field(description="The uploaded checkout whose e2e tests run.")


class HelixTestRun(BaseModel):
    id: str
    status: Literal["running", "passed", "failed"]
    tests: list[str]
    helm_revision: int | None = Field(default=None, description="The Helix release revision the run started on.")
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    detail: str | None = Field(default=None, description="The end of pytest's output when the run failed.")


def _pythonpath(root: Path) -> str:
    """Put the checkout's own packages first, so the tests use the client code they were written against."""
    paths = [root / "packages" / "nemo_nb", *sorted(root.glob("packages/*/src")), *sorted(root.glob("plugins/*/src"))]
    return os.pathsep.join(str(path) for path in paths)


async def _pytest(work: Path, api_url: str) -> tuple[int, str]:
    report = work / "report.xml"
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pytest",
            *TESTS,
            "--kubernetes",
            "--cluster-url",
            api_url,
            "-p",
            "no:cacheprovider",
            f"--junitxml={report}",
            cwd=work,
            env={**os.environ, "PYTHONPATH": _pythonpath(work)},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
        return process.returncode or 0, report.read_text() if report.is_file() else output.decode(errors="replace")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _result(run_id: str, helm_revision: int | None, code: int, report_or_output: str) -> HelixTestRun:
    run = HelixTestRun(id=run_id, status="passed" if code == 0 else "failed", tests=TESTS, helm_revision=helm_revision)
    try:
        suite = ET.fromstring(report_or_output).find("testsuite")
    except ET.ParseError:
        suite = None
    if suite is None:
        # pytest stopped before writing a report, for example on a conftest import error.
        return run.model_copy(update={"status": "failed", "detail": report_or_output[-2000:]})
    failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    run.passed, run.failed, run.skipped = int(suite.get("tests", 0)) - failed - skipped, failed, skipped
    if code:
        run.detail = "; ".join(
            f"{case.get('classname')}::{case.get('name')}"
            for case in suite.iter("testcase")
            if case.find("failure") is not None or case.find("error") is not None
        )
    return run


def e2e_router() -> APIRouter:
    router = APIRouter()

    @router.post("/test-runs", response_model=HelixTestRun, status_code=202)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[RunPerms.CREATE])
    async def create_test_run(
        workspace: str, body: CreateTestRunRequest, client: AsyncNemoClient = Depends(get_nemo_client)
    ) -> HelixTestRun:
        """Start the uploaded checkout's e2e tests against the Helix API."""
        release, namespace = deploy._target()
        code, output = await deploy._helm("status", release, "-n", namespace, "-o", "json")
        if code:
            raise HTTPException(status_code=502, detail=f"helm status failed: {output.strip()}")
        current = json.loads(output)
        status = current["info"]["status"]
        if status.startswith("pending") or any(not task.done() for task in deploy._upgrades.values()):
            raise HTTPException(status_code=409, detail=f"{release} is being upgraded ({status})")
        if any(not task.done() for _, task in _runs.values()):
            raise HTTPException(status_code=409, detail="A test run is already running")
        try:
            download = await AsyncFilesClient.from_client(client).download_file(
                workspace=workspace, name=body.source.fileset, path=body.source.archive
            )
            archive = await download.read()
        except NemoHTTPError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except NemoTransportError as exc:
            raise HTTPException(status_code=502, detail=f"Couldn't reach the Helix API: {exc}") from exc

        work = Path(tempfile.mkdtemp(prefix="helix-test-"))
        try:
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(work, filter="data")
            missing = [test for test in TESTS if not (work / test).is_file()]
            if missing:
                raise HTTPException(status_code=422, detail=f"The archive has no {', '.join(missing)}")
        except tarfile.TarError as exc:
            shutil.rmtree(work, ignore_errors=True)
            raise HTTPException(status_code=422, detail=f"Can't extract the archive: {exc}") from exc
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
        run_id = uuid.uuid4().hex[:12]
        _runs[run_id] = current["version"], asyncio.create_task(_pytest(work, HelixOperatorConfig.get().api_url))
        return HelixTestRun(id=run_id, status="running", tests=TESTS, helm_revision=current["version"])

    @router.get("/test-runs/{run_id}", response_model=HelixTestRun)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[RunPerms.READ])
    async def get_test_run(workspace: str, run_id: str) -> HelixTestRun:
        """Read a test run."""
        if run_id not in _runs:
            raise HTTPException(status_code=404, detail=f"No test run {run_id}")
        helm_revision, task = _runs[run_id]
        if not task.done():
            return HelixTestRun(id=run_id, status="running", tests=TESTS, helm_revision=helm_revision)
        if task.exception():
            return _result(run_id, helm_revision, 1, f"pytest didn't run: {task.exception()!r}")
        return _result(run_id, helm_revision, *task.result())

    return router
