# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration test for the analyst agent.

This exercises the whole loop against *real* services and *real* LLM APIs:

1. Start ClickHouse (auto-provisioned via the platform's docker helper).
2. Start the NeMo Platform (``nemo services run``) with Intake + Insights.
3. Clear all spans for the test project.
4. Run the research agent on three questions (concurrently) so it logs
   traces to Intake.
5. Run the analyst agent (``nemo agents analyst run``).
6. Assert the analyst created at least one Insight.

Required setup (the test is **opt-in** because it costs real tokens and needs
Docker):

- ``NMP_INSIGHTS_E2E=1`` — opt in; otherwise the test skips.
- ``NVIDIA_API_KEY`` and ``TAVILY_API_KEY`` — for the research agent's NIM
  model + Tavily search. Read from the example's ``.env`` (or the shell).
  The test also runs ``nemo setup --auto`` against its isolated Platform, so
  the NVIDIA provider supplies the persisted default and fast model entities
  used by the analyst. ``NEMO_DEFAULT_INFERENCE_KEY`` and its base URL can be
  supplied in addition when the optimizer models use another provider; they do
  not replace the NVIDIA credential required by the research agent under test.
- Docker — required to auto-start ClickHouse if one isn't already at
  ``NMP_INTAKE_CLICKHOUSE_URL`` (default ``http://localhost:8123``). A missing
  Docker daemon fails the test.

Optional overrides: ``NMP_INSIGHTS_E2E_PORT`` (default ``18080``),
``NEMO_PLATFORM_DIR`` (default ``~/code/nemo-platform``, where the ClickHouse
helper script lives), and ``NMP_INTAKE_CLICKHOUSE_{URL,USER,PASSWORD,DATABASE}``.

Run it from the example's venv so every dependency
(``nemo-platform[services]``, the NAT langchain provider, and the Insights
plugin) is present::

    cd examples/research-agent
    NMP_INSIGHTS_E2E=1 uv run pytest tests/test_analyst_e2e.py -s

The test is isolated from any platform you already have running: it binds a
dedicated port, uses a throwaway ``NMP_DATA_DIR``, and scopes all data to the
``research-agent-e2e-test`` project so it never touches real ``research-agent``
spans or Insights.
"""

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
from nemo_insights_plugin.analyst.observability import ANALYST_OBSERVABILITY_AGENT_NAME

# Scope every artifact this test produces to a dedicated project/agent name so
# it is non-destructive to the real `research-agent` data.
TEST_AGENT = "research-agent-e2e-test"
WORKSPACE = "default"

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
TMP_DIR = EXAMPLE_DIR / "tmp"

QUESTIONS = [
    "Who won the 2025 Nobel Prize in Physics?",
    "What were the main findings of the 2025 ICLR best-paper award?",
    "List three open-source vector databases and one differentiator each.",
]

PORT = int(os.environ.get("NMP_INSIGHTS_E2E_PORT", "18080"))
BASE_URL = f"http://127.0.0.1:{PORT}"
CLICKHOUSE_URL = os.environ.get("NMP_INTAKE_CLICKHOUSE_URL", "http://localhost:8123")
CLICKHOUSE_USER = os.environ.get("NMP_INTAKE_CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.environ.get("NMP_INTAKE_CLICKHOUSE_PASSWORD", "")
CLICKHOUSE_DB = os.environ.get("NMP_INTAKE_CLICKHOUSE_DATABASE", "intake")
PLATFORM_DIR = Path(os.environ.get("NEMO_PLATFORM_DIR", str(Path.home() / "code" / "nemo-platform")))

MIN_TRACES = 3
SERVER_START_TIMEOUT_S = 180
SPAN_VISIBLE_TIMEOUT_S = 120


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NMP_INSIGHTS_E2E") != "1",
        reason="opt-in only; set NMP_INSIGHTS_E2E=1 to run the real end-to-end test",
    ),
]


def _load_dotenv(path: Path) -> dict[str, str]:
    """Parse the example's ``.env`` into a dict (no external dependency)."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _subprocess_env() -> dict[str, str]:
    """Shell env + example ``.env``, plus the test's service/ClickHouse pins."""
    env = dict(os.environ)
    env.update(_load_dotenv(EXAMPLE_DIR / ".env"))
    env.update(
        {
            "NMP_BASE_URL": BASE_URL,
            "NMP_CONFIG_FILE": str(TMP_DIR / "e2e-nmp-config.yaml"),
            "NMP_INTAKE_CLICKHOUSE_URL": CLICKHOUSE_URL,
            "NMP_INTAKE_CLICKHOUSE_USER": CLICKHOUSE_USER,
            "NMP_INTAKE_CLICKHOUSE_PASSWORD": CLICKHOUSE_PASSWORD,
            "NMP_INTAKE_CLICKHOUSE_DATABASE": CLICKHOUSE_DB,
        }
    )
    return env


# The `nemo` / `nat` console-script wrappers are not reliably written into the
# venv (a `uv sync` that reverts an editable install can drop them), but their
# entry-point callables are always importable. Invoke those directly so the
# test does not depend on the script wrappers existing on disk.
_CLI_CALLABLES = {
    "nemo": ("nemo_platform.cli.app", "cli"),
    "nat": ("nat.cli.main", "run_cli"),
}


def _cli_cmd(name: str, *args: str) -> list[str]:
    """Build a command that runs CLI *name* via its entry-point callable."""
    module, func = _CLI_CALLABLES[name]
    return [sys.executable, "-c", f"from {module} import {func}; {func}()", *args]


def _require_keys() -> None:
    env = _subprocess_env()
    missing = [key for key in ("NVIDIA_API_KEY", "TAVILY_API_KEY") if not env.get(key)]
    if missing:
        pytest.skip(f"missing required API keys: {', '.join(missing)}")


@pytest.fixture(scope="module", autouse=True)
def required_keys() -> None:
    """Skip before provisioning services when the opt-in credentials are absent."""
    _require_keys()


# --------------------------------------------------------------------------- #
# ClickHouse                                                                   #
# --------------------------------------------------------------------------- #
def _clickhouse_reachable() -> bool:
    try:
        resp = httpx.get(f"{CLICKHOUSE_URL}/ping", timeout=2.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module")
def clickhouse() -> None:
    """Ensure ClickHouse is up, auto-starting it via the platform's docker helper.

    Per the test contract, a missing Docker daemon is a hard failure (not a
    skip) once the suite is opted into.
    """
    if _clickhouse_reachable():
        return

    if shutil.which("docker") is None:
        pytest.fail("Docker is required to start ClickHouse for the e2e test, but `docker` was not found")

    script = PLATFORM_DIR / "services" / "intake" / "scripts" / "spans" / "run_clickhouse.sh"
    if not script.exists():
        pytest.fail(
            f"ClickHouse helper script not found at {script}; set NEMO_PLATFORM_DIR to your nemo-platform checkout"
        )

    result = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Failed to start ClickHouse via {script}:\n{result.stdout}\n{result.stderr}")

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if _clickhouse_reachable():
            return
        time.sleep(1.0)
    pytest.fail("ClickHouse did not become reachable after starting the container")


def _clear_spans_for_project() -> None:
    """Delete all spans for the test project directly in ClickHouse.

    Intake exposes no span-delete HTTP route, and the spans table is keyed on
    the `project.name` attribute, so we issue a synchronous ALTER ... DELETE.
    The schema is lazily created by Intake on first span query, so this is a
    no-op (table-missing) on a brand-new database.
    """
    import clickhouse_connect

    parsed = urlparse(CLICKHOUSE_URL)
    client = clickhouse_connect.get_client(
        host=parsed.hostname or "localhost",
        port=parsed.port or 8123,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DB,
    )
    try:
        client.command(
            "ALTER TABLE spans DELETE WHERE attributes_string['project.name'] = "
            "{project:String} SETTINGS mutations_sync=2",
            parameters={"project": TEST_AGENT},
        )
    except Exception as exc:  # noqa: BLE001 - table may not exist yet on a fresh DB
        if "doesn't exist" not in str(exc) and "UNKNOWN_TABLE" not in str(exc):
            raise
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# Platform server                                                             #
# --------------------------------------------------------------------------- #
def _server_ready() -> bool:
    """True once both the Intake span route and the Insights route are live."""
    try:
        spans = httpx.get(
            f"{BASE_URL}/apis/intake/v2/workspaces/{WORKSPACE}/spans",
            params={"page_size": 1},
            timeout=3.0,
        )
        insights = httpx.get(
            f"{BASE_URL}/apis/insights/v2/workspaces/{WORKSPACE}/insights",
            params={"page_size": 1},
            timeout=3.0,
        )
    except httpx.HTTPError:
        return False
    return spans.status_code == 200 and insights.status_code == 200


@pytest.fixture(scope="module")
def platform_server(clickhouse: None) -> Iterator[str]:  # noqa: ARG001 - ordering dep
    """Start `nemo services run` on a dedicated port with isolated storage."""
    TMP_DIR.mkdir(exist_ok=True)
    data_dir = TMP_DIR / "e2e-nmp-data"
    log_path = TMP_DIR / "e2e_platform_server.log"

    env = _subprocess_env()
    # Isolate entity/Insight storage from any platform already running locally.
    env["NMP_DATA_DIR"] = str(data_dir)

    log = open(log_path, "w")
    proc = subprocess.Popen(
        _cli_cmd("nemo", "services", "run", "--host", "127.0.0.1", "--port", str(PORT)),
        cwd=str(EXAMPLE_DIR),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    try:
        deadline = time.monotonic() + SERVER_START_TIMEOUT_S
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log.flush()
                pytest.fail(
                    f"platform server exited early (code {proc.returncode}); see {log_path}\n"
                    + log_path.read_text()[-2000:]
                )
            if _server_ready():
                break
            time.sleep(2.0)
        else:
            pytest.fail(f"platform server not ready after {SERVER_START_TIMEOUT_S}s; see {log_path}")

        setup_result = subprocess.run(
            _cli_cmd(
                "nemo",
                "--base-url",
                BASE_URL,
                "setup",
                "--auto",
                "--workspace",
                WORKSPACE,
                "--no-start-services",
                "--no-install-skills",
                "--no-deploy-agent",
            ),
            cwd=str(EXAMPLE_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if setup_result.returncode != 0:
            pytest.fail(f"isolated `nemo setup --auto` failed:\n{setup_result.stdout}\n{setup_result.stderr}")
        yield BASE_URL
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        log.close()


# --------------------------------------------------------------------------- #
# SDK helpers (reuse the Insights plugin's own client/preflight code)         #
# --------------------------------------------------------------------------- #
def _count_traces() -> int:
    from nemo_insights_plugin.analyst.analyst_backend import make_analyst_backend
    from nemo_insights_plugin.platform_client import make_client

    async def _run() -> int:
        client = make_client(BASE_URL)
        backend = make_analyst_backend(client=client, insights_output=None)
        try:
            return await backend.count_agent_sessions(agent=TEST_AGENT, workspace=WORKSPACE)
        finally:
            await client.close()

    return asyncio.run(_run())


def _list_insight_ids() -> list[str]:
    from nemo_insights_plugin.platform_client import make_client

    async def _run() -> list[str]:
        client = make_client(BASE_URL)
        try:
            page = await client.insights.insights.list_insights(workspace=WORKSPACE, agent=TEST_AGENT, page_size=100)
        finally:
            await client.close()
        return [insight.id for insight in page.data]

    return asyncio.run(_run())


def _delete_insights(insight_ids: list[str]) -> None:
    from nemo_insights_plugin.platform_client import make_client

    async def _run() -> None:
        client = make_client(BASE_URL)
        try:
            for insight_id in insight_ids:
                await client.insights.insights.delete(workspace=WORKSPACE, insight_id=insight_id)
        finally:
            await client.close()

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# Research-agent workflow templating                                          #
# --------------------------------------------------------------------------- #
def _write_research_workflow() -> Path:
    """Copy the example workflow, re-scoping it to the test project + port."""
    source = (EXAMPLE_DIR / "workflow.yml").read_text()
    templated = source.replace("http://localhost:8080", BASE_URL).replace("research-agent", TEST_AGENT)
    TMP_DIR.mkdir(exist_ok=True)
    out = TMP_DIR / "e2e_research_workflow.yml"
    out.write_text(templated)
    return out


def _run_research_agents_parallel(workflow: Path, questions: list[str]) -> None:
    """Run every research-agent question concurrently (each is its own process).

    Distinct invocations get distinct trace/session ids, so running them in
    parallel only changes wall-clock time, not the resulting span data.
    """
    env = _subprocess_env()
    with ThreadPoolExecutor(max_workers=len(questions)) as pool:
        futures = {
            pool.submit(
                subprocess.run,
                _cli_cmd("nat", "run", "--config_file", str(workflow), "--input", q),
                cwd=str(EXAMPLE_DIR),
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            ): q
            for q in questions
        }
        failures: list[str] = []
        for future, question in futures.items():
            result = future.result()
            if result.returncode != 0:
                failures.append(f"research agent failed for {question!r}:\n{result.stdout}\n{result.stderr}")
    assert not failures, "\n\n".join(failures)


def _wait_for_traces(minimum: int, timeout_s: int) -> int:
    deadline = time.monotonic() + timeout_s
    count = 0
    while time.monotonic() < deadline:
        count = _count_traces()
        if count >= minimum:
            return count
        time.sleep(3.0)
    return count


def _wait_for_analyst_spans(timeout_s: int) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    spans: list[dict] = []
    while time.monotonic() < deadline:
        response = httpx.get(
            f"{BASE_URL}/apis/intake/v2/workspaces/{WORKSPACE}/spans",
            params={
                "filter[agent_name]": ANALYST_OBSERVABILITY_AGENT_NAME,
                "page_size": 100,
            },
            timeout=3.0,
        )
        if response.status_code == 200:
            spans = response.json()["data"]
            if spans:
                return spans
        time.sleep(3.0)
    return spans


# --------------------------------------------------------------------------- #
# The test                                                                    #
# --------------------------------------------------------------------------- #
def test_analyst_creates_insight_end_to_end(platform_server: str) -> None:  # noqa: ARG001
    # 1. Clean slate: drop any spans + Insights left by a previous run.
    _clear_spans_for_project()
    _delete_insights(_list_insight_ids())
    assert _list_insight_ids() == []

    # 2. Exercise the research agent (questions run concurrently) so Intake
    #    accumulates traces.
    workflow = _write_research_workflow()
    _run_research_agents_parallel(workflow, QUESTIONS)

    # 3. Traces must be queryable before the analyst's preflight gate passes.
    trace_count = _wait_for_traces(MIN_TRACES, SPAN_VISIBLE_TIMEOUT_S)
    assert trace_count >= MIN_TRACES, f"expected >= {MIN_TRACES} traces for {TEST_AGENT}, saw {trace_count}"

    # 4. Run the analyst.
    result = subprocess.run(
        _cli_cmd(
            "nemo",
            "agents",
            "analyst",
            "run",
            "--agent",
            TEST_AGENT,
            "--workspace",
            WORKSPACE,
            "--base-url",
            BASE_URL,
        ),
        cwd=str(EXAMPLE_DIR),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"analyst failed:\n{result.stdout}\n{result.stderr}"

    # 5. The analyst must have persisted at least one Insight.
    insight_ids = _list_insight_ids()
    assert len(insight_ids) >= 1, (
        f"analyst created no Insights for {TEST_AGENT}.\n"
        f"--- analyst stdout ---\n{result.stdout}\n--- analyst stderr ---\n{result.stderr}"
    )

    # 6. The analyst's own Nooa spans should be queryable in Intake.
    analyst_spans = _wait_for_analyst_spans(SPAN_VISIBLE_TIMEOUT_S)
    assert analyst_spans, (
        f"expected Intake spans for {ANALYST_OBSERVABILITY_AGENT_NAME}; "
        f"--- analyst stdout ---\n{result.stdout}\n--- analyst stderr ---\n{result.stderr}"
    )
    assert {span["agent_name"] for span in analyst_spans} == {ANALYST_OBSERVABILITY_AGENT_NAME}
    assert all(span["session_id"] for span in analyst_spans)
