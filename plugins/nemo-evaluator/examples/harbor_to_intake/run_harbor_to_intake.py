# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pull a benchmark from Harbor Hub, run it in Docker, publish the scores to Intake.

Four steps, in order:

1. ``harbor download`` pulls each task from Harbor Hub into one directory of task
   folders — the layout the SDK's Harbor runtime already consumes.
2. :class:`AgentEvaluator` scores the tasks against a Harbor runner that executes
   each one in Docker.
   The default agent is ``codex``, working inside the task container, so what gets
   published is a real trajectory. ``--agent oracle`` replays each task's bundled
   reference solution instead: no credentials, 1.0 on everything, and a
   single-step trajectory, since a replay has no agent turns to record.
3. The Experiment and Evaluation are created up front, because ATIF ingest
   rejects a name the platform has not seen and ``publish_to_intake`` never
   creates one itself.
4. :func:`publish_to_intake` writes one ATIF trajectory per trial and one
   evaluator-result row per metric output, then the script reads them back
   through the Intake API to show what landed.

Prerequisites (see ``README.md``): Python >= 3.12 with the ``harbor`` extra, a
running Docker daemon, a platform running at least ``auth,entities,intake``, and —
for the default agent — a logged-in ``codex``.

Run it from the repository root::

    uv run plugins/nemo-evaluator/examples/harbor_to_intake/run_harbor_to_intake.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tomllib
import urllib.request
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import quote

from nemo_evaluator.intake.publish import PublishReport, publish_to_intake
from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    HarborAgentTaskRunner,
    HarborRuntimeConfig,
    HarborTasksetLoader,
    discover_harbor_tasks,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig
from nemo_helix import APIError, AsyncNeMoHelix
from nemo_helix.types.intake.trace_filter_param import TraceFilterParam
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.intake.client import AsyncIntakeClient

#: Tasks to pull and run. Terminal-Bench 2.1 is Apache-2.0 and its tasks ship prebuilt images, so a
#: run pulls rather than builds; these two are among its quickest.
DEFAULT_TASKS = ("terminal-bench/git-leak-recovery", "terminal-bench/largest-eigenval")

#: Label for the benchmark the tasks come from. Only used to name the Evaluation, so a run is
#: identifiable in Studio — tasks are fetched individually, not as a dataset.
DEFAULT_DATASET_NAME = "terminal-bench-2-1"

#: Agents that replay a task's bundled reference solution (or do nothing) rather than calling a
#: model. They need no API key, which is what makes them usable as a pipeline smoke test.
KEYLESS_AGENTS = frozenset({"oracle", "nop"})

#: Read for any ``nvidia_nim/`` model. Terminus runs the model on the host, not in the task
#: container, so an exported variable is all the agent needs. Agents pointed at another
#: provider bring their own credentials and are not gated on this.
AGENT_KEY_ENV = "NVIDIA_NIM_API_KEY"
NIM_MODEL_PREFIX = "nvidia_nim/"

#: Harbor's codex agent authenticates either with ``OPENAI_API_KEY`` or with the ``auth.json`` a
#: local ``codex login`` writes; the latter needs ``CODEX_FORCE_AUTH_JSON`` to be opted into.
CODEX_AUTH_JSON = Path.home() / ".codex" / "auth.json"

#: Entity-store names are capped here and must not end in a hyphen.
ENTITY_NAME_MAX_LENGTH = 63
# Leads a derived Evaluation name whose dataset cannot: entity names must start with a letter.
EVALUATION_NAME_PREFIX = "eval-"


def _harbor_cli() -> str:
    """Locate the ``harbor`` console script, preferring the one beside this interpreter."""
    candidate = Path(sys.executable).parent / "harbor"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("harbor")
    if found is None:
        raise SystemExit('harbor is not installed: uv pip install "harbor>=0.20,<0.21"')
    return found


def _task_id_of(folder: Path) -> str | None:
    """The task id a downloaded folder declares, or None if it does not declare a usable one.

    Every field is checked rather than trusted: this reads a directory that a previous run left
    behind, which may be half-written or not a Harbor task at all.
    """
    try:
        config = tomllib.loads((folder / "task.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    task = config.get("task")
    if not isinstance(task, dict):
        return None
    name = task.get("name")
    return name if isinstance(name, str) and name else None


def _ensure_tasks(tasks: list[str], tasks_dir: Path) -> Path:
    """Download each task from Harbor Hub, into one directory of task folders.

    Fetching tasks individually rather than the whole dataset keeps the download to the handful
    actually run. ``--export`` puts each task folder directly under ``tasks_dir``, which is itself
    the "directory whose subdirectories are task folders" shape the runtime discovers.
    """
    tasks_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        # ``--export`` writes <tasks_dir>/<bare name>/, dropping any org prefix, so two orgs
        # publishing the same task name land on one folder. An empty directory means an
        # interrupted download, not a cached task.
        folder = tasks_dir / task.rsplit("/", 1)[-1]
        if folder.is_dir() and any(folder.iterdir()):
            cached = _task_id_of(folder)
            if cached == task:
                print(f"Task already present: {folder.name}")
                continue
            if cached is None:
                raise SystemExit(
                    f"{folder} exists but does not declare a task id, so it cannot be used or "
                    f"safely overwritten. Delete it and re-run to fetch {task}."
                )
            raise SystemExit(
                f"{folder} already holds {cached}, which shares a name with {task}. "
                "Harbor exports both to the same folder; run them with separate --tasks-dir."
            )
        # flush: the subprocess writes straight to this stdout, so an unflushed line lands after it.
        print(f"Downloading {task} from Harbor Hub...", flush=True)
        result = subprocess.run(
            [_harbor_cli(), "download", task, "--export", "-o", str(tasks_dir)],
            check=False,
        )
        if result.returncode != 0 or not folder.is_dir():
            raise SystemExit(f"harbor download {task} failed; see the output above.")
    return tasks_dir


def _validate_tasks(dataset_dir: Path, tasks: list[str]) -> None:
    """Reject task ids the downloaded folders do not actually provide.

    A task's id comes from its ``task.toml``, not its folder name, so a download can succeed and
    still not supply the id being asked for. Filtering to nothing otherwise reaches the evaluator
    as an empty task list, which fails with "at least one task is required".
    """
    if not tasks:
        return
    available = {task.id for task in discover_harbor_tasks(dataset_dir)}
    unknown = sorted(set(tasks) - available)
    if unknown:
        sample = ", ".join(sorted(available)[:5])
        raise SystemExit(
            f"Unknown task id(s) under {dataset_dir}: {', '.join(unknown)}. {len(available)} available, e.g. {sample}"
        )


def _run_evaluation_name(dataset: str, run_id: str) -> str:
    """Derive an Evaluation name for one run that the entity store will accept.

    Entity names are capped at 63 characters, so a long ``--dataset`` plus a run id overflows; the
    run id is the part that has to survive, since it is what keeps runs apart. They must also start
    with a *lowercase* letter, which neither a numeric ``--dataset`` nor a bare run id is
    guaranteed to do.
    """
    suffix = f"-{run_id}"
    # An ``org/name`` dataset carries a slash, which entity names do not allow, and they are
    # lowercase-only.
    stem = dataset.rsplit("/", 1)[-1].lower()[: ENTITY_NAME_MAX_LENGTH - len(suffix)].rstrip("-")
    # An empty, hyphen-only, or digit-leading stem cannot lead, so fall back to a fixed prefix
    # rather than emitting a name the entity store will reject with a 422.
    if not stem or not ("a" <= stem[0] <= "z"):
        stem = f"{EVALUATION_NAME_PREFIX}{stem}"[: ENTITY_NAME_MAX_LENGTH - len(suffix)].rstrip("-")
    return f"{stem}{suffix}"


def _ensure_codex_auth() -> None:
    """Make the local ``codex login`` usable, or say which credential is missing.

    Harbor's codex agent defaults to ``OPENAI_API_KEY`` and only reads ``~/.codex/auth.json`` when
    told to, so opt in on the caller's behalf when that file is the only credential present.
    """
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_AUTH_JSON_PATH"):
        return
    if os.environ.get("CODEX_FORCE_AUTH_JSON"):
        return
    if not CODEX_AUTH_JSON.exists():
        raise SystemExit(
            "The codex agent needs credentials: run `codex login`, or export OPENAI_API_KEY. "
            "Pass --agent oracle to run without either."
        )
    os.environ["CODEX_FORCE_AUTH_JSON"] = "1"
    print(f"codex auth: using {CODEX_AUTH_JSON}")


def _preflight(base_url: str, agent: str, model: str | None) -> None:
    """Check what can be checked without a client, before the evaluation rather than after it.

    A real-agent run costs both time and tokens, so a missing prerequisite should surface now.
    """
    if find_spec("harbor") is None:
        raise SystemExit('harbor is not installed: uv pip install "harbor>=0.20,<0.21"')
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode != 0:
        raise SystemExit("Docker is not available; Harbor runs every task in a container.")
    # Only the NIM provider reads this key. Other agents authenticate their own way, so demanding
    # it of them would block a perfectly valid `--agent claude-code` run.
    needs_nim_key = agent not in KEYLESS_AGENTS and (model or "").startswith(NIM_MODEL_PREFIX)
    if needs_nim_key and not os.environ.get(AGENT_KEY_ENV):
        raise SystemExit(
            f"{AGENT_KEY_ENV} is unset, and model {model!r} resolves through NVIDIA NIM. Export a "
            "key, or pass --agent oracle to replay each task's reference solution without one."
        )
    if agent == "codex":
        _ensure_codex_auth()
    try:
        with urllib.request.urlopen(f"{base_url}/health/ready", timeout=5) as response:
            ready = response.status == 200
    except OSError as error:
        raise SystemExit(f"No platform at {base_url}: {error}") from error
    if not ready:
        raise SystemExit(f"Platform at {base_url} is not ready.")


async def _probe_intake(async_sdk: AsyncNeMoHelix, workspace: str) -> None:
    """Confirm Intake can reach ClickHouse, which platform readiness does not cover.

    Intake starts and reports itself ready when ClickHouse is unreachable, serving its
    ClickHouse-backed endpoints with 503 instead. Querying one now turns that into an error before
    the evaluation runs rather than after it.
    """
    probe: TraceFilterParam = {"session_id": "harbor-to-intake-preflight"}
    try:
        async for _ in async_sdk.intake.traces.list(workspace=workspace, filter=probe):
            break
    except APIError as error:
        raise SystemExit(f"Intake cannot serve queries — is ClickHouse reachable? {error}") from error


async def _evaluate(
    dataset_dir: Path,
    tasks: list[str],
    *,
    agent: str,
    model: str | None,
    jobs_dir: Path,
) -> AgentEvalResult:
    """Run the selected tasks through Harbor and score them.

    The evaluator needs the tasks to score and a target that produces trials for them; the Harbor
    runtime supplies both. ``run_harbor_eval`` is a one-call wrapper around exactly this.
    """
    config = HarborRuntimeConfig(
        jobs_dir=jobs_dir,
        agent_name=agent,
        # Ignored by oracle/nop, which never call a model.
        agent_model_name=None if agent in KEYLESS_AGENTS else model,
        n_attempts=1,
        n_concurrent_trials=2,
        quiet=False,
    )
    wanted = set(tasks)
    taskset = [task for task in HarborTasksetLoader(dataset_dir).load().tasks if task.id in wanted]
    runner = HarborAgentTaskRunner(config=config, task_names=tasks)
    result = await AgentEvaluator().run(tasks=taskset, target=runner, config=AgentEvalRunConfig())

    print(f"\nRan {result.summary.task_count} task(s) as {result.summary.trial_count} trial(s)  [run {result.run_id}]")
    for aggregate in result.summary.scores.scores:
        print(f"  {aggregate.name}: mean={aggregate.mean}")
    for score in result.scores:
        reward = score.outputs[0].value if score.outputs else None
        print(f"  {score.task_id}: reward={reward} status={score.status.value}")
    for trial in result.trials:
        if trial.error is not None:
            print(f"  {trial.id}: error={trial.error.type}: {trial.error.message}")
    return result


async def _publish(
    async_sdk: AsyncNeMoHelix,
    result: AgentEvalResult,
    *,
    workspace: str,
    experiment: str,
    evaluation: str,
    dataset_name: str,
    agent: str,
    model: str | None,
) -> PublishReport:
    """Create the Experiment and Evaluation, then publish the scored run under them."""
    group = await async_sdk.experiments.create(
        workspace=workspace, name=experiment, description="Harbor -> Intake demo", exist_ok=True
    )
    await async_sdk.evaluations.create(
        workspace=workspace,
        name=evaluation,
        experiment_ids=[group.id],
        dataset_name=dataset_name,
        dataset_version="v1",
        exist_ok=True,
    )

    report = await publish_to_intake(
        result,
        client=client_from_platform(async_sdk, AsyncIntakeClient),
        experiment_id=evaluation,
        workspace=workspace,
        agent_name=agent,
        # oracle and nop never call a model, so publishing the --model default would record a
        # model that did not run against the trial.
        model_name="none" if agent in KEYLESS_AGENTS else (model or "none"),
    )
    print(f"\nPublished {report.trial_count} trial(s), {report.evaluator_result_count} score row(s) to {evaluation}")
    for omitted in report.skipped:
        print(f"  skipped {omitted.trial_id}/{omitted.name}: {omitted.reason}")
    return report


async def _read_back(async_sdk: AsyncNeMoHelix, report: PublishReport, *, workspace: str) -> None:
    """Query Intake for what was just written — the trajectory and its score rows."""
    print("\nRead back from Intake:")
    for published in report.published_trials:
        trace_filter: TraceFilterParam = {"session_id": published.session_id}
        traces = [trace async for trace in async_sdk.intake.traces.list(workspace=workspace, filter=trace_filter)]
        rows = await async_sdk.intake.spans.evaluator_results.list(published.span_id, workspace=workspace)
        print(f"  {published.trial_id}: {len(traces)} trajectory, span {published.span_id}")
        for row in rows:
            value = row.string_value if row.data_type == "TEXT" else row.value
            print(f"    {row.name} = {value}  ({row.data_type})")


def _print_studio_links(
    base_url: str,
    report: PublishReport,
    *,
    workspace: str,
    experiment: str,
    evaluation: str,
) -> None:
    """Print Studio URLs for what was just published.

    Studio mounts its SPA under ``/studio`` and the paths mirror two of the destinations Studio
    itself publishes in ``nhx.studio.studio_links``: ``experiment_detail`` for the Evaluation and
    ``intake_session`` for a single trial's trajectory. The links resolve only when Studio is
    among the running services.
    """
    root = f"{base_url.rstrip('/')}/studio/workspaces/{quote(workspace)}"
    print("\nView in Studio:")
    print(f"  Evaluation: {root}/experiment/{quote(experiment)}/{quote(evaluation)}")
    for published in report.published_trials:
        print(f"  {published.trial_id}: {root}/intake/sessions/{quote(published.session_id)}")


async def _main(args: argparse.Namespace) -> None:
    _preflight(args.base_url, args.agent, args.model)
    dataset_dir = _ensure_tasks(args.tasks, args.tasks_dir)
    _validate_tasks(dataset_dir, args.tasks)
    async with AsyncNeMoHelix(base_url=args.base_url, max_retries=2) as async_sdk:
        await _probe_intake(async_sdk, args.workspace)
        result = await _evaluate(dataset_dir, args.tasks, agent=args.agent, model=args.model, jobs_dir=args.jobs_dir)
        # One Evaluation per run by default: a stable name makes every re-run pile more test-case
        # rows into the same list, which is rarely what you want to look at.
        evaluation = args.evaluation or _run_evaluation_name(args.dataset, result.run_id)
        report = await _publish(
            async_sdk,
            result,
            workspace=args.workspace,
            experiment=args.experiment,
            evaluation=evaluation,
            dataset_name=args.dataset,
            agent=args.agent,
            model=args.model,
        )
        await _read_back(async_sdk, report, workspace=args.workspace)
    _print_studio_links(
        args.base_url,
        report,
        workspace=args.workspace,
        experiment=args.experiment,
        evaluation=evaluation,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("NHX_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--workspace", default="default")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help="Label for the benchmark, used to name the Evaluation. Tasks are fetched individually.",
    )
    parser.add_argument(
        "--tasks",
        # "+" not "*": a bare --tasks would otherwise parse to [] and fail much later, inside the
        # evaluator, with its generic "at least one task is required".
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Harbor Hub task ids to download and run, as 'org/task'.",
    )
    parser.add_argument(
        "--agent",
        default="codex",
        help="Harbor agent. Use 'oracle' to replay each task's reference solution without an API key.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.6-luna",
        help="Model for the agent; it must be one your agent's account can use. Ignored by oracle/nop.",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        # Under $HOME, not /tmp: Harbor bind-mounts the container's /logs back out to collect the
        # verifier reward, and some macOS Docker backends only share $HOME.
        default=Path.home() / ".cache" / "harbor-to-intake" / "tasks",
        help="Where Harbor Hub tasks are downloaded to.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path.home() / ".cache" / "harbor-to-intake" / "jobs",
        help="Directory Harbor writes its job results into.",
    )
    parser.add_argument("--experiment", default="harbor-demo", help="Experiment (group) name.")
    parser.add_argument(
        "--evaluation",
        default=None,
        help="Evaluation name to publish under. Defaults to one per run, so repeat runs do not "
        "stack rows under a single Evaluation; pin a name to accumulate them deliberately.",
    )
    asyncio.run(_main(parser.parse_args()))
