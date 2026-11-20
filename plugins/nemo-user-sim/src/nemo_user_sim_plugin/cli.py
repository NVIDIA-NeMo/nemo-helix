# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo user-sim``: set up persona data, run simulations, read results and export training data.

The raw job commands (``nemo user-sim simulate --spec ...``) are generated from the job classes.
These commands are the friendlier path the skill drives.
"""

import json
import logging
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Annotated

import click
import typer
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.cli import NemoCLI, create_typer_app
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_options import WorkspaceOption
from nemo_helix_plugin.cli_state import cli_state, resolve_cli_workspace
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_user_sim_plugin import scenarios
from nemo_user_sim_plugin.client import UserSimClient
from nemo_user_sim_plugin.personas import dataset_path, download_personas
from nemo_user_sim_plugin.probes import PROBES
from nemo_user_sim_plugin.report import SEGMENTS, conversations, failure_rates, session_link
from nemo_user_sim_plugin.types.endpoints import Job, JobCollection, JobRequest, ProbeSuggestions, SuggestionRequest
from rich.console import Console
from rich.table import Table
from typer import prompt
from usersim.engine.probes.safety_agentic.scorer import BOUNDARY_HELD_AXIS as BOUNDARY_HELD

FINISHED = {"completed", "error", "cancelled"}
ERROR_LABELS = {"agent_error": "agent error", "simulator_error": "simulation failed", "step_limit": "ran too long"}

LocaleOption = Annotated[
    str,
    typer.Option(
        "--locale",
        help="Persona locale, such as en_US or ja_JP.",
        autocompletion=lambda incomplete: _installed_locales(),
    ),
]


ROLE_LABELS = {
    "user_model": "simulated user",
    "judge_model": "judge",
    "summary_model": "summary",
    "evaluator_model": "evaluator",
}


def _result(client: UserSimClient, workspace: str, job: str, name: str) -> dict | None:
    """A simulate job's JSON result, or None when the job didn't save one."""
    try:
        return json.loads(client.download_result(workspace=workspace, collection="simulate", job=job, name=name).read())
    except NotFoundError:
        return None


def _pick(title: str, options: list[str], *, many: bool = False, default: str = "1") -> list[str]:
    typer.echo(title)
    for number, option in enumerate(options, 1):
        typer.echo(f"  {number}. {option}")
    answer = prompt("Numbers, separated by commas" if many else "Number", default=default)
    picked = [options[int(part) - 1] for part in answer.replace(" ", "").split(",") if part]
    return picked if many else picked[:1]


def run_target(agent: str | None, model: str | None, *, use_agent_tools: bool) -> dict:
    """The run's target. ``use_agent_tools`` is sent only when on, so other runs' requests don't change."""
    if agent:
        return {"kind": "agent", "agent": agent, **({"use_agent_tools": True} if use_agent_tools else {})}
    return {"kind": "model", "model": model}


# The settings a rerun keeps from its source run. The draw (users, seed, preview) comes from the source rows.
RERUN_KEEPS = ("target", "probes", "locale", "max_turns", "scenarios")


def rerun_request(source: dict, run: str, conversations: list[str]) -> dict:
    """A simulate request that reruns ``conversations`` of ``run`` with the source run's settings."""
    kept = {key: source[key] for key in RERUN_KEEPS if source.get(key) is not None}
    return {**kept, "rerun": {"job": run, "conversations": conversations}}


def interview(*, agents: list[str], models: list[str], suggest: Callable[[str], list[str]] = lambda agent: []) -> dict:
    """Ask for what `run` needs when it's started without flags. ``suggest`` names probes that fit an agent."""
    kind = prompt("Test an agent or a model?", default="agent", type=click.Choice(["agent", "model"]))
    targets = agents if kind == "agent" else models
    if not targets:
        raise typer.BadParameter(f"No {kind}s found in this workspace.")
    (target,) = _pick(f"Which {kind}?", targets)
    names = [probe.name for probe in PROBES]
    labels = [f"{probe.label}: {probe.tests} ({probe.name})" for probe in PROBES]
    suggested = [str(names.index(name) + 1) for name in suggest(target) if name in names] if kind == "agent" else []
    title = "Which probes? The defaults fit this agent." if suggested else "Which probes?"
    picked = _pick(title, labels, many=True, default=",".join(suggested) or "1")
    probes = [names[labels.index(label)] for label in picked]
    users = int(prompt("How many simulated users?", default="20"))
    return {
        "agent": target if kind == "agent" else None,
        "model": target if kind == "model" else None,
        "probes": probes,
        "users": users,
    }


def _suggestions(client: UserSimClient, workspace: str, agent: str) -> ProbeSuggestions:
    return client.suggest_probes(workspace=workspace, body=SuggestionRequest(agent=agent)).data()


def _agent_names(workspace: str | None = None) -> list[str]:
    return [agent.name for agent in CLIContext().typed_client(AgentsClient).list_agents(workspace=workspace)]


def _model_names(workspace: str | None = None) -> list[str]:
    models = CLIContext().typed_client(InferenceGatewayClient).list_openai_models(workspace=workspace).data().data
    return [model.id.split("/", 1)[-1] for model in models or []]


def _complete(names: Callable[[], list[str]]) -> Callable[[click.Context, str], list[str]]:
    """A tab completer that never breaks the shell: a platform that can't be reached completes nothing."""

    def complete(ctx: click.Context, incomplete: str) -> list[str]:
        logging.getLogger().setLevel(logging.CRITICAL)
        try:
            return [name for name in names() if name.startswith(incomplete)]
        except Exception:
            return []

    return complete


def _installed_locales() -> list[str]:
    datasets = dataset_path("en_US").parent
    return sorted(path.stem for path in datasets.glob("*.parquet")) if datasets.is_dir() else []


def _local(timestamp: datetime) -> str:
    """The API returns UTC without a zone; show it in the reader's time."""
    return timestamp.replace(tzinfo=timestamp.tzinfo or UTC).astimezone().strftime("%Y-%m-%d %H:%M")


def _runs(client: UserSimClient, workspace: str) -> list[Job]:
    return sorted(
        client.list_jobs(workspace=workspace, collection="simulate"),
        key=lambda run: run.created_at or datetime.min,
        reverse=True,
    )


def _progress_line(client: UserSimClient, workspace: str, job: str, step: str | None) -> str:
    """One line on where a running simulate job is: waiting, conversations finished so far or scoring."""
    if step is None:
        return "Waiting for the job to start."
    if step == "simulate":
        summary = _result(client, workspace, job, "summary")
        if summary is None:
            return "Simulating: starting the conversations."
        totals = summary["totals"]
        line = (
            f"Simulating: {totals['completed'] + totals['failed']} of {totals['conversations']} conversations finished"
        )
        return line + (f", {totals['failed']} couldn't run." if totals["failed"] else ".")
    if step == "evaluate":
        return "Simulation done. The judges are scoring the conversations."
    return f"Running step {step}."


def _wait(
    client: UserSimClient,
    workspace: str,
    job: str,
    timeout: int,
    collection: JobCollection = "simulate",
    *,
    progress: bool = False,
) -> str:
    """Poll until the job finishes and return its status. Exits with code 1 when ``timeout`` passes first.

    With ``progress``, prints a line whenever the job moves on and at least once a minute while it doesn't,
    so a long run never looks stuck.
    """
    started = time.monotonic()
    deadline = started + timeout
    said, said_at = None, started
    while True:
        status = client.get_job_status(workspace=workspace, collection=collection, name=job).data()
        state = status.status.value
        if state in FINISHED:
            return state
        if time.monotonic() > deadline:
            typer.echo(f"{job} is still running after {timeout}s. Check later with: nemo user-sim report {job}")
            raise typer.Exit(code=1)
        if progress:
            step = next((s.name for s in status.steps or [] if s.status.value == "active"), None)
            # Between steps nothing is active; keep the last line rather than say the job hasn't started.
            line = said if step is None and said is not None else _progress_line(client, workspace, job, step)
            now = time.monotonic()
            if line != said or now - said_at >= 60:
                elapsed = int(now - started)
                typer.echo(f"{line} ({elapsed // 60}m {elapsed % 60:02d}s)")
                said, said_at = line, now
        sleep(5 if progress else 10)


def _upload_scenarios(files: FilesClient, workspace: str, found: dict[str, bytes]) -> str:
    """Check a local scenario set with UserSim's loader, upload it and return its fileset."""
    with tempfile.TemporaryDirectory() as checked:
        scenarios.install(found, Path(checked))
    fileset = f"{scenarios.FILESET_PREFIX}{scenarios.set_id(found)}"
    files.create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(name=fileset, purpose=FilesetPurpose.DATASET),
        exist_ok=True,
    )
    for name, content in found.items():
        files.upload_file(workspace=workspace, name=fileset, path=name, content=content)
    for optional in (scenarios.BANK, scenarios.ACTIONS, scenarios.OUTCOMES, scenarios.TOPICS):
        if optional not in found:  # don't leave an earlier upload's file behind
            try:
                files.delete_file(workspace=workspace, name=fileset, path=optional)
            except NotFoundError:
                pass
    return fileset


def _finish(client: UserSimClient, workspace: str, job: str, timeout: int, base_url: str) -> None:
    """Wait for a simulate job and print its report. Exits with code 1 when the job didn't complete."""
    status = _wait(client, workspace, job, timeout, progress=True)
    if status != "completed":
        typer.echo(f"{job} ended with status {status}. See why with: nemo jobs get-logs {job}")
        raise typer.Exit(code=1)
    _print_report(client, workspace, job, base_url)


def _picked_reason(row: dict) -> str:
    """A scenario set's own name for the pick, plus any other check the conversation failed."""
    others = [axis for axis in (row["reason"] or "").split(", ") if axis and axis != BOUNDARY_HELD]
    pick = f"{row['picked']['label']}: {row['picked']['tool']}"
    return f"{pick}; also failed on {', '.join(others)}" if others else pick


def _print_report(client: UserSimClient, workspace: str, job: str, base_url: str) -> None:
    summary = _result(client, workspace, job, "summary")
    if summary is None:
        typer.echo(f"Job {job} has no results yet.")
        return
    rows = conversations(summary, _result(client, workspace, job, "scores"))
    totals = summary["totals"]
    scored = [row for row in rows if row["passed"] is not None]
    typer.echo(
        f"{totals['completed']} of {totals['conversations']} conversations completed, {totals['failed']} failed."
    )
    if not scored and totals["completed"]:
        failure = _result(client, workspace, job, "scoring_error")
        typer.echo(
            failure["message"] if failure else f"No conversation was scored. See why with: nemo jobs get-logs {job}"
        )
    if scored:
        passed = sum(row["passed"] for row in scored)
        typer.echo(f"{passed} of {len(scored)} scored conversations passed.")
        for segment in SEGMENTS:
            rates = failure_rates(rows, segment)
            worst = max(rates, key=rates.get, default=None)
            if len(rates) > 1 and rates[worst] > 0:
                typer.echo(f"Highest failure rate by {segment}: {worst} ({rates[worst]:.0%}).")
    picks = Counter(row["picked"]["label"] for row in rows if row.get("picked"))
    for label, count in sorted(picks.items()):
        typer.echo(f"{label}: {count} of {len(scored)}.")
    for row in rows:
        if row["status"] == "failed" or row["passed"] is False:
            reason = (
                f"failed to run ({ERROR_LABELS.get(row['error']['category'], 'error')}): {row['error']['message']}"
                if row["error"]
                else _picked_reason(row)
                if row.get("picked")
                else f"failed on {row['reason'] or 'scoring'}"
            )
            typer.echo(f"- {row['persona_name']} ({row['probe_type']}): {reason}")
            if link := session_link(row, base_url, workspace):
                typer.echo(f"  {link}")


class UserSimCLI(NemoCLI):
    name = "user-sim"
    description = "Test agents and models with census-grounded simulated users."

    def get_cli(self) -> typer.Typer:
        app = create_typer_app(help=self.description)

        @app.command()
        @handle_errors
        def setup(locale: LocaleOption = "en_US") -> None:
            """Download the Nemotron-Personas dataset for a locale from NGC (about 1 GB per locale)."""
            if dataset_path(locale).is_file():
                typer.echo(f"Personas for {locale} are already installed at {dataset_path(locale)}.")
                return
            typer.echo(f"Downloading Nemotron-Personas for {locale} with your NGC CLI config...")
            typer.echo(f"Installed at {download_personas(locale)}.")

        @app.command()
        @handle_errors
        def doctor(ctx: typer.Context, locale: LocaleOption = "en_US", workspace: WorkspaceOption = None) -> None:
            """Check that the plugin service is up, its models answer and persona data is installed."""
            healthy = True
            client = cli_state(ctx).typed_client(UserSimClient)
            try:
                client.health()
                typer.echo("✓ UserSim service is reachable.")
            except Exception as exc:
                typer.echo(f"✗ UserSim service isn't reachable: {exc}")
                raise typer.Exit(code=1) from exc
            try:
                statuses = client.check_models(workspace=resolve_cli_workspace(ctx, workspace)).data()
            except Exception as exc:
                healthy = False
                typer.echo(f"✗ Couldn't check the models: {exc}")
                statuses = []
            for status in statuses:
                role = ROLE_LABELS[status.role]
                if status.ok:
                    typer.echo(f"✓ The {role} model ({status.model}) answers.")
                else:
                    healthy = False
                    typer.echo(f"✗ The {role} model ({status.model}) didn't answer: {status.error}")
            if dataset_path(locale).is_file():
                typer.echo(f"✓ Personas for {locale} are installed.")
            else:
                healthy = False
                typer.echo(f"✗ Personas for {locale} are missing. Run: nemo user-sim setup --locale {locale}")
            if not healthy:
                raise typer.Exit(code=1)

        @app.command()
        @handle_errors
        def run(
            ctx: typer.Context,
            agent: Annotated[
                str | None, typer.Option(help="Registered agent to test.", autocompletion=_complete(_agent_names))
            ] = None,
            model: Annotated[
                str | None,
                typer.Option(help="Model to test instead of an agent.", autocompletion=_complete(_model_names)),
            ] = None,
            probe: Annotated[
                list[str] | None,
                typer.Option(
                    help="Probe to run. Repeatable. See them with: nemo user-sim probes",
                    autocompletion=_complete(lambda: [probe.name for probe in PROBES]),
                ),
            ] = None,
            users: Annotated[int, typer.Option(help="Number of simulated conversations.")] = 50,
            locale: LocaleOption = "en_US",
            seed: Annotated[
                int | None, typer.Option(help="Sampling seed. The same seed can still pick different personas.")
            ] = None,
            max_turns: Annotated[int, typer.Option(help="Maximum user turns per conversation.")] = 5,
            publish_to_intake: Annotated[bool, typer.Option(help="Send conversations to Intake.")] = True,
            evaluate: Annotated[bool, typer.Option(help="Score conversations after simulating.")] = True,
            wait: Annotated[bool, typer.Option(help="Wait for the job and print its report.")] = True,
            timeout: Annotated[int, typer.Option(help="Seconds to wait before giving up.")] = 7200,
            scenarios_from: Annotated[
                str | None,
                typer.Option(
                    "--scenarios",
                    help=(
                        "Custom scenarios: a folder with topics.yaml (general_open_ended) or bank.yaml and "
                        "optional actions.yaml (safety_agentic), one of those files, or a fileset that holds "
                        "them. Local files are checked and uploaded first."
                    ),
                ),
            ] = None,
            use_agent_tools: Annotated[
                bool,
                typer.Option(
                    "--use-agent-tools",
                    help=(
                        "Keep the agent's own tools during the run. "
                        "Only for tools that are safe to call for real, like search."
                    ),
                ),
            ] = False,
            workspace: WorkspaceOption = None,
        ) -> None:
            """Simulate users talking to an agent or model, then score the conversations.

            Run it without --agent or --model to be asked for each choice.
            """
            if agent is not None and model is not None:
                raise typer.BadParameter("Pass exactly one of --agent or --model.")
            if use_agent_tools and model is not None:
                raise typer.BadParameter("--use-agent-tools needs --agent.")
            state = cli_state(ctx)
            workspace = resolve_cli_workspace(ctx, workspace)
            client = state.typed_client(UserSimClient)
            if agent is None and model is None:
                if not sys.stdin.isatty():
                    raise typer.BadParameter("Pass exactly one of --agent or --model.")
                choice = interview(
                    agents=_agent_names(workspace),
                    models=_model_names(workspace),
                    suggest=lambda name: [s.name for s in _suggestions(client, workspace, name).probes],
                )
                agent, model, probe, users = choice["agent"], choice["model"], choice["probes"], choice["users"]
                if use_agent_tools and model is not None:
                    raise typer.BadParameter("--use-agent-tools needs --agent.")
            fileset = None
            if scenarios_from is not None:
                local = Path(scenarios_from).expanduser()
                files_client = state.typed_client(FilesClient)
                try:
                    if local.exists():
                        found = scenarios.read_local(local)
                        fileset = _upload_scenarios(files_client, workspace, found)
                        typer.echo(f"Uploaded the scenarios to fileset {fileset}.")
                    else:
                        fileset = scenarios_from
                        found = scenarios.fetch(files_client, workspace, fileset)
                except scenarios.ScenarioError as exc:
                    raise typer.BadParameter(str(exc)) from exc
                probe = sorted({*(probe or []), *scenarios.probes(found)})
            if agent and not probe:
                inferred = _suggestions(client, workspace, agent)
                typer.echo(f"Probes inferred from {agent}'s {' and '.join(inferred.read)}:")
                for suggestion in inferred.probes:
                    typer.echo(f"  {suggestion.name}: {suggestion.reason}")
                probe = [suggestion.name for suggestion in inferred.probes]
            spec = {
                "target": run_target(agent, model, use_agent_tools=use_agent_tools),
                "probes": probe or ["general_open_ended"],
                "users": users,
                "locale": locale,
                "seed": seed,
                "max_turns": max_turns,
                "publish_to_intake": publish_to_intake,
                "evaluate": evaluate,
                "scenarios": fileset,
            }
            job = client.create_job(workspace=workspace, collection="simulate", body=JobRequest(spec=spec)).data()
            typer.echo(f"Submitted {job.name}.")
            # The list, not the run page: Studio serves a path ending in a dotted job name as a missing file.
            base_url = (state.get_base_url() or "").rstrip("/")
            typer.echo(f"Watch it in Studio: {base_url}/studio/workspaces/{workspace}/plugin/user-sim")
            if wait:
                _finish(client, workspace, job.name, timeout, state.get_base_url() or "")

        @app.command()
        @handle_errors
        def report(
            ctx: typer.Context,
            job: Annotated[str, typer.Argument(help="Simulate job name.")],
            workspace: WorkspaceOption = None,
        ) -> None:
            """Print a simulate job's results: pass rate, weakest segments and failed conversations."""
            state = cli_state(ctx)
            client = state.typed_client(UserSimClient)
            _print_report(client, resolve_cli_workspace(ctx, workspace), job, state.get_base_url() or "")

        @app.command()
        @handle_errors
        def probes() -> None:
            """Show the probes a simulation can run and what each one tests."""
            table = Table("Probe", "Name", "Tests for", "Suits")
            for probe in PROBES:
                table.add_row(probe.name, probe.label, probe.tests, probe.suits)
            Console().print(table)

        @app.command("list")
        @handle_errors
        def list_runs(ctx: typer.Context, workspace: WorkspaceOption = None) -> None:
            """List simulation runs, newest first."""
            client = cli_state(ctx).typed_client(UserSimClient)
            runs = _runs(client, resolve_cli_workspace(ctx, workspace))
            if not runs:
                typer.echo("No simulations yet. Start one with: nemo user-sim run")
                return
            table = Table("Run", "Tested", "Probes", "Users", "Status", "Started")
            for run in runs:
                target = run.spec.get("target") or {}
                table.add_row(
                    run.name,
                    str(target.get("agent") or target.get("model") or ""),
                    ", ".join(run.spec.get("probes") or []),
                    str(run.spec.get("users", "")),
                    run.status.value if run.status else "",
                    _local(run.created_at) if run.created_at else "",
                )
            Console().print(table)

        @app.command()
        @handle_errors
        def delete(
            ctx: typer.Context,
            runs: Annotated[list[str] | None, typer.Argument(help="Runs to delete.")] = None,
            failed: Annotated[bool, typer.Option("--failed", help="Delete every run that ended in error.")] = False,
            yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask for confirmation.")] = False,
            workspace: WorkspaceOption = None,
        ) -> None:
            """Delete simulation runs. Their conversations stay in Traces."""
            client = cli_state(ctx).typed_client(UserSimClient)
            workspace = resolve_cli_workspace(ctx, workspace)
            names = list(runs or [])
            if failed:
                names += [run.name for run in _runs(client, workspace) if run.status and run.status.value == "error"]
            if not names:
                raise typer.BadParameter("Name the runs to delete, or pass --failed.")
            if not yes:
                typer.confirm(f"Delete {len(names)} run(s): {', '.join(names)}?", abort=True)
            for name in names:
                client.delete_job(workspace=workspace, collection="simulate", name=name)
                typer.echo(f"Deleted {name}.")

        @app.command()
        @handle_errors
        def cancel(
            ctx: typer.Context,
            run: Annotated[str, typer.Argument(help="Run to stop.")],
            workspace: WorkspaceOption = None,
        ) -> None:
            """Stop a running simulation. Conversations it already finished are kept."""
            client = cli_state(ctx).typed_client(UserSimClient)
            client.cancel_job(workspace=resolve_cli_workspace(ctx, workspace), collection="simulate", name=run)
            typer.echo(f"Cancelling {run}.")

        @app.command()
        @handle_errors
        def rerun(
            ctx: typer.Context,
            run: Annotated[str, typer.Argument(help="The run the conversations come from.")],
            conversation: Annotated[
                list[str],
                typer.Option("--conversation", help="Trajectory ID of a conversation to run again. Repeatable."),
            ],
            wait: Annotated[bool, typer.Option(help="Wait for the job and print its report.")] = True,
            timeout: Annotated[int, typer.Option(help="Seconds to wait before giving up.")] = 7200,
            workspace: WorkspaceOption = None,
        ) -> None:
            """Run conversations of an earlier run again, with the same personas and topics.

            The agent runs as it's registered now, so use it to check a fix: rerun the conversation that failed
            and a few that passed, and look for regressions.
            """
            state = cli_state(ctx)
            workspace = resolve_cli_workspace(ctx, workspace)
            client = state.typed_client(UserSimClient)
            source = client.get_job(workspace=workspace, collection="simulate", name=run).data()
            spec = source.spec if isinstance(source.spec, dict) else source.spec.model_dump()
            body = JobRequest(spec=rerun_request(spec, run, list(conversation)))
            job = client.create_job(workspace=workspace, collection="simulate", body=body).data()
            typer.echo(f"Submitted {job.name}, rerunning {len(conversation)} conversation(s) of {run}.")
            if wait:
                _finish(client, workspace, job.name, timeout, state.get_base_url() or "")

        @app.command("training-data")
        @handle_errors
        def training_data(
            ctx: typer.Context,
            fileset: Annotated[str, typer.Option(help="Dataset fileset to write the training data to.")],
            runs: Annotated[list[str] | None, typer.Argument(help="Runs whose conversations to export.")] = None,
            all_completed: Annotated[bool, typer.Option("--all-completed", help="Export every completed run.")] = False,
            generous: Annotated[
                bool, typer.Option(help="Keep borderline conversations a judge flagged. Strict by default.")
            ] = False,
            require_review: Annotated[
                bool, typer.Option(help="Keep only conversations a reviewer gave a thumbs up in Studio.")
            ] = False,
            wait: Annotated[bool, typer.Option(help="Wait for the export to finish.")] = True,
            workspace: WorkspaceOption = None,
        ) -> None:
            """Turn the passing conversations of one or more runs into one chat training dataset."""
            client = cli_state(ctx).typed_client(UserSimClient)
            workspace = resolve_cli_workspace(ctx, workspace)
            names = list(runs or [])
            if all_completed:
                names += [
                    run.name for run in _runs(client, workspace) if run.status and run.status.value == "completed"
                ]
            if not names:
                raise typer.BadParameter("Name the runs to export, or pass --all-completed.")
            spec = {
                "jobs": names,
                "fileset": fileset,
                "quality": "generous" if generous else "strict",
                "require_review": require_review,
            }
            job = client.create_job(workspace=workspace, collection="export", body=JobRequest(spec=spec)).data()
            typer.echo(f"Exporting {len(names)} run(s) as {job.name}.")
            if not wait:
                return
            status = _wait(client, workspace, job.name, timeout=1800, collection="export", progress=True)
            if status != "completed":
                typer.echo(f"{job.name} ended with status {status}. See why with: nemo jobs get-logs {job.name}")
                raise typer.Exit(code=1)
            typer.echo(f"Training data is in fileset {fileset}.")

        return app
