# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimentalist plugin CLI — ``nemo agents experimentalist ...`` subcommands.

Registered under ``nemo.cli.agents`` and mounted by ``AgentsCLI`` as
``nemo agents experimentalist <verb>``.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import httpx
import typer
import yaml
from nemo_experimentalist_plugin.client import make_client
from nemo_experimentalist_plugin.preflight import (
    Probes,
    check_artifacts,
    check_datasets,
    check_environment,
    check_profile,
)
from nemo_experimentalist_plugin.profile import AgentProfile, load_profile
from nemo_experimentalist_plugin.resolve import (
    ResolveError,
    build_effective_experiment_plan,
    resolve_effective_insight,
    resolve_experiment_inputs,
)
from nemo_insights_plugin.contracts.checks import (
    CheckResult,
    advisories,
    format_report,
    required_failures,
)
from nemo_insights_plugin.contracts.profile import (
    EnvFileError,
    ProfileError,
    discover_profile,
    load_env_file,
    resolve_base_url,
)
from nemo_platform_plugin.cli import NemoCLI
from nemo_platform_plugin.client.errors import NemoClientError
from nooa import GenerationError

DEFAULT_WORKSPACE = "default"

_PREFLIGHT_PROBES: Probes | None = None  # test seam; None → real probes

run_experimentalist = None  # lazily imported by the run command; tests monkeypatch it

_PLATFORM_CLIENT_ERRORS = (NemoClientError, httpx.HTTPError, OSError, RuntimeError, ValueError)

# TODO: Add remote train/validation dataset support when remote experiment mode is implemented.


def _default_experiment_dir(profile: AgentProfile | None) -> Path:
    if profile is None:
        experiments_root = Path("tmp")
    else:
        experiments_root = profile.profile_dir / ".nemo-optimizer" / "experiments"
    experiments_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    while True:
        candidate = experiments_root / f"{timestamp}-{uuid.uuid4().hex}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate


class ExperimentalistCLI(NemoCLI):
    """``nemo agents experimentalist ...`` subcommands."""

    name: ClassVar[str] = "experimentalist"
    description: ClassVar[str] = "NeMo Experimentalist commands."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(help=self.description, no_args_is_help=True)

        @app.callback()
        def _root() -> None:
            """Force subcommand dispatch even when only one verb is registered."""

        @app.command("run")
        def run(
            agent: str | None = typer.Option(
                None,
                "--agent",
                help=(
                    "Baseline agent: a local directory or a git URL with an optional ref "
                    "(e.g. ssh://git@host/group/repo.git@main). Optional in Mode 1 (the insight "
                    "supplies the agent) and overrides the insight's agent when given. A git "
                    "source records provenance and enables --config storage.publish_winner to open a "
                    "draft PR/MR for the winner against that ref."
                ),
            ),
            ethos: str | None = typer.Option(
                None,
                "--ethos",
                help="URI of a markdown file describing the agent under test (its Ethos).",
            ),
            insight: str | None = typer.Option(
                None,
                "--insight",
                help=(
                    "Insight for Mode 1: a local insight file OR a platform insight id "
                    "(surfaced in Studio). A path that exists on disk is read locally; "
                    "otherwise it is fetched from the platform. Default: "
                    "<profile-dir>/.nemo-optimizer/insights.yaml when it exists (where "
                    "`nemo agents analyst run` writes by default)."
                ),
            ),
            insight_id: str | None = typer.Option(
                None,
                "--insight-id",
                help=(
                    "Selector for a local multi-insight file: exact id/title first; "
                    "if none matches, a decimal value is a zero-based index."
                ),
            ),
            no_insight: bool = typer.Option(
                False,
                "--no-insight",
                help="Disable insight use (Mode 2), including the shared profile insight file.",
            ),
            profile_path: Path | None = typer.Option(
                None,
                "--profile",
                help="Path to optimizer.yaml. Default: discovered by walking up from the cwd.",
                exists=True,
                dir_okay=False,
                readable=True,
            ),
            train_dataset: str | None = typer.Option(
                None,
                "--train-dataset",
                help="Train dataset: local path or harbor registry ref. Falls back to the profile.",
            ),
            validation_dataset: str | None = typer.Option(
                None,
                "--validation-dataset",
                help="Validation dataset: local path or harbor registry ref. Falls back to the profile.",
            ),
            task_template: str | None = typer.Option(
                None,
                "--task-template",
                help="Evaluator-specific task-template URI. Required with --insight.",
            ),
            experiment_dir: Path | None = typer.Option(
                None,
                "--experiment-dir",
                "--output",
                "--experiments-output",
                "-o",
                help=(
                    "Local experiment directory; writes eval-and-optimize/ here. "
                    "Default: <profile-dir>/.nemo-optimizer/experiments/<timestamp> "
                    "when a profile governs the run, else ./tmp."
                ),
                file_okay=False,
                dir_okay=True,
            ),
            workspace: str | None = typer.Option(
                None,
                "--workspace",
                help="Intake/NMP workspace for traces and run/candidate metadata. Falls back to the profile.",
            ),
            base_url: str | None = typer.Option(
                None,
                "--base-url",
                help=(
                    "Base URL of the running NMP instance. Default: NMP_BASE_URL "
                    "(shell or profile-dir .env), else http://localhost:8080."
                ),
                envvar="NMP_BASE_URL",
            ),
            config: Path | None = typer.Option(
                None,
                "--config",
                help="YAML or JSON configuration for the optimizer run.",
                exists=True,
                file_okay=True,
                dir_okay=False,
                readable=True,
            ),
            framework_skills: list[Path] = typer.Option(
                [],
                "--framework-skills",
                help="Path to a directory of framework skills to load into the optimizer agents. May be specified multiple times.",
                exists=True,
                file_okay=False,
                dir_okay=True,
                readable=True,
            ),
        ) -> None:
            """Run local optimization for a baseline agent (local dir or git source)."""

            if no_insight and (insight is not None or insight_id is not None):
                typer.echo("--no-insight cannot be combined with --insight or --insight-id", err=True)
                raise typer.Exit(code=1)

            async def _flow() -> str:
                """Resolve inputs (profile + flags) and run the Experimentalist."""
                profile, profile_load_error = _load_profile_or_error(profile_path)
                # Resolved AFTER the profile-dir .env load so an NMP_BASE_URL set there
                # takes effect; typer's envvar/flag binding (shell) wins.
                base_url_resolved = resolve_base_url(base_url)
                effective_insight = resolve_effective_insight(
                    profile=profile,
                    insight=insight,
                    insight_id=insight_id,
                    disabled=no_insight,
                )
                if no_insight:
                    typer.echo("Insight disabled: --no-insight (Mode 2)", err=True)
                elif effective_insight.is_profile_default:
                    # `nemo agents analyst run` writes here by default: the verbs connect flag-free.
                    typer.echo(
                        f"Insight file: {effective_insight.ref} (default; pass --insight to override)",
                        err=True,
                    )
                # Phase 1 — cheap environment checks BEFORE resolution: certain failures
                # (missing models, docker down, harbor absent) surface as the grouped
                # report before any dataset download. check_profile is
                # doctor-only on purpose: flags may fully specify the run, and resolution
                # reports missing inputs with the skeleton.
                _preflight_or_exit(
                    check_environment(
                        profile=profile,
                        insight=effective_insight.ref,
                        insight_id=effective_insight.selector,
                        base_url=base_url_resolved,
                        enforce_insight_agent=agent is None,
                        probes=_PREFLIGHT_PROBES,
                    )
                )
                config_payload = _load_config_payload(config)
                try:
                    plan = build_effective_experiment_plan(
                        profile=profile,
                        agent=agent,
                        ethos=ethos,
                        insight=effective_insight.ref,
                        insight_id=effective_insight.selector,
                        no_insight=no_insight,
                        train_dataset=train_dataset,
                        validation_dataset=validation_dataset,
                        task_template=task_template,
                        workspace=workspace,
                        config_payload=config_payload,
                        framework_skills=framework_skills or None,
                    )
                except ResolveError as exc:
                    if profile_load_error is not None:
                        raise ResolveError(f"{profile_load_error}\n{exc}") from None
                    raise
                if profile_load_error is not None:
                    typer.echo(f"⚠ ignoring unusable optimizer.yaml: {profile_load_error}", err=True)
                # Phase 2 validates effective source/template/storage before
                # creating an output directory or downloading datasets.
                _preflight_or_exit(
                    check_artifacts(
                        profile,
                        task_template=plan.task_template,
                        agent_source=plan.agent,
                        storage=plan.config.storage.model_dump(exclude_unset=True),
                        evaluator=plan.config.outcome_evaluator_config,
                        require_template=plan.insight is not None,
                        probes=_PREFLIGHT_PROBES,
                    )
                )
                if experiment_dir is None:
                    experiment_dir_resolved = _default_experiment_dir(profile)
                    typer.echo(f"Experiment dir: {experiment_dir_resolved}", err=True)
                else:
                    experiment_dir_resolved = experiment_dir
                inputs = await resolve_experiment_inputs(
                    profile=profile,
                    scratch_dir=experiment_dir_resolved / "resolved",
                    plan=plan,
                )
                global run_experimentalist
                if run_experimentalist is None:
                    from nemo_experimentalist_plugin.experimentalist.run import (
                        run_experimentalist as _run_experimentalist,  # noqa: PLC0415
                    )

                    run_experimentalist = _run_experimentalist
                client = make_client(base_url_resolved)
                try:
                    return await run_experimentalist(
                        agent=inputs.agent,
                        ethos=inputs.ethos,
                        insight=inputs.insight,
                        train_dataset=inputs.train_dataset,
                        validation_dataset=inputs.validation_dataset,
                        task_template=inputs.task_template,
                        experiment_dir=experiment_dir_resolved,
                        workspace=inputs.workspace,
                        client=client,
                        config=inputs.config,
                        framework_skills_dirs=inputs.framework_skills_dirs,
                    )
                finally:
                    await client.close()

            try:
                output_text = asyncio.run(_flow())
            except GenerationError as exc:
                # The harness raises this for a model that will not follow a return
                # contract, but also for API errors, timeouts, exhausted output
                # tokens, and unusable type annotations. Report what it said rather
                # than guessing which one it was.
                typer.echo(f"Error: a model-driven step of the run failed.\n\n{exc}", err=True)
                raise typer.Exit(code=1) from None
            except (OSError, ValueError, yaml.YAMLError) as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=1) from None
            typer.echo(output_text)

        @app.command("components")
        def components(
            role: str | None = typer.Option(None, "--role", help="Show only this role."),
        ) -> None:
            """List the components this install can resolve by name.

            Includes anything a `pip install`ed package registered, which is how a
            developer checks their own component was picked up.
            """
            from nemo_experimentalist_plugin.experimentalist.registry import Component, load_plugins

            load_plugins()
            rows = sorted(Component._registry.items())
            if role is not None:
                rows = [row for row in rows if row[0][0] == role]
            if not rows:
                typer.echo(f"No components registered{f' for role {role!r}' if role else ''}.")
                raise typer.Exit(1)
            width = max(len(registered_role) for (registered_role, _), _ in rows)
            for (registered_role, registered_name), component in rows:
                where = f"{component.__module__}.{component.__qualname__}"
                typer.echo(f"{registered_role:<{width}}  {registered_name:<24}  {where}")

        @app.command("doctor")
        def doctor(
            insight: str | None = typer.Option(None, "--insight", help="Optional insight ref to verify."),
            insight_id: str | None = typer.Option(
                None,
                "--insight-id",
                help="Select an exact id/title or zero-based index from a local multi-insight file.",
            ),
            profile_path: Path | None = typer.Option(None, "--profile", help="Path to optimizer.yaml."),
            base_url: str | None = typer.Option(
                None,
                "--base-url",
                help=(
                    "Base URL of the running NMP instance. Default: NMP_BASE_URL "
                    "(shell or profile-dir .env), else http://localhost:8080."
                ),
                envvar="NMP_BASE_URL",
            ),
        ) -> None:
            """Diagnose Experimentalist setup: profile, artifacts, models, platform, runtime."""
            found = profile_path or discover_profile()
            profile_obj, profile_error = None, None
            env_results: list[CheckResult] = []
            if found is not None:
                try:
                    profile_obj = load_profile(found)
                except ProfileError as exc:
                    profile_error = str(exc)
                try:
                    _announce_env_file(found.parent)
                except EnvFileError as exc:
                    env_results.append(
                        CheckResult(
                            name="profile-env",
                            group="profile",
                            status="fail",
                            severity="required",
                            message=str(exc),
                            hint="check that the file is readable UTF-8 text, or remove the broken .env file",
                        )
                    )
            base_url_resolved = resolve_base_url(base_url)
            insight_results: list[CheckResult] = []
            try:
                effective_insight = resolve_effective_insight(
                    profile=profile_obj,
                    insight=insight,
                    insight_id=insight_id,
                )
            except ResolveError as exc:
                effective_insight = None
                insight_results.append(
                    CheckResult(
                        name="insight-resolution",
                        group="platform",
                        status="fail",
                        severity="required",
                        message=str(exc),
                    )
                )
            plan = None
            plan_results: list[CheckResult] = []
            if profile_obj is not None and effective_insight is not None:
                try:
                    plan = build_effective_experiment_plan(
                        profile=profile_obj,
                        insight=effective_insight.ref,
                        insight_id=effective_insight.selector,
                    )
                except ResolveError as exc:
                    plan_results.append(
                        CheckResult(
                            name="experiment-plan",
                            group="profile",
                            status="fail",
                            severity="required",
                            message=str(exc),
                            hint="fix optimizer.yaml or its referenced ethos/experiment_config",
                        )
                    )
            insight_ref = effective_insight.ref if effective_insight is not None else None
            results = check_profile(profile_obj, profile_error) + env_results + plan_results + insight_results
            results += check_environment(
                profile=profile_obj,
                insight=insight_ref,
                insight_id=effective_insight.selector if effective_insight is not None else None,
                base_url=base_url_resolved,
                probes=_PREFLIGHT_PROBES,
            )
            results.append(asyncio.run(_check_platform_client_bootstrap(base_url_resolved)))
            if profile_obj is not None:
                if plan is not None:
                    results += check_artifacts(
                        profile_obj,
                        task_template=plan.task_template,
                        agent_source=plan.agent,
                        storage=plan.config.storage.model_dump(),
                        evaluator=plan.config.outcome_evaluator_config,
                        require_template=plan.insight is not None,
                        probes=_PREFLIGHT_PROBES,
                    )
                results += check_datasets(profile_obj, require_tasks=insight_ref is None)
            typer.echo(format_report(results))
            if required_failures(results):
                raise typer.Exit(code=1)

        return app


async def _check_platform_client_bootstrap(base_url: str) -> CheckResult:
    """Verify doctor can construct the same Platform client used by a run."""
    try:
        client = make_client(base_url)
    except _PLATFORM_CLIENT_ERRORS as exc:
        return CheckResult(
            name="platform-client-bootstrap",
            group="platform",
            status="fail",
            severity="required",
            message=f"Platform client initialization failed ({type(exc).__name__})",
            hint="check --base-url/NMP_BASE_URL and the active authentication context",
        )
    try:
        await client.close()
    except _PLATFORM_CLIENT_ERRORS as exc:
        return CheckResult(
            name="platform-client-bootstrap",
            group="platform",
            status="fail",
            severity="required",
            message=f"Platform client cleanup failed ({type(exc).__name__})",
            hint="check the active authentication context and retry",
        )
    return CheckResult(
        name="platform-client-bootstrap",
        group="platform",
        status="pass",
        severity="required",
        message="Platform client initialized with the effective authentication path",
    )


def _load_profile_or_error(profile_path: Path | None) -> tuple[AgentProfile | None, str | None]:
    """Load the explicit or discovered profile; announce a discovered one.

    An explicitly named ``--profile`` must load (ProfileError propagates). A
    discovered optimizer.yaml that fails to load is returned as an error
    string — it may be a malformed or foreign file, and flags can still fully
    specify the run. A successfully discovered profile is echoed to stderr so
    a stray parent-directory optimizer.yaml can never silently govern a run.
    """
    found = profile_path or discover_profile()
    if found is None:
        return None, None
    try:
        profile = load_profile(found)
    except ProfileError as exc:
        if profile_path is not None:
            raise
        _announce_env_file(found.parent)
        return None, str(exc)
    if profile_path is None:
        typer.echo(f"Using profile: {found} (agent: {profile.agent})", err=True)
    _announce_env_file(found.parent)
    return profile, None


def _announce_env_file(profile_dir: Path) -> None:
    """Load ``<profile_dir>/.env`` into the process env (set keys win) and say so."""
    env_path = profile_dir / ".env"
    loaded = load_env_file(env_path)
    if loaded:
        typer.echo(f"Loaded .env from {env_path} ({len(loaded)} vars)", err=True)


def _load_config_payload(config: Path | None) -> dict | None:
    """Read a --config file; an explicitly passed empty file is an error, not
    a silent fallback to the profile's experiment_config."""
    if config is None:
        return None
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"--config {config} is empty; remove the flag or add settings to the file")
    return payload


def _preflight_or_exit(results: list[CheckResult]) -> None:
    """Auto-run gate: hard-error on required failures, warn on advisories."""
    if required_failures(results):
        typer.echo(format_report(results), err=True)
        raise typer.Exit(code=1)
    for warning in advisories(results):
        typer.echo(f"⚠ {warning.message}" + (f" ({warning.hint})" if warning.hint else ""), err=True)
