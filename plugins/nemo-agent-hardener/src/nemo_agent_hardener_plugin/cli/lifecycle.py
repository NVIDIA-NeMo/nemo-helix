# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agent-hardener doctor | setup | init | refresh | status`` — host provisioning and target lifecycle.

agent-hardener is never imported: it runs in its own venv, invoked by subprocess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from nemo_agent_hardener_plugin.cli import checks, credentials, provisioning
from nemo_agent_hardener_plugin.cli._shared import (
    command_context,
    json_string_list,
    models_from_flags,
    parse_env_pairs,
    preflight_models,
)
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.filesets import upload_project_dir


def _project_init_body(
    ctx: Any,
    *,
    project_dir: str,
    name: str | None,
    dockerfile: str | None,
    start_command: str | None,
    binaries: list[str],
    harness: str | None,
    relay_confirmed: bool,
) -> dict[str, Any]:
    """Upload a project bundle, derive what it states, and fail naming whatever is still missing.

    The derivation runs server-side — the same endpoint Studio calls — so a CLI user and a Studio user
    get a manifest from one code path rather than two that drift.
    """
    root = Path(project_dir).expanduser().resolve()
    if not root.is_dir():
        typer.secho(f"Error: --project-dir {project_dir!r} is not a directory.", fg="red")
        raise typer.Exit(code=1)

    try:
        fileset = upload_project_dir(ctx.sdk, root, workspace=ctx.workspace)
    except Exception as exc:
        typer.secho(f"Error: could not upload the project — {exc}", fg="red")
        raise typer.Exit(code=1) from exc
    typer.echo(f"  uploaded  {root.name} -> {fileset}")

    try:
        derived = ctx.sdk.agent_hardener.manifests.inspect_project(
            fileset, dockerfile=dockerfile, workspace=ctx.workspace
        )
    except Exception as exc:
        typer.secho(f"Error: could not read the project — {exc}", fg="red")
        raise typer.Exit(code=1) from exc

    for warning in derived.get("warnings", []):
        typer.secho(f"  ! {warning}", fg="yellow")

    supplied = {
        "dockerfile": dockerfile,
        "start_command": start_command,
        "binaries": binaries,
        "harness": harness,
        "relay_integration_confirmed": relay_confirmed or None,
    }
    # Report every gap at once. Discovering them one flag per run is the slowest possible way to learn
    # what a project could not say about itself.
    missing = [field for field in derived.get("unresolved", []) if not supplied.get(field)]
    if missing:
        flags = {
            "dockerfile": "--dockerfile",
            "start_command": "--start-command",
            "binaries": "--binary",
            "harness": "--harness",
            "relay_integration_confirmed": "--relay-confirmed",
        }
        typer.secho(
            f"Error: the project does not state {', '.join(missing)}. "
            f"Pass {', '.join(flags[field] for field in missing)}.",
            fg="red",
        )
        raise typer.Exit(code=1)

    for field, value in (("dockerfile", derived.get("dockerfile")), ("start_command", derived.get("start_command"))):
        if not supplied.get(field) and value:
            typer.echo(f"  derived   {field}: {value}")

    body: dict[str, Any] = {
        "name": name or root.name,
        "source_type": "project",
        "project_fileset": fileset,
        "relay_integration_confirmed": relay_confirmed,
    }
    for key, value in (
        ("dockerfile", dockerfile),
        ("start_command", start_command),
        ("binaries", binaries or None),
        ("harness", harness),
    ):
        if value:
            body[key] = value
    return body


def register(app: typer.Typer) -> None:
    """Attach the provisioning and target-lifecycle commands to *app*."""

    @app.command()
    def doctor() -> None:
        """Read-only preflight: agent-hardener venv, garak venv, Docker daemon, OpenShell gateway."""
        config = AgentHardenerConfig.get()
        typer.echo("Agent Hardener preflight:")
        results = checks.run_checks(config)
        checks.print_checks(results)
        if all(check.ok for check in results):
            typer.secho("\nAll checks passed.", fg="green")
            return
        typer.secho("\nSome checks failed — run `nemo agent-hardener setup`.", fg="yellow")
        raise typer.Exit(code=1)

    @app.command()
    def setup(
        force: bool = typer.Option(False, "--force", "-f", help="Recreate the venv even if it already exists."),
    ) -> None:
        """Provision agent-hardener's venv, the garak venv, and the inference credential, then check prereqs.

        agent-hardener's setup registers the OpenShell gateway (best-effort). Docker and the
        OpenShell CLI/service install stay instructed — they need sudo/brew and are unreliable
        under a sandbox.
        """
        config = AgentHardenerConfig.get()
        provisioning.provision_venv(config, force=force)
        provisioning.run_agent_hardener_setup(config, force=force)
        credentials.provision_operator_env(config, force=force)

        typer.echo("\nChecking host prerequisites:")
        results = checks.run_checks(config)
        checks.print_checks(results)
        failed = [check.label for check in results if not check.ok]
        if failed:
            typer.secho(
                f"\nStill needed: {', '.join(failed)}. Follow the hints above, then re-run `nemo agent-hardener doctor`.",
                fg="yellow",
            )
            raise typer.Exit(code=1)
        typer.secho("\nSetup complete. Next: nemo agent-hardener init --agent <name>", fg="green")

    @app.command()
    def init(
        agent: str | None = typer.Option(
            None, "--agent", help="Registered NeMo Platform agent to war-game (name or workspace/name)."
        ),
        project_dir: str | None = typer.Option(
            None,
            "--project-dir",
            help="Bring your own: a directory holding the Dockerfile that builds your agent. Uploaded and "
            "read server-side; everything the project states is derived, and you are asked only for the rest.",
        ),
        dockerfile: str | None = typer.Option(
            None, "--dockerfile", help="Which Dockerfile builds the agent, when the project holds more than one."
        ),
        start_command: str | None = typer.Option(
            None,
            "--start-command",
            help="Command that serves the agent inside the sandbox. Derived from an exec-form ENTRYPOINT/CMD; "
            "required when the Dockerfile uses a shell form. Must be absolute — OpenShell replaces PATH.",
        ),
        binary: list[str] = typer.Option(
            None,
            "--binary",
            help="Glob matching the victim's interpreter, repeatable. Derived from the image's venv; override "
            "when the image puts it elsewhere.",
        ),
        harness: str | None = typer.Option(
            None,
            "--harness",
            help="Which harness the agent runs (deepagents, hermes, langchain, langgraph, other). Decides "
            "whether a guardrail can refuse a tool call, and cannot be read from the project.",
        ),
        relay_confirmed: bool = typer.Option(
            False,
            "--relay-confirmed",
            help="Confirm NeMo Relay is attached to the agent. Without it the victim emits no telemetry and "
            "the run cannot be scored.",
        ),
        name: str | None = typer.Option(
            None, "--name", help="Saved-manifest name (the id later phases reference). Defaults to the agent name."
        ),
        workspace: str | None = typer.Option(None, "--workspace", help="Agent workspace."),
        output: str = typer.Option("agent-hardener.yaml", "--output", "-o", help="Where to write the rendered YAML."),
        egress: list[str] = typer.Option(
            None,
            "--egress",
            help="Host[:port] the victim may reach, repeatable. A bare host opens 443 only — add "
            "'host:80' for plain HTTP. Config-only agents need this; their tool hosts can't be discovered.",
        ),
        secrets: list[str] = typer.Option(None, "--secrets", help="Override the derived secret names (repeatable)."),
        env: list[str] = typer.Option(
            None,
            "--env",
            help="Non-secret env var for the victim as KEY=VALUE, repeatable. Credentials belong in "
            "--secrets, which names them and resolves values from the platform Secrets store.",
        ),
        port: int | None = typer.Option(None, "--port", help="Victim port (default: the agent's deployment port)."),
        attack_model: str | None = typer.Option(
            None, "--attack-model", help="Default model for garak's red-team + detector."
        ),
        attack_base_url: str | None = typer.Option(
            None, "--attack-base-url", help="Custom OpenAI-compatible endpoint for the attack model."
        ),
        attack_key_secret: str | None = typer.Option(
            None, "--attack-key-secret", help="Secrets-store name holding the attack endpoint's API key."
        ),
        analysis_model: str | None = typer.Option(
            None, "--analysis-model", help="Default model for the defenders and the benign validator."
        ),
        analysis_base_url: str | None = typer.Option(
            None, "--analysis-base-url", help="Custom OpenAI-compatible endpoint for the analysis model."
        ),
        analysis_key_secret: str | None = typer.Option(
            None, "--analysis-key-secret", help="Secrets-store name holding the analysis endpoint's API key."
        ),
        safety_model: str | None = typer.Option(
            None,
            "--safety-model",
            help="Default model the generated guardrail uses to screen traffic. Unset reuses the agent's own.",
        ),
        assume_yes: bool = typer.Option(
            False, "--yes", "-y", help="Accept agent-hardener's detected answers instead of being prompted."
        ),
    ) -> None:
        """Save a reusable war-game target from a registered agent.

        Resolution happens server-side — the same path Studio takes — so the manifest a CLI user
        gets and the one Studio gets are produced by one code path.
        """
        if bool(agent) == bool(project_dir):
            typer.secho("Error: pass exactly one of --agent or --project-dir.", fg="red")
            raise typer.Exit(code=1)

        ctx = command_context(workspace)
        body: dict[str, Any]
        if project_dir:
            body = _project_init_body(
                ctx,
                project_dir=project_dir,
                name=name,
                dockerfile=dockerfile,
                start_command=start_command,
                binaries=list(binary or []),
                harness=harness,
                relay_confirmed=relay_confirmed,
            )
        else:
            body = {"name": name or str(agent).split("/")[-1], "agent": agent}
        if port:
            body["port"] = port
        if egress:
            body["egress"] = list(egress)
        if secrets:
            body["secrets"] = list(secrets)
        if env:
            body["env"] = parse_env_pairs(list(env))
        # No merge here, unlike `manifest set`: nothing is stored yet, so these are the initial selection.
        models = models_from_flags(
            attack_model=attack_model,
            attack_base_url=attack_base_url,
            attack_key_secret=attack_key_secret,
            analysis_model=analysis_model,
            analysis_base_url=analysis_base_url,
            analysis_key_secret=analysis_key_secret,
            safety_model=safety_model,
        )
        if models:
            preflight_models(ctx, models)
            body["models"] = models
        try:
            manifest = ctx.agent_hardener.manifests.create(workspace=ctx.workspace, **body)
        except Exception as exc:
            typer.secho(f"Error: could not create manifest — {exc}", fg="red")
            raise typer.Exit(code=1) from exc

        # A project manifest already printed these while deriving; repeating them here reads as two
        # separate problems rather than one.
        if manifest.get("source_type") != "project":
            for warning in json_string_list(manifest.get("warnings")):
                typer.secho(f"  ! {warning}", fg="yellow")
        manifest_name = str(manifest.get("name") or body["name"])
        secrets = json_string_list(manifest.get("secrets"))
        egress = json_string_list(manifest.get("egress"))
        typer.secho(f"Saved manifest '{manifest_name}'", fg="green")
        source = manifest.get("agent") or manifest.get("project_fileset") or "?"
        image = manifest.get("dockerfile") or "(generic, built from the project)"
        typer.echo(
            f"  source    {source}\n"
            f"  image     {image}\n"
            f"  workflow  {manifest.get('workflow') or '(none)'}\n"
            f"  victim    port {manifest.get('port', '?')}\n"
            f"  secrets   {', '.join(secrets)}\n"
            f"  egress    {', '.join(egress) or '(none — outbound calls are blocked)'}"
        )

        # Runnable, not just readable: `run --config` takes this file. Prefer
        # `run --manifest-id`, which uses the stored manifest and its frozen target fileset.
        if manifest.get("manifest_yaml"):
            out_path = Path(output)
            out_path.write_text(
                f"# Rendered from manifest '{manifest_name}'.\n"
                f"# Runnable with `run --config {output}`; `run --manifest-id {manifest_name}` "
                f"uses the stored manifest instead.\n"
                f"{manifest['manifest_yaml']}",
                encoding="utf-8",
            )
            typer.echo(f"  rendered  {out_path}")
        typer.echo(f"\nNext: nemo agent-hardener synth-benign --manifest-id {manifest_name}")

    @app.command()
    def refresh(
        manifest_id: str = typer.Option(..., "--manifest-id", help="Saved manifest to re-resolve."),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the manifest."),
    ) -> None:
        """Re-resolve a saved manifest against its agent as it is now.

        A manifest is a frozen target, so editing the agent — new model, new tool, redeploy —
        does not change it on its own. Run this to take those changes deliberately. Your egress,
        secrets, models, defenders and cached benign suite are kept.
        """
        ctx = command_context(workspace, preflight=False)
        try:
            manifest = ctx.agent_hardener.manifests.refresh(manifest_id, workspace=ctx.workspace)
        except Exception as exc:
            typer.secho(f"Error: could not refresh manifest '{manifest_id}' — {exc}", fg="red")
            raise typer.Exit(code=1) from exc

        for warning in json_string_list(manifest.get("warnings")):
            typer.secho(f"  ! {warning}", fg="yellow")
        secrets = json_string_list(manifest.get("secrets"))
        egress = json_string_list(manifest.get("egress"))
        typer.secho(f"Refreshed manifest '{manifest_id}' from {manifest.get('agent', '?')}", fg="green")
        typer.echo(
            f"  victim    port {manifest.get('port', '?')}\n"
            f"  secrets   {', '.join(secrets)}\n"
            f"  egress    {', '.join(egress) or '(none — outbound calls are blocked)'}"
        )

    @app.command()
    def status(
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace to read runs from."),
        limit: int = typer.Option(5, "--limit", help="How many recent runs to show."),
    ) -> None:
        """Show recent Agent Hardener runs."""
        # No preflight: reading run records doesn't need Docker/OpenShell/the venvs.
        ctx = command_context(workspace, preflight=False)
        ws = ctx.workspace
        runs = ctx.agent_hardener.runs.list(workspace=ws, limit=limit)
        if not runs:
            typer.echo(f"No Agent Hardener runs in workspace '{ws}'.")
            return
        for record in runs:
            mark = typer.style("✓", fg="green") if record.get("status") == "completed" else typer.style("✗", fg="red")
            typer.echo(
                f"  {mark} {record.get('created_at', '?')}  {record.get('agent', '?')}  "
                f"{record.get('status', '?')} (exit {record.get('returncode', '?')})  [{record.get('name', '')}]"
            )
