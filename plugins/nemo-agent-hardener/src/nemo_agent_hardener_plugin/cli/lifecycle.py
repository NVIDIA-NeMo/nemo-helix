# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agent-hardener doctor | setup | init | refresh | status`` — host provisioning and target lifecycle.

agent-hardener is never imported: it runs in its own venv, invoked by subprocess.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from nemo_agent_hardener_plugin.cli import checks, credentials, provisioning
from nemo_agent_hardener_plugin.cli._shared import (
    CommandContext,
    command_context,
    models_from_flags,
    parse_env_pairs,
    preflight_models,
)
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.filesets import upload_project_dir


def _check_byo(project_dir: Path, dockerfile: str, binaries: list[str]) -> None:
    """Reject a BYO request that would only fail later, on a different machine."""
    if not binaries:
        typer.secho(
            "Error: --dockerfile requires at least one --binary glob, e.g. --binary '/app/.venv/bin/**'.", fg="red"
        )
        raise typer.Exit(code=1)
    if Path(dockerfile).is_absolute():
        # The run re-downloads the bundle elsewhere, so only a project-relative path resolves there.
        typer.secho(f"Error: --dockerfile must be relative to {project_dir}, not an absolute path.", fg="red")
        raise typer.Exit(code=1)
    candidate = (project_dir / dockerfile).resolve()
    if not candidate.is_relative_to(project_dir.resolve()) or not candidate.is_file():
        typer.secho(f"Error: --dockerfile not found inside the project: {dockerfile}", fg="red")
        raise typer.Exit(code=1)


def _project_init_body(
    ctx: CommandContext,
    project_dir: Path,
    *,
    name: str | None,
    workflow: str | None,
    dockerfile: str | None,
    binaries: list[str],
    port: int | None,
    egress: list[str],
    secrets: list[str],
    assume_yes: bool,
) -> dict[str, object]:
    """Run agent-hardener's own ``init`` on a local project, then upload it and return the create body.

    Delegating to the local binary keeps agent-hardener the single owner of both the detection and the
    questions it asks — the operator answers them at their terminal, which the server-side path
    (``init --yes`` behind an HTTP request) structurally cannot offer. The platform then stores the
    manifest agent-hardener produced instead of rebuilding it.
    """
    if not project_dir.is_dir():
        typer.secho(f"Error: {project_dir} is not a directory.", fg="red")
        raise typer.Exit(code=1)
    if dockerfile:
        _check_byo(project_dir, dockerfile, binaries)

    manifest_name = name or project_dir.resolve().name
    with tempfile.TemporaryDirectory() as tmp:
        rendered = Path(tmp) / "agent-hardener.yaml"
        # `--project-dir .` with cwd set mirrors the server, so the stored manifest is path-independent.
        cmd = [
            str(ctx.config.agent_hardener_bin),
            "init",
            "--force",
            "--project-dir",
            ".",
            "--name",
            manifest_name,
            "-o",
            str(rendered),
        ]
        if assume_yes:
            cmd.append("--yes")
        if workflow:
            cmd += ["--workflow", workflow]
        if dockerfile:
            cmd += ["--dockerfile", dockerfile]
            for glob in binaries:
                cmd += ["--binary", glob]
        if port:
            cmd += ["--port", str(port)]
        if secrets:
            cmd += ["--secrets", ",".join(secrets)]
        for host in egress:
            cmd += ["--egress", host]
        provisioning.run_subprocess(
            cmd, "build the manifest with `agent-hardener init`", cwd=str(project_dir), timeout=None
        )
        if not rendered.is_file():
            typer.secho("Error: `agent-hardener init` produced no manifest.", fg="red")
            raise typer.Exit(code=1)
        manifest_yaml = rendered.read_text(encoding="utf-8")

    try:
        fileset = upload_project_dir(ctx.sdk, project_dir, workspace=ctx.workspace)
    except Exception as exc:
        typer.secho(f"Error: could not upload {project_dir} — {exc}", fg="red")
        raise typer.Exit(code=1) from exc
    typer.echo(f"Uploaded {project_dir} as {fileset}")

    return {
        "name": manifest_name,
        "source_type": "project",
        "project_fileset": fileset,
        "manifest_yaml": manifest_yaml,
        "launch_mode": "byo" if dockerfile else "workflow",
    }


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
            None, "--agent", help="Deployed NeMo Platform agent to target (name or workspace/name)."
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
        project_dir: str | None = typer.Option(
            None, "--project-dir", help="Local NAT project to upload and war-game (alternative to --agent)."
        ),
        workflow: str | None = typer.Option(
            None, "--workflow", help="Workflow path within the project (project source; default: detected)."
        ),
        dockerfile: str | None = typer.Option(
            None,
            "--dockerfile",
            help="Project-relative Dockerfile to build the victim from instead of a generic image (project "
            "source) — for agents needing system packages or a custom base image. Composes with --workflow; "
            "requires --binary. The image must carry a 'sandbox' user/group, iproute2, and `nat` on the "
            "default PATH.",
        ),
        binary: list[str] = typer.Option(
            None,
            "--binary",
            help="In-container glob scoping which processes may egress, repeatable (e.g. '/app/.venv/bin/**'). "
            "Required with --dockerfile.",
        ),
        port: int | None = typer.Option(None, "--port", help="Victim port (project source; default: detected)."),
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
        """Save a reusable war-game target: a deployed agent (--agent) or a NAT project (--project-dir).

        --agent resolves server-side, the path Studio also takes. --project-dir runs agent-hardener's
        own interactive init here, so you answer its questions, then uploads the project and
        stores the manifest it produced.
        """
        if bool(agent) == bool(project_dir):
            typer.secho("Error: pass exactly one of --agent or --project-dir.", fg="red")
            raise typer.Exit(code=1)
        if dockerfile and agent:
            typer.secho("Error: --dockerfile applies to --project-dir only; a deployed agent has no image.", fg="red")
            raise typer.Exit(code=1)

        ctx = command_context(workspace)
        body: dict[str, object]
        if project_dir:
            body = _project_init_body(
                ctx,
                Path(project_dir),
                name=name,
                workflow=workflow,
                dockerfile=dockerfile,
                binaries=list(binary or []),
                port=port,
                egress=list(egress or []),
                secrets=list(secrets or []),
                assume_yes=assume_yes,
            )
        else:
            body = {"name": name or str(agent).split("/")[-1], "source_type": "agent", "agent": agent}
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
            manifest = ctx.sdk.agent_hardener.manifests.create(workspace=ctx.workspace, **body)
        except Exception as exc:
            typer.secho(f"Error: could not create manifest — {exc}", fg="red")
            raise typer.Exit(code=1) from exc

        for warning in manifest.get("warnings") or []:
            typer.secho(f"  ! {warning}", fg="yellow")
        manifest_name = str(manifest.get("name") or body["name"])
        typer.secho(f"Saved manifest '{manifest_name}'", fg="green")
        source = manifest.get("agent") or manifest.get("project_fileset") or "?"
        image = manifest.get("dockerfile") or "(generic, built from the project)"
        typer.echo(
            f"  source    {source}\n"
            f"  image     {image}\n"
            f"  workflow  {manifest.get('workflow') or '(none)'}\n"
            f"  victim    port {manifest.get('port', '?')}\n"
            f"  secrets   {', '.join(manifest.get('secrets') or [])}\n"
            f"  egress    {', '.join(manifest.get('egress') or []) or '(none — outbound calls are blocked)'}"
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
            manifest = ctx.sdk.agent_hardener.manifests.refresh(manifest_id, workspace=ctx.workspace)
        except Exception as exc:
            typer.secho(f"Error: could not refresh manifest '{manifest_id}' — {exc}", fg="red")
            raise typer.Exit(code=1) from exc

        for warning in manifest.get("warnings") or []:
            typer.secho(f"  ! {warning}", fg="yellow")
        typer.secho(f"Refreshed manifest '{manifest_id}' from {manifest.get('agent', '?')}", fg="green")
        typer.echo(
            f"  victim    port {manifest.get('port', '?')}\n"
            f"  secrets   {', '.join(manifest.get('secrets') or [])}\n"
            f"  egress    {', '.join(manifest.get('egress') or []) or '(none — outbound calls are blocked)'}"
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
        runs = ctx.sdk.agent_hardener.runs.list(workspace=ws, limit=limit)
        if not runs:
            typer.echo(f"No Agent Hardener runs in workspace '{ws}'.")
            return
        for record in runs:
            mark = typer.style("✓", fg="green") if record.get("status") == "completed" else typer.style("✗", fg="red")
            typer.echo(
                f"  {mark} {record.get('created_at', '?')}  {record.get('agent', '?')}  "
                f"{record.get('status', '?')} (exit {record.get('returncode', '?')})  [{record.get('name', '')}]"
            )
