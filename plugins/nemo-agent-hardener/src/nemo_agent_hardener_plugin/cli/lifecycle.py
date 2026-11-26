# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agent-hardener doctor | setup | init | stage | refresh | status`` — host provisioning and target lifecycle.

agent-hardener is never imported: it runs in its own venv, invoked by subprocess.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path
from typing import Any

import typer
import yaml
from nemo_agent_hardener_plugin.cli import checks, credentials, provisioning
from nemo_agent_hardener_plugin.cli._shared import (
    command_context,
    json_mapping,
    json_string_list,
    models_from_flags,
    parse_env_pairs,
    preflight_models,
)
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.filesets import download_and_extract_project, upload_project_dir
from nemo_helix_plugin.client.errors import NotFoundError


def _with_local_project(ctx: Any, manifest: dict[str, Any], out_path: Path) -> str:
    """The manifest YAML with ``project_dir`` pointing at a local copy of its stored victim bundle.

    The stored ``project_dir`` names where the service built the bundle, which does not exist on this
    machine, so ``run --config`` would fail on it. The bundle is downloaded next to *out_path*; on any
    failure the YAML is returned unchanged with a warning, since ``run --manifest-id`` still works.
    """
    manifest_yaml = str(manifest["manifest_yaml"])
    is_project = manifest.get("source_type") == "project"
    bundle = manifest.get("project_fileset") if is_project else manifest.get("agent_fileset")
    try:
        data = yaml.safe_load(manifest_yaml)
    except yaml.YAMLError:
        return manifest_yaml
    if not (bundle and isinstance(data, dict) and isinstance(data.get("agent"), dict)):
        return manifest_yaml
    workdir = out_path.parent / f"{out_path.stem}-victim"
    # Replace only a copy an earlier `init` wrote, never a directory of the user's that shares the name.
    if workdir.exists() and not {p.name for p in workdir.iterdir()} <= {"bundle", "project"}:
        typer.secho(f"  ! {workdir} exists and is not a victim bundle; leaving project_dir as stored.", fg="yellow")
        return manifest_yaml
    try:
        shutil.rmtree(workdir, ignore_errors=True)
        project = download_and_extract_project(ctx.sdk, str(bundle), workdir)
    except Exception as exc:
        typer.secho(f"  ! could not download the victim bundle for --config runs: {exc}", fg="yellow")
        return manifest_yaml
    data["agent"]["project_dir"] = os.path.relpath(project, out_path.parent)
    return yaml.safe_dump(data, sort_keys=False)


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
    stored: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Upload a project bundle, derive what it states, and fail naming whatever is still missing.

    The derivation runs server-side — the same endpoint Studio calls — so a CLI user and a Studio user
    get a manifest from one code path rather than two that drift. *stored* is the manifest a re-init
    updates: whatever it already answered is not asked for again (the server carries it over).
    """
    root = Path(project_dir).expanduser().resolve()
    if not root.is_dir():
        typer.secho(f"Error: --project-dir {project_dir!r} is not a directory.", fg="red")
        _say_unchanged(stored)
        raise typer.Exit(code=1)

    kept = _stored_project_answers(stored)
    try:
        fileset = upload_project_dir(ctx.sdk, root, workspace=ctx.workspace).ref
    except Exception as exc:
        typer.secho(f"Error: could not upload the project — {exc}", fg="red")
        _say_unchanged(stored)
        raise typer.Exit(code=1) from exc
    typer.echo(f"  uploaded  {root.name} -> {fileset}")

    try:
        derived = ctx.agent_hardener.manifests.inspect_project(
            fileset, dockerfile=dockerfile or kept.get("dockerfile") or None, workspace=ctx.workspace
        )
    except Exception as exc:
        typer.secho(f"Error: could not read the project — {exc}", fg="red")
        _say_unchanged(stored)
        raise typer.Exit(code=1) from exc

    # On a re-init, mark what the saved manifest did not already warn about: that is what this upload changed.
    known = json_string_list(stored.get("warnings")) if stored else None
    for warning in derived.get("warnings", []):
        tag = "(new) " if known is not None and warning not in known else ""
        typer.secho(f"  ! {tag}{warning}", fg="yellow")

    supplied = {
        "dockerfile": dockerfile or kept.get("dockerfile"),
        "start_command": start_command or kept.get("start_command"),
        "binaries": binaries or kept.get("binaries"),
        "harness": harness or kept.get("harness"),
        "relay_integration_confirmed": relay_confirmed or kept.get("relay_integration_confirmed") or None,
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
        _say_unchanged(stored)
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


def _say_unchanged(stored: dict[str, Any] | None) -> None:
    """After a failed re-init, say the saved manifest is intact: an update writes only once everything succeeded."""
    if stored:
        typer.secho(
            f"  Manifest '{stored.get('name')}' was not changed; it still holds the previous upload and settings.",
            fg="yellow",
        )


def _stored_project_answers(stored: dict[str, Any] | None) -> dict[str, Any]:
    """What a saved project manifest already answered that `init` would otherwise ask for."""
    if not stored:
        return {}
    try:
        data = yaml.safe_load(str(stored.get("manifest_yaml") or "")) or {}
    except yaml.YAMLError:
        data = {}
    agent = data.get("agent") if isinstance(data, dict) else None
    agent = agent if isinstance(agent, dict) else {}
    return {
        "dockerfile": stored.get("dockerfile") or None,
        "binaries": json_string_list(stored.get("binaries")) or None,
        "start_command": agent.get("start_command") or None,
        "harness": agent.get("harness") or None,
        "relay_integration_confirmed": bool(agent.get("relay_integration_confirmed")) or None,
    }


def _saved_manifest(ctx: Any, name: str) -> dict[str, Any] | None:
    """The manifest saved under *name*, or None when there is none."""
    try:
        return ctx.agent_hardener.manifests.get(name, workspace=ctx.workspace)
    except NotFoundError:
        return None


def _name_taken_message(existing: dict[str, Any], *, agent: str | None) -> str:
    """Why `init` will not reuse *existing*'s name, and what to run instead. Mirrors the server's 409."""
    name = existing.get("name")
    if agent and existing.get("source_type") != "project":
        return (
            f"Error: manifest '{name}' already exists for agent '{existing.get('agent')}'. To take the agent's "
            f"latest changes, run `nemo agent-hardener refresh --manifest-id {name}`; to save another, pass --name."
        )
    origin = "an uploaded project" if existing.get("source_type") == "project" else f"agent '{existing.get('agent')}'"
    return f"Error: manifest '{name}' already exists and was created from {origin}; pass --name to save another."


def _changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """One line per setting a re-init changed, so an update is never silent about what it replaced."""
    lines: list[str] = []
    before_answers, after_answers = _stored_project_answers(before), _stored_project_answers(after)
    for label, old, new in (
        ("dockerfile", before.get("dockerfile"), after.get("dockerfile")),
        ("start", before_answers.get("start_command"), after_answers.get("start_command")),
        ("port", before.get("port"), after.get("port")),
    ):
        if old != new:
            lines.append(f"  {label:<9} {old or '(none)'} -> {new or '(none)'}")
    for label, key in (("binaries", "binaries"), ("secrets", "secrets"), ("egress", "egress")):
        old_items, new_items = json_string_list(before.get(key)), json_string_list(after.get(key))
        added = [item for item in new_items if item not in old_items]
        removed = [item for item in old_items if item not in new_items]
        if added or removed:
            parts = [f"+{item}" for item in added] + [f"-{item}" for item in removed]
            lines.append(f"  {label:<9} {' '.join(parts)}")
    return lines


def register(app: typer.Typer) -> None:
    """Attach the provisioning and target-lifecycle commands to *app*."""

    @app.command()
    def doctor() -> None:
        """Read-only preflight: agent-hardener venv, garak venv, Docker daemon, OpenShell CLI and gateway versions."""
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
            None, "--agent", help="Registered NeMo Helix agent to war-game (name or workspace/name)."
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
        backend: list[str] = typer.Option(
            None,
            "--backend",
            help="Host service the victim calls, as NAME:PORT[,PORT], repeatable (e.g. a remote MCP server "
            "running on the host). Opens the sandbox-to-host route for those ports.",
        ),
        secrets: list[str] = typer.Option(
            None,
            "--secrets",
            help="Secret the victim needs, repeatable: NAME takes its value from the operator env or --env-file; "
            "NAME=[workspace/]secret reads it from the platform Secrets store at run time. Replaces the derived names.",
        ),
        env: list[str] = typer.Option(
            None,
            "--env",
            help="Non-secret env var for the victim as KEY=VALUE, repeatable. Stored in plaintext; credentials "
            "belong in --secrets NAME=workspace/secret.",
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
            help="Default model the generated guardrail uses to screen traffic. Unset uses the analysis model.",
        ),
        safety_base_url: str | None = typer.Option(
            None,
            "--safety-base-url",
            help="Custom OpenAI-compatible endpoint for the safety model. Unset uses the analysis endpoint.",
        ),
        safety_key_secret: str | None = typer.Option(
            None, "--safety-key-secret", help="Secrets-store name holding the safety endpoint's API key."
        ),
        assume_yes: bool = typer.Option(
            False, "--yes", "-y", help="Accept agent-hardener's detected answers instead of being prompted."
        ),
    ) -> None:
        """Save a reusable war-game target from a registered agent.

        Resolution happens server-side — the same path Studio takes — so the manifest a CLI user
        gets and the one Studio gets are produced by one code path.

        Re-running with `--project-dir` on an existing project manifest's name updates it in place: the
        project is re-uploaded and re-read, and your settings are kept. An agent manifest takes its
        agent's changes through `refresh` instead.
        """
        if bool(agent) == bool(project_dir):
            typer.secho("Error: pass exactly one of --agent or --project-dir.", fg="red")
            raise typer.Exit(code=1)

        ctx = command_context(workspace)
        target = name or (Path(project_dir).expanduser().resolve().name if project_dir else str(agent).split("/")[-1])
        # Look before uploading: a name that cannot be reused should fail in a second, not after the upload.
        existing = _saved_manifest(ctx, target)
        if existing is not None:
            if not (project_dir and existing.get("source_type") == "project"):
                typer.secho(_name_taken_message(existing, agent=agent), fg="red")
                raise typer.Exit(code=1)
            typer.echo(
                f"Manifest '{target}' exists; updating it from {project_dir}. Its egress, secrets, env, backends, "
                "models, defenders, intensity, rounds and benign suite are kept unless you pass them again."
            )
        body: dict[str, Any]
        if project_dir:
            body = _project_init_body(
                ctx,
                project_dir=project_dir,
                name=target,
                dockerfile=dockerfile,
                start_command=start_command,
                binaries=list(binary or []),
                harness=harness,
                relay_confirmed=relay_confirmed,
                stored=existing,
            )
            if existing is not None:
                body["update_existing"] = True
        else:
            body = {"name": target, "agent": agent}
        if port:
            body["port"] = port
        if egress:
            body["egress"] = list(egress)
        if backend:
            body["backends"] = list(backend)
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
            safety_base_url=safety_base_url,
            safety_key_secret=safety_key_secret,
        )
        if models:
            try:
                preflight_models(ctx, models)
            except typer.Exit:
                _say_unchanged(existing)
                raise
            body["models"] = models
        try:
            manifest = ctx.agent_hardener.manifests.create(workspace=ctx.workspace, **body)
        except Exception as exc:
            detail = getattr(exc, "detail", None) or exc
            typer.secho(
                f"Error: could not {'update' if existing else 'create'} manifest '{target}' — {detail}", fg="red"
            )
            _say_unchanged(existing)
            raise typer.Exit(code=1) from exc

        # A project manifest already printed these while deriving; repeating them here reads as two
        # separate problems rather than one.
        if manifest.get("source_type") != "project":
            for warning in json_string_list(manifest.get("warnings")):
                typer.secho(f"  ! {warning}", fg="yellow")
        manifest_name = str(manifest.get("name") or body["name"])
        secrets = json_string_list(manifest.get("secrets"))
        egress = json_string_list(manifest.get("egress"))
        if existing is None:
            typer.secho(f"Saved manifest '{manifest_name}'", fg="green")
        else:
            typer.secho(f"Updated manifest '{manifest_name}'", fg="green")
            for line in _changes(existing, manifest) or ["  (nothing the project states changed)"]:
                typer.echo(line)
            if existing.get("victim_image") and not manifest.get("victim_image"):
                typer.secho(
                    f"  ! victim image {existing['victim_image']} was built from the previous upload and is cleared;"
                    f" rebuild it: nemo agent-hardener stage --manifest-id {manifest_name}",
                    fg="yellow",
                )
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
                f"{_with_local_project(ctx, manifest, out_path)}",
                encoding="utf-8",
            )
            typer.echo(f"  rendered  {out_path}")
        if existing is not None and existing.get("benign_suite"):
            typer.echo(f"\nNext: nemo agent-hardener run --manifest-id {manifest_name} (the benign suite is kept)")
        else:
            typer.echo(f"\nNext: nemo agent-hardener synth-benign --manifest-id {manifest_name}")

    @app.command()
    def stage(
        manifest_id: str = typer.Option(..., "--manifest-id", help="Saved manifest whose victim to stage."),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the manifest."),
    ) -> None:
        """Stage a saved manifest's victim build context, to build the victim image yourself.

        Needed where war-games run in a container (NeMo Helix on Kubernetes): those jobs cannot build the
        victim, so they run it from a registry image you build from this context. Needs `nemo agent-hardener
        setup`, not Docker or OpenShell.
        """
        # No preflight: staging needs the agent-hardener venv only, which the job itself checks.
        ctx = command_context(workspace, preflight=False)
        result = ctx.agent_hardener.stage_victim_context(manifest_id=manifest_id, workspace=ctx.workspace)
        if result.get("status") != "completed":
            error = json_mapping(result.get("error"))
            typer.secho(f"Error: could not stage manifest '{manifest_id}' — {error.get('message', result)}", fg="red")
            if remediation := error.get("remediation"):
                typer.echo(f"  {remediation}")
            raise typer.Exit(code=1)

        archive = shlex.quote(str(result["context_archive"]))
        image = "<registry>/<image>:<tag>"
        scope = f" --workspace {shlex.quote(workspace)}" if workspace else ""
        typer.secho(f"Staged the victim build context: {archive}", fg="green")
        typer.echo(
            "\nBuild and push it to a registry the OpenShell gateway can pull from, then record the image:\n"
            f"  docker build -t {image} - < {archive}\n"
            f"  docker push {image}\n"
            f"  nemo agent-hardener manifest set {shlex.quote(manifest_id)}{scope} --victim-image {image}"
        )

    @app.command()
    def refresh(
        manifest_id: str = typer.Option(..., "--manifest-id", help="Saved manifest to re-resolve."),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the manifest."),
    ) -> None:
        """Re-resolve a saved manifest against its agent as it is now.

        A manifest is a frozen target, so editing the agent — new model, new tool, redeploy —
        does not change it on its own. Run this to take those changes deliberately. Your egress,
        secrets, models, defenders and cached benign suite are kept. A project manifest has no agent
        to refresh from: re-run `init --project-dir <dir>` with the same name instead.
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
