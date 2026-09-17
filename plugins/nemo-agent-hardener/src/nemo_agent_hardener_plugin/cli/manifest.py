# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agent-hardener manifest ...`` — inspect and edit a saved manifest's stored defaults.

These persist. The same knobs exist as per-launch flags on ``run``, which apply to one launch and
leave the frozen target untouched (Studio draws the same line: "Save as default" vs "Start").
"""

from __future__ import annotations

import json

import typer
from nemo_agent_hardener_plugin.cli._shared import (
    ATTACK_INTENSITIES,
    command_context,
    merge_models,
    models_from_flags,
    parse_env_pairs,
    preflight_models,
    validated_defenders,
    validated_intensity,
)
from nemo_agent_hardener_plugin.jobs.manifest import DEFENDER_ENTRIES


def build_app() -> typer.Typer:
    """The ``manifest`` sub-command group."""
    app = typer.Typer(
        help="Inspect and edit a saved manifest's stored defaults.", no_args_is_help=True, add_completion=False
    )

    @app.command("show")
    def manifest_show(
        name: str = typer.Argument(..., help="Saved manifest to display."),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the manifest."),
    ) -> None:
        """Print a saved manifest's stored war-game defaults."""
        # No preflight: reading a manifest record doesn't need Docker/OpenShell/the venvs.
        ctx = command_context(workspace, preflight=False)
        try:
            record = ctx.sdk.agent_hardener.manifests.get(name, workspace=ctx.workspace)
        except Exception as exc:
            typer.secho(f"Error: could not read manifest {name!r} — {exc}", fg="red")
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(record, indent=2, default=str))

    @app.command("set")
    def manifest_set(
        name: str = typer.Argument(..., help="Saved manifest to edit."),
        rounds: int | None = typer.Option(None, "--rounds", min=1, help="Default hardening rounds."),
        defender: list[str] = typer.Option(
            None,
            "--defender",
            help=f"Default defender to enable, repeatable ({', '.join(sorted(DEFENDER_ENTRIES))}). "
            "Replaces the stored selection.",
        ),
        attack_intensity: str | None = typer.Option(
            None,
            "--attack-intensity",
            help=f"Default attacker effort preset: {', '.join(ATTACK_INTENSITIES)}.",
        ),
        port: int | None = typer.Option(None, "--port", help="Default victim port."),
        egress: list[str] = typer.Option(
            None, "--egress", help="Allow-listed host[:port], repeatable. Replaces the stored list."
        ),
        env: list[str] = typer.Option(
            None,
            "--env",
            help="Non-secret env var as KEY=VALUE, repeatable. Replaces the stored map. Credentials "
            "belong in the manifest's secrets, which resolve from the platform Secrets store.",
        ),
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
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the manifest."),
    ) -> None:
        """Change a saved manifest's stored defaults (the baseline every later run starts from).

        This persists; it does not launch anything. To deviate for a single run without touching the
        saved baseline, pass the same flags to `nemo agent-hardener run` instead.
        """
        body: dict[str, object] = {}
        if rounds is not None:
            body["rounds"] = rounds
        if defender:
            body["defenders"] = validated_defenders(list(defender))
        if attack_intensity is not None:
            body["attack_intensity"] = validated_intensity(attack_intensity)
        if port is not None:
            body["port"] = port
        if egress:
            body["egress"] = list(egress)
        if env:
            body["env"] = parse_env_pairs(list(env))
        chosen_models = models_from_flags(
            attack_model=attack_model,
            attack_base_url=attack_base_url,
            attack_key_secret=attack_key_secret,
            analysis_model=analysis_model,
            analysis_base_url=analysis_base_url,
            analysis_key_secret=analysis_key_secret,
            safety_model=safety_model,
        )
        if not body and not chosen_models:
            typer.secho("Error: pass at least one field to set.", fg="red")
            raise typer.Exit(code=1)

        ctx = command_context(workspace, preflight=False)
        if chosen_models:
            preflight_models(ctx, chosen_models)
        try:
            if chosen_models:
                # PATCH replaces `models` wholesale, so merge over the stored selection first — otherwise
                # setting one group would silently clear the others.
                stored = ctx.sdk.agent_hardener.manifests.get(name, workspace=ctx.workspace).get("models") or {}
                body["models"] = merge_models(stored, chosen_models)
            ctx.sdk.agent_hardener.manifests.update(name, workspace=ctx.workspace, **body)
        except Exception as exc:
            typer.secho(f"Error: could not update manifest {name!r} — {exc}", fg="red")
            raise typer.Exit(code=1) from exc
        typer.secho(f"Updated manifest '{name}': {', '.join(sorted(body))}", fg="green")

    return app
