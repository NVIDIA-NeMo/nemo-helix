# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agent-hardener run | synth-benign | sanity-check`` — the war-game cycle itself."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from nemo_agent_hardener_plugin.cli._shared import (
    ATTACK_INTENSITIES,
    command_context,
    models_from_flags,
    validated_defenders,
    validated_intensity,
)
from nemo_agent_hardener_plugin.config import missing_secrets
from nemo_agent_hardener_plugin.jobs.defenses import defense_ids, select_defense_ids
from nemo_agent_hardener_plugin.jobs.manifest import DEFENDER_ENTRIES


def register(app: typer.Typer) -> None:
    """Attach the war-game commands to *app*."""

    @app.command()
    def run(
        config_file: str | None = typer.Option(
            None, "--config", "-c", help="Local manifest produced by `init` (default: agent-hardener.yaml)."
        ),
        manifest_id: str | None = typer.Option(
            None, "--manifest-id", help="Saved manifest to run (reuses its cached benign suite)."
        ),
        env_file: str | None = typer.Option(None, "--env-file", help="Dotenv with the agent's secrets."),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace for the run."),
        benign_suite: str | None = typer.Option(
            None,
            "--benign-suite",
            help="Benign-suite CSV (tool,payload,label,rationale,persona) to use as-is, overriding the cache.",
        ),
        rounds: int | None = typer.Option(
            None, "--rounds", min=1, help="Iterative attack/defend/validate hardening rounds (default: 1)."
        ),
        defender: list[str] = typer.Option(
            None,
            "--defender",
            help=f"Defender to enable, repeatable ({', '.join(sorted(DEFENDER_ENTRIES))}). "
            "Default: the manifest's saved selection. Requires --manifest-id.",
        ),
        attack_intensity: str | None = typer.Option(
            None,
            "--attack-intensity",
            help=f"Attacker (garak) effort preset: {', '.join(ATTACK_INTENSITIES)}. Requires --manifest-id.",
        ),
        port: int | None = typer.Option(
            None, "--port", help="Override the victim port for this run only. Requires --manifest-id."
        ),
        replay_hitlog: str | None = typer.Option(
            None,
            "--replay-hitlog",
            help="Fileset ref of a recorded garak hitlog to replay instead of attacking live.",
        ),
        attack_model: str | None = typer.Option(
            None, "--attack-model", help="Model for garak's red-team + detector. Default: agent-hardener's built-in."
        ),
        attack_base_url: str | None = typer.Option(
            None, "--attack-base-url", help="Custom OpenAI-compatible endpoint for the attack model."
        ),
        attack_key_secret: str | None = typer.Option(
            None, "--attack-key-secret", help="Secrets-store name holding the attack endpoint's API key."
        ),
        analysis_model: str | None = typer.Option(
            None,
            "--analysis-model",
            help="Model for the defenders and the benign validator (synth + judge). Default: agent-hardener's built-in.",
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
            help="Model the generated guardrail uses to screen traffic. Unset reuses the agent's own model. "
            "Not preflighted — it runs inside the victim against agent-hardener's own endpoint, so a bad name "
            "surfaces only when the guardrail runs.",
        ),
    ) -> None:
        """Run the attack/defend/validate war-game against a local manifest or a saved manifest.

        The override flags apply to this launch only — they never edit the saved manifest, so a run
        can deviate from the frozen baseline without breaking comparability. Use
        `nemo agent-hardener manifest set` to change the stored defaults instead.
        """
        ctx = command_context(workspace)
        if config_file and manifest_id:
            typer.secho("Pass either --config or --manifest-id, not both.", fg="red")
            raise typer.Exit(code=1)
        if benign_suite and not Path(benign_suite).is_file():
            typer.secho(f"Benign suite CSV {benign_suite} not found.", fg="red")
            raise typer.Exit(code=1)

        # The saved-manifest path materializes server-side and checks victim secrets in the job; the
        # local-file path validates the manifest exists and its secrets are satisfiable up front.
        if not manifest_id:
            config_file = config_file or "agent-hardener.yaml"
            if not Path(config_file).exists():
                typer.secho(
                    f"Manifest {config_file} not found — run `nemo agent-hardener init --agent <name>` first.",
                    fg="red",
                )
                raise typer.Exit(code=1)
            env_files = [ctx.config.operator_env_file] + ([Path(env_file)] if env_file else [])
            missing = missing_secrets(Path(config_file), env_files=env_files)
            if missing:
                typer.secho(
                    f"Missing required secrets: {', '.join(missing)}. Provide them via --env-file "
                    "or export them, then re-run.",
                    fg="red",
                )
                raise typer.Exit(code=1)

        # Validate before launching: an unknown preset or defender key is silently absorbed
        # downstream (unknown intensity reads as "standard", an unknown defender leaves the full
        # default set in place), so a typo would otherwise war-game the wrong configuration.
        try:
            result = ctx.sdk.agent_hardener.run(
                config=config_file,
                manifest_id=manifest_id,
                env_file=env_file,
                workspace=ctx.workspace,
                benign_suite=benign_suite,
                rounds=rounds,
                port=port,
                defenders=validated_defenders(list(defender or [])),
                attack_intensity=validated_intensity(attack_intensity),
                replay_hitlog_fileset=replay_hitlog,
                models=models_from_flags(
                    attack_model=attack_model,
                    attack_base_url=attack_base_url,
                    attack_key_secret=attack_key_secret,
                    analysis_model=analysis_model,
                    analysis_base_url=analysis_base_url,
                    analysis_key_secret=analysis_key_secret,
                    safety_model=safety_model,
                ),
            )
        except ValueError as exc:
            typer.secho(f"Error: {exc}", fg="red")
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(result, indent=2, default=str))
        raise typer.Exit(code=0 if result.get("status") == "completed" else 1)

    @app.command(name="synth-benign")
    def synth_benign(
        manifest_id: str = typer.Option(..., "--manifest-id", help="Saved manifest to synthesize a suite for."),
        env_file: str | None = typer.Option(None, "--env-file", help="Dotenv with the agent's secrets."),
        yes: bool = typer.Option(
            False, "--yes", "-y", help="Run the interview but auto-accept each recommended default (no prompts)."
        ),
        no_interactive: bool = typer.Option(
            False, "--no-interactive", help="Skip the interview entirely (leaner, rules-only suite; use in CI)."
        ),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the manifest."),
    ) -> None:
        """Synthesize the benign request suite for a saved manifest and cache it on the manifest.

        Brings the victim sandbox up, runs agent-hardener's interview/review (interactive by default), then
        tears it down — the reviewed suite is stored on the manifest so a later `run --manifest-id` reuses it.
        """
        ctx = command_context(workspace)
        if yes and no_interactive:
            typer.secho("Pass either --yes or --no-interactive, not both.", fg="red")
            raise typer.Exit(code=1)
        interview = "skip" if no_interactive else "auto" if yes else "interactive"

        result = ctx.sdk.agent_hardener.synth_benign(
            manifest_id=manifest_id,
            env_file=env_file,
            interview=interview,
            workspace=ctx.workspace,
        )
        if result.get("status") == "completed":
            typer.secho(
                f"Cached {result.get('suite_size', 0)} benign requests on manifest '{manifest_id}'.", fg="green"
            )
            typer.echo(f"\nNext: nemo agent-hardener run --manifest-id {manifest_id}")
            raise typer.Exit(code=0)
        typer.echo(json.dumps(result, indent=2, default=str))
        raise typer.Exit(code=1)

    @app.command(name="sanity-check")
    def sanity_check(
        manifest_id: str = typer.Option(..., "--manifest-id", help="Saved manifest to validate against."),
        mitigations_file: str = typer.Option(
            ..., "--mitigations", help="Path to the run's mitigations.json (its 'defenses' list is selected from)."
        ),
        replay_hitlog: str = typer.Option(
            ...,
            "--replay-hitlog",
            help="Fileset ref of the recorded garak hitlog to replay (run's hitlog_fileset).",
        ),
        keep: list[str] = typer.Option(
            None,
            "--keep",
            help="Defense id to keep (repeatable). Default: keep all. Mutually exclusive with --exclude.",
        ),
        exclude: list[str] = typer.Option(None, "--exclude", help="Defense id to drop (repeatable)."),
        env_file: str | None = typer.Option(None, "--env-file", help="Dotenv with the agent's secrets."),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace for the run."),
    ) -> None:
        """Freeze a chosen subset of a run's recommended defenses and replay the recorded attacks + benign.

        Runs the war-game cycle one last time with the mitigation-generating defenders disabled and the
        chosen defenses frozen as the victim baseline — a sanity check that reports which attacks the
        selection blocks and which benign requests it wrongly blocks (false positives). Get the
        ``mitigations.json`` and the hitlog fileset ref from a completed run (`nemo agent-hardener status`).
        """
        ctx = command_context(workspace)
        if keep and exclude:
            typer.secho("Pass either --keep or --exclude, not both.", fg="red")
            raise typer.Exit(code=1)
        path = Path(mitigations_file)
        if not path.exists():
            typer.secho(f"Mitigations file {mitigations_file} not found.", fg="red")
            raise typer.Exit(code=1)
        mitigations = json.loads(path.read_text(encoding="utf-8"))
        selected = select_defense_ids(defense_ids(mitigations), keep=keep or None, exclude=exclude or None)
        typer.echo(f"Sanity-checking {len(selected)} defense(s): {', '.join(selected) or '(none)'}")

        result = ctx.sdk.agent_hardener.sanity_check(
            manifest_id=manifest_id,
            mitigations=mitigations,
            selected_defense_ids=selected,
            replay_hitlog_fileset=replay_hitlog,
            env_file=env_file,
            workspace=ctx.workspace,
        )
        typer.echo(json.dumps(result, indent=2, default=str))
