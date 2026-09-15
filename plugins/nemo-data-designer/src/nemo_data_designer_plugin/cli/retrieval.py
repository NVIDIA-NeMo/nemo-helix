# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec-building helpers for retrieval job and preview commands."""

from __future__ import annotations

import json
import shlex

import typer
from nemo_data_designer_plugin.jobs.retrieval_spec import RetrievalGenerateJobConfig, RetrievalPrepareJobConfig

retrieval_app = typer.Typer(
    name="retrieval",
    help=(
        "Build specs for Nemotron retrieval SDG commands. "
        "These helpers print the auto-generated retrieval-generate, retrieval-prepare, "
        "or retrieval-preview command to submit."
    ),
    no_args_is_help=True,
)


@retrieval_app.command("generate")
def retrieval_generate(
    corpus: str = typer.Option(..., "--corpus", help="Corpus fileset ref or hf:// URI."),
    provider: str = typer.Option(..., "--provider", help="Inference Gateway provider (workspace/name)."),
    chat_model: str = typer.Option(..., "--chat-model", help="Chat model for artifact extraction, Q&A, and judging."),
    embed_model: str = typer.Option(..., "--embed-model", help="Embedding model."),
    workspace: str = typer.Option("default", "--workspace", "-w"),
    spec_out: bool = typer.Option(False, "--print-spec", help="Print JSON spec instead of a submission command."),
) -> None:
    """Build a spec for the auto-generated ``retrieval-generate`` job command."""
    spec = RetrievalGenerateJobConfig(
        corpus=corpus,
        provider=provider,
        artifact_extraction_model=chat_model,
        qa_generation_model=chat_model,
        quality_judge_model=chat_model,
        embed_model=embed_model,
    )
    payload = spec.model_dump(mode="json")
    if spec_out:
        typer.echo(json.dumps(payload, indent=2))
        return
    _echo_submit_command("retrieval-generate", workspace, payload)


@retrieval_app.command("prepare")
def retrieval_prepare(
    sdg_input: str | None = typer.Option(
        None,
        "--sdg-input",
        help="Stage 0 fileset, fileset#file, or hf:// URI (optionally /path to a file).",
    ),
    generation_file: str | None = typer.Option(
        None,
        "--generation-file",
        help="Relative Stage 0 filename inside sdg_input when the ref is a directory.",
    ),
    train_input_file: str | None = typer.Option(None, "--train-input-file"),
    enable_mining: bool = typer.Option(
        False,
        "--mine/--no-mine",
        help="Run GPU hard-negative mining after conversion. Conversion-only is the default.",
    ),
    workspace: str = typer.Option("default", "--workspace", "-w"),
) -> None:
    """Build a spec for the auto-generated ``retrieval-prepare`` job command."""
    if (sdg_input is None) == (train_input_file is None):
        raise typer.BadParameter("Provide exactly one of --sdg-input or --train-input-file.")
    if generation_file is None:
        spec = RetrievalPrepareJobConfig(
            sdg_input=sdg_input,
            train_input_file=train_input_file,
            enable_mining=enable_mining,
        )
    else:
        spec = RetrievalPrepareJobConfig(
            sdg_input=sdg_input,
            train_input_file=train_input_file,
            enable_mining=enable_mining,
            generation_file=generation_file,
        )
    _echo_submit_command("retrieval-prepare", workspace, spec.model_dump(mode="json"))


@retrieval_app.command("preview")
def retrieval_preview(
    corpus: str = typer.Option(..., "--corpus", help="Corpus fileset ref or hf:// URI."),
    provider: str = typer.Option(..., "--provider", help="Inference Gateway provider (workspace/name)."),
    chat_model: str = typer.Option(..., "--chat-model", help="Chat model for artifact extraction, Q&A, and judging."),
    embed_model: str = typer.Option(..., "--embed-model", help="Embedding model."),
    workspace: str = typer.Option("default", "--workspace", "-w"),
) -> None:
    """Build a spec for the auto-generated ``retrieval-preview`` function command."""
    generate = RetrievalGenerateJobConfig(
        corpus=corpus,
        provider=provider,
        artifact_extraction_model=chat_model,
        qa_generation_model=chat_model,
        quality_judge_model=chat_model,
        embed_model=embed_model,
    )
    spec = {"generate": generate.model_dump(mode="json"), "num_records": 1}
    _echo_submit_command("retrieval-preview", workspace, spec)


def _echo_submit_command(job_name: str, workspace: str, spec: dict) -> None:
    typer.echo("Submit with:")
    typer.echo(
        "  "
        + shlex.join(
            [
                "nemo",
                "data-designer",
                job_name,
                "--workspace",
                workspace,
                "--spec",
                json.dumps(spec),
            ]
        )
    )
