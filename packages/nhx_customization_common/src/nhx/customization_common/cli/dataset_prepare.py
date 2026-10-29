# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo customization dataset prepare``: a Parquet, CSV or JSONL table as a training dataset.

The rows are read on this machine, converted to the requested format and split, and
uploaded to a new ``dataset`` fileset through the Files service. Nothing runs on the platform.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import typer
from nemo_helix_plugin.cli_options import WorkspaceOption
from nemo_helix_plugin.cli_state import resolve_cli_workspace
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.commands import resolve_submit_auth_headers, resolve_submit_base_url
from nemo_helix_plugin.files.client import FilesClient
from nhx.customization_common.cli.uploads import UploadError, UploadReport, create_dataset_fileset
from nhx.customization_common.dataset_prepare import (
    SPLIT_FILES,
    TRAINED_BY,
    DatasetFormat,
    DatasetPrepareError,
    InputFormat,
    map_columns,
    read_rows,
    split_rows,
    to_training_rows,
    write_split,
)

_COLUMNS_PANEL = "Column mapping"

_SHORT_HELP = "Convert a Parquet, CSV or JSONL table into a Customizer training dataset and upload it as a fileset."

#: What the help adds after a format's backends.
_FORMAT_NOTES = {DatasetFormat.PROMPT_COMPLETION: " Raw text with no chat template, for base models."}

# A bullet per format: command help strips each line's indentation, so the list cannot rely on it.
_FORMATS_HELP = "Formats (--format), and the backends that train on each:\n" + "\n".join(
    f"- {fmt}: {'; or '.join(TRAINED_BY[fmt])}.{_FORMAT_NOTES.get(fmt, '')}" for fmt in DatasetFormat
)

_GRPO_HELP = (
    "GRPO is not supported: rl GRPO (training.type: grpo) trains on NeMo Gym rows that each environment "
    "defines, not on a table. Write them for the environment and upload them with --upload-dataset "
    "on the rl command."
)

_GROUP_HELP = (
    f"{_SHORT_HELP}\n\n{_FORMATS_HELP}\n\n{_GRPO_HELP}\n\n"
    "The prepare command's help lists the source, split and column options."
)

_PREPARE_HELP = (
    f"{_SHORT_HELP}\n\n"
    "Reads SOURCE on this machine: a local file or directory, or a fileset. Writes its rows in the format "
    "--format names, drops rows with an empty field, and splits the rest with a seeded shuffle. Then uploads "
    "the training and validation splits to the platform as a new dataset fileset, NAME (train.jsonl, "
    "validation.jsonl), and the test split, when there is one, as a second fileset, NAME-test (test.jsonl). "
    "Prints the fileset references as JSON, to put in the job JSON. It creates filesets, not a training job.\n\n"
    f"{_FORMATS_HELP}\n\n{_GRPO_HELP}\n\n"
    "Data Designer output: download the job's artifacts result (nemo jobs results download artifacts "
    "--job JOB -o artifacts.tar), extract it, and pass its parquet-files directory."
)


def add_dataset_commands(app: typer.Typer) -> None:
    """Register ``nemo customization dataset prepare``."""
    dataset = typer.Typer(name="dataset", help=_GROUP_HELP, short_help=_SHORT_HELP, no_args_is_help=True)

    @dataset.command("prepare", help=_PREPARE_HELP, short_help=_SHORT_HELP)
    def prepare(
        typer_ctx: typer.Context,
        source: str = typer.Argument(
            ...,
            help="A local file or directory, or a fileset as WORKSPACE/NAME[#PATH]. Data Designer: its parquet-files.",
        ),
        name: str = typer.Option(
            ...,
            "--name",
            help="Dataset fileset to create on the platform and upload the splits to. The test split goes to NAME-test.",
        ),
        dataset_format: DatasetFormat = typer.Option(
            ..., "--format", help="Row format to write; it decides which backends can train on the dataset."
        ),
        input_format: InputFormat | None = typer.Option(
            None,
            "--input-format",
            help="Format of the source files. Default: from the file extensions, which must all be the same.",
        ),
        val_fraction: float = typer.Option(0.1, "--val-fraction", help="Share of rows for validation."),
        test_fraction: float | None = typer.Option(
            None,
            "--test-fraction",
            help="Share of rows held out for evaluation. Default: 0 for preference, else 0.1.",
        ),
        seed: int = typer.Option(42, "--seed", help="Shuffle seed. The same seed gives the same split."),
        prompt_column: str | None = typer.Option(
            None,
            "--prompt-column",
            help="The user prompt (chat, preference, prompt-completion). Default: instruction or prompt.",
            rich_help_panel=_COLUMNS_PANEL,
        ),
        response_column: str | None = typer.Option(
            None,
            "--response-column",
            help="The answer to train on (chat, prompt-completion). Default: response or completion.",
            rich_help_panel=_COLUMNS_PANEL,
        ),
        system_column: str | None = typer.Option(
            None,
            "--system-column",
            help="A system prompt added to each row (chat, preference). Default: none.",
            rich_help_panel=_COLUMNS_PANEL,
        ),
        messages_column: str | None = typer.Option(
            None,
            "--messages-column",
            help="Whole chat conversations, as a list of {role, content} (chat). Default: messages.",
            rich_help_panel=_COLUMNS_PANEL,
        ),
        chosen_column: str | None = typer.Option(
            None,
            "--chosen-column",
            help="The preferred answer (preference). Default: chosen.",
            rich_help_panel=_COLUMNS_PANEL,
        ),
        rejected_column: str | None = typer.Option(
            None,
            "--rejected-column",
            help="The dispreferred answer (preference). Default: rejected.",
            rich_help_panel=_COLUMNS_PANEL,
        ),
        query_column: str | None = typer.Option(
            None,
            "--query-column",
            help="The query (embedding). Default: query.",
            rich_help_panel=_COLUMNS_PANEL,
        ),
        positive_column: str | None = typer.Option(
            None,
            "--positive-column",
            help="The document that matches the query (embedding). Default: pos_doc.",
            rich_help_panel=_COLUMNS_PANEL,
        ),
        negatives_column: str | None = typer.Option(
            None,
            "--negatives-column",
            help="A list of documents that do not match the query (embedding). Default: neg_doc.",
            rich_help_panel=_COLUMNS_PANEL,
        ),
        text_column: str | None = typer.Option(
            None,
            "--text-column",
            help="The text to train on (text). Default: text.",
            rich_help_panel=_COLUMNS_PANEL,
        ),
        exist_ok: bool = typer.Option(
            False, "--exist-ok", help="Reuse a fileset that already exists. Its files are left as they are."
        ),
        workspace: WorkspaceOption = None,
    ) -> None:
        workspace = resolve_cli_workspace(typer_ctx, workspace)
        headers = resolve_submit_auth_headers(typer_ctx) or None
        files = FilesClient(base_url=resolve_submit_base_url(typer_ctx), workspace=workspace, default_headers=headers)
        named = {
            "prompt": prompt_column,
            "response": response_column,
            "system": system_column,
            "messages": messages_column,
            "chosen": chosen_column,
            "rejected": rejected_column,
            "query": query_column,
            "positive": positive_column,
            "negatives": negatives_column,
            "text": text_column,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rows_path = _fetch_source(source, files=files, workspace=workspace, into=Path(tmp) / "source")
                result = _prepare(
                    rows_path,
                    out=Path(tmp) / "out",
                    name=name,
                    files=files,
                    workspace=workspace,
                    dataset_format=dataset_format,
                    input_format=input_format,
                    named=named,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                    seed=seed,
                    exist_ok=exist_ok,
                )
        except (DatasetPrepareError, UploadError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        except NemoHTTPError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        typer.echo(json.dumps(result, indent=2))

    app.add_typer(dataset, name="dataset")


def _fetch_source(source: str, *, files: FilesClient, workspace: str, into: Path) -> Path:
    """The local file or directory holding *source*'s rows, downloaded into *into* when a fileset."""
    local = Path(source).expanduser()
    if local.exists():
        return local
    if "/" not in source.split("#", 1)[0]:
        raise DatasetPrepareError(f"{source} is not a local path. Pass a fileset as WORKSPACE/NAME[#PATH].")
    # Imported here, as ``nemo files download`` does: fsspec is only needed for a fileset.
    from filesets import parse_fileset_ref
    from filesets.transfer import download

    ref_workspace, fileset, path = parse_fileset_ref(source, workspace_fallback=workspace)
    download(files, remote_path=path, fileset=fileset, workspace=ref_workspace, local_path=str(into))
    return into


def _prepare(
    rows_path: Path,
    *,
    out: Path,
    name: str,
    files: FilesClient,
    workspace: str,
    dataset_format: DatasetFormat,
    input_format: InputFormat | None,
    named: dict[str, str | None],
    val_fraction: float,
    test_fraction: float | None,
    seed: int,
    exist_ok: bool,
) -> dict[str, object]:
    """Convert, split, write and upload the rows; return what to report."""
    rows = read_rows(rows_path, input_format)
    if not rows:
        raise DatasetPrepareError(f"{rows_path} has no rows.")
    mapping = map_columns(dataset_format, rows[0].keys(), named)
    training_rows, dropped = to_training_rows(rows, mapping)
    if not training_rows:
        raise DatasetPrepareError(f"All {dropped} rows have an empty field; none is left to train on.")
    if test_fraction is None:
        test_fraction = 0.0 if dataset_format is DatasetFormat.PREFERENCE else 0.1
    if dataset_format is DatasetFormat.PREFERENCE and val_fraction == 0:
        raise DatasetPrepareError("DPO needs a validation split. Set --val-fraction above 0.")
    splits = split_rows(training_rows, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)

    write_split(splits.train, out / name / SPLIT_FILES["train"])
    if splits.validation:
        write_split(splits.validation, out / name / SPLIT_FILES["validation"])
    report = UploadReport()
    result: dict[str, object] = {
        "dataset": create_dataset_fileset(
            files, name=name, source=out / name, workspace=workspace, exist_ok=exist_ok, report=report
        ),
    }
    if splits.test:
        test_name = f"{name}-test"
        write_split(splits.test, out / test_name / SPLIT_FILES["test"])
        result["test_dataset"] = create_dataset_fileset(
            files, name=test_name, source=out / test_name, workspace=workspace, exist_ok=exist_ok, report=report
        )
    result["format"] = str(dataset_format)
    result["trained_by"] = list(TRAINED_BY[dataset_format])
    result["columns"] = dict(mapping.columns)
    result["rows"] = {"train": len(splits.train), "validation": len(splits.validation), "test": len(splits.test)}
    if dropped:
        result["dropped"] = dropped
        typer.echo(f"Dropped {dropped} rows with an empty field.", err=True)
    if report.reused:
        typer.echo(
            f"Reused {', '.join(report.reused)} as it is: its files were not replaced with these rows.", err=True
        )
    return result
