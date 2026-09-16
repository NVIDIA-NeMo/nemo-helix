# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Any, cast

import typer
from nemo_helix_ext.cli.core.code_generator import handle_code_generation
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import format_output
from nemo_helix_ext.cli.core.stdin_utils import (
    read_data_input_with_flags,
    read_payload,
    validate_required_fields,
)
from nemo_helix_ext.cli.core.types import EntityOutputFormatOption

app = cast(Any, None)  # override-skip: provided by generated file


@app.command("preview")
@handle_errors
def preview_data_designer(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    config: Annotated[
        str | None,
        typer.Option(
            "--config",
            help="Configuration for NeMo Data Designer.This class defines the main configuration structure for NeMo Data Designer, which orchestrates the generation of synthetic data.Attributes: columns: Required list of column configurations defining how each column should be generated. Must contain at least one column. model_configs: Optional list of model configurations for LLM-based generation. Each model config defines the model, provider, and inference parameters. seed_config: Optional seed dataset settings to use for generation. constraints: Optional list of column constraints. profilers: Optional list of column profilers for analyzing generated data characteristics. (JSON string)",
        ),
    ] = None,
    num_records: Annotated[int | None, typer.Option("--num-records")] = None,
    input_file: Annotated[
        str | None,
        typer.Option("--input-file", help="Path to JSON file (use '-' for stdin)", rich_help_panel="Input Options"),
    ] = None,
    input_data: Annotated[
        str | None,
        typer.Option("--input-data", help="Input data for the request (JSON or YAML)", rich_help_panel="Input Options"),
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Generate preview Data Designer

    [bold red]Required fields:[/] config

    [green]Examples:[/]
    nemo data_designer preview --input-file config.json
    nemo data_designer preview --input-data '{"field": "value"}'
    echo '{"json": "data"}' | nemo data_designer preview --input-file -
    nemo data_designer preview --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if config is not None:
        input_payload["config"] = read_payload("config", config)
    if num_records is not None:
        input_payload["num_records"] = num_records
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["config"],
        "data_designer preview",
        {
            "config": "Configuration for NeMo Data Designer.This class defines the main configuration structure for NeMo Data Designer, which orchestrates the generation of synthetic data.Attributes: columns: Required list of column configurations defining how each column should be generated. Must contain at least one column. model_configs: Optional list of model configurations for LLM-based generation. Each model config defines the model, provider, and inference parameters. seed_config: Optional seed dataset settings to use for generation. constraints: Optional list of column constraints. profilers: Optional list of column profilers for analyzing generated data characteristics. (JSON string)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)
    if handle_code_generation(["data_designer"], "preview", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.data_designer.preview(**all_kwargs)

    # TODO: figure out how to properly output JSON
    for item in result:
        format_output(
            item,
            is_list=False,
            output_format=output_format,
            no_truncate=state.get_no_truncate(),
            timestamp_format=state.get_timestamp_format(),
        )
