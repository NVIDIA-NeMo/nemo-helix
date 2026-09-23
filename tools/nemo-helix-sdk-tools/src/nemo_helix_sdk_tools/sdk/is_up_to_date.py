# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import difflib
import logging
from pathlib import Path
from typing import Annotated

import click
import typer
import yaml
from nemo_helix_sdk_tools.sdk.core.common import get_sdk_info

logger = logging.getLogger(__name__)


def is_up_to_date(
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Output dir to save the diffs.", click_type=click.Path(path_type=Path)),
    ] = None,
) -> None:
    """
    Checks that the SDK is up to date with the OpenAPI spec and Stainless config.

    Stainless takes 2 input files and generates the SDK code based on them.
    At the generation time, we save the files that were used to generate that version of the SDK (in the .nhxcontext dir).

    If the OpenAPI spec or Stainless config from the NeMo Helix is different than the saved files, it means the SDK needs to be regenerated.
    """
    sdk_info = get_sdk_info()

    nhxcontext_dir = sdk_info.sdk_dir / ".nhxcontext"
    if not nhxcontext_dir.exists():
        logger.warning(f"SDK directory {nhxcontext_dir} does not exist, skipping the checks.")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    sdk_openapi = nhxcontext_dir / "openapi.yaml"
    sdk_stainless = nhxcontext_dir / "stainless.yaml"

    _check_no_reviewme_names_in_stainless(sdk_info.stainless_config_file)

    openapi_diff = _diff(sdk_openapi, sdk_info.openapi_spec_file)
    stainless_diff = _diff(sdk_stainless, sdk_info.stainless_config_file)

    if openapi_diff:
        logger.error(f"OpenAPI spec is out of sync with the SDK: \n{openapi_diff}")
        if output_dir:
            (output_dir / "openapi_diff.txt").write_text(openapi_diff)

    if stainless_diff:
        logger.error(f"Stainless config is out of sync with the SDK: \n{stainless_diff}")
        if output_dir:
            (output_dir / "stainless_diff.txt").write_text(stainless_diff)

    if openapi_diff or stainless_diff:
        raise RuntimeError(
            "OpenAPI spec or Stainless config is out of sync with the SDK. "
            "First, make sure you are set up with the Stainless (see ./sdk/README.md) and then run `sdk/stainless.sh sync` to update the SDK."
        )

    logger.info("OpenAPI spec and Stainless config are up to date with the SDK!")


def _diff(nhx_file: Path, sdk_file: Path) -> str:
    """
    Generate a unified diff between:
     - the file that was used to generate the current SDK and
     - the file that is the current source of truth in the NeMo Helix.
    """
    nhx_content = _normalize_yaml(nhx_file.read_text())
    sdk_content = _normalize_yaml(sdk_file.read_text())

    diff = difflib.unified_diff(
        nhx_content.splitlines(keepends=True),
        sdk_content.splitlines(keepends=True),
        fromfile=str(nhx_file),
        tofile=str(sdk_file),
    )

    return "".join(diff)


def _normalize_yaml(spec: str) -> str:
    yaml_spec = yaml.safe_load(spec)
    return yaml.dump(yaml_spec, sort_keys=True, default_flow_style=False)


def _check_no_reviewme_names_in_stainless(stainless_file: Path) -> None:
    """
    Check that there are no names containing 'reviewme' in the Stainless config.
    """
    stainless_content = stainless_file.read_text()
    if "reviewme_" in stainless_content.lower():
        raise RuntimeError(f"'reviewme_' found in Stainless config {stainless_file}. Please update the config.")
