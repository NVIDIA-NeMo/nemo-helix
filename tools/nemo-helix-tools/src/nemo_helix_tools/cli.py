# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import typer
from nemo_helix_tools.common.logging import setup_logging
from nemo_helix_tools.license.cli import app as license_app
from nemo_helix_tools.vendor.vendor_package import app as vendor_app

app = typer.Typer(
    help="License and package-layout maintenance tooling for NeMo Helix",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    rich_markup_mode="markdown",
)


@app.callback()
def callback(verbose: Annotated[bool, typer.Option(help="Verbose mode")] = False) -> None:
    setup_logging(verbose=verbose, show_path=True, enable_link_path=True, stderr=True)


app.add_typer(vendor_app)
app.add_typer(license_app, name="license")
