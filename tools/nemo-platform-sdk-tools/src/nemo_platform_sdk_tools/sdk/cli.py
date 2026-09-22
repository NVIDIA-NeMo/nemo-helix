# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import typer
from nemo_platform_sdk_tools.common.logging import setup_logging
from nemo_platform_sdk_tools.sdk.is_up_to_date import is_up_to_date
from nemo_platform_sdk_tools.sdk.openapi_stainless_mapper import app as openapi_stainless_mapper_app
from nemo_platform_sdk_tools.sdk.post_generation_update import app as post_generation_update_app
from nemo_platform_sdk_tools.sdk.vendor.vendor_package import app as vendor_app

app = typer.Typer(help="Tools for managing NeMo Platform SDKs", no_args_is_help=True, rich_markup_mode="markdown")


@app.callback()
def callback(verbose: Annotated[bool, typer.Option(help="Verbose mode")] = False) -> None:
    """NeMo Platform SDK management commands."""
    setup_logging(verbose=verbose, show_path=True, enable_link_path=True, stderr=True)


app.command()(is_up_to_date)
app.add_typer(openapi_stainless_mapper_app)
app.add_typer(post_generation_update_app)
app.add_typer(vendor_app)
