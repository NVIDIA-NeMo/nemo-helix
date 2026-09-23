# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common logging configuration for NeMo Helix SDK tools."""

import logging
import os

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    *, verbose: bool = False, enable_link_path: bool = False, show_path: bool = False, stderr: bool = False
) -> None:
    """
    Set up logging configuration for SDK tools modules.

    Args:
        verbose: Enable debug level logging
        enable_link_path: Enable clickable file paths in log output
        stderr: If True, output logs to stderr instead of stdout (useful for CLI commands
                that need clean stdout for scripting)
    """
    is_ci = os.getenv("CI") is not None or os.getenv("GITLAB_CI") is not None
    log_level = logging.DEBUG if verbose else logging.INFO
    FORMAT = "%(message)s"

    if is_ci:
        logging.basicConfig(
            level=log_level,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        console = Console(stderr=stderr)
        logging.basicConfig(
            level=log_level,
            format=FORMAT,
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    console=console,
                    rich_tracebacks=True,
                    enable_link_path=enable_link_path,
                    show_path=show_path,
                    markup=False,
                )
            ],
        )
