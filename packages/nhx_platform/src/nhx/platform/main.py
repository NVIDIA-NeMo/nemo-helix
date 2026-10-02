#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy task-only compatibility entrypoint for ``nemo-helix``.

Usage:
  nemo-helix run task --task nhx.hello_world.tasks.hello_world
  nemo-helix run task --task nhx.hello_world.tasks.hello_world --config '{"key": "value"}'
  nemo-helix run task --task nhx.hello_world.tasks.hello_world --env FOO=bar --env BAZ=qux
  nemo-helix run task --task nhx.platform_seed
"""

from __future__ import annotations

import argparse
import logging
import os
import runpy
import sys

from nhx.common.config import get_common_service_config
from nhx.common.jobs.constants import TASK_CONFIG_ENVVAR
from nhx.common.observability import initialize_obs, setup_global_instrumentations
from nhx.common.observability.otel import settings as otel_settings
from nhx.platform_runner.health import get_platform_resource_attributes
from nhx.platform_runner.loader import load_service as _load_service
from rich_argparse import RawTextRichHelpFormatter

logger = logging.getLogger(__name__)
load_service = _load_service


def run_task(module_name: str, env_vars: list[str], config: str | None) -> int:
    """Execute a Python module as a task using ``runpy``."""
    for env_var in env_vars:
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            os.environ[key] = value

    if config:
        os.environ[TASK_CONFIG_ENVVAR] = config
        os.environ["NEMO_JOB_STEP_CONFIG"] = config

    sys.argv = [module_name]
    logger.info("Running task module: %s", module_name)
    runpy.run_module(module_name, run_name="__main__")
    return 0


def main() -> None:
    """Main entrypoint function."""
    parser = argparse.ArgumentParser(
        description="Nemo Helix",
        formatter_class=RawTextRichHelpFormatter,
        epilog="""
Examples:
  nemo-helix run task --task nhx.hello_world.tasks.hello_world
  nemo-helix run task --task nhx.hello_world.tasks.hello_world --config '{"key": "value"}'
  nemo-helix run task --task nhx.hello_world.tasks.hello_world --env FOO=bar --env BAZ=qux
  nemo-helix run task --task nhx.platform_seed
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    platform_parser = subparsers.add_parser(
        "run",
        help="Legacy task compatibility entrypoint",
        formatter_class=RawTextRichHelpFormatter,
    )
    run_subparsers = platform_parser.add_subparsers(dest="run_command")
    task_parser = run_subparsers.add_parser(
        "task", help="Run a Python module as a task", formatter_class=RawTextRichHelpFormatter
    )

    task_parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Python module to run (e.g. nhx.hello_world.tasks.hello_world, nhx.platform_seed)",
    )
    task_parser.add_argument(
        "--env",
        type=str,
        action="append",
        default=[],
        help="Environment variable in KEY=VALUE format (repeatable)",
    )
    task_parser.add_argument(
        "--config",
        type=str,
        help=f"Configuration JSON passed via {TASK_CONFIG_ENVVAR} env var",
    )

    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        sys.exit(1)

    if args.run_command != "task":
        logger.error("Service startup moved to `nemo services run`; `nemo-helix` only supports `run task`.")
        sys.exit(1)

    service_config = get_common_service_config()
    if otel_settings.log_format != service_config.log_format:
        otel_settings.log_format = service_config.log_format
    if otel_settings.log_level != service_config.log_level:
        otel_settings.log_level = service_config.log_level
    initialize_obs(resource_attributes=get_platform_resource_attributes())
    setup_global_instrumentations()
    sys.exit(run_task(args.task, args.env, args.config))


if __name__ == "__main__":
    main()
