# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse


def get_args(args=None):
    """Get the command line arguments for the Guardrails Server."""

    parser = argparse.ArgumentParser(description="Configuration for the Guardrails Server")
    parser.add_argument(
        "--config-store",
        type=str,
        default="/config-store",
        help="Path to the configuration directory",
    )
    parser.add_argument(
        "--default-config-id",
        type=str,
        default="default",
        help="The default configuration id to use",
    )

    parser.add_argument("--fastapi-port", type=str, default="7373", help="The default fastapi port")

    parser.add_argument("--fastapi-host", type=str, default="0.0.0.0")

    parser.add_argument("--fastapi-reload", type=str, default=False)

    return parser.parse_args(args)
