# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the shared model_entity container task."""

import argparse
import sys

from nhx.customization_common.tasks.model_entity.run import run


def main() -> int:
    parser = argparse.ArgumentParser(description="NeMo customization model entity task")
    parser.add_argument(
        "--service-name",
        required=True,
        help="Platform service identity for SDK auth/telemetry (e.g. customizer, unsloth, rl)",
    )
    args = parser.parse_args()
    return run(service_name=args.service_name)


if __name__ == "__main__":
    sys.exit(main())
