# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the shared file_io container task."""

import argparse
import sys

from nhx.customization_common.tasks.file_io.run import run


def main() -> int:
    parser = argparse.ArgumentParser(description="NeMo customization file I/O task")
    parser.add_argument(
        "--service-source",
        required=True,
        help="Value stamped on upload-created filesets (custom_fields.service_source)",
    )
    parser.add_argument(
        "--service-name",
        required=True,
        help="Platform service identity for SDK auth/telemetry (e.g. customizer, unsloth, rl)",
    )
    args = parser.parse_args()
    return run(service_source=args.service_source, service_name=args.service_name)


if __name__ == "__main__":
    sys.exit(main())
