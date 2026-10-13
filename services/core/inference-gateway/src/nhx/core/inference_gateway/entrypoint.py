# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

logger = logging.getLogger(__name__)


def main(stop_signal=None):
    parser = argparse.ArgumentParser(description="Run the Inference Gateway service.")
    parser.add_argument(
        "command",
        choices=["migrate"],
        help="The command to run: `migrate` performs a database migration.",
    )

    args = parser.parse_args()

    if args.command == "migrate":
        logger.info("Running migrations is not supported by the Inference Gateway service.")


if __name__ == "__main__":
    main()
