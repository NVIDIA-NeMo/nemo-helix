# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default entrypoint for the nhx-customizer-tasks image (help / task listing).

Production job steps invoke a specific module directly, e.g.
``python -m nhx.customization_common.tasks.file_io --service-source automodel --service-name customizer``.
"""

from __future__ import annotations

import argparse
import sys

_TASK_MODULES = (
    (
        "file_io",
        "nhx.customization_common.tasks.file_io",
        "Download/upload model and dataset files (pass --service-source / --service-name)",
    ),
    (
        "model_entity",
        "nhx.customization_common.tasks.model_entity",
        "Create output model entity (pass --service-name)",
    ),
    (
        "model_spec",
        "nhx.core.models.tasks.model_spec",
        "Populate model entity spec from checkpoint metadata",
    ),
    (
        "lora_sidecar",
        "nhx.core.models.sidecars.adapters.main",
        "LoRA adapter sidecar for NIM/vLLM deployments",
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nhx.automodel.tasks",
        description="NeMo customization CPU task image. The jobs compiler runs one module per step.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python -m nhx.automodel.tasks --help\n"
        "  python -m nhx.customization_common.tasks.file_io --service-source automodel --service-name customization\n"
        "  python -m nhx.customization_common.tasks.model_entity --service-name customization\n\n"
        "GPU training uses backend-specific training images:\n"
        "  python -m nhx.automodel.tasks.training\n"
        "  python -m nhx.automodel.tasks.retrieval_mine\n",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List task modules and exit (default when no job config is provided).",
    )
    args = parser.parse_args(argv)
    if args.list or len(argv or sys.argv[1:]) == 0:
        print("Task modules:\n")
        for name, module, summary in _TASK_MODULES:
            print(f"  {name:14}  {module}")
            print(f"                  {summary}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
