#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-configure workspace for the LLM judge eval.

This script runs after the NeMo Helix API server is healthy but before the agent starts.
It creates the workspace so the agent can focus on evaluation tasks.

Note: Secrets cannot be pre-configured here because ANTHROPIC_API_KEY is not
available at ENTRYPOINT time (Harbor injects it when docker-exec'ing the agent).
"""

import subprocess
import sys

NHX = "/app/.venv/bin/nhx"


def run_cmd(cmd: list[str], description: str) -> bool:
    """Run a CLI command and report result."""
    print(f"  {description}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        print(f"  OK: {description}")
    else:
        print(f"  FAIL: {description} (exit {result.returncode})")
        if result.stderr.strip():
            print(f"    stderr: {result.stderr.strip()}")
    return result.returncode == 0


def setup() -> None:
    if not run_cmd(
        [NHX, "workspaces", "create", "eval-judge-workspace"],
        "Create workspace eval-judge-workspace",
    ):
        raise RuntimeError("Failed to create workspace eval-judge-workspace")


if __name__ == "__main__":
    try:
        setup()
        print("\n=== Infrastructure Setup Complete ===")
    except Exception as e:
        print(f"Error during setup: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
