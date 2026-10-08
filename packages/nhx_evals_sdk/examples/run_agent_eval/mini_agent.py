#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tiny, dependency-free stand-in agent CLI used as the default "workflow".

Stands in for whatever real agent a workflow backend would launch (``nat run``,
``codex exec``, ...). The contract ``WorkflowAgentRuntime`` relies on: read the
task-input JSON (``--input-json``), do the work inside ``--workspace`` (here:
write the requested file), print a final answer to stdout, exit non-zero on
failure. Any executable honoring this contract drops in unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Toy file-writing agent.")
    parser.add_argument("--instruction", type=Path, required=True, help="Path to the human-readable instruction.")
    parser.add_argument("--workspace", type=Path, required=True, help="Directory the agent may write into.")
    parser.add_argument("--input-json", type=Path, required=True, help="Path to the structured task inputs.")
    args = parser.parse_args()

    spec = json.loads(args.input_json.read_text(encoding="utf-8")) if args.input_json.exists() else {}
    create_file = spec.get("create_file")
    content = spec.get("content", "")

    if not isinstance(create_file, str) or not create_file:
        print("No 'create_file' directive found in task inputs; nothing to do.", file=sys.stderr)
        return 2

    args.workspace.mkdir(parents=True, exist_ok=True)
    target = args.workspace / create_file
    target.write_text(content, encoding="utf-8")

    print(f"Created {create_file} ({len(content.encode('utf-8'))} bytes) in the workspace. Task complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
