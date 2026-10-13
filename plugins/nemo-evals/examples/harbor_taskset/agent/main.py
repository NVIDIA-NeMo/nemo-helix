# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container entry point: solve one task and write the answer + a trace."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from agent import FALLBACK, HelloAgent
from tracing import write_trace

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/app/artifacts"))
OUTPUT_PATH = ARTIFACTS_DIR / "output.txt"


def main() -> None:
    """Run the agent on --prompt and write /app/artifacts/output.txt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--session-id", default=os.environ.get("HARBOR_SESSION_ID", "local"))
    args = parser.parse_args()

    answer = HelloAgent().solve(args.prompt)
    handler = "fallback" if answer == FALLBACK else "handle_greeting"

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(answer + "\n", encoding="utf-8")

    trace_path = write_trace(args.session_id, args.prompt, answer, handler)
    print(f"answer={answer!r} handler={handler} output={OUTPUT_PATH} trace={trace_path}")


if __name__ == "__main__":
    main()
