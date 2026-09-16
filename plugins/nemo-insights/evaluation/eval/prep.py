# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency prep for monorepo CI jobs: uv syncs + tau2 judge repoint.

The tau2 checkout is expected beside the nemo-helix checkout.
Flags: --sync-insights {true,false}, --install-tau2 {true,false}.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

JUDGE_VARS = ("DEFAULT_LLM_NL_ASSERTIONS", "DEFAULT_LLM_ENV_INTERFACE", "DEFAULT_LLM_EVAL_USER_SIMULATOR")
PLUGIN_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ROOT = PLUGIN_ROOT.parents[1]
TAU2_ROOT = PLATFORM_ROOT.parent / "tau2-bench"


def repoint_judge(config_text: str, judge_model: str) -> tuple[str, int]:
    total = 0
    for var in JUDGE_VARS:
        config_text, n = re.subn(
            rf"^{var} = .*$",
            lambda _match, var=var: f'{var} = "{judge_model}"',
            config_text,
            flags=re.M,
        )
        total += n
    return config_text, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-insights", choices=["true", "false"], default="true")
    parser.add_argument("--install-tau2", choices=["true", "false"], default="true")
    args = parser.parse_args()

    if args.sync_insights == "true":
        subprocess.run(["uv", "sync", "--group", "insights"], check=True, cwd=PLATFORM_ROOT)

    config = TAU2_ROOT / "src/tau2/config.py"
    text, count = repoint_judge(config.read_text(encoding="utf-8"), os.environ["TAU2_JUDGE_LLM"])
    if count != 3:
        sys.exit(f"judge repoint failed: {count}/3 defaults matched")
    config.write_text(text, encoding="utf-8")

    if args.install_tau2 == "true":
        subprocess.run(["uv", "sync"], check=True, cwd=TAU2_ROOT)
        subprocess.run(["uv", "run", "tau2", "check-data"], check=True, cwd=TAU2_ROOT)


if __name__ == "__main__":
    main()
