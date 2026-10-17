# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a running NeMo Helix instance without adding a production CLI command.

Uses the current ``~/.config/nhx`` context. ``NHX_BASE_URL`` and
``NHX_CURRENT_CONTEXT`` override that context the same way the CLI does.
Pass ``--context`` to select a different saved context. When authentication
is enabled, the logged-in user is used.

Examples:
  uv run python tools/instance_check/check.py
  uv run python tools/instance_check/check.py --context prod
  uv run python tools/instance_check/check.py --workspace my-workspace --keep
  uv run python tools/instance_check/check.py --model system/served-model-name
  uv run python tools/instance_check/check.py --model-image docker.io/library/python:3.12-alpine
  uv run python tools/instance_check/check.py --require-token-exchange
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from nemo_helix_plugin.client.config.config import ConfigParams, get_context

_SUITE = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the check script flags."""
    parser = argparse.ArgumentParser(
        description="Check a running NeMo Helix instance: jobs, secrets, and deployments.",
    )
    parser.add_argument("--context", help="Saved config context. Defaults to the current context.")
    parser.add_argument(
        "--workspace",
        help="Workspace to use. When omitted, a temporary workspace is created and deleted.",
    )
    parser.add_argument("--keep", action="store_true", help="Leave the temporary workspace in place.")
    parser.add_argument(
        "--image",
        help="Task image for deployment probes. Defaults to the profile task image, or nhx-tasks from the launcher image.",
    )
    parser.add_argument("--job-profile", help="Jobs execution profile.")
    parser.add_argument("--executor", help="Deployments executor name.")
    parser.add_argument("--models", action="store_true", help="Call an existing inference-gateway model.")
    parser.add_argument(
        "--model",
        help="Model to call, as name or workspace/name. Implies --models.",
    )
    parser.add_argument(
        "--deploy-model",
        action="store_true",
        help="Accepted for compatibility. A CPU model deployment runs on every check.",
    )
    parser.add_argument(
        "--model-image",
        help="Override the model-deploy image. The default is docker.io/library/python:3.12-alpine.",
    )
    parser.add_argument(
        "--require-token-exchange",
        action="store_true",
        help="Fail when token exchange is enabled but a workload only received principal injection.",
    )
    parser.add_argument("--timeout", type=float, default=600, help="Seconds to wait for each workload.")
    return parser.parse_args(argv)


def run_check(args: argparse.Namespace) -> int:
    """Resolve the platform context and run the acceptance pytest file."""
    if importlib.util.find_spec("pytest") is None:
        print(
            "This check needs pytest. Run it from the NeMo Helix dev environment: "
            "uv run python tools/instance_check/check.py",
            file=sys.stderr,
        )
        return 1
    overrides: ConfigParams | None = {"current_context": args.context} if args.context else None
    try:
        sdk_context = get_context(overrides=overrides)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    base_url = str(sdk_context.cluster.base_url).rstrip("/")
    print(f"Checking {base_url} (context {sdk_context.context_name})", flush=True)
    env = os.environ.copy()
    env["NHX_BASE_URL"] = base_url
    env["NHX_CURRENT_CONTEXT"] = sdk_context.context_name
    env["NHX_CHECK_TIMEOUT"] = str(args.timeout)
    _set_flag(env, "NHX_CHECK_WORKSPACE", args.workspace)
    _set_flag(env, "NHX_CHECK_IMAGE", args.image)
    _set_flag(env, "NHX_CHECK_JOB_PROFILE", args.job_profile)
    _set_flag(env, "NHX_CHECK_EXECUTOR", args.executor)
    _set_flag(env, "NHX_CHECK_MODEL", args.model)
    _set_flag(env, "NHX_CHECK_MODEL_IMAGE", args.model_image)
    if args.keep:
        env["NHX_CHECK_KEEP"] = "1"
    if args.models or args.model:
        env["NHX_CHECK_MODELS"] = "1"
    if args.require_token_exchange:
        env["NHX_CHECK_REQUIRE_TOKEN_EXCHANGE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(_SUITE / "instance_check.py"),
            "-s",
            "-v",
            "--tb=short",
            "-c",
            str(_SUITE / "pytest.ini"),
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        env=env,
        check=False,
    )
    return completed.returncode


def _set_flag(env: dict[str, str], name: str, value: str | None) -> None:
    if value:
        env[name] = value


def main() -> None:
    """CLI entry point."""
    sys.exit(run_check(parse_args()))


if __name__ == "__main__":
    main()
