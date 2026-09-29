# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership-scoped cleanup of OpenSandbox sandboxes created for one evaluation.

Run with the Harbor 0.20 runner's interpreter, which is the only environment that carries the
OpenSandbox SDK::

    <harbor_dir>/.venv/bin/python -m scaled_evals.harbor_opensandbox_cleanup \\
        --selector nemo-scaled-evals-deployment=<id> --selector nemo-scaled-evals-evaluation=<id>

It kills every sandbox matching all selectors, across every page, then waits until none of them is
still live. It prints one JSON line and exits 0 only when nothing matching is left.

Names shared by the service (no OpenSandbox SDK) and ``NemoOpenSandboxEnvironment`` (Harbor's venv)
live here because this module imports nothing outside the standard library at load time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any

DEPLOYMENT_METADATA_KEY = "nemo-scaled-evals-deployment"
EVALUATION_METADATA_KEY = "nemo-scaled-evals-evaluation"
BENCHMARK_RUN_METADATA_KEY = "nemo-scaled-evals-benchmark-run"
REQUIRED_SELECTOR_KEYS = (DEPLOYMENT_METADATA_KEY, EVALUATION_METADATA_KEY)
APPLIED_EGRESS_FILENAME = "nemo-applied-egress.json"
_TERMINAL_STATES = frozenset({"Terminated", "Failed"})
_ENV_ALIASES = {"OPEN_SANDBOX_DOMAIN": "OPENSANDBOX_DOMAIN", "OPEN_SANDBOX_API_KEY": "OPENSANDBOX_API_KEY"}


def ownership_selector(*, deployment_id: str, evaluation_id: str) -> dict[str, str]:
    return {DEPLOYMENT_METADATA_KEY: deployment_id, EVALUATION_METADATA_KEY: evaluation_id}


def validate_selector(selector: Mapping[str, str]) -> None:
    """Refuse any selector that could match sandboxes outside one evaluation of one deployment."""
    missing = [key for key in REQUIRED_SELECTOR_KEYS if not str(selector.get(key) or "").strip()]
    if missing:
        raise ValueError(f"cleanup selector is incomplete; missing {', '.join(missing)}")


def connection_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy of ``source`` with ``OPEN_SANDBOX_*`` aliased to the SDK names; raise if either is unset."""
    env = dict(os.environ if source is None else source)
    for platform_name, sdk_name in _ENV_ALIASES.items():
        if not env.get(sdk_name) and env.get(platform_name):
            env[sdk_name] = env[platform_name]
    missing = [name for name in _ENV_ALIASES.values() if not env.get(name, "").strip()]
    if missing:
        raise ValueError(f"missing OpenSandbox setting(s): {', '.join(missing)}")
    return env


async def _manager(env: Mapping[str, str], protocol: str) -> Any:
    from opensandbox.config import ConnectionConfig  # ty: ignore[unresolved-import]  # Harbor 0.20 venv only
    from opensandbox.manager import SandboxManager  # ty: ignore[unresolved-import]

    return await SandboxManager.create(
        connection_config=ConnectionConfig(
            domain=env["OPENSANDBOX_DOMAIN"],
            api_key=env["OPENSANDBOX_API_KEY"],
            protocol=protocol,
            request_timeout=timedelta(seconds=60),
        )
    )


async def _live(manager: Any, selector: Mapping[str, str]) -> list[str]:
    from opensandbox.models.sandboxes import SandboxFilter  # ty: ignore[unresolved-import]

    live: list[str] = []
    page = 1
    while True:
        result = await manager.list_sandbox_infos(SandboxFilter(metadata=dict(selector), page=page))
        for info in result.sandbox_infos:
            state = str(getattr(getattr(info, "status", None), "state", "") or "")
            if state not in _TERMINAL_STATES:
                live.append(info.id)
        if not result.pagination.has_next_page:
            return live
        page += 1


async def destroy_owned_sandboxes(
    env: Mapping[str, str],
    protocol: str,
    selector: Mapping[str, str],
    *,
    timeout_s: float = 120.0,
    poll_s: float = 5.0,
) -> dict[str, list[str]]:
    """Kill every live sandbox matching ``selector`` and wait for them to go away."""
    validate_selector(selector)
    manager = await _manager(env, protocol)
    killed: list[str] = []
    failed: list[str] = []
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            live = await _live(manager, selector)
            if not live:
                return {"killed": killed, "failed": failed, "remaining": []}
            for sandbox_id in live:
                if sandbox_id in killed:
                    continue
                try:
                    await manager.kill_sandbox(sandbox_id)
                    killed.append(sandbox_id)
                except Exception as exc:  # noqa: BLE001 - reported per sandbox, retried next pass
                    failed.append(f"{sandbox_id}: {type(exc).__name__}: {exc}")
            if time.monotonic() >= deadline:
                return {"killed": killed, "failed": failed, "remaining": live}
            await asyncio.sleep(poll_s)
    finally:
        await manager.close()


def _parse_selector(values: Sequence[str]) -> dict[str, str]:
    selector: dict[str, str] = {}
    for value in values:
        key, sep, item = value.partition("=")
        if not sep or not key:
            raise ValueError(f"selectors must be key=value, got {value!r}")
        selector[key] = item
    return selector


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", action="append", default=[], help="metadata key=value (repeatable)")
    parser.add_argument("--protocol", choices=("http", "https"), required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        selector = _parse_selector(args.selector)
        validate_selector(selector)
        report = asyncio.run(destroy_owned_sandboxes(connection_env(), args.protocol, selector, timeout_s=args.timeout))
    except Exception as exc:  # noqa: BLE001 - the caller records one structured failure
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 2
    print(json.dumps(report), flush=True)
    return 0 if not report["remaining"] else 1


if __name__ == "__main__":
    sys.exit(main())
