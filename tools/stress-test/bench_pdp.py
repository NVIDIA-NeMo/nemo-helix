# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time the embedded authorization engine (OPA WASM) against the size of its policy data.

Every authenticated Helix request, including each internal entity call, pays one ``allow``
evaluation. The auth service runs it synchronously on the event loop, so its cost is paid by every
request in the process. This loads the real static policy data plus synthetic role bindings, then
times ``set_data`` (paid by each thread after a data refresh) and ``evaluate("allow")`` for a few
representative inputs. Nothing here talks to a network or a database.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import statistics
import time
from typing import Any

from nhx.core.auth.app.bundle import build_authorization_data
from nhx.core.auth.app.embedded_pdp.engine import ENTRYPOINT_MAP, OPAPolicy
from nhx.core.auth.app.embedded_pdp.policy_wasm import ensure_embedded_policy_wasm
from nhx.core.auth.app.scope_families import with_scope_families

USER = "bench-user@example.com"


def _with_bindings(static: dict[str, Any], bindings: int, role: str, per_principal: int) -> dict[str, Any]:
    data = copy.deepcopy(static)
    principals = data["authz"].setdefault("principals", {})
    workspaces = data["authz"].setdefault("workspaces", {})
    for index in range(bindings):
        principal = USER if index < per_principal else f"user-{index // per_principal}@example.com"
        workspace = f"ws-{index}"
        workspaces.setdefault(workspace, {})
        principals.setdefault(principal, {"workspaces": {}})["workspaces"][workspace] = [role]
    return with_scope_families(data)


def _inputs() -> dict[str, dict[str, Any]]:
    return {
        "user GET workspace": {
            "principal_id": "bench-user-id",
            "principal_email": USER,
            "method": "GET",
            "path": "/apis/entities/v2/workspaces/ws-0",
            "caller_kind": "principal",
        },
        "user GET jobs list": {
            "principal_id": "bench-user-id",
            "principal_email": USER,
            "method": "GET",
            "path": "/apis/jobs/v2/workspaces/ws-0/jobs",
            "caller_kind": "principal",
        },
        "user denied (no binding)": {
            "principal_id": "nobody-id",
            "principal_email": "nobody@example.com",
            "method": "GET",
            "path": "/apis/entities/v2/workspaces/ws-0",
            "caller_kind": "principal",
        },
        "service:jobs PATCH step status": {
            "principal_id": "service:jobs",
            "method": "PATCH",
            "path": "/apis/jobs/v2/workspaces/ws-0/jobs/job-0/attempts/attempt-0/steps/step-0/status",
            "caller_kind": "service",
        },
        "service:jobs GET entity": {
            "principal_id": "service:jobs",
            "method": "GET",
            "path": "/apis/entities/v2/workspaces/ws-0/entities/jobs/job-0",
            "caller_kind": "service",
        },
    }


def _allowed(result: Any) -> object:
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict):
        result = result.get("result", result)
    return result.get("allowed") if isinstance(result, dict) else None


def _time(fn, repeat: int) -> tuple[list[float], Any]:
    seconds = []
    value = None
    for _ in range(repeat):
        began = time.perf_counter()
        value = fn()
        seconds.append(time.perf_counter() - began)
    return seconds, value


def _ms(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    return {
        "p50_ms": round(statistics.median(ordered) * 1000, 2),
        "p90_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))] * 1000, 2),
        "max_ms": round(ordered[-1] * 1000, 2),
    }


def run(args: argparse.Namespace) -> int:
    wasm = ensure_embedded_policy_wasm(auto_build=False)
    static = asyncio.run(build_authorization_data(None))
    authz = static["authz"]
    roles = sorted(authz.get("roles", {}), key=lambda name: -len(authz["roles"][name].get("permissions", [])))
    role = args.role or roles[0]
    print(
        json.dumps(
            {
                "wasm": str(wasm),
                "endpoints": len(authz.get("endpoints", {})),
                "roles": len(authz.get("roles", {})),
                "static_principals": len(authz.get("principals", {})),
                "role_used": role,
                "static_json_bytes": len(json.dumps(static)),
            }
        )
    )
    policy = OPAPolicy(str(wasm))
    for bindings in args.bindings:
        data = _with_bindings(static, bindings, role, args.per_principal)
        set_seconds, _ = _time(lambda: policy.set_data(data), args.set_repeat)
        row: dict[str, Any] = {
            "bindings": bindings,
            "data_json_bytes": len(json.dumps(data)),
            "set_data": _ms(set_seconds),
        }
        for label, payload in _inputs().items():
            entrypoint = ENTRYPOINT_MAP["allow"]
            seconds, result = _time(lambda payload=payload: policy.evaluate(payload, entrypoint), args.repeat)
            row[label] = {**_ms(seconds), "allowed": _allowed(result)}
        print(json.dumps(row))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bindings", type=int, nargs="+", default=[10, 100, 1000, 10000])
    parser.add_argument("--per-principal", type=int, default=10, help="Workspaces bound to each synthetic principal")
    parser.add_argument("--role", help="Role to bind; defaults to the role with the most permissions")
    parser.add_argument("--repeat", type=int, default=50, help="Evaluations per input")
    parser.add_argument("--set-repeat", type=int, default=3, help="set_data calls per data size")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
