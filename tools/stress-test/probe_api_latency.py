# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only latency ladder for an authenticated Helix deployment.

Each rung adds one layer to the request path, so the differences between rungs split the time of an
authenticated request into network, authentication and authorization, and entity-store work:

- ``/status`` and ``/apis/auth/discovery`` are unauthenticated: network plus ingress.
- ``workspaces.get`` is one authenticated request that reads one entity.
- ``hello`` is ``workspaces.get`` plus one internal loopback hop (the handler calls ``workspaces.get``).
  A principal without the hello-world permission gets a 403 from the middleware, which is then the
  cost of authentication plus one authorization round trip with no handler work.
- A malformed bearer token is rejected before authorization; no token at all skips token
  validation and goes straight to an anonymous authorization round trip.
- ``/apis/auth/authenticate`` runs the same bearer-token resolution as the middleware (workload
  resolvers, then OIDC validation) and nothing else: no authorization, no account resolution.
- ``jobs list`` and ``job status`` are the jobs API, which reads several entities per request.

Privilege checks record the status code and the reported total for listings we may or may not be
allowed to see cluster-wide. Every request is a GET.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from nemo_helix_plugin.client.client import NemoClient

SERVER_TIMING_HEADERS = ("x-envoy-upstream-service-time", "server-timing")


@dataclass(frozen=True)
class Rung:
    label: str
    path: str
    authenticated: bool = True
    headers: tuple[tuple[str, str], ...] = ()


def _headers(client: NemoClient, rung: Rung) -> dict[str, str]:
    if rung.headers:
        return dict(rung.headers)
    if not rung.authenticated:
        return {}
    return dict(client.request_headers() or {})


def _get(http: httpx.Client, client: NemoClient, rung: Rung) -> dict[str, object]:
    headers = _headers(client, rung)
    began = time.perf_counter()
    try:
        response = http.get(rung.path, headers=headers)
    except httpx.HTTPError as exc:
        return {"rung": rung.label, "error": f"{type(exc).__name__}: {exc}", "s": time.perf_counter() - began}
    elapsed = time.perf_counter() - began
    record: dict[str, object] = {"rung": rung.label, "status": response.status_code, "s": elapsed}
    for name in SERVER_TIMING_HEADERS:
        if name in response.headers:
            record[name] = response.headers[name]
    return record


def _total(response: httpx.Response) -> object:
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict):
        pagination = body.get("pagination")
        if isinstance(pagination, dict):
            return pagination.get("total_results")
        if isinstance(body.get("data"), list):
            return len(body["data"])
    return None


def _names(http: httpx.Client, client: NemoClient, path: str) -> list[str]:
    response = http.get(path, headers=_headers(client, Rung("names", path)))
    if response.status_code != 200:
        return []
    return [item["name"] for item in response.json().get("data", []) if isinstance(item, dict) and "name" in item]


def _choose_targets(
    http: httpx.Client, client: NemoClient, workspace: str | None
) -> tuple[str, str | None, str | None]:
    """Pick a workspace to read, a workspace with no jobs, and one existing job name."""
    workspaces = _names(http, client, "/apis/entities/v2/workspaces?page_size=100")
    if workspace:
        workspaces = [workspace, *[name for name in workspaces if name != workspace]]
    if not workspaces:
        raise SystemExit("no visible workspaces")
    empty: str | None = None
    job: tuple[str, str] | None = None
    for name in workspaces:
        jobs = _names(http, client, f"/apis/jobs/v2/workspaces/{name}/jobs?page_size=5")
        if jobs and job is None:
            job = (name, jobs[0])
        if not jobs and empty is None:
            empty = name
    target = job[0] if job else workspaces[0]
    return target, empty, (job[1] if job else None)


def _ladder(target: str, empty: str | None, job: str | None) -> list[Rung]:
    rungs = [
        Rung("unauth /status", "/status", authenticated=False),
        Rung("unauth /apis/auth/discovery", "/apis/auth/discovery", authenticated=False),
        Rung(
            "malformed token (401 before authorization)",
            f"/apis/entities/v2/workspaces/{target}",
            headers=(("Authorization", "Bearer not a token"),),
        ),
        Rung("no token (anonymous authorization only)", f"/apis/entities/v2/workspaces/{target}", authenticated=False),
        Rung("authenticate (token validation only)", "/apis/auth/authenticate"),
        Rung("workspaces.get", f"/apis/entities/v2/workspaces/{target}"),
        Rung("hello-world hello", f"/apis/hello-world/v2/workspaces/{target}/hello"),
        Rung("jobs list", f"/apis/jobs/v2/workspaces/{target}/jobs?page_size=10"),
    ]
    if empty:
        rungs.append(Rung("jobs list (empty workspace)", f"/apis/jobs/v2/workspaces/{empty}/jobs?page_size=10"))
    if job:
        rungs.append(Rung("job status", f"/apis/jobs/v2/workspaces/{target}/jobs/{job}/status"))
    return rungs


def _privileges(target: str) -> list[Rung]:
    return [
        Rung("workspaces list", "/apis/entities/v2/workspaces?page_size=1"),
        Rung("role bindings list", "/apis/auth/v2/iam/role-bindings?page_size=1"),
        Rung("workspace members", f"/apis/entities/v2/workspaces/{target}/members?page_size=1"),
        Rung("jobs list, all workspaces", "/apis/jobs/v2/workspaces/-/jobs?page_size=1"),
        Rung("steps list, all workspaces", "/apis/jobs/v2/workspaces/-/jobs/-/steps?page_size=1"),
        Rung("web session", "/apis/auth/v2/session"),
    ]


def _summarize(records: list[dict[str, Any]]) -> dict[str, object]:
    seconds = sorted(float(r["s"]) for r in records if "error" not in r)
    statuses = sorted({str(r.get("status", "error")) for r in records})
    upstream = sorted(
        float(str(r["x-envoy-upstream-service-time"])) / 1000 for r in records if "x-envoy-upstream-service-time" in r
    )
    summary: dict[str, object] = {"n": len(records), "statuses": statuses}
    if seconds:
        summary["p50_s"] = round(statistics.median(seconds), 3)
        summary["p90_s"] = round(seconds[min(len(seconds) - 1, int(len(seconds) * 0.9))], 3)
        summary["max_s"] = round(seconds[-1], 3)
    if upstream:
        summary["upstream_p50_s"] = round(statistics.median(upstream), 3)
    return summary


def run(args: argparse.Namespace) -> int:
    client = NemoClient.from_config(context=args.context)
    base_url = client.base_url.rstrip("/")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=base_url, timeout=args.timeout) as http, out.open("a", encoding="utf-8") as handle:
        target, empty, job = _choose_targets(http, client, args.workspace)
        meta = {"t": time.time(), "base_url": base_url, "target": target, "empty": empty, "job": job}
        handle.write(json.dumps({"meta": meta}) + "\n")
        print(json.dumps(meta), file=sys.stderr)

        privileges = []
        for rung in [] if args.skip_privileges else _privileges(target):
            response = http.get(rung.path, headers=_headers(client, rung))
            row = {"privilege": rung.label, "status": response.status_code, "total": _total(response)}
            privileges.append(row)
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({"privileges": privileges}, indent=1))

        results: dict[str, dict[str, object]] = {}
        ladder = [r for r in _ladder(target, empty, job) if not args.rungs or any(s in r.label for s in args.rungs)]
        for concurrency in args.concurrency:
            for rung in ladder:
                records: list[dict[str, object]] = []
                if concurrency == 1:
                    for _ in range(args.samples):
                        records.append(_get(http, client, rung))
                        time.sleep(args.pause)
                else:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        records = list(pool.map(lambda _: _get(http, client, rung), range(args.samples)))
                for record in records:
                    record["concurrency"] = concurrency
                    handle.write(json.dumps(record) + "\n")
                handle.flush()
                summary = _summarize(records)
                results[f"c{concurrency} {rung.label}"] = summary
                print(f"c{concurrency} {rung.label}: {json.dumps(summary)}", file=sys.stderr)
        print(json.dumps({"ladder": results}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", required=True)
    parser.add_argument("--workspace", help="Workspace to read first; defaults to one with jobs")
    parser.add_argument("--out", required=True, help="JSONL file for every sample")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--pause", type=float, default=0.2, help="Seconds between sequential samples")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--rungs", nargs="+", help="Only run rungs whose label contains one of these strings")
    parser.add_argument("--skip-privileges", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
