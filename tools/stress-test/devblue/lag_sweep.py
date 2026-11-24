# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count database round trips per request by slowing Postgres and watching latency grow.

The platform reaches Postgres through toxiproxy, whose latency toxic delays every server-to-client
chunk by a fixed amount. A request that makes N sequential database round trips therefore slows by
N times the injected delay, so the slope of request latency against injected delay is N. That
counts every round trip, including pool pre-pings, prepares, and transaction BEGIN/COMMIT, which a
statement log alone would undercount.

Statement counts come from pg_stat_statements over a direct connection that bypasses toxiproxy.
They include background work (the jobs controller, the role-binding refresh), so treat them as an
upper bound.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import asyncpg
import httpx

ADMIN = {"X-NHX-Principal-Id": "lag-user", "X-NHX-Principal-Email": "lag-admin@example.com"}
NOBODY = {"X-NHX-Principal-Id": "nobody", "X-NHX-Principal-Email": "nobody@example.com"}
JOBS_SERVICE = {"X-NHX-Principal-Id": "service:jobs"}

STATEMENT_COUNT = """
select coalesce(sum(calls), 0) from pg_stat_statements
where dbid = (select oid from pg_database where datname = current_database())
  and query not ilike '%pg_stat_statements%'
"""
TOP_STATEMENTS = """
select calls, left(regexp_replace(query, '\\s+', ' ', 'g'), 160) as query from pg_stat_statements
where dbid = (select oid from pg_database where datname = current_database())
  and query not ilike '%pg_stat_statements%'
order by calls desc limit 20
"""


@dataclass(frozen=True)
class Rung:
    label: str
    path: str
    headers: dict[str, str]


def _rungs(workspace: str) -> list[Rung]:
    ws = f"/apis/entities/v2/workspaces/{workspace}"
    return [
        Rung("status, no auth", "/status", {}),
        Rung("anonymous authorization (no account lookup)", ws, {}),
        Rung("user denied (hello-world 403)", f"/apis/hello-world/v2/workspaces/{workspace}/hello", NOBODY),
        Rung("service:jobs workspaces.get", ws, JOBS_SERVICE),
        Rung("admin workspaces.get", ws, ADMIN),
        Rung("admin hello-world (one loopback hop)", f"/apis/hello-world/v2/workspaces/{workspace}/hello", ADMIN),
        Rung("admin jobs list", f"/apis/jobs/v2/workspaces/{workspace}/jobs?page_size=10", ADMIN),
    ]


async def _set_latency(http: httpx.AsyncClient, toxiproxy: str, proxy: str, ms: int) -> None:
    await http.delete(f"{toxiproxy}/proxies/{proxy}/toxics/lag")
    if ms:
        response = await http.post(
            f"{toxiproxy}/proxies/{proxy}/toxics",
            json={"name": "lag", "type": "latency", "stream": "downstream", "attributes": {"latency": ms, "jitter": 0}},
        )
        response.raise_for_status()


async def _measure(
    http: httpx.AsyncClient, pg: asyncpg.Connection, base_url: str, rung: Rung, samples: int
) -> dict[str, object]:
    for _ in range(2):
        await http.get(base_url + rung.path, headers=rung.headers)
    await pg.execute("select pg_stat_statements_reset()")
    seconds: list[float] = []
    statuses: set[int] = set()
    for _ in range(samples):
        began = time.perf_counter()
        response = await http.get(base_url + rung.path, headers=rung.headers)
        seconds.append(time.perf_counter() - began)
        statuses.add(response.status_code)
    statements = await pg.fetchval(STATEMENT_COUNT)
    return {
        "p50_ms": round(statistics.median(seconds) * 1000, 2),
        "max_ms": round(max(seconds) * 1000, 2),
        "statuses": sorted(statuses),
        "statements_per_request": round(int(statements) / samples, 1),
    }


def _slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs, ys = zip(*points, strict=True)
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator if denominator else None


async def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rungs = [r for r in _rungs(args.workspace) if not args.rungs or any(s in r.label for s in args.rungs)]
    pg = await asyncpg.connect(args.pg_dsn)
    await pg.execute("create extension if not exists pg_stat_statements")
    results: dict[str, dict[int, dict[str, object]]] = {r.label: {} for r in rungs}
    with out.open("a", encoding="utf-8") as handle:
        await _sweep(args, rungs, pg, results, handle)
    return 0


async def _sweep(
    args: argparse.Namespace,
    rungs: list[Rung],
    pg: asyncpg.Connection,
    results: dict[str, dict[int, dict[str, object]]],
    handle: TextIO,
) -> None:
    async with httpx.AsyncClient(timeout=args.timeout) as http:
        handle.write(json.dumps({"meta": {"t": time.time(), "label": args.label, "latencies": args.latencies}}) + "\n")
        try:
            for ms in args.latencies:
                await _set_latency(http, args.toxiproxy, args.proxy, ms)
                await asyncio.sleep(1)
                for rung in rungs:
                    row = await _measure(http, pg, args.base_url, rung, args.samples)
                    results[rung.label][ms] = row
                    handle.write(json.dumps({"label": args.label, "latency_ms": ms, "rung": rung.label, **row}) + "\n")
                    handle.flush()
                    print(f"{ms:>3} ms  {rung.label}: {json.dumps(row)}", flush=True)
                if args.top and ms == args.latencies[0]:
                    await pg.execute("select pg_stat_statements_reset()")
                    for _ in range(args.samples):
                        await http.get(args.base_url + rungs[-1].path, headers=rungs[-1].headers)
                    top = [dict(r) for r in await pg.fetch(TOP_STATEMENTS)]
                    handle.write(
                        json.dumps({"label": args.label, "top_statements_for": rungs[-1].label, "rows": top}) + "\n"
                    )
                    print(json.dumps({"top_statements_for": rungs[-1].label, "rows": top}, indent=1))
        finally:
            await _set_latency(http, args.toxiproxy, args.proxy, 0)
            await pg.close()

        summary = {}
        for label, by_ms in results.items():
            slope = _slope([(ms, float(str(row["p50_ms"]))) for ms, row in sorted(by_ms.items())])
            summary[label] = {
                "round_trips_per_request": round(slope, 1) if slope is not None else None,
                **{f"p50_ms@{ms}": row["p50_ms"] for ms, row in sorted(by_ms.items())},
                "statements_per_request@0": by_ms.get(args.latencies[0], {}).get("statements_per_request"),
            }
        handle.write(json.dumps({"label": args.label, "summary": summary}) + "\n")
        print(json.dumps({"label": args.label, "summary": summary}, indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8090")
    parser.add_argument("--toxiproxy", default="http://pg-lag:8474")
    parser.add_argument("--proxy", default="pg")
    parser.add_argument("--pg-dsn", default="postgresql://nhx:nhx-lag@pg-lag:5432/nhx", help="Direct, unproxied")
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--latencies", type=int, nargs="+", default=[0, 10, 45], help="Injected delay per round trip")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rungs", nargs="+", help="Only rungs whose label contains one of these strings")
    parser.add_argument("--top", action="store_true", help="Also dump the statement mix for the last rung")
    parser.add_argument("--label", default="baseline", help="Name for this run in the output")
    parser.add_argument("--out", required=True)
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
