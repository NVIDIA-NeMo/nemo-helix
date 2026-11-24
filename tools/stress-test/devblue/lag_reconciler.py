# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count database round trips per reconciler step sync by slowing Postgres.

The jobs reconciler syncs steps one after another, so the time between two "Updating job step
status" log lines is one step's sync. With toxiproxy adding a fixed delay to every database round
trip, that time grows by (round trips per step) x (delay): the slope across delays is the count.

For each delay the tool waits for ``--steps`` completed steps in the platform log, then reports the
mean seconds per step, the mean time between reconciler passes, and the Postgres statements that
ran per step (from pg_stat_statements over a direct, unproxied connection). Statement counts include
any other database work in that window, so they are an upper bound for the reconciler alone.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

import asyncpg

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)")
STEP_DONE = "Updating job step status from"
PASS_START = "Got "
PASS_MARK = "job steps to reconcile"


def _set_latency(toxiproxy: str, proxy: str, ms: int) -> None:
    base = f"{toxiproxy}/proxies/{proxy}/toxics"
    try:
        urllib.request.urlopen(urllib.request.Request(f"{base}/lag", method="DELETE"))
    except urllib.error.HTTPError:
        pass
    if ms:
        body = {"name": "lag", "type": "latency", "stream": "downstream", "attributes": {"latency": ms, "jitter": 0}}
        request = urllib.request.Request(base, data=json.dumps(body).encode(), method="POST")
        urllib.request.urlopen(request)


def _events(log: Path, offset: int) -> tuple[list[float], list[float]]:
    """Return (step-done times, pass-start times) in epoch seconds for log text after ``offset``."""
    with log.open("rb") as handle:
        handle.seek(offset)
        text = handle.read().decode("utf-8", errors="replace")
    steps: list[float] = []
    passes: list[float] = []
    for raw in text.splitlines():
        line = ANSI.sub("", raw)
        match = STAMP.match(line)
        if not match:
            continue
        when = datetime.datetime.fromisoformat(match.group(1)).replace(tzinfo=datetime.timezone.utc).timestamp()
        if STEP_DONE in line:
            steps.append(when)
        elif PASS_START in line and PASS_MARK in line:
            passes.append(when)
    return steps, passes


async def _measure(args: argparse.Namespace, pg: asyncpg.Connection, log: Path, ms: int) -> dict[str, object]:
    _set_latency(args.toxiproxy, args.proxy, ms)
    await asyncio.sleep(args.settle)
    await pg.execute("select pg_stat_statements_reset()")
    offset = log.stat().st_size
    began = time.time()
    steps: list[float] = []
    passes: list[float] = []
    while time.time() - began < args.timeout:
        await asyncio.sleep(2)
        steps, passes = _events(log, offset)
        if len(steps) >= args.steps + 1:
            break
    calls = int(
        await pg.fetchval(
            "select coalesce(sum(calls), 0) from pg_stat_statements "
            "where dbid = (select oid from pg_database where datname = current_database()) "
            "and query not ilike '%pg_stat_statements%'"
        )
    )
    gaps = [b - a for a, b in zip(steps, steps[1:], strict=False)]
    # Gaps that span a pass boundary include the pass's fetch and sweeps, so keep the shorter half.
    within_pass = sorted(gaps)[: max(1, (len(gaps) * 2) // 3)] if gaps else []
    elapsed = (steps[-1] - steps[0]) if len(steps) > 1 else 0.0
    return {
        "latency_ms": ms,
        "steps_seen": len(steps),
        "s_per_step_mean": round(statistics.fmean(gaps), 3) if gaps else None,
        "s_per_step_within_pass": round(statistics.fmean(within_pass), 3) if within_pass else None,
        "s_between_passes": round(statistics.fmean([b - a for a, b in zip(passes, passes[1:], strict=False)]), 3)
        if len(passes) > 1
        else None,
        "statements_total": calls,
        "statements_per_step": round(calls / max(1, len(steps) - 1), 1) if len(steps) > 1 else None,
        "window_s": round(elapsed, 1),
    }


def _slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs, ys = zip(*points, strict=True)
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator if denominator else None


async def run(args: argparse.Namespace) -> int:
    log = Path(args.log)
    pg = await asyncpg.connect(args.pg_dsn)
    await pg.execute("create extension if not exists pg_stat_statements")
    rows: list[dict[str, object]] = []
    try:
        for ms in args.latencies:
            row = await _measure(args, pg, log, ms)
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        _set_latency(args.toxiproxy, args.proxy, 0)
        await pg.close()
    points = [
        (int(str(r["latency_ms"])), float(str(r["s_per_step_within_pass"])))
        for r in rows
        if r["s_per_step_within_pass"] is not None
    ]
    slope = _slope([(ms, seconds * 1000) for ms, seconds in points])
    summary = {
        "label": args.label,
        "round_trips_per_step": round(slope, 1) if slope is not None else None,
        "rows": rows,
    }
    with Path(args.out).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")
    print(json.dumps({"label": args.label, "round_trips_per_step": summary["round_trips_per_step"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default="/work/logs/lag-platform.log")
    parser.add_argument("--toxiproxy", default="http://pg-lag:8474")
    parser.add_argument("--proxy", default="pg")
    parser.add_argument("--pg-dsn", default="postgresql://nhx:nhx-lag@pg-lag:5432/nhx", help="Direct, unproxied")
    parser.add_argument("--latencies", type=int, nargs="+", default=[0, 30, 60], help="Injected delay per round trip")
    parser.add_argument("--steps", type=int, default=4, help="Completed steps to observe at each delay")
    parser.add_argument("--settle", type=float, default=6.0, help="Seconds to let a new delay take hold")
    parser.add_argument("--timeout", type=float, default=420.0, help="Give up on a delay after this many seconds")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--out", required=True)
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
