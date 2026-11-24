# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print which Postgres statements ran during a quiet window, most frequent first.

Resets pg_stat_statements, waits, then lists the top statements by call count. Run it while the
jobs reconciler is looping to see what a step sync asks the database. BEGIN and ROLLBACK count as
statements (track_utility is on), and each is a round trip.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import asyncpg

FILTER = (
    "dbid = (select oid from pg_database where datname = current_database()) and query not ilike '%pg_stat_statements%'"
)


async def run(args: argparse.Namespace) -> int:
    pg = await asyncpg.connect(args.pg_dsn)
    try:
        await pg.execute("create extension if not exists pg_stat_statements")
        await pg.execute("select pg_stat_statements_reset()")
        began = time.time()
        await asyncio.sleep(args.seconds)
        window = time.time() - began
        total = await pg.fetchval(f"select coalesce(sum(calls), 0) from pg_stat_statements where {FILTER}")
        rows = await pg.fetch(
            "select calls, left(regexp_replace(query, '\\s+', ' ', 'g'), $1) as query "
            f"from pg_stat_statements where {FILTER} order by calls desc limit $2",
            args.width,
            args.top,
        )
    finally:
        await pg.close()
    print(json.dumps({"window_s": round(window, 1), "total_calls": int(total)}))
    for row in rows:
        print(f"{row['calls']:>7}  {row['query']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pg-dsn", default="postgresql://nhx:nhx-lag@pg-lag:5432/nhx", help="Direct, unproxied")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument("--width", type=int, default=150)
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
