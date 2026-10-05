#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Watch a benchmark run's members and report what fan-out actually did.

eval-smoke.sh drives one evaluation at a time, which cannot show the three
things that only appear once a benchmark fans out: how fast the controller
submits, whether the concurrency caps hold, and whether a status ever moves
backward. Those need a sampler rather than a wait-for-terminal poll, because
each is a property of the *sequence* of observations, not of the final state.

Reads the member list repeatedly and prints a JSON summary. Exits non-zero if
any member never settled, a cap was exceeded, or a status regressed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
IN_FLIGHT = frozenset({"provisioning", "running"})
# A member is "submitted" once it leaves queued. Evaluation responses do not
# expose dispatch_job_name, so this is the closest API-visible proxy for the
# controller having handed the row to Platform Jobs.
UNSUBMITTED = frozenset({"queued", "blocked"})

# A manual retry legitimately moves a failed member back to queued, and a
# cancel legitimately lands on an already-terminal row. This probe does
# neither, so here any change out of a terminal status is the bug.
_RANK = {"blocked": 0, "queued": 1, "provisioning": 2, "running": 3}


def _members(base: str, run_id: str) -> list[dict]:
    """Return every member evaluation of the run, following pagination."""
    out: list[dict] = []
    cursor = None
    while True:
        url = f"{base}/v1/benchmark-runs/{run_id}/evaluations?limit=100"
        if cursor:
            url += f"&cursor={urllib.parse.quote(cursor)}"
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed http scheme, local port-forward
            page = json.load(resp)
        out.extend(page.get("data", []))
        cursor = page.get("next_cursor")
        if not cursor:
            return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("run_id")
    ap.add_argument("--expect-members", type=int, required=True)
    ap.add_argument("--member-cap", type=int, default=0, help="0 disables the cap assertion")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    started = time.monotonic()
    last: dict[str, str] = {}
    regressions: list[str] = []
    ramp: list[list[float]] = []
    max_in_flight = 0
    cap_breaches: list[str] = []
    final: dict[str, str] = {}

    while True:
        elapsed = round(time.monotonic() - started, 1)
        try:
            rows = _members(args.base, args.run_id)
        except (urllib.error.URLError, TimeoutError) as exc:
            # A port-forward blip should not fail a 20-minute run.
            print(f"  [{elapsed:7.1f}s] poll failed: {exc}", file=sys.stderr)
            time.sleep(args.interval)
            continue

        statuses = {r["id"]: r["status"] for r in rows}
        for eid, status in statuses.items():
            before = last.get(eid)
            if before is None or before == status:
                continue
            if before in TERMINAL or _RANK.get(status, 9) < _RANK.get(before, 9):
                regressions.append(f"{eid}: {before} -> {status}")
        last = statuses

        in_flight = sum(1 for s in statuses.values() if s in IN_FLIGHT)
        max_in_flight = max(max_in_flight, in_flight)
        if args.member_cap and in_flight > args.member_cap:
            cap_breaches.append(f"{in_flight} in flight at {elapsed}s, cap is {args.member_cap}")

        submitted = sum(1 for s in statuses.values() if s not in UNSUBMITTED)
        if not ramp or ramp[-1][1] != submitted:
            ramp.append([elapsed, submitted])

        settled = sum(1 for s in statuses.values() if s in TERMINAL)
        print(
            f"  [{elapsed:7.1f}s] members={len(statuses)} submitted={submitted} "
            f"in_flight={in_flight} settled={settled}",
            file=sys.stderr,
        )

        if len(statuses) >= args.expect_members and settled == len(statuses):
            final = statuses
            break
        if time.monotonic() - started > args.timeout:
            final = statuses
            break
        time.sleep(args.interval)

    # Submissions per minute over the window it took to submit them all, which
    # is the number AALGO-656 moves. One-per-pass at a 10s interval is ~6/min.
    submit_window = next((t for t, n in ramp if n >= args.expect_members), None)
    per_min = round(args.expect_members / (submit_window / 60), 1) if submit_window else None

    counts: dict[str, int] = {}
    for status in final.values():
        counts[status] = counts.get(status, 0) + 1
    unsettled = [eid for eid, s in final.items() if s not in TERMINAL]

    summary = {
        "members_seen": len(final),
        "expected_members": args.expect_members,
        "final_status_counts": counts,
        "unsettled": unsettled,
        "max_in_flight": max_in_flight,
        "member_cap": args.member_cap or None,
        "cap_breaches": cap_breaches,
        "regressions": regressions,
        "submit_all_seconds": submit_window,
        "submissions_per_minute": per_min,
        "submit_ramp": ramp,
    }
    print(json.dumps(summary, indent=2))

    problems = []
    if len(final) != args.expect_members:
        problems.append(f"saw {len(final)} members, expected {args.expect_members}")
    if unsettled:
        problems.append(f"{len(unsettled)} members never settled")
    if cap_breaches:
        problems.append(f"concurrency cap exceeded: {cap_breaches[0]}")
    if regressions:
        problems.append(f"status moved backward: {regressions[0]}")
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
