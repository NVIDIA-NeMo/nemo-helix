# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual check of the Eval Author Intake tools against a live platform.

Not a test. Pytest does not collect this file. No automation runs it. See README.md in
this directory.

The unit tests fake every Intake call, so they prove the logic and never prove that
Intake answers the way the logic expects. This script proves the second half.

Read-only: every call is a GET.

    uv run --frozen python plugins/nemo-eval-author/tests/manual/intake_tool_checks.py
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from nemo_experimentalist_plugin.eval_author import traces
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.workspaces.client import AsyncWorkspacesClient

DISCOVERY_SPAN_BUDGET = 200


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        suffix = f" :: {detail}" if detail else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{suffix}")

    def note(self, text: str) -> None:
        print(f"        {text}")

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")


async def discover(client: AsyncNemoClient) -> tuple[str, str] | None:
    """Find the workspace with the most agent-scoped spans, and its busiest agent."""
    best: tuple[int, str, str] | None = None
    intake = AsyncIntakeClient.from_client(client)
    async for workspace in (await AsyncWorkspacesClient.from_client(client).list_workspaces()).items():
        counts: dict[str, int] = {}
        scanned = 0
        try:
            response = await intake.list_spans(
                workspace=workspace.name, query_params={"mode": "summary", "page_size": 100}
            )
            async for span in response.items():
                scanned += 1
                agent = getattr(span, "agent_name", None)
                if agent:
                    counts[agent] = counts.get(agent, 0) + 1
                if scanned >= DISCOVERY_SPAN_BUDGET:
                    break
        except Exception:
            continue
        if not counts:
            continue
        agent, count = max(counts.items(), key=lambda item: item[1])
        if best is None or count > best[0]:
            best = (count, workspace.name, agent)
    if best is None:
        return None
    return best[1], best[2]


async def check_raw_span_queries(client: AsyncNemoClient, report: Report, workspace: str, agent: str) -> None:
    report.section("query_spans: rows, order, and plain dicts")
    result = await traces.query_spans(client, workspace=workspace, limit=5)
    report.check("returns rows", result["count"] > 0, f"count={result['count']} truncated={result['truncated']}")
    if not result["count"]:
        return
    report.check("rows are plain dicts", isinstance(result["spans"][0], dict))
    stamps = [row.get("started_at") for row in result["spans"]]
    report.check("newest first", stamps == sorted(stamps, reverse=True))

    report.section("query_spans: the server honors a composed filter")
    for kind in ("LLM", "TOOL"):
        got = await traces.query_spans(
            client, workspace=workspace, filter={"agent_name": agent, "kind": kind}, limit=25
        )
        kinds = {row.get("kind") for row in got["spans"]}
        if got["count"]:
            report.check(f"kind={kind} narrows server-side", kinds == {kind}, f"count={got['count']}")
        else:
            report.note(f"no {kind} spans for agent {agent!r}, skipped")

    scoped = await traces.query_spans(client, workspace=workspace, filter={"agent_name": agent}, limit=25)
    agents = {row.get("agent_name") for row in scoped["spans"]}
    report.check("agent_name narrows server-side", agents == {agent}, f"saw {agents}")

    report.section("query_spans: group_by")
    grouped = await traces.query_spans(client, workspace=workspace, group_by="trace_id", limit=5)
    report.check("grouped shape", grouped.get("grouped_by") == "trace_id" and "groups" in grouped)
    if grouped["groups"]:
        report.note(f"first group: {grouped['groups'][0]}")
    try:
        await traces.query_spans(client, workspace=workspace, group_by="agent_name", limit=1)
        report.check("only trace_id and session_id group", False, "agent_name grouping was accepted")
    except traces.TraceQueryError as exc:
        report.check("only trace_id and session_id group", True, str(exc)[:110])


async def check_raw_trace_queries(client: AsyncNemoClient, report: Report, workspace: str) -> None:
    report.section("query_traces: rollups and the $in filter")
    result = await traces.query_traces(client, workspace=workspace, limit=5)
    report.check("returns rows", result["count"] > 0, f"count={result['count']}")
    if not result["count"]:
        return
    report.check("preview carries rollups", "span_count" in result["traces"][0])

    ids = [row["id"] for row in result["traces"][:3]]
    got = await traces.query_traces(client, workspace=workspace, filter={"id": {"$in": ids}}, limit=10)
    report.check(
        "$in returns exactly those ids",
        {row["id"] for row in got["traces"]} == set(ids),
        f"asked {len(ids)}, got {got['count']}",
    )

    report.section("composability: filter spans, group to traces, then read those traces")
    grouped = await traces.query_spans(
        client, workspace=workspace, filter={"kind": "TOOL"}, group_by="trace_id", limit=5
    )
    trace_ids = [group["group"]["trace_id"] for group in grouped["groups"]]
    if not trace_ids:
        report.note("no TOOL spans in this workspace, skipped")
        return
    composed = await traces.query_traces(
        client, workspace=workspace, filter={"id": {"$in": trace_ids}}, limit=len(trace_ids)
    )
    report.check(
        "round trip resolves every grouped id",
        composed["count"] == len(trace_ids),
        f"{len(trace_ids)} traces used a TOOL span, resolved {composed['count']}",
    )


async def check_find_agent_traces(
    client: AsyncNemoClient, report: Report, workspace: str, agent: str
) -> dict[str, Any] | None:
    report.section("find_agent_traces")
    found = await traces.find_agent_traces(client, agent=agent, workspace=workspace, limit=5)
    report.check("returns traces", found["count"] > 0, f"count={found['count']} truncated={found['truncated']}")
    if not found["count"]:
        return None
    for entry in found["traces"]:
        report.note(
            f"{entry['trace_ref']}  status={entry['status']:<8} spans={entry['span_count']} "
            f"errors={entry['error_count']} at={entry['started_at']}"
        )
    stamps = [entry["started_at"] for entry in found["traces"]]
    report.check("newest first", stamps == sorted(stamps, reverse=True))
    report.check("no duplicate traces", len({e["trace_id"] for e in found["traces"]}) == found["count"])

    report.section("the summary merge is authoritative, not window-scoped")
    target = found["traces"][0]
    counted = await traces.query_spans(client, workspace=workspace, filter={"trace_id": target["trace_id"]}, limit=1000)
    report.check(
        "span_count equals the true span count",
        target["span_count"] == counted["count"],
        f"summary={target['span_count']} counted={counted['count']}",
    )
    report.section("discovery is one grouped call, sorted by the server")
    # find_agent_traces used to page spans and collect distinct ids, which only ever
    # found the traces inside the scanned window. It now asks Intake to group and sort.
    # Prove the server really serves that, since a fake client cannot.
    grouped = await traces.query_spans(
        client, workspace=workspace, filter={"agent_name": agent}, group_by="trace_id", sort="-started_at", limit=5
    )
    starts = [row["started_at"] for row in grouped["groups"]]
    report.check("groups come back newest first", starts == sorted(starts, reverse=True), f"{starts}")
    report.check(
        "the grouped ids are the traces find_agent_traces returned",
        {row["group"]["trace_id"] for row in grouped["groups"]} == {e["trace_id"] for e in found["traces"]},
    )
    by_size = await traces.query_spans(
        client, workspace=workspace, filter={"agent_name": agent}, group_by="trace_id", sort="-span_count", limit=5
    )
    sizes = [row["span_count"] for row in by_size["groups"]]
    report.check("groups also sort by size", sizes == sorted(sizes, reverse=True), f"{sizes}")

    report.section("find_agent_traces: since, and the empty case")
    wide = await traces.find_agent_traces(
        client, agent=agent, workspace=workspace, since=datetime.now(UTC) - timedelta(days=3650), limit=3
    )
    report.check("a wide since returns traces", wide["count"] > 0, f"count={wide['count']}")
    future = await traces.find_agent_traces(
        client, agent=agent, workspace=workspace, since=datetime.now(UTC) + timedelta(days=1), limit=3
    )
    report.check("a future since returns nothing", future["count"] == 0, f"count={future['count']}")

    empty = await traces.find_agent_traces(client, agent="no-such-agent-zzz", workspace=workspace)
    report.check(
        "an empty result is a note, not an error", empty["count"] == 0 and "no-such-agent-zzz" in empty["note"]
    )
    return target


async def check_read_trace(client: AsyncNemoClient, report: Report, workspace: str, trace_id: str) -> None:
    report.section("read_trace: all three ref spellings")
    explorer = None
    for ref in (trace_id, f"intake://{trace_id}", f"intake://traces/{trace_id}"):
        explorer = await traces.read_trace(client, ref, workspace=workspace)
        report.check(f"read {ref[:30]}", explorer is not None, f"agents={explorer.agent_count}")
    if explorer is not None:
        overview = await explorer.get_overview(concise=True)
        for line in overview.splitlines()[:6]:
            report.note(line)


async def check_errors(client: AsyncNemoClient, report: Report, workspace: str, agent: str) -> None:
    report.section("errors name a corrective action")
    cases = (
        ("an unknown filter field", traces.query_spans(client, workspace=workspace, filter={"not_a_field": "x"})),
        ("an unknown workspace", traces.find_agent_traces(client, agent=agent, workspace="no-such-workspace-zzz")),
        ("a missing trace", traces.read_trace(client, "trace-does-not-exist-zzz", workspace=workspace)),
        # Intake used to publish this span filter and answer it with HTTP 500. Since
        # nemo-platform#1225 it is unpublished, so it must now read as a rejected field.
        (
            "a filter Intake once answered with 500",
            traces.query_spans(client, workspace=workspace, filter={"dataset_name": "x"}),
        ),
    )
    for label, coro in cases:
        try:
            await coro
            report.check(label, False, "it succeeded")
        except traces.TraceQueryError as exc:
            report.check(label, True, str(exc)[:130])


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--workspace", default=None, help="Skip discovery and use this workspace.")
    parser.add_argument("--agent", default=None, help="Skip discovery and use this agent_name.")
    args = parser.parse_args()
    if bool(args.workspace) != bool(args.agent):
        parser.error("--workspace and --agent must be given together, or neither.")

    async with AsyncNemoClient(base_url=args.base_url) as client:
        if args.workspace and args.agent:
            workspace, agent = args.workspace, args.agent
        else:
            print("Looking for a workspace that holds agent-scoped spans...")
            discovered = await discover(client)
            if discovered is None:
                print(
                    "No spans found with an agent_name in any workspace. Ingest some traces, "
                    "or pass --workspace and --agent."
                )
                return 1
            workspace, agent = discovered
        print(f"Using workspace {workspace!r} and agent {agent!r}.")

        report = Report()
        await check_raw_span_queries(client, report, workspace, agent)
        await check_raw_trace_queries(client, report, workspace)
        target = await check_find_agent_traces(client, report, workspace, agent)
        if target is not None:
            await check_read_trace(client, report, workspace, target["trace_id"])
        await check_errors(client, report, workspace, agent)

    print(f"\n{'=' * 66}\nPASS {report.passed}   FAIL {report.failed}\n{'=' * 66}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
