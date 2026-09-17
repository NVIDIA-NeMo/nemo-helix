# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ask a live Intake which filter fields and operators it really serves.

Not a test. Pytest does not collect this file. No automation runs it. See README.md in
this directory.

The docstrings of ``query_spans`` and ``query_traces`` are the only guide the agent has
when it builds a query, so a wrong docstring sends the agent down a dead end. Neither the
filter schema nor the "Unknown filter field" error list settles the question: the schema
says what is declared, and the error lists the entity's columns, which is wider than what
is filterable. Only the server knows.

So probe every field, and record the operators too. Run this after any change to the
Intake filter schemas. If this report and the docstrings disagree, fix the docstrings.

Read-only: every call is a GET.

    uv run --frozen python plugins/nemo-eval-author/tests/manual/intake_vocabulary_probe.py
"""

import argparse
import asyncio
import re
from typing import Any

from nemo_experimentalist_plugin.eval_author import traces
from nemo_platform_plugin.client.client import AsyncNemoClient

# Every column Intake names in its "Unknown filter field" message, which is a superset of
# what it can actually filter. Probing the superset is the point.
SPAN_FIELDS = (
    "agent_id",
    "agent_name",
    "created_at",
    "dataset_id",
    "dataset_name",
    "dataset_version",
    "entity_type",
    "evaluation_id",
    "id",
    "kind",
    "model",
    "name",
    "parent_span_id",
    "project",
    "prompt_name",
    "prompt_version",
    "provider",
    "session_id",
    "source",
    "started_at",
    "status",
    "test_case_id",
    "tool_name",
    "trace_id",
    "updated_at",
    "workspace",
)
TRACE_FIELDS = (
    "created_at",
    "entity_type",
    "evaluation_id",
    "id",
    "name",
    "project",
    "session_id",
    "started_at",
    "status",
    "test_case_id",
    "updated_at",
    "workspace",
)
DATE_FIELDS = frozenset({"created_at", "updated_at", "started_at", "ended_at"})
# A plausible value per enum field, so a rejection means the field, never the value.
ENUM_VALUES = {"kind": "LLM", "status": "success", "source": "openinference", "entity_type": "span"}
PROBE = "probe-value-zzz"

OPERATOR_CASES = (
    ("span", "started_at gte", {"started_at": {"gte": "2020-01-01T00:00:00"}}),
    ("span", "started_at lte", {"started_at": {"lte": "2030-01-01T00:00:00"}}),
    ("span", "started_at gte+lte", {"started_at": {"gte": "2020-01-01T00:00:00", "lte": "2030-01-01T00:00:00"}}),
    ("span", "trace_id $in", {"trace_id": {"$in": ["a", "b"]}}),
    ("span", "status $in", {"status": {"$in": ["error", "success"]}}),
    ("span", "agent_name $in", {"agent_name": {"$in": ["a", "b"]}}),
    ("trace", "started_at gte+lte", {"started_at": {"gte": "2020-01-01T00:00:00", "lte": "2030-01-01T00:00:00"}}),
    ("trace", "id $in", {"id": {"$in": ["a", "b"]}}),
    ("trace", "session_id $in", {"session_id": {"$in": ["a", "b"]}}),
    ("trace", "status $in", {"status": {"$in": ["error", "success"]}}),
)


def value_for(field: str) -> Any:
    if field in DATE_FIELDS:
        return {"gte": "2020-01-01T00:00:00"}
    return ENUM_VALUES.get(field, PROBE)


def detail(exc: Exception) -> str:
    """Pull the reason of the server out of the wrapper message.

    The wrapper adds a recovery hint at both ends, so a plain truncation shows our own
    boilerplate instead of the answer.
    """
    text = str(exc)
    match = re.search(r"'detail':\s*(['\"])(.*?)\1", text, re.DOTALL)
    return match.group(2) if match else text[:80]


def uses_sentinel(field: str) -> bool:
    """True when the probe value is one nothing can hold, so the reply must be empty."""
    return field not in DATE_FIELDS and field not in ENUM_VALUES


async def run(
    kind: str,
    client: AsyncNemoClient,
    workspace: str,
    filter: dict[str, Any],
    *,
    expect_no_rows: bool = False,
) -> str:
    query = traces.query_spans if kind == "span" else traces.query_traces
    try:
        result = await query(client, workspace=workspace, filter=filter, limit=1)
    except traces.TraceQueryError as exc:
        text = str(exc)
        if "failed internally" in text or "HTTP 500" in text:
            return "BROKEN (HTTP 500)"
        if "Unknown filter field" in text:
            return "no (not a field)"
        return f"no ({detail(exc)})"
    if expect_no_rows and result["count"]:
        # Accepting a request is not applying a filter. Nothing holds the sentinel value,
        # so a row here means Intake took the field and then ignored it, which would make
        # every answer built on that field wrong.
        return "ACCEPTED BUT NOT APPLIED"
    return "WORKS"


async def probe_fields(client: AsyncNemoClient, workspace: str, kind: str, fields: tuple[str, ...]) -> None:
    print(f"\n{'=' * 66}\n{kind.upper()} FILTER FIELDS\n{'=' * 66}")
    works: list[str] = []
    broken: list[str] = []
    ignored: list[str] = []
    for field in fields:
        verdict = await run(kind, client, workspace, {field: value_for(field)}, expect_no_rows=uses_sentinel(field))
        print(f"  {field:<18} {verdict}")
        if verdict == "WORKS":
            works.append(field)
        elif verdict.startswith("BROKEN"):
            broken.append(field)
        elif verdict.startswith("ACCEPTED"):
            ignored.append(field)
    print(f"\n  filterable ({len(works)}): {', '.join(works)}")
    unconfirmed = [field for field in works if not uses_sentinel(field)]
    if unconfirmed:
        # An enum or a date needs a real value, so these queries return rows either way
        # and an empty reply cannot be required. intake_tool_checks.py asserts that kind
        # and agent_name narrow for real.
        print(f"  accepted, application not confirmed: {', '.join(unconfirmed)}")
    if ignored:
        print(f"\n  ACCEPTED BUT NOT APPLIED ({len(ignored)}): {', '.join(ignored)}")
        print("  Intake takes these filters and returns rows that do not match.")
        print("  Treat any answer that relies on them as wrong.")
    if broken:
        print(f"\n  BROKEN ({len(broken)}): {', '.join(broken)}")
        print("  Intake publishes these filters and answers them with HTTP 500.")
        print("  Do not document them. Confirm they are still broken before removing this note.")


async def probe_operators(client: AsyncNemoClient, workspace: str) -> None:
    print(f"\n{'=' * 66}\nOPERATORS (which operators Intake accepts per field)\n{'=' * 66}")
    for kind, label, filter in OPERATOR_CASES:
        # The $in cases pass values nothing holds, so they must come back empty. The date
        # ranges are deliberately wide, so rows are expected and only acceptance is shown.
        print(f"  {kind:<6} {label:<22} {await run(kind, client, workspace, filter, expect_no_rows='$in' in label)}")


async def probe_group_by(client: AsyncNemoClient, workspace: str) -> None:
    print(f"\n{'=' * 66}\nGROUP BY\n{'=' * 66}")
    for field in ("trace_id", "session_id", "agent_name", "kind"):
        try:
            await traces.query_spans(client, workspace=workspace, group_by=field, limit=1)
            print(f"  {field:<18} WORKS")
        except traces.TraceQueryError as exc:
            print(f"  {field:<18} no ({detail(exc)})")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--workspace", default="default", help="Any workspace. It need not hold spans.")
    args = parser.parse_args()

    async with AsyncNemoClient(base_url=args.base_url) as client:
        await probe_fields(client, args.workspace, "span", SPAN_FIELDS)
        await probe_fields(client, args.workspace, "trace", TRACE_FIELDS)
        await probe_operators(client, args.workspace)
        await probe_group_by(client, args.workspace)


if __name__ == "__main__":
    asyncio.run(main())
