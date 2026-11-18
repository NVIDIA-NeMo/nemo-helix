# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score every generate_data_designer_job_request call in Intake and tag its span accepted or rejected."""

import argparse
import json
import os
import urllib.parse
import urllib.request
from typing import Any

TOOL_NAME = "generate_data_designer_job_request"
METRIC_NAME = "tool_call_accepted/v1"
REJECTION_MARKERS = ("TOOL CALL REJECTED:", "422 Unprocessable Entity")


def _request(base_url: str, path: str, *, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{base_url}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET"
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _list_all(base_url: str, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({**params, "page": page, "page_size": 100})
        result = _request(base_url, f"{path}?{query}")
        rows.extend(result["data"])
        if page >= result["pagination"]["total_pages"]:
            return rows
        page += 1


def _tool_output_text(span: dict[str, Any]) -> str:
    try:
        return json.loads(span["output"])["content"][0]["text"]
    except (TypeError, KeyError, IndexError, json.JSONDecodeError):
        return span.get("error_message") or span.get("output") or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="data-designer-toolcall")
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--base-url", default=os.environ.get("NHX_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--dry-run", action="store_true", help="Print the scores without writing evaluator results.")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    intake = f"/apis/intake/v2/workspaces/{args.workspace}"

    spans = _list_all(
        base_url,
        f"{intake}/spans",
        {"filter[agent_name]": args.agent, "filter[tool_name]": TOOL_NAME, "sort": "started_at"},
    )
    # Relay records each call twice; the OTel `execute_tool` span is the one that carries the tool result.
    calls = [span for span in spans if span["name"].startswith("execute_tool")]
    already_scored = {
        result["span_id"]
        for result in _list_all(base_url, f"{intake}/evaluator-results", {"filter[name]": METRIC_NAME})
    }

    rejected = 0
    for span in calls:
        text = _tool_output_text(span)
        is_rejected = span["status"] != "success" or text.startswith(REJECTION_MARKERS)
        rejected += is_rejected
        reasons = [line.removeprefix("- ") for line in text.splitlines()[1:]] if is_rejected else []
        print(f"{span['started_at'][:19]}  {'REJECTED' if is_rejected else 'accepted'}  {span['session_id']}")
        for reason in reasons:
            print(f"    - {reason}")
        if args.dry_run or span["span_id"] in already_scored:
            continue
        _request(
            base_url,
            f"{intake}/evaluator-results",
            body={
                "span_id": span["span_id"],
                "session_id": span["session_id"],
                "name": METRIC_NAME,
                "data_type": "BOOLEAN",
                "value": 0 if is_rejected else 1,
                "comment": "\n".join(reasons) if is_rejected else "Accepted by the Data Designer job request API.",
            },
        )

    total = len(calls)
    rate = rejected / total if total else 0.0
    print(f"\n{rejected}/{total} tool calls rejected (tool_error_rate={rate:.0%})")


if __name__ == "__main__":
    main()
