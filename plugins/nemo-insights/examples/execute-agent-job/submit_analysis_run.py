# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit an Insights Analyst demo run through the analysis-runs SDK.

Everything here is the supported surface — ``client.insights.analysis_runs``,
mirrored by ``nemo insights analysis-runs`` — rather than hand-rolled HTTP, so
the demo shows what an end user would actually write. The SDK translates the
request into the generic ``agents.execute`` job: the route composes the Analyst
config from the supplied model pair and submits it as an inline agent
definition, so there is no Analyst Agent entity to provision.

With ``--wait`` it polls the run to a terminal state and prints the
``analysis-report`` result the ``insights.analysis`` execute extension saved —
the durable comparison point against the existing ``AnalyzeJob``, which saves
the same report under the same result name.

The equivalent one-liner, with the model pair taken from your CLI config::

    nemo insights analysis-runs create --agent demo-agent --wait
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import httpx
from nemo_insights_plugin.platform_client import make_client
from nemo_platform import AsyncNeMoPlatform

REPORT_RESULT_NAME = "analysis-report"


async def main() -> None:
    args = _parse_args()
    if args.dry_run:
        print(json.dumps(_request_preview(args), indent=2))
        return

    client = make_client(args.base_url)
    try:
        response = await client.insights.analysis_runs.create(
            workspace=args.workspace,
            agent=args.target_agent,
            default_model=args.default_model,
            fast_model=args.fast_model,
            timeout_seconds=args.timeout_seconds,
        )
        run_name = response.run.name
        print(f"Created analysis run '{run_name}' (job status: {response.job_status}).")
        if not args.wait:
            print(f"Poll it with: nemo insights analysis-runs get {run_name} --workspace {args.workspace}")
            return

        final = await client.insights.analysis_runs.wait(
            workspace=args.workspace,
            name=run_name,
            timeout=args.poll_timeout,
            poll_interval=args.poll_interval,
            on_status=lambda status: print(f"  status: {status}"),
        )
        print(f"Run '{run_name}' finished with job status '{final.job_status}'.")
        await _print_report(client, args.base_url, args.workspace, run_name)
        if final.job_status != "completed":
            raise SystemExit(1)
    finally:
        await client.close()


def _request_preview(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "workspace": args.workspace,
        "agent": args.target_agent,
        "default_model": args.default_model,
        "fast_model": args.fast_model,
        "timeout_seconds": args.timeout_seconds,
    }


async def _print_report(client: AsyncNeMoPlatform, base_url: str, workspace: str, job_name: str) -> None:
    """Print the saved analysis report, or explain why it is absent.

    Job results are an Agents/Jobs concern, so this reads them through
    ``client.agents.jobs.execute``. Only the byte download still needs raw
    HTTP: the jobs SDK exposes the result listing but not the file itself.
    """
    listing = await client.agents.jobs.execute.list_results(job_name, workspace=workspace)
    names = [result["name"] for result in listing["data"]]
    if REPORT_RESULT_NAME not in names:
        print(f"No '{REPORT_RESULT_NAME}' result was saved. Available results: {', '.join(names) or '(none)'}")
        return

    url = (
        f"{base_url.rstrip('/')}/apis/agents/v2/workspaces/{workspace}"
        f"/jobs/execute/{job_name}/results/{REPORT_RESULT_NAME}/download"
    )
    async with httpx.AsyncClient(timeout=60.0) as http:
        download = await http.get(url, follow_redirects=True)
    download.raise_for_status()
    print(f"\n--- {REPORT_RESULT_NAME} ---")
    print(download.text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--target-agent", default="demo-agent")
    parser.add_argument(
        "--default-model", required=True, help="Model Entity ref, e.g. default/nvidia-nemotron-3-super-120b-a12b."
    )
    parser.add_argument(
        "--fast-model", required=True, help="Model Entity ref, e.g. default/nvidia-nemotron-3-nano-30b-a3b."
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0, help="Execute-job Fabric timeout.")
    parser.add_argument("--wait", action="store_true", help="Poll the run and print its analysis report.")
    parser.add_argument("--poll-timeout", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
