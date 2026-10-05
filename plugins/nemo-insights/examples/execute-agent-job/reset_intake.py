# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove the demo telemetry written by ``seed_intake.py`` so the demo can start fresh.

Deletes, for one workspace:

- every annotation on a session that has spans from the demo source, through
  ``DELETE /apis/intake/v2/workspaces/{workspace}/annotations/{annotation_id}``;
- every span from the demo source, plus its ``trace_index`` rows, directly in
  ClickHouse. Intake has no public span delete API, and ``trace_index`` is
  filled by a materialized view on insert, so both tables need a delete.

Spans from any other source are left alone. Annotations carry no source, so
every annotation on a demo session is deleted, whoever created it.

ClickHouse is located the same way Intake locates it: ``NHX_INTAKE_CLICKHOUSE_URL``
(with ``_USER``, ``_PASSWORD``, ``_DATABASE``) when set, otherwise the running
container Intake provisioned locally in Docker.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import clickhouse_connect
import httpx
from clickhouse_connect.driver.client import Client

import docker

DEFAULT_SOURCE = "insights-demo"
MANAGED_CONTAINER_LABEL = "nhx.nvidia.com/component=intake-clickhouse"
CLICKHOUSE_HTTP_PORT_KEY = "8123/tcp"


def main() -> None:
    args = _parse_args()
    clickhouse = _connect_clickhouse()
    base_url = args.base_url.rstrip("/")
    params = {"workspace": args.workspace, "source": args.source}

    session_ids = [
        str(row[0])
        for row in clickhouse.query(
            "SELECT DISTINCT session_id FROM spans WHERE workspace = %(workspace)s AND source_format = %(source)s",
            parameters=params,
        ).result_rows
    ]
    span_count = _count(clickhouse, "spans", params)
    trace_count = _count(clickhouse, "trace_index", params)

    with httpx.Client(timeout=args.request_timeout) as client:
        annotation_ids = [
            annotation["annotation_id"]
            for session_id in session_ids
            for annotation in _list_annotations(client, base_url, args.workspace, session_id)
        ]
        print(
            f"Source '{args.source}' in workspace '{args.workspace}': {span_count} span row(s), "
            f"{trace_count} trace row(s), {len(annotation_ids)} annotation(s) across {len(session_ids)} session(s)."
        )
        if args.dry_run:
            print("Dry run; nothing deleted.")
            return

        for annotation_id in annotation_ids:
            response = client.delete(
                f"{base_url}/apis/intake/v2/workspaces/{args.workspace}/annotations/{annotation_id}"
            )
            if response.status_code not in (200, 204):
                raise SystemExit(f"Annotation delete failed ({response.status_code}): {response.text}")

    clickhouse.command(
        "DELETE FROM spans WHERE workspace = %(workspace)s AND source_format = %(source)s", parameters=params
    )
    clickhouse.command(
        "DELETE FROM trace_index WHERE workspace = %(workspace)s AND source_format = %(source)s", parameters=params
    )

    remaining = _count(clickhouse, "spans", params)
    if remaining:
        raise SystemExit(f"{remaining} span row(s) from '{args.source}' remain after the delete.")
    print(f"Deleted {len(annotation_ids)} annotation(s), {span_count} span row(s), and {trace_count} trace row(s).")


def _connect_clickhouse() -> Client:
    database = os.environ.get("NHX_INTAKE_CLICKHOUSE_DATABASE", "intake")
    url = os.environ.get("NHX_INTAKE_CLICKHOUSE_URL")
    if url:
        return clickhouse_connect.get_client(
            dsn=url,
            username=os.environ.get("NHX_INTAKE_CLICKHOUSE_USER", "default"),
            password=os.environ.get("NHX_INTAKE_CLICKHOUSE_PASSWORD", ""),
            database=database,
        )

    containers = docker.from_env().containers.list(filters={"label": MANAGED_CONTAINER_LABEL})
    if len(containers) != 1:
        raise SystemExit(
            f"Expected one running Intake ClickHouse container, found {len(containers)}. "
            "Start the platform, or set NHX_INTAKE_CLICKHOUSE_URL to target ClickHouse directly."
        )
    container = containers[0]
    bindings = (container.ports or {}).get(CLICKHOUSE_HTTP_PORT_KEY) or []
    if not bindings or not bindings[0].get("HostPort"):
        raise SystemExit(f"Container {container.name} does not publish the ClickHouse HTTP port.")
    env = dict(entry.split("=", 1) for entry in container.attrs["Config"]["Env"] if "=" in entry)
    return clickhouse_connect.get_client(
        host="127.0.0.1",
        port=int(bindings[0]["HostPort"]),
        username=env.get("CLICKHOUSE_USER", "default"),
        password=env.get("CLICKHOUSE_PASSWORD", ""),
        database=database,
    )


def _count(clickhouse: Client, table: str, params: dict[str, str]) -> int:
    result = clickhouse.query(
        f"SELECT count() FROM {table} FINAL WHERE workspace = %(workspace)s AND source_format = %(source)s",
        parameters=params,
    )
    return int(result.result_rows[0][0])


def _list_annotations(client: httpx.Client, base_url: str, workspace: str, session_id: str) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    page = 1
    while True:
        response = client.get(
            f"{base_url}/apis/intake/v2/workspaces/{workspace}/annotations",
            params={"filter[session_id]": session_id, "page": page, "page_size": 1000},
        )
        response.raise_for_status()
        body = response.json()
        annotations.extend(body["data"])
        if page >= body["pagination"]["total_pages"]:
            return annotations
        page += 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Intake source whose spans are removed.")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted without deleting.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
