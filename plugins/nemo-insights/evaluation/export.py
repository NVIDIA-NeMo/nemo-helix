# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drain the intake read API into per-workspace JSONL files (bundle capture).

The capture side of evaluation export bundles: for each workspace, page ALL spans
(``mode="detailed"``), annotations, and evaluator results through the platform
SDK — exactly the surface the analyst's remote backend reads — and write one
JSON document per line under ``<out_dir>/export/<workspace>/``.

Every query carries an explicit lower bound (``since``, else epoch): the read
API silently injects a 30-day default lookback when none is given, so an
"unbounded" drain would quietly lose older spans.
"""

import asyncio
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO, cast
from urllib.parse import urlparse

from nemo_helix import AsyncNeMoHelix
from nemo_helix_ext.config.config import Config

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
PAGE_SIZE = 1000
TRACE_EXPORT_CONCURRENCY = 8

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def make_client(base_url: str) -> AsyncNeMoHelix:
    """Async SDK client; platform auth only for remote URLs (mirrors ingest's client)."""
    host = (urlparse(base_url).hostname or "").lower()
    config_path = Config.get_default_config_path()
    if host in _LOOPBACK_HOSTS or not config_path.exists():
        return AsyncNeMoHelix(base_url=base_url, timeout=60.0)
    return AsyncNeMoHelix(base_url=base_url, config_path=config_path, timeout=60.0)


def _dump(item) -> dict:
    """One SDK model -> plain JSON-able dict (drop null fields, like the analyst does)."""
    return item.model_dump(mode="json", exclude_none=True)


async def _drain_to_jsonl(paginator, path: Path, *, on_doc: Callable[[dict], None] | None = None) -> int:
    """Write every paginated doc to *path*, one JSON document per line; return the count."""
    with path.open("w", encoding="utf-8") as fh:
        return await _drain_to_stream(paginator, fh, on_doc=on_doc)


async def _drain_to_stream(paginator, stream: TextIO, *, on_doc: Callable[[dict], None] | None = None) -> int:
    """Append every paginated doc to an open JSONL stream."""
    count = 0
    async for item in paginator:
        doc = _dump(item)
        stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
        if on_doc is not None:
            on_doc(doc)
        count += 1
    return count


@dataclass(frozen=True)
class ExperimentScope:
    """The immutable trace/session membership selected from one Experiment."""

    experiment: str
    experiment_id: str
    evaluation_names: list[str]
    trace_ids: list[str]
    session_ids: list[str]
    expected_spans: int

    def manifest(self) -> dict:
        return {
            "kind": "experiment",
            "experiment": self.experiment,
            "experiment_id": self.experiment_id,
            "evaluation_names": self.evaluation_names,
            "trace_ids": self.trace_ids,
            "session_ids": self.session_ids,
            "expected_spans": self.expected_spans,
        }


class _StartBounds:
    """Min/max span ``started_at`` across everything exported (manifest time bounds)."""

    def __init__(self) -> None:
        self.min: datetime | None = None
        self.max: datetime | None = None

    def note(self, doc: dict) -> None:
        raw = doc.get("started_at")
        if not raw:
            return
        try:
            ts = datetime.fromisoformat(str(raw))
        except ValueError:
            return
        if self.min is None or ts < self.min:
            self.min = ts
        if self.max is None or ts > self.max:
            self.max = ts


def export_workspaces(
    base_url: str,
    workspaces: list[str],
    out_dir: Path,
    *,
    since: datetime | None,
    experiment: str | None = None,
    selection: dict | None = None,
    client: AsyncNeMoHelix | None = None,
) -> dict:
    """Drain spans/annotations/evaluator-results per workspace into JSONL files.

    Writes ``out_dir/export/<workspace>/{spans,annotations,evaluator_results}.jsonl``
    and returns ``{"workspaces": {ws: {collection: count}}, "min_start_time": ...,
    "max_start_time": ...}`` (time bounds from span ``started_at``; ISO strings or
    None when no spans matched).
    """
    return asyncio.run(
        _export_workspaces(
            base_url,
            workspaces,
            out_dir,
            since=since,
            experiment=experiment,
            selection=selection,
            client=client,
        )
    )


async def _resolve_experiment_scope(
    client: AsyncNeMoHelix,
    *,
    workspace: str,
    experiment_name: str,
    lower: str,
) -> ExperimentScope:
    """Resolve an Experiment to complete traces and their expected span count."""
    experiment = await client.experiments.retrieve(experiment_name, workspace=workspace)
    evaluation_names = sorted(
        [
            evaluation.name
            async for evaluation in client.evaluations.list(
                workspace=workspace,
                filter=cast(Any, {"experiment_id": experiment.id}),
                page_size=PAGE_SIZE,
                sort="name",
            )
        ]
    )
    if experiment.evaluation_count is not None and len(evaluation_names) != experiment.evaluation_count:
        raise RuntimeError(
            f"{workspace}/{experiment_name}: Experiment reports {experiment.evaluation_count} evaluations "
            f"but the membership query returned {len(evaluation_names)}"
        )

    traces: dict[str, tuple[str, int]] = {}
    for evaluation_name in evaluation_names:
        paginator = client.intake.traces.list(
            workspace=workspace,
            page_size=PAGE_SIZE,
            mode="preview",
            sort="started_at",
            filter=cast(Any, {"evaluation_id": evaluation_name, "started_at": {"gte": lower}}),
        )
        async for trace in paginator:
            if trace.span_count is None:
                raise RuntimeError(f"{workspace}/{trace.id}: preview response omitted span_count")
            value = (trace.session_id, trace.span_count)
            previous = traces.setdefault(trace.id, value)
            if previous != value:
                raise RuntimeError(f"{workspace}/{trace.id}: conflicting trace membership metadata")
    if not traces:
        raise RuntimeError(f"{workspace}/{experiment_name}: no traces matched the Experiment")

    trace_ids = sorted(traces)
    return ExperimentScope(
        experiment=experiment_name,
        experiment_id=experiment.id,
        evaluation_names=evaluation_names,
        trace_ids=trace_ids,
        session_ids=sorted({traces[trace_id][0] for trace_id in trace_ids}),
        expected_spans=sum(traces[trace_id][1] for trace_id in trace_ids),
    )


async def _export_scoped_workspace(
    client: AsyncNeMoHelix,
    *,
    workspace: str,
    ws_dir: Path,
    scope: ExperimentScope,
    bounds: _StartBounds,
) -> dict[str, int]:
    """Export complete selected traces and their session-level auxiliary rows."""
    epoch = EPOCH.isoformat()
    parts = ws_dir / ".span-parts"
    parts.mkdir()
    semaphore = asyncio.Semaphore(TRACE_EXPORT_CONCURRENCY)

    async def export_trace(index: int, trace_id: str) -> tuple[Path, int]:
        path = parts / f"{index:06d}.jsonl"
        async with semaphore:
            count = await _drain_to_jsonl(
                client.intake.spans.list(
                    workspace=workspace,
                    page_size=PAGE_SIZE,
                    mode="detailed",
                    sort="started_at",
                    filter=cast(Any, {"trace_id": trace_id, "started_at": {"gte": epoch}}),
                ),
                path,
                on_doc=bounds.note,
            )
        return path, count

    try:
        exports = await asyncio.gather(
            *(export_trace(index, trace_id) for index, trace_id in enumerate(scope.trace_ids))
        )
        with (ws_dir / "spans.jsonl").open("wb") as stream:
            for path, _count in exports:
                with path.open("rb") as part:
                    shutil.copyfileobj(part, stream)
    finally:
        shutil.rmtree(parts)
    n_spans = sum(count for _path, count in exports)
    if n_spans != scope.expected_spans:
        raise RuntimeError(
            f"{workspace}/{scope.experiment}: selected traces reported {scope.expected_spans} spans "
            f"but export returned {n_spans}; the source changed during capture"
        )

    async def drain_sessions(collection, path: Path) -> int:
        count = 0
        with path.open("w", encoding="utf-8") as stream:
            for session_id in scope.session_ids:
                count += await _drain_to_stream(
                    collection.list(
                        workspace=workspace,
                        page_size=PAGE_SIZE,
                        sort="created_at",
                        filter=cast(Any, {"session_id": session_id, "created_at": {"gte": epoch}}),
                    ),
                    stream,
                )
        return count

    n_annotations = await drain_sessions(client.intake.annotations, ws_dir / "annotations.jsonl")
    n_results = await drain_sessions(client.intake.evaluator_results, ws_dir / "evaluator_results.jsonl")
    return {"spans": n_spans, "annotations": n_annotations, "evaluator_results": n_results}


async def _export_workspaces(
    base_url: str,
    workspaces: list[str],
    out_dir: Path,
    *,
    since: datetime | None,
    experiment: str | None = None,
    selection: dict | None = None,
    client: AsyncNeMoHelix | None = None,
) -> dict:
    if (experiment is not None or selection is not None) and len(workspaces) != 1:
        raise ValueError("experiment-scoped export requires exactly one workspace")
    if experiment is not None and selection is not None:
        raise ValueError("pass experiment or selection, not both")
    lower = (since or EPOCH).isoformat()
    bounds = _StartBounds()
    counts: dict[str, dict[str, int]] = {}
    selections: dict[str, dict] = {}
    client = client if client is not None else make_client(base_url)
    try:
        for workspace in workspaces:
            ws_dir = out_dir / "export" / workspace
            ws_dir.mkdir(parents=True, exist_ok=True)
            scope = (
                await _resolve_experiment_scope(
                    client,
                    workspace=workspace,
                    experiment_name=experiment,
                    lower=lower,
                )
                if experiment is not None
                else ExperimentScope(
                    experiment=str(selection["experiment"]),
                    experiment_id=str(selection["experiment_id"]),
                    evaluation_names=list(selection["evaluation_names"]),
                    trace_ids=list(selection["trace_ids"]),
                    session_ids=list(selection["session_ids"]),
                    expected_spans=int(selection["expected_spans"]),
                )
                if selection is not None
                else None
            )
            if scope is not None:
                counts[workspace] = await _export_scoped_workspace(
                    client,
                    workspace=workspace,
                    ws_dir=ws_dir,
                    scope=scope,
                    bounds=bounds,
                )
                selections[workspace] = scope.manifest()
                workspace_counts = counts[workspace]
                print(
                    f"exported {workspace}/{scope.experiment}: {workspace_counts['spans']} spans, "
                    f"{workspace_counts['annotations']} annotations, "
                    f"{workspace_counts['evaluator_results']} evaluator results"
                )
                continue
            spans = client.intake.spans.list(
                workspace=workspace,
                page_size=PAGE_SIZE,
                mode="detailed",
                sort="started_at",
                filter=cast(Any, {"started_at": {"gte": lower}}),
            )
            n_spans = await _drain_to_jsonl(spans, ws_dir / "spans.jsonl", on_doc=bounds.note)
            annotations = client.intake.annotations.list(
                workspace=workspace,
                page_size=PAGE_SIZE,
                sort="created_at",
                filter=cast(Any, {"created_at": {"gte": lower}}),
            )
            n_annotations = await _drain_to_jsonl(annotations, ws_dir / "annotations.jsonl")
            results = client.intake.evaluator_results.list(
                workspace=workspace,
                page_size=PAGE_SIZE,
                sort="created_at",
                filter=cast(Any, {"created_at": {"gte": lower}}),
            )
            n_results = await _drain_to_jsonl(results, ws_dir / "evaluator_results.jsonl")
            counts[workspace] = {
                "spans": n_spans,
                "annotations": n_annotations,
                "evaluator_results": n_results,
            }
            print(f"exported {workspace}: {n_spans} spans, {n_annotations} annotations, {n_results} evaluator results")
    finally:
        await client.close()
    return {
        "workspaces": counts,
        "min_start_time": bounds.min.isoformat() if bounds.min else None,
        "max_start_time": bounds.max.isoformat() if bounds.max else None,
        "selections": selections,
    }
