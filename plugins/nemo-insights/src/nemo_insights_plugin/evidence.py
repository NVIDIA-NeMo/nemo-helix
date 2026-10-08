# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Insight evidence and compatibility with previously stored references."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class SpanEvidence(BaseModel):
    span_id: str = Field(min_length=1)
    url: str | None = None


class TraceEvidence(BaseModel):
    trace_id: str = Field(min_length=1)
    url: str | None = None
    spans: list[SpanEvidence] = Field(default_factory=list)


class EvidenceCompatibility(BaseModel):
    """Read legacy records while always writing the evidence representation."""

    @model_validator(mode="before")
    @classmethod
    def convert_legacy_evidence(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        refs = result.pop("trace_refs", []) or []
        links = result.pop("trace_links", {}) or {}
        span_refs = result.pop("span_refs", {}) or {}
        span_links = result.pop("span_links", {}) or {}
        if "evidence" in result:
            return result
        if not any(key in value for key in ("trace_refs", "trace_links", "span_refs", "span_links")):
            return result
        traces: dict[str, Any] = {ref: {"trace_id": ref, "url": links.get(ref), "spans": []} for ref in refs}
        for ref, url in links.items():
            traces.setdefault(ref, {"trace_id": ref, "url": url, "spans": []})
        # Legacy span references are grouped by trace ID. Also accept explicit
        # trace_id/span_id records, which retain the parent without a lookup.
        if isinstance(span_refs, list):
            grouped: dict[str, list] = {}
            for span in span_refs:
                grouped.setdefault(span["trace_id"], []).append(span)
            span_refs = grouped
        nested_links = {trace_id: urls for trace_id, urls in span_links.items() if isinstance(urls, dict)}
        for trace_id in dict.fromkeys([*span_refs, *nested_links]):
            trace = traces.setdefault(trace_id, {"trace_id": trace_id, "url": links.get(trace_id), "spans": []})
            urls = nested_links.get(trace_id, {})
            spans = span_refs.get(trace_id, [])
            for span in spans:
                item = dict(span) if isinstance(span, dict) else {"span_id": span}
                item.pop("trace_id", None)
                url = urls.get(item["span_id"], span_links.get(item["span_id"]))
                item.setdefault("url", url if isinstance(url, str) else None)
                trace["spans"].append(item)
            seen = {span["span_id"] for span in trace["spans"]}
            trace["spans"].extend({"span_id": sid, "url": url} for sid, url in urls.items() if sid not in seen)
        result["evidence"] = list(traces.values())
        return result


def merge_evidence(existing: list[TraceEvidence], added: list[TraceEvidence]) -> list[TraceEvidence]:
    """Preserve evidence order and saved links while adding traces and spans."""
    merged: dict[str, TraceEvidence] = {}
    for item in [*existing, *added]:
        if item.trace_id not in merged:
            merged[item.trace_id] = TraceEvidence(trace_id=item.trace_id)
        trace = merged[item.trace_id]
        trace.url = item.url or trace.url
        spans = {span.span_id: span for span in trace.spans}
        for span in item.spans:
            if span.span_id not in spans:
                spans[span.span_id] = span.model_copy(deep=True)
            else:
                spans[span.span_id].url = span.url or spans[span.span_id].url
        trace.spans = list(spans.values())
    return list(merged.values())
