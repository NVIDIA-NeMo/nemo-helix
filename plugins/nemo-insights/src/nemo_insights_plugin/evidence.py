# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Insight evidence and compatibility with previously stored references."""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class SpanEvidence(BaseModel):
    span_id: str = Field(min_length=1)
    url: str | None = None


class TraceEvidence(BaseModel):
    trace_id: str = Field(min_length=1)
    url: str | None = None
    spans: list[SpanEvidence] = Field(default_factory=list)


class EvidenceCompatibility(BaseModel):
    """Read legacy records while always writing the evidence representation."""

    @field_validator("evidence", check_fields=False)
    @classmethod
    def normalize_evidence(cls, value: list[TraceEvidence] | None) -> list[TraceEvidence] | None:
        return merge_evidence([], value) if value is not None else None

    @model_validator(mode="before")
    @classmethod
    def convert_legacy_evidence(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        legacy_fields = ("trace_refs", "trace_links", "span_refs", "span_links")
        result = dict(value)
        legacy = {key: result.pop(key) for key in legacy_fields if key in result}
        # Explicit evidence wins, including an empty list. An unrelated PATCH
        # must leave evidence unset rather than replacing it with an empty list.
        if "evidence" not in result and legacy:
            result["evidence"] = _convert_legacy_references(legacy)
        return result


def _group_legacy_span_refs(span_refs: dict[str, list[Any]] | list[Any]) -> dict[str, list[Any]]:
    """Accept trace-to-span mappings or flat records with explicit parent IDs."""
    if isinstance(span_refs, dict):
        return span_refs
    grouped: dict[str, list[Any]] = {}
    for span in span_refs:
        if not isinstance(span, dict):
            continue
        trace_id = span.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id.strip():
            continue
        grouped.setdefault(trace_id, []).append(span)
    return grouped


def _convert_legacy_spans(
    refs: list[Any], links_for_trace: dict[str, Any], all_links: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach span URLs, preferring explicit URLs over grouped or flat links."""
    spans = []
    for ref in refs:
        span = dict(ref) if isinstance(ref, dict) else {"span_id": ref}
        span.pop("trace_id", None)
        span_id = span["span_id"]
        url = links_for_trace.get(span_id, all_links.get(span_id))
        span.setdefault("url", url if isinstance(url, str) else None)
        spans.append(span)

    # Nested links retain their parent, so they can supply otherwise unlisted spans.
    referenced_ids = {span["span_id"] for span in spans}
    for span_id, url in links_for_trace.items():
        if span_id not in referenced_ids:
            spans.append({"span_id": span_id, "url": url})
    return spans


def _convert_legacy_references(legacy: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one evidence entry per trace, retaining the original reference order."""
    trace_refs = legacy.get("trace_refs") or []
    trace_links = legacy.get("trace_links") or {}
    span_refs = _group_legacy_span_refs(legacy.get("span_refs") or {})
    span_links = legacy.get("span_links") or {}
    links_by_trace = {trace_id: links for trace_id, links in span_links.items() if isinstance(links, dict)}
    trace_ids = dict.fromkeys([*trace_refs, *trace_links, *span_refs, *links_by_trace])
    return [
        {
            "trace_id": trace_id,
            "url": trace_links.get(trace_id),
            "spans": _convert_legacy_spans(span_refs.get(trace_id, []), links_by_trace.get(trace_id, {}), span_links),
        }
        for trace_id in trace_ids
    ]


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
