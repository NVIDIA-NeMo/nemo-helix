// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The plugin's own narrow span model. Components depend on this, not on Phoenix's
// wire format, so the backend can change without touching the UI.

export type SpanStatus = 'OK' | 'ERROR' | 'UNSET';

export interface Span {
  id: string;
  traceId: string;
  parentId: string | null;
  name: string;
  kind: string;
  /** Epoch milliseconds. */
  start: number;
  end: number;
  status: SpanStatus;
  statusMessage: string;
  attributes: Record<string, unknown>;
}

export interface TraceSummary {
  traceId: string;
  name: string;
  start: number;
  durationMs: number;
  status: SpanStatus;
  spanCount: number;
  tokens: number | null;
}

export interface SpanRow {
  span: Span;
  depth: number;
}

/** Phoenix-shaped item returned by the external Intake adapter. */
export interface PhoenixSpan {
  name: string;
  context: { trace_id: string; span_id: string };
  span_kind: string;
  parent_id: string | null;
  start_time: string;
  end_time: string | null;
  status_code: string;
  status_message: string;
  attributes: Record<string, unknown>;
}

export const fromPhoenix = (s: PhoenixSpan): Span => ({
  id: s.context.span_id,
  traceId: s.context.trace_id,
  parentId: s.parent_id,
  name: s.name,
  kind: s.span_kind,
  start: Date.parse(s.start_time),
  // An unfinished Intake span runs until this fetch's observation time.
  end: s.end_time === null ? Math.max(Date.parse(s.start_time), Date.now()) : Date.parse(s.end_time),
  status: s.status_code === 'OK' || s.status_code === 'ERROR' ? s.status_code : 'UNSET',
  statusMessage: s.status_message,
  attributes: s.attributes,
});

export const groupTraces = (spans: Span[]): TraceSummary[] => {
  const byTrace = new Map<string, Span[]>();
  for (const s of spans) byTrace.set(s.traceId, [...(byTrace.get(s.traceId) ?? []), s]);

  return [...byTrace.entries()]
    .map(([traceId, group]) => {
      const ids = new Set(group.map((s) => s.id));
      const roots = group.filter((s) => s.parentId === null || !ids.has(s.parentId));
      const root = roots.sort((a, b) => a.start - b.start)[0] ?? group[0];
      const start = Math.min(...group.map((s) => s.start));
      const end = Math.max(...group.map((s) => s.end));
      const tokens = group.reduce<number | null>((sum, s) => {
        const n = s.attributes['llm.token_count.total'];
        return typeof n === 'number' ? (sum ?? 0) + n : sum;
      }, null);
      return {
        traceId,
        name: root.name,
        start,
        durationMs: end - start,
        status: group.some((s) => s.status === 'ERROR') ? ('ERROR' as const) : root.status,
        spanCount: group.length,
        tokens,
      };
    })
    .sort((a, b) => b.start - a.start);
};

/** Depth-first rows (children ordered by start time) for rendering a span tree. */
export const toRows = (spans: Span[]): SpanRow[] => {
  const ids = new Set(spans.map((s) => s.id));
  const children = new Map<string | null, Span[]>();
  for (const s of spans) {
    const parent = s.parentId !== null && ids.has(s.parentId) ? s.parentId : null;
    children.set(parent, [...(children.get(parent) ?? []), s]);
  }
  const rows: SpanRow[] = [];
  const walk = (parent: string | null, depth: number) => {
    for (const span of (children.get(parent) ?? []).sort((a, b) => a.start - b.start)) {
      rows.push({ span, depth });
      walk(span.id, depth + 1);
    }
  };
  walk(null, 0);
  return rows;
};

export const formatDuration = (ms: number): string =>
  ms < 1000 ? `${Math.round(ms)} ms` : ms < 60_000 ? `${(ms / 1000).toFixed(1)} s` : `${(ms / 60_000).toFixed(1)} min`;
