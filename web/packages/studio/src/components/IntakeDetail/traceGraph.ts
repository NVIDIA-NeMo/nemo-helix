// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SpanStatus } from '@nemo/sdk/generated/platform/schema';
import type {
  DagEdge,
  DagNode,
  DagNodeData,
  DagNodeStatus,
} from '@studio/components/DagCanvas/types';
import {
  getSpanKindColorClass,
  getSpanKindConfig,
} from '@studio/components/SpanKindBadge/spanKindConfig';
import {
  compareSpansByStartedAt,
  formatDurationMs,
  formatInteger,
  getSpanDisplayName,
  getSpanDurationMs,
  type SpanTableRow,
} from '@studio/util/intakeTelemetry';

export type TraceGraphMode = 'grouped' | 'all';

export interface TraceGraphModel {
  nodes: DagNode[];
  edges: DagEdge[];
  spansByNode: ReadonlyMap<string, readonly SpanTableRow[]>;
  nodeBySpanId: ReadonlyMap<string, string>;
}

interface BuildTraceGraphOptions {
  highlightLongestPath?: boolean;
}

const graphStatus = (spans: SpanTableRow[]): DagNodeStatus => {
  if (spans.some((span) => span.status === SpanStatus.error)) return 'error';
  if (spans.some((span) => span.status === SpanStatus.cancelled)) return 'cancelled';
  if (spans.every((span) => span.status === SpanStatus.success)) return 'success';
  return 'unknown';
};

export const getTraceGraphDisplayName = (span: SpanTableRow): string => {
  const name = getSpanDisplayName(span);
  if (span.kind === 'AGENT' && /^user-\d+$/.test(name)) return 'User message';
  if (span.kind === 'AGENT' && /^system-\d+$/.test(name)) return 'System message';
  const operation = name.startsWith('mcp__') ? (name.split('__').at(-1) ?? name) : name;
  if (!/^[a-z0-9]+(?:[-_][a-z0-9]+)*$/.test(operation)) return operation;
  return operation
    .split(/[-_]/)
    .filter(Boolean)
    .map((word) => `${word[0]?.toUpperCase() ?? ''}${word.slice(1)}`)
    .join(' ');
};

const nodeData = (spans: SpanTableRow[], grouped: boolean): DagNodeData => {
  const representative = spans[0];
  const kind = getSpanKindConfig(representative.kind);
  const durationMs = spans.reduce((total, span) => total + (getSpanDurationMs(span) ?? 0), 0);
  const tokens = spans.reduce((total, span) => total + (span.total_tokens ?? 0), 0);
  const unit = representative.kind === 'TOOL' || representative.kind === 'LLM' ? 'call' : 'span';
  const countLabel = `${spans.length.toLocaleString()} ${unit}${spans.length === 1 ? '' : 's'}`;
  const errorCount = spans.filter((span) => span.status === SpanStatus.error).length;
  const cancelledCount = spans.filter((span) => span.status === SpanStatus.cancelled).length;
  const unknownCount = spans.filter((span) => span.status === SpanStatus.unknown).length;
  const statusSummary = [
    errorCount > 0 ? `${errorCount.toLocaleString()} failed` : null,
    cancelledCount > 0 ? `${cancelledCount.toLocaleString()} cancelled` : null,
    unknownCount > 0 ? `${unknownCount.toLocaleString()} unknown` : null,
  ].filter(Boolean);

  return {
    title: getTraceGraphDisplayName(representative),
    type: kind.label,
    description: grouped
      ? [countLabel, ...statusSummary].join(' · ')
      : formatDurationMs(durationMs),
    tags: tokens > 0 ? [`${formatInteger(tokens)} tokens`] : undefined,
    icon: kind.icon,
    status: graphStatus(spans),
    colorClassName: getSpanKindColorClass(representative.kind),
  };
};

const edgeId = (source: string, target: string): string => `${source}->${target}`;

const longestPath = (spans: SpanTableRow[]): string[] => {
  const spanById = new Map(spans.map((span) => [span.span_id, span]));
  const childrenByParent = new Map<string, SpanTableRow[]>();
  for (const span of spans) {
    if (!span.parent_span_id || !spanById.has(span.parent_span_id)) continue;
    const children = childrenByParent.get(span.parent_span_id) ?? [];
    children.push(span);
    childrenByParent.set(span.parent_span_id, children);
  }
  interface PathState {
    span: SpanTableRow;
    length: number;
    previous?: PathState;
  }

  const roots = spans.filter((span) => !span.parent_span_id || !spanById.has(span.parent_span_id));
  const stack = [...(roots.length > 0 ? roots : spans.slice(0, 1))]
    .reverse()
    .map((span): PathState => ({ span, length: 1 }));
  const visited = new Set<string>();
  let longest: PathState | undefined;

  while (stack.length > 0) {
    const path = stack.pop()!;
    if (visited.has(path.span.span_id)) continue;
    visited.add(path.span.span_id);
    const children = (childrenByParent.get(path.span.span_id) ?? []).filter(
      ({ span_id }) => !visited.has(span_id)
    );
    if (children.length === 0) {
      if (!longest || path.length > longest.length) longest = path;
      continue;
    }
    for (const child of [...children].reverse()) {
      stack.push({ span: child, length: path.length + 1, previous: path });
    }
  }

  const spanIds: string[] = [];
  for (let path = longest; path; path = path.previous) spanIds.push(path.span.span_id);
  return spanIds.reverse();
};

export const getMostTokensSpan = (spans: SpanTableRow[]): SpanTableRow | undefined => {
  let highest: SpanTableRow | undefined;
  for (const span of spans) {
    if (span.kind !== 'LLM' || !span.total_tokens || span.total_tokens <= 0) continue;
    if (!highest || span.total_tokens > (highest.total_tokens ?? 0)) highest = span;
  }
  return highest;
};

const buildAllSpansGraph = (
  spans: SpanTableRow[],
  { highlightLongestPath = false }: BuildTraceGraphOptions
): TraceGraphModel => {
  const spanIds = new Set(spans.map((span) => span.span_id));
  const longestPathIds = highlightLongestPath ? longestPath(spans) : [];
  const highlightedNodes = new Set(longestPathIds);
  const highlightedEdges = new Set(
    longestPathIds.slice(1).map((spanId, index) => edgeId(longestPathIds[index], spanId))
  );
  const nodes = spans.map((span) => ({
    id: span.span_id,
    data: {
      ...nodeData([span], false),
      ...(highlightLongestPath
        ? {
            highlighted: highlightedNodes.has(span.span_id),
          }
        : undefined),
    },
  }));
  const edges = spans.flatMap((span) =>
    span.parent_span_id && spanIds.has(span.parent_span_id)
      ? [
          {
            id: edgeId(span.parent_span_id, span.span_id),
            source: span.parent_span_id,
            target: span.span_id,
            ...(highlightLongestPath
              ? {
                  highlighted: highlightedEdges.has(edgeId(span.parent_span_id, span.span_id)),
                  muted: !highlightedEdges.has(edgeId(span.parent_span_id, span.span_id)),
                }
              : undefined),
          },
        ]
      : []
  );

  return {
    nodes,
    edges,
    spansByNode: new Map(spans.map((span) => [span.span_id, [span]])),
    nodeBySpanId: new Map(spans.map((span) => [span.span_id, span.span_id])),
  };
};

const operationIdentity = (span: SpanTableRow): string => {
  if (span.kind === 'AGENT' && /^user-\d+$/.test(span.name ?? '')) return 'AGENT:user-message';
  if (span.kind === 'AGENT' && /^system-\d+$/.test(span.name ?? '')) {
    return 'AGENT:system-message';
  }
  if (span.kind === 'TOOL') return JSON.stringify([span.kind, span.name, span.tool_name]);
  if (span.kind === 'LLM') {
    return JSON.stringify([span.kind, span.name, span.provider, span.model]);
  }
  return JSON.stringify([span.kind, span.name, span.agent_name]);
};

const groupKey = (span: SpanTableRow, spanById: ReadonlyMap<string, SpanTableRow>): string => {
  const parent = span.parent_span_id ? spanById.get(span.parent_span_id) : undefined;
  return JSON.stringify([
    span.hierarchyDepth,
    operationIdentity(span),
    parent ? operationIdentity(parent) : null,
  ]);
};

const buildGroupedGraph = (spans: SpanTableRow[]): TraceGraphModel => {
  const spanById = new Map(spans.map((span) => [span.span_id, span]));
  const groups = new Map<string, SpanTableRow[]>();
  for (const span of spans) {
    const key = groupKey(span, spanById);
    groups.set(key, [...(groups.get(key) ?? []), span]);
  }

  const nodeIdByGroup = new Map(
    [...groups.entries()].map(([key, groupedSpans]) => [key, `group-${groupedSpans[0].span_id}`])
  );
  const nodeBySpanId = new Map<string, string>();
  const spansByNode = new Map<string, readonly SpanTableRow[]>();
  const nodes = [...groups.entries()].map(([key, groupedSpans]) => {
    const id = nodeIdByGroup.get(key)!;
    groupedSpans.sort(compareSpansByStartedAt);
    groupedSpans.forEach((span) => nodeBySpanId.set(span.span_id, id));
    spansByNode.set(id, groupedSpans);
    return { id, data: nodeData(groupedSpans, true) };
  });

  const edgeCounts = new Map<string, { source: string; target: string; count: number }>();
  for (const span of spans) {
    const parent = span.parent_span_id ? spanById.get(span.parent_span_id) : undefined;
    if (!parent) continue;
    const source = nodeIdByGroup.get(groupKey(parent, spanById));
    const target = nodeIdByGroup.get(groupKey(span, spanById));
    if (!source || !target || source === target) continue;
    const id = edgeId(source, target);
    const edge = edgeCounts.get(id);
    edgeCounts.set(id, { source, target, count: (edge?.count ?? 0) + 1 });
  }
  const edges: DagEdge[] = [...edgeCounts.entries()].map(([id, edge]) => ({
    id,
    source: edge.source,
    target: edge.target,
    label: edge.count > 1 ? `${edge.count} links` : undefined,
  }));

  return { nodes, edges, spansByNode, nodeBySpanId };
};

export const buildTraceGraph = (
  spans: SpanTableRow[],
  mode: TraceGraphMode,
  options: BuildTraceGraphOptions = {}
): TraceGraphModel => {
  const uniqueSpans = [...new Map(spans.map((span) => [span.span_id, span])).values()];
  return mode === 'grouped'
    ? buildGroupedGraph(uniqueSpans)
    : buildAllSpansGraph(uniqueSpans, options);
};
