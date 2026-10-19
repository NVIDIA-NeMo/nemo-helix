// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SpanKind, SpanStatus } from '@nemo/sdk/generated/platform/schema';
import { buildTraceGraph, getMostTokensSpan } from '@studio/components/IntakeDetail/traceGraph';
import type { SpanTableRow } from '@studio/util/intakeTelemetry';

const makeSpan = (spanId: string, name: string, parentSpanId?: string): SpanTableRow => ({
  span_id: spanId,
  session_id: 'session-1',
  workspace: 'default',
  parent_span_id: parentSpanId,
  kind: SpanKind.TOOL,
  name,
  source: 'otel',
  trace_id: 'trace-1',
  started_at: '2026-08-21T12:00:00Z',
  ended_at: '2026-08-21T12:00:01Z',
  status: SpanStatus.success,
  total_tokens: 10,
  ingested_at: '2026-08-21T12:00:02Z',
  hierarchyDepth: parentSpanId ? 1 : 0,
});

describe('buildTraceGraph', () => {
  it('keeps every span and parent relationship in all spans mode', () => {
    const spans = [makeSpan('root', 'Agent'), makeSpan('child', 'Search', 'root')];

    const graph = buildTraceGraph(spans, 'all');

    expect(graph.nodes.map(({ id }) => id)).toEqual(['root', 'child']);
    expect(graph.edges).toEqual([{ id: 'root->child', source: 'root', target: 'child' }]);
    expect(graph.spansByNode.get('child')).toEqual([spans[1]]);
  });

  it('highlights the root-to-leaf path containing the most spans', () => {
    const root = makeSpan('root', 'Agent');
    const slowShallow = {
      ...makeSpan('slow-shallow', 'Slow check', 'root'),
      ended_at: '2026-08-21T12:00:03Z',
    };
    const deepParent = {
      ...makeSpan('deep-parent', 'First deep step', 'root'),
      ended_at: '2026-08-21T12:00:00.050Z',
    };
    const deepChild = {
      ...makeSpan('deep-child', 'Second deep step', 'deep-parent'),
      ended_at: '2026-08-21T12:00:00.100Z',
      hierarchyDepth: 2,
    };

    const graph = buildTraceGraph([root, slowShallow, deepParent, deepChild], 'all', {
      highlightLongestPath: true,
    });

    expect(graph.nodes.find(({ id }) => id === 'root')?.data.highlighted).toBe(true);
    expect(graph.nodes.find(({ id }) => id === 'slow-shallow')?.data.highlighted).toBe(false);
    expect(graph.nodes.find(({ id }) => id === 'deep-parent')?.data.highlighted).toBe(true);
    expect(graph.nodes.find(({ id }) => id === 'deep-child')?.data.highlighted).toBe(true);
    expect(graph.edges.find(({ target }) => target === 'deep-child')?.highlighted).toBe(true);
    expect(graph.edges.find(({ target }) => target === 'slow-shallow')?.muted).toBe(true);
  });

  it('calculates the longest path without recursion for a maximum-size trace page', () => {
    const spans = Array.from({ length: 1000 }, (_, index) =>
      makeSpan(`span-${index}`, `Step ${index}`, index === 0 ? undefined : `span-${index - 1}`)
    );

    const graph = buildTraceGraph(spans, 'all', { highlightLongestPath: true });

    expect(graph.nodes.filter(({ data }) => data.highlighted)).toHaveLength(1000);
    expect(graph.edges.filter(({ highlighted }) => highlighted)).toHaveLength(999);
  });

  it('ignores duplicate span summaries', () => {
    const root = makeSpan('root', 'Agent');
    const child = makeSpan('child', 'Search', 'root');

    const graph = buildTraceGraph([root, child, child], 'all');

    expect(graph.nodes).toHaveLength(2);
    expect(graph.edges).toHaveLength(1);
  });

  it('groups repeated operations and counts their relationships', () => {
    const spans = [
      makeSpan('root', 'Agent'),
      makeSpan('search-1', 'Search', 'root'),
      makeSpan('search-2', 'Search', 'root'),
    ];

    const graph = buildTraceGraph(spans, 'grouped');
    const searchNode = graph.nodes.find(({ data }) => data.title === 'Search');

    expect(graph.nodes).toHaveLength(2);
    expect(graph.edges).toHaveLength(1);
    expect(graph.edges[0].label).toBe('2 links');
    expect(searchNode?.data.description).toBe('2 calls');
    expect(searchNode?.data.tags).toEqual(['20 tokens']);
    expect(graph.nodeBySpanId.get('search-1')).toBe(searchNode?.id);
    expect(graph.nodeBySpanId.get('search-2')).toBe(searchNode?.id);
    expect(graph.spansByNode.get(searchNode?.id ?? '')).toEqual([spans[1], spans[2]]);
  });

  it('shows the number of failed calls in a mixed group', () => {
    const success = makeSpan('search-1', 'Search');
    const failure = { ...makeSpan('search-2', 'Search'), status: SpanStatus.error };

    const graph = buildTraceGraph([success, failure], 'grouped');

    expect(graph.nodes).toHaveLength(1);
    expect(graph.nodes[0].data.description).toBe('2 calls · 1 failed');
    expect(graph.nodes[0].data.status).toBe('error');
  });

  it('keeps cancelled and unknown statuses visible', () => {
    const cancelled = { ...makeSpan('cancelled', 'Search'), status: SpanStatus.cancelled };
    const unknown = { ...makeSpan('unknown', 'Search'), status: SpanStatus.unknown };

    const graph = buildTraceGraph([cancelled, unknown], 'grouped');

    expect(graph.nodes[0].data.description).toBe('2 calls · 1 cancelled · 1 unknown');
    expect(graph.nodes[0].data.status).toBe('cancelled');
    expect(buildTraceGraph([unknown], 'grouped').nodes[0].data.status).toBe('unknown');
  });

  it('uses readable labels for ATIF messages and MCP tools', () => {
    const user = { ...makeSpan('user', 'user-1'), kind: SpanKind.AGENT };
    const tool = makeSpan('tool', 'mcp__documents__calculate_total', 'user');
    const agent = { ...makeSpan('agent', 'document-review'), kind: SpanKind.AGENT };

    const graph = buildTraceGraph([user, tool, agent], 'grouped');

    expect(graph.nodes.map(({ data }) => data.title)).toEqual([
      'User message',
      'Calculate Total',
      'Document Review',
    ]);
  });

  it('keeps same-named tools from different servers separate', () => {
    const first = { ...makeSpan('first', 'mcp__server_a__search'), tool_name: 'search' };
    const second = { ...makeSpan('second', 'mcp__server_b__search'), tool_name: 'search' };

    const graph = buildTraceGraph([first, second], 'grouped');

    expect(graph.nodes).toHaveLength(2);
    expect(graph.nodes.map(({ data }) => data.title)).toEqual(['Search', 'Search']);
  });

  it('keeps recursive operations at different depths separate', () => {
    const outer = makeSpan('outer', 'Review');
    const search = makeSpan('search', 'Search', 'outer');
    const inner = { ...makeSpan('inner', 'Review', 'search'), hierarchyDepth: 2 };

    const graph = buildTraceGraph([outer, search, inner], 'grouped');

    expect(graph.nodes).toHaveLength(3);
    expect(graph.edges).toHaveLength(2);
    expect(graph.edges.every(({ source, target }) => source !== target)).toBe(true);
  });

  it('keeps a span whose parent is outside the loaded page', () => {
    const orphan = makeSpan('orphan', 'Search', 'missing');

    const graph = buildTraceGraph([orphan], 'all');

    expect(graph.nodes).toHaveLength(1);
    expect(graph.edges).toHaveLength(0);
  });

  it('finds the individual LLM span with the most tokens', () => {
    const rollup = { ...makeSpan('root', 'Agent'), kind: SpanKind.AGENT, total_tokens: 10_000 };
    const smaller = { ...makeSpan('small', 'Draft', 'root'), kind: SpanKind.LLM, total_tokens: 80 };
    const highest = {
      ...makeSpan('high', 'Answer', 'root'),
      kind: SpanKind.LLM,
      total_tokens: 120,
    };
    const tool = { ...makeSpan('tool', 'Search', 'root'), total_tokens: 500 };

    expect(getMostTokensSpan([rollup, smaller, highest, tool])).toBe(highest);
  });

  it('returns no most-tokens span when LLM usage is unavailable', () => {
    const missing = {
      ...makeSpan('missing', 'Draft'),
      kind: SpanKind.LLM,
      total_tokens: undefined,
    };
    const zero = { ...makeSpan('zero', 'Answer'), kind: SpanKind.LLM, total_tokens: 0 };

    expect(getMostTokensSpan([missing, zero])).toBeUndefined();
  });
});
