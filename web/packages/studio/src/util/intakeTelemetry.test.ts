// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Span, SpanEvaluationContext } from '@nemo/sdk/generated/platform/schema';
import { SpanKind } from '@nemo/sdk/generated/platform/schema/SpanKind';
import { SpanStatus } from '@nemo/sdk/generated/platform/schema/SpanStatus';
import {
  buildSpanHierarchyRows,
  buildSpanTree,
  compareSpansByStartedAt,
  formatCost,
  getEvaluationContextSummary,
  getSpansDurationMs,
  hasEvaluationContext,
  type SpanTreeNode,
} from '@studio/util/intakeTelemetry';

const makeSpan = (span: Partial<Span> & Pick<Span, 'span_id' | 'started_at'>): Span => ({
  session_id: 'session-1',
  workspace: 'default',
  kind: SpanKind.AGENT,
  source: 'otel',
  status: SpanStatus.success,
  ingested_at: '2026-05-20T00:00:00Z',
  ...span,
});

describe('evaluation context helpers', () => {
  it('prefers canonical names', () => {
    const context: SpanEvaluationContext = {
      evaluation_name: 'new-evaluation',
      test_case_name: 'new-test-case',
      evaluation_id: 'deprecated-evaluation',
      test_case_id: 'deprecated-test-case',
    };

    expect(getEvaluationContextSummary(context)).toBe('new-evaluation');
    expect(hasEvaluationContext(context)).toBe(true);
  });

  it('continues to display deprecated fields during the compatibility window', () => {
    const context: SpanEvaluationContext = { evaluation_id: 'legacy-evaluation' };

    expect(getEvaluationContextSummary(context)).toBe('legacy-evaluation');
    expect(hasEvaluationContext(context)).toBe(true);
  });
});

describe('intakeTelemetry span hierarchy helpers', () => {
  it('formats sub-cent costs without trailing zero padding', () => {
    expect(formatCost(0.0032)).toBe('$0.0032');
  });

  it('sorts spans by started_at while preserving server order for ties', () => {
    const first = makeSpan({ span_id: 'span-z', started_at: '2026-05-20T00:00:00Z' });
    const second = makeSpan({ span_id: 'span-a', started_at: '2026-05-20T00:00:00Z' });
    const c = makeSpan({ span_id: 'span-c', started_at: '2026-05-20T00:00:01Z' });

    expect([c, first, second].sort(compareSpansByStartedAt).map((span) => span.span_id)).toEqual([
      'span-z',
      'span-a',
      'span-c',
    ]);
  });

  it('builds nested rows by parent_span_id', () => {
    const rows = buildSpanHierarchyRows([
      makeSpan({
        span_id: 'child-2',
        parent_span_id: 'root',
        started_at: '2026-05-20T00:00:03Z',
      }),
      makeSpan({ span_id: 'root', started_at: '2026-05-20T00:00:01Z' }),
      makeSpan({
        span_id: 'grandchild',
        parent_span_id: 'child-1',
        started_at: '2026-05-20T00:00:04Z',
      }),
      makeSpan({
        span_id: 'child-1',
        parent_span_id: 'root',
        started_at: '2026-05-20T00:00:02Z',
      }),
    ]);

    expect(rows.map((row) => [row.span_id, row.hierarchyDepth])).toEqual([
      ['root', 0],
      ['child-1', 1],
      ['grandchild', 2],
      ['child-2', 1],
    ]);
  });

  it('keeps multiple roots ordered by start time', () => {
    const rows = buildSpanHierarchyRows([
      makeSpan({ span_id: 'root-2', started_at: '2026-05-20T00:00:03Z' }),
      makeSpan({ span_id: 'root-1', started_at: '2026-05-20T00:00:01Z' }),
      makeSpan({
        span_id: 'child-1',
        parent_span_id: 'root-1',
        started_at: '2026-05-20T00:00:02Z',
      }),
    ]);

    expect(rows.map((row) => [row.span_id, row.hierarchyDepth])).toEqual([
      ['root-1', 0],
      ['child-1', 1],
      ['root-2', 0],
    ]);
  });

  it('keeps spans visible when their parent is not in the current page', () => {
    const rows = buildSpanHierarchyRows([
      makeSpan({
        span_id: 'orphan-child',
        parent_span_id: 'missing-parent',
        started_at: '2026-05-20T00:00:01Z',
      }),
    ]);

    expect(rows.map((row) => [row.span_id, row.hierarchyDepth, row.hierarchyStatus])).toEqual([
      ['orphan-child', 0, 'parent_outside_page'],
    ]);
  });

  it('keeps cyclic spans visible with an unresolved hierarchy marker', () => {
    const rows = buildSpanHierarchyRows([
      makeSpan({
        span_id: 'span-1',
        parent_span_id: 'span-2',
        started_at: '2026-05-20T00:00:01Z',
      }),
      makeSpan({
        span_id: 'span-2',
        parent_span_id: 'span-1',
        started_at: '2026-05-20T00:00:02Z',
      }),
    ]);

    expect(rows.map((row) => [row.span_id, row.hierarchyDepth, row.hierarchyStatus])).toEqual([
      ['span-1', 0, 'cycle_or_unreachable'],
      ['span-2', 1, undefined],
    ]);
  });
});

const flattenTree = (nodes: SpanTreeNode[]): [string, number][] =>
  nodes.flatMap((node) => [
    [node.span.span_id, node.depth] as [string, number],
    ...flattenTree(node.children),
  ]);

describe('buildSpanTree', () => {
  it('preserves server order when sibling timestamps match', () => {
    const tree = buildSpanTree([
      makeSpan({ span_id: 'root', started_at: '2026-05-20T00:00:00Z' }),
      makeSpan({
        span_id: 'server-first-sorts-later',
        parent_span_id: 'root',
        started_at: '2026-05-20T00:00:01Z',
      }),
      makeSpan({
        span_id: 'server-second-sorts-first',
        parent_span_id: 'root',
        started_at: '2026-05-20T00:00:01Z',
      }),
    ]);

    expect(tree[0].children.map((node) => node.span.span_id)).toEqual([
      'server-first-sorts-later',
      'server-second-sorts-first',
    ]);
  });

  it('nests spans by parent_span_id, ordered by start time', () => {
    const tree = buildSpanTree([
      makeSpan({ span_id: 'child-2', parent_span_id: 'root', started_at: '2026-05-20T00:00:03Z' }),
      makeSpan({ span_id: 'root', started_at: '2026-05-20T00:00:01Z' }),
      makeSpan({
        span_id: 'grandchild',
        parent_span_id: 'child-1',
        started_at: '2026-05-20T00:00:04Z',
      }),
      makeSpan({ span_id: 'child-1', parent_span_id: 'root', started_at: '2026-05-20T00:00:02Z' }),
    ]);

    expect(tree).toHaveLength(1);
    expect(flattenTree(tree)).toEqual([
      ['root', 0],
      ['child-1', 1],
      ['grandchild', 2],
      ['child-2', 1],
    ]);
  });

  it('keeps multiple roots ordered by start time', () => {
    const tree = buildSpanTree([
      makeSpan({ span_id: 'root-2', started_at: '2026-05-20T00:00:03Z' }),
      makeSpan({ span_id: 'root-1', started_at: '2026-05-20T00:00:01Z' }),
      makeSpan({
        span_id: 'child-1',
        parent_span_id: 'root-1',
        started_at: '2026-05-20T00:00:02Z',
      }),
    ]);

    expect(tree.map((node) => node.span.span_id)).toEqual(['root-1', 'root-2']);
    expect(tree[0].children.map((node) => node.span.span_id)).toEqual(['child-1']);
  });

  it('surfaces spans whose parent is outside the current page', () => {
    const tree = buildSpanTree([
      makeSpan({
        span_id: 'orphan-child',
        parent_span_id: 'missing-parent',
        started_at: '2026-05-20T00:00:01Z',
      }),
    ]);

    expect(tree).toEqual([
      expect.objectContaining({ depth: 0, hierarchyStatus: 'parent_outside_page', children: [] }),
    ]);
    expect(tree[0].span.span_id).toBe('orphan-child');
  });

  it('surfaces cyclic spans with an unresolved-hierarchy marker', () => {
    const tree = buildSpanTree([
      makeSpan({ span_id: 'span-1', parent_span_id: 'span-2', started_at: '2026-05-20T00:00:01Z' }),
      makeSpan({ span_id: 'span-2', parent_span_id: 'span-1', started_at: '2026-05-20T00:00:02Z' }),
    ]);

    expect(flattenTree(tree)).toEqual([
      ['span-1', 0],
      ['span-2', 1],
    ]);
    expect(tree[0].hierarchyStatus).toBe('cycle_or_unreachable');
  });
});

describe('getSpansDurationMs', () => {
  it('spans the earliest start to the latest end', () => {
    expect(
      getSpansDurationMs([
        makeSpan({
          span_id: 'a',
          started_at: '2026-05-20T00:00:00Z',
          ended_at: '2026-05-20T00:00:02Z',
        }),
        makeSpan({
          span_id: 'b',
          started_at: '2026-05-20T00:00:01Z',
          ended_at: '2026-05-20T00:00:05Z',
        }),
      ])
    ).toBe(5000);
  });

  it('returns undefined when no usable timestamps exist', () => {
    expect(getSpansDurationMs([])).toBeUndefined();
  });
});
