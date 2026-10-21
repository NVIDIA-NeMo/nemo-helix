// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Span, SpanKind, SpanStatus } from '@nemo/sdk/generated/platform/schema';
import {
  findReplySpan,
  findToolCallSpans,
} from '@studio/routes/EvaluationSessionDetailRoute/transcriptSpans';

const span = (
  span_id: string,
  kind: SpanKind,
  parent_span_id?: string,
  ended_at: string | null = '2026-10-16T00:01:00Z'
): Span => ({
  span_id,
  kind,
  parent_span_id,
  ended_at: ended_at ?? undefined,
  session_id: 'session',
  workspace: 'default',
  source: 'test',
  started_at: '2026-10-16T00:00:00Z',
  ingested_at: '2026-10-16T00:00:00Z',
  status: SpanStatus.success,
});

// Fabric DeepAgent shape: the outer agent's LLM calls a DeepAgent tool next to the real agent.
const fabricSpans = [
  span('outer', SpanKind.AGENT),
  span('outer-llm', SpanKind.LLM, 'outer'),
  span('handoff', SpanKind.TOOL, 'outer-llm', null),
  span('inner', SpanKind.AGENT, 'outer-llm'),
  span('inner-llm', SpanKind.LLM, 'inner'),
  span('get_demand', SpanKind.TOOL, 'inner-llm', null),
  span('user-turn', SpanKind.AGENT, 'inner', null),
];

describe('findReplySpan', () => {
  it('uses the only agent span', () => {
    const spans = [span('agent', SpanKind.AGENT), span('llm', SpanKind.LLM, 'agent')];
    expect(findReplySpan(spans)?.span_id).toBe('agent');
  });

  it('skips a wrapper agent for the deepest agent that finished', () => {
    expect(findReplySpan(fabricSpans)?.span_id).toBe('inner');
  });

  it('breaks ties by the agent that ended last', () => {
    const spans = [
      span('root', SpanKind.CHAIN),
      span('first', SpanKind.AGENT, 'root', '2026-10-16T00:01:00Z'),
      span('last', SpanKind.AGENT, 'root', '2026-10-16T00:02:00Z'),
    ];
    expect(findReplySpan(spans)?.span_id).toBe('last');
  });

  it('falls back to the root span without a finished agent', () => {
    const spans = [span('wrapper', SpanKind.CHAIN), span('tool', SpanKind.TOOL, 'wrapper')];
    expect(findReplySpan(spans)?.span_id).toBe('wrapper');
  });

  it('returns nothing without spans', () => {
    expect(findReplySpan([])).toBeUndefined();
  });
});

describe('findToolCallSpans', () => {
  it('leaves out tool spans that hand off to a sub-agent', () => {
    expect(findToolCallSpans(fabricSpans).map((tool) => tool.span_id)).toEqual(['get_demand']);
  });
});
