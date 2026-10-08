// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildEvaluationContextEntries,
  buildSessionHighlightMetrics,
  buildTraceHighlightMetrics,
  buildTraceSummaryEntries,
} from '@studio/components/IntakeDetail/IntakeComponents/traceKeyValues';
import { mockSessionById, mockTraceById } from '@studio/mocks/intake/telemetry';

describe('traceKeyValues', () => {
  it('builds trace summary entries without headline metrics', () => {
    const trace = mockTraceById('trace-agent-run-001');
    expect(trace).toBeDefined();

    const entries = buildTraceSummaryEntries(trace!, { workspace: 'default' });
    const labels = entries.map((entry) => entry.label);

    expect(labels).toEqual(expect.arrayContaining(['Name', 'Trace ID', 'Root Span', 'Session ID']));
    // Status/timing and token/cost values are surfaced in the header, not metadata.
    expect(labels).not.toEqual(
      expect.arrayContaining([
        'Started',
        'Ended',
        'Cached Tokens',
        'Input Cost',
        'Output Cost',
        'Spans',
        'Status',
        'Total Cost',
        'Total Tokens',
      ])
    );
  });

  it('leaves the agent out of the summary entries since the header links it', () => {
    const trace = mockTraceById('trace-agent-run-001');
    expect(trace).toBeDefined();

    const entries = buildTraceSummaryEntries(
      { ...trace!, agent_name: 'react-agent' },
      { workspace: 'default' }
    );

    expect(entries.map(({ id }) => id)).not.toContain('agent_name');
  });

  it('builds session headline metrics without a trace error count', () => {
    const session = mockSessionById('session-agent-run-001');
    expect(session).toBeDefined();

    const metrics = buildSessionHighlightMetrics(session!);

    expect(metrics.map(({ id }) => id)).toEqual([
      'span_count',
      'duration_ms',
      'total_tokens',
      'cost_usd',
    ]);
  });

  it('builds trace headline metrics from trace totals', () => {
    const trace = mockTraceById('trace-agent-run-003');
    expect(trace).toBeDefined();

    const metrics = buildTraceHighlightMetrics(trace!);

    expect(metrics.map(({ value }) => value)).toEqual(['2', '9s 130ms', '1,194', '$0.0021']);
  });

  it('includes evaluation context entries when present', () => {
    const trace = mockTraceById('trace-agent-run-001');
    expect(trace).toBeDefined();

    const entries = buildEvaluationContextEntries(trace!.evaluation_context);

    expect(entries.map((entry) => entry.label)).toEqual(['Evaluation Name', 'Test Case Name']);
  });

  it('returns no evaluation context entries when context is absent', () => {
    const trace = mockTraceById('trace-agent-run-002');
    expect(trace).toBeDefined();

    expect(buildEvaluationContextEntries(trace!.evaluation_context)).toEqual([]);
  });
});
