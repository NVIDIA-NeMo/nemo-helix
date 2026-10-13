// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChainSpanContent } from '@studio/components/IntakeDetail/SpanTemplates/ChainSpanContent';
import { TraceSpanTree } from '@studio/components/IntakeDetail/TraceDetailSpanTree';
import { mockSpanById, mockTraceById } from '@studio/mocks/intake/telemetry';
import { render, screen } from '@studio/tests/util/render';
import { buildSpanTree, type SessionTrajectory } from '@studio/util/intakeTelemetry';
import userEvent from '@testing-library/user-event';

const LONG_TRACE_NAME =
  'Answer a customer policy question with enough detail that the trajectory label is truncated';

const makeTrajectory = (): SessionTrajectory => {
  const trace = { ...mockTraceById('trace-agent-run-001')!, name: LONG_TRACE_NAME };
  const spans = [mockSpanById('span-root-001')!, mockSpanById('span-llm-001')!];
  return { trace, spans, spanTree: buildSpanTree(spans) };
};

describe('TraceSpanTree', () => {
  it('distinguishes a trace summary from its identically named root span', () => {
    const trajectory = makeTrajectory();
    const root = trajectory.spans[0]!;
    render(
      <TraceSpanTree
        trajectories={[{ ...trajectory, trace: { ...trajectory.trace, name: root.name } }]}
        activeSpanId={null}
        onSelectSpan={vi.fn()}
      />
    );
    expect(screen.getByTitle('View trace')).toHaveTextContent(`Trace: ${root.name}`);
    expect(screen.getByText(root.name!, { selector: 'span.truncate' })).toBeVisible();
  });

  it('shows recorded rollout duration separately from the derived timing basis', () => {
    const span = {
      ...mockSpanById('span-root-001')!,
      source: 'gym',
      raw_attributes: JSON.stringify({
        'gym.observed_duration_ms': 3334,
        'gym.timing': 'observed_child_window',
      }),
    };
    render(<ChainSpanContent span={span} workspace={span.workspace} />);
    expect(screen.getByText('Recorded rollout duration')).toBeVisible();
    expect(screen.getByText('3s 334ms')).toBeVisible();
    expect(screen.getByText('Observed child operations')).toBeVisible();
  });

  it('collapses trace and span branches while preserving branch selection', async () => {
    const user = userEvent.setup();
    const onSelectTrace = vi.fn();
    const onSelectSpan = vi.fn();
    render(
      <TraceSpanTree
        trajectories={[makeTrajectory()]}
        activeTraceId="trace-agent-run-001"
        activeSpanId={null}
        onSelectTrace={onSelectTrace}
        onSelectSpan={onSelectSpan}
      />
    );

    const traceTrigger = screen.getByTitle('View trace');
    const rootSpanLabel = screen.getByText('Answer customer policy question', {
      selector: 'span.truncate',
    });
    expect(traceTrigger).toHaveAttribute('data-state', 'open');
    expect(rootSpanLabel).toBeVisible();
    await user.click(traceTrigger);
    expect(traceTrigger).toHaveAttribute('data-state', 'closed');
    expect(rootSpanLabel).not.toBeVisible();
    expect(onSelectTrace).toHaveBeenCalledWith('trace-agent-run-001');

    await user.click(traceTrigger);
    const childSpanLabel = screen.getByText('Generate final response', {
      selector: 'span.truncate',
    });
    expect(childSpanLabel).toBeVisible();
    await user.click(rootSpanLabel);
    expect(childSpanLabel).not.toBeVisible();
    expect(onSelectSpan).toHaveBeenCalledWith('span-root-001', 'trace-agent-run-001');
  });

  it('shows the full trajectory label in a tooltip on hover', async () => {
    const user = userEvent.setup();
    render(
      <TraceSpanTree trajectories={[makeTrajectory()]} activeSpanId={null} onSelectSpan={vi.fn()} />
    );

    const traceLabel = screen.getByText(`Trace: ${LONG_TRACE_NAME}`, { selector: 'span.truncate' });
    await user.hover(traceLabel);
    const tooltip = await screen.findByRole('tooltip', { name: `Trace: ${LONG_TRACE_NAME}` });
    expect(tooltip).toHaveAttribute('data-state', 'open');
  });
});
