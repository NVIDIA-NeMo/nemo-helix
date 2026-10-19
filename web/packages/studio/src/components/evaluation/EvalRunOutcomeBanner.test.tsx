// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { evalRunOutcomeOf } from '@studio/api/evaluation/runOutcome';
import {
  EvalRunOutcomeBadge,
  EvalRunOutcomeBanner,
} from '@studio/components/evaluation/EvalRunOutcomeBanner';
import { render, screen } from '@testing-library/react';

const outcome = (overrides: Record<string, unknown>) => ({
  evaluation: {
    unit: 'trials',
    total: 3,
    errored: 0,
    scored: 3,
    failed: false,
    message: 'rollup message',
    ...overrides,
  },
});

describe('EvalRunOutcomeBanner', () => {
  it('shows the evaluator message as an error with the log action when the run failed', () => {
    render(
      <EvalRunOutcomeBanner
        statusDetails={outcome({ errored: 3, scored: 0, failed: true })}
        slotActions={<button type="button">Download Logs</button>}
      />
    );
    expect(screen.getByText('rollup message')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Download Logs' })).toBeInTheDocument();
  });

  it('warns without the log action when only some units scored', () => {
    render(
      <EvalRunOutcomeBanner
        statusDetails={outcome({ errored: 1, scored: 2 })}
        slotActions={<button type="button">Download Logs</button>}
      />
    );
    expect(screen.getByText('rollup message')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Download Logs' })).not.toBeInTheDocument();
  });

  it('renders nothing for a clean run or a job without the rollup', () => {
    const { container, rerender } = render(<EvalRunOutcomeBanner statusDetails={outcome({})} />);
    expect(container).toBeEmptyDOMElement();
    rerender(<EvalRunOutcomeBanner statusDetails={{ message: 'Job exited with code 1' }} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe('EvalRunOutcomeBadge', () => {
  it('summarises the scored count for a run with failures', () => {
    render(<EvalRunOutcomeBadge outcome={evalRunOutcomeOf(outcome({ errored: 1, scored: 2 }))} />);
    expect(screen.getByText('2 of 3 scored')).toBeInTheDocument();
  });

  it('renders nothing for a clean run', () => {
    const { container } = render(<EvalRunOutcomeBadge outcome={evalRunOutcomeOf(outcome({}))} />);
    expect(container).toBeEmptyDOMElement();
  });
});
