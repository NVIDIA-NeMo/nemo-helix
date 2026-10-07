// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentEvalResultSummary } from '@nemo/sdk/generated/evals/schema';
import { EvalMetricCoverageTable } from '@studio/components/evaluation/EvalMetricCoverageTable';
import { render, screen } from '@testing-library/react';

const summary = (overrides: Partial<AgentEvalResultSummary>): AgentEvalResultSummary => ({
  task_count: 2,
  trial_count: 3,
  score_count: 2,
  error_count: 0,
  metric_coverage: { judge: { judge: { total: 3, scored: 2, failed: 1, missing: 0 } } },
  ...overrides,
});

describe('EvalMetricCoverageTable', () => {
  it('reads naturally when exactly one trial errored', () => {
    render(<EvalMetricCoverageTable summary={summary({ error_count: 1 })} />);
    expect(
      screen.getByText('3 trials across 2 tasks; 1 trial reported an error.')
    ).toBeInTheDocument();
  });

  it('pluralises every count otherwise', () => {
    render(<EvalMetricCoverageTable summary={summary({ task_count: 1, error_count: 2 })} />);
    expect(
      screen.getByText('3 trials across 1 task; 2 trials reported errors.')
    ).toBeInTheDocument();
  });

  it('renders nothing without coverage rows', () => {
    const { container } = render(
      <EvalMetricCoverageTable summary={summary({ metric_coverage: {} })} />
    );
    expect(container).toBeEmptyDOMElement();
  });
});
