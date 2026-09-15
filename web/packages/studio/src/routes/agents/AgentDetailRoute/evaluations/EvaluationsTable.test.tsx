// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EvalJobRow } from '@studio/api/evaluation/utils';
import { EvaluationsTable } from '@studio/routes/agents/AgentDetailRoute/evaluations/EvaluationsTable';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { LG_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';

const makeEval = (overrides: Partial<AgentEvaluationRow> & { name: string }): AgentEvaluationRow =>
  ({
    experiments: [],
    created_at: '2024-12-17T16:08:56.880768',
    metadata: {},
    ...overrides,
  }) as AgentEvaluationRow;

const renderTable = (evaluations: AgentEvaluationRow[], jobs: EvalJobRow[] = []) =>
  render(
    <TestProviders>
      <MemoryRouter>
        <EvaluationsTable workspace="default" evaluations={evaluations} jobs={jobs} />
      </MemoryRouter>
    </TestProviders>
  );

describe('EvaluationsTable Duration column', () => {
  it('formats the recorded run duration from eval_duration_sec metadata', async () => {
    renderTable([makeEval({ name: 'eval-with-duration', metadata: { eval_duration_sec: '12' } })]);

    // StudioDataView renders the header label (column/filter menu) and each cell in more than one
    // place (a responsive duplicate view), so neither text is unique — assert presence, not count.
    expect(
      (await screen.findAllByText('Duration', undefined, { timeout: LG_SELECTOR_TIMEOUT })).length
    ).toBeGreaterThan(0);
    // 12s only comes from eval_duration_sec=12 -> 12000ms -> formatDurationMs -> "12s".
    expect(
      (await screen.findAllByText('12s', undefined, { timeout: LG_SELECTOR_TIMEOUT })).length
    ).toBeGreaterThan(0);
  });

  it('drops sub-second precision once the run reaches a minute', async () => {
    // 612.013s -> 612013ms -> "10m 12s" (the 13ms is dropped past a minute).
    renderTable([makeEval({ name: 'long-eval', metadata: { eval_duration_sec: '612.013' } })]);

    // Exact-match: if the 13ms were not dropped every cell would read "10m 12s 13ms" and match zero.
    expect(
      (await screen.findAllByText('10m 12s', undefined, { timeout: LG_SELECTOR_TIMEOUT })).length
    ).toBeGreaterThan(0);
  });
});

describe('EvaluationsTable Job column', () => {
  it('links to the generic job detail route, not the evaluation result route', async () => {
    const job: EvalJobRow = {
      id: 'job-1',
      name: 'my-eval-job',
      status: 'completed',
      created_at: '2024-12-17T16:08:56.880768',
      kind: 'dataset',
      agentName: 'my-agent',
      configLabel: null,
      evaluationName: null,
    };
    renderTable([], [job]);

    const links = await screen.findAllByRole(
      'link',
      { name: 'my-eval-job' },
      {
        timeout: LG_SELECTOR_TIMEOUT,
      }
    );
    expect(links.length).toBeGreaterThan(0);
    for (const link of links) {
      expect(link).toHaveAttribute('href', '/workspaces/default/jobs/my-eval-job');
    }
  });
});
