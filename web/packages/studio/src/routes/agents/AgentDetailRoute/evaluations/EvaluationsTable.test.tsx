// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getListExperimentsQueryKey } from '@nemo/sdk/generated/platform/experiments';
import { type EvalJobRow, toEvalJobRow } from '@studio/api/evaluation/utils';
import { DEFAULT_EXPERIMENT_AGENT_KEY } from '@studio/components/evaluation/defaultExperiment';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { makeEvaluatorJob } from '@studio/mocks/evaluation/jobs';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { EvaluationsTable } from '@studio/routes/agents/AgentDetailRoute/evaluations/EvaluationsTable';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { getAgentRunEvaluationRoute } from '@studio/routes/utils';
import { LG_SELECTOR_TIMEOUT, LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute, waitFor, within } from '@studio/tests/util/render';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
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
  it('keeps the seventh publication joined to its job and recorded results', async () => {
    const jobs = Array.from({ length: 7 }, (_, index) =>
      toEvalJobRow(
        makeEvaluatorJob({
          id: `job-${index}`,
          name: `job-${index}`,
          spec: {
            target: { kind: 'fabric', source: { agent: 'my-agent' } },
            publication: { intake: { evaluation_id: `published-${index}` } },
          },
        })
      )
    );
    renderTable(
      jobs.map((_, index) =>
        makeEval({
          name: `published-${index}`,
          // Entity id deliberately differs from the publication name used for the join.
          id: `entity-${index}`,
          aggregate_scores: { 'judge.accuracy': { mean: 0.83, count: 1, failed_count: 0 } },
          tokens: { mean: 1234, sum: 2468, count: 2 },
        })
      ),
      jobs
    );

    const links = await screen.findAllByRole(
      'link',
      { name: 'job-6' },
      { timeout: LG_SELECTOR_TIMEOUT }
    );
    for (const link of links) {
      expect(link).toHaveAttribute('href', '/workspaces/default/jobs/job-6');
    }
    expect((await screen.findAllByText('0.83')).length).toBeGreaterThan(0);
    expect(screen.getAllByText('1,234').length).toBeGreaterThan(0);
    expect(screen.getAllByText('2,468').length).toBeGreaterThan(0);
  });

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
      outcome: null,
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

describe('EvaluationsTable Scores column', () => {
  it('shows how many attempts failed to score beside the mean they are counted in', async () => {
    renderTable([
      makeEval({
        name: 'eval-with-failures',
        aggregate_scores: {
          'judge.accuracy': { mean: 0.5, count: 4, failed_count: 2 },
          'exact.match': { mean: 1, count: 4, failed_count: 0 },
        },
      }),
    ]);

    expect(
      (await screen.findAllByText('(2 failed)', undefined, { timeout: LG_SELECTOR_TIMEOUT })).length
    ).toBeGreaterThan(0);
    // A clean evaluator gets no failure note.
    expect(screen.queryByText('(0 failed)')).not.toBeInTheDocument();
  });

  it('folds evaluators past the first into a "+N more" chip that lists them on hover', async () => {
    renderTable([
      makeEval({
        name: 'eval-with-many',
        aggregate_scores: {
          'judge.accuracy': { mean: 0.5, count: 4, failed_count: 0 },
          'judge.tone': { mean: 0.7, count: 4, failed_count: 0 },
          'judge.safety': { mean: 0.9, count: 4, failed_count: 1 },
          'judge.empathy': { mean: 0.6, count: 4, failed_count: 0 },
        },
      }),
    ]);

    const more = (
      await screen.findAllByTestId('more-scores', undefined, { timeout: LG_SELECTOR_TIMEOUT })
    )[0];
    expect(more).toHaveTextContent('+3 more');
    expect(more).toHaveTextContent('(1 failed)');
    expect(screen.queryByText('tone')).not.toBeInTheDocument();
    expect(screen.queryByText('empathy')).not.toBeInTheDocument();

    await userEvent.hover(more);
    expect(await screen.findByText(/safety 0\.90 \(1 failed\)/)).toBeInTheDocument();
    expect(screen.getByText(/tone 0\.70/)).toBeInTheDocument();
    expect(screen.getByText(/empathy 0\.60/)).toBeInTheDocument();
  });
});

describe('EvaluationsTable experiment linking', () => {
  const AGENT = 'my-agent';
  const DEFAULT_EXPERIMENT = { id: 'grp_default', name: 'my-agent-runs' };

  const renderAgentTable = (evaluations: AgentEvaluationRow[], jobs: EvalJobRow[] = []) =>
    renderRoute(undefined, {
      history: '/workspaces/default',
      routes: [
        {
          path: '/workspaces/:workspace',
          element: (
            <EvaluationsTable
              workspace="default"
              agentName={AGENT}
              evaluations={evaluations}
              jobs={jobs}
            />
          ),
        },
        { path: ROUTES.workspace.agentEvaluationNew, element: <LocationDisplay /> },
      ],
    });

  const unlinked = makeEval({
    name: 'quick-run',
    id: 'eval_quick',
    experiment_ids: [DEFAULT_EXPERIMENT.id],
    experiments: [{ ...DEFAULT_EXPERIMENT } as AgentEvaluationRow['experiments'][number]],
  });

  it('marks a run still in the default experiment as not linked', async () => {
    renderAgentTable([unlinked]);

    expect(
      (await screen.findAllByText('Not linked', undefined, { timeout: LG_SELECTOR_TIMEOUT })).length
    ).toBeGreaterThan(0);
    expect(screen.queryByText('my-agent-runs')).not.toBeInTheDocument();
  });

  it("re-runs a row's job from the wizard's re-run form", async () => {
    const job = toEvalJobRow(
      makeEvaluatorJob({
        id: 'job-1',
        name: 'job-1',
        spec: { publication: { intake: { evaluation_id: 'quick-run' } } },
      })
    );
    const user = userEvent.setup();
    renderAgentTable([unlinked], [job]);

    await screen.findAllByText('quick-run', undefined, { timeout: LG_SELECTOR_TIMEOUT });
    await user.click(screen.getAllByRole('button', { name: /actions/i })[0]);
    await user.click(await screen.findByRole('menuitem', { name: 'Re-run' }));

    expect(await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(
      getAgentRunEvaluationRoute('default', AGENT, 'job-1')
    );
  });

  it('links an unlinked run by moving it out of the default experiment', async () => {
    let patched: unknown;
    server.use(
      http.get(mockApiUrl(getListExperimentsQueryKey, ':workspace'), () =>
        HttpResponse.json({
          data: [
            { id: 'grp_primary', name: 'primary-use-cases-benchmark', workspace: 'default' },
            {
              ...DEFAULT_EXPERIMENT,
              workspace: 'default',
              metadata: { [DEFAULT_EXPERIMENT_AGENT_KEY]: AGENT },
            },
          ],
        })
      ),
      http.patch(
        `${PLATFORM_BASE_URL}/apis/intake/v2/workspaces/:workspace/evaluations/:name`,
        async ({ request }) => {
          patched = await request.json();
          return HttpResponse.json({ ...unlinked, experiment_ids: ['grp_primary'] });
        }
      )
    );
    const user = userEvent.setup();
    renderAgentTable([unlinked]);

    await screen.findAllByText('quick-run', undefined, { timeout: LG_SELECTOR_TIMEOUT });
    await user.click(screen.getAllByRole('button', { name: /actions/i })[0]);
    await user.click(await screen.findByRole('menuitem', { name: 'Link to experiment' }));
    const dialog = await screen.findByRole('dialog');
    await user.click(within(dialog).getByRole('combobox', { name: 'Experiment' }));
    expect(screen.queryByRole('option', { name: 'my-agent-runs' })).not.toBeInTheDocument();
    await user.click(await screen.findByRole('option', { name: 'primary-use-cases-benchmark' }));
    await user.click(within(dialog).getByRole('button', { name: 'Add' }));

    await waitFor(() => expect(patched).toEqual({ experiment_ids: ['grp_primary'] }));
  });
});
