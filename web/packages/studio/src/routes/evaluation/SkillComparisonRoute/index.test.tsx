// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  evalsGetTasksetRun,
  evalsListTasksetRuns,
} from '@nemo/sdk/generated/evals/evals-plugin-taskset-run-routes';
import type { TasksetRunStatus } from '@nemo/sdk/generated/evals/schema';
import { ROUTES } from '@studio/constants/routes';
import { makeSkillComparisonReport } from '@studio/mocks/evaluation/skillComparison';
import { SkillComparisonRoute } from '@studio/routes/evaluation/SkillComparisonRoute';
import { getSkillComparisonsRoute } from '@studio/routes/utils';
import { renderRoute } from '@studio/tests/util/render';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@nemo/sdk/generated/evals/evals-plugin-taskset-run-routes', async (importOriginal) => ({
  ...(await importOriginal<
    typeof import('@nemo/sdk/generated/evals/evals-plugin-taskset-run-routes')
  >()),
  evalsGetTasksetRun: vi.fn(),
  evalsListTasksetRuns: vi.fn(),
}));

const renderComparison = () =>
  renderRoute(undefined, {
    history: getSkillComparisonsRoute(DEFAULT_WORKSPACE),
    routes: [{ path: ROUTES.workspace.skillComparisons, element: <SkillComparisonRoute /> }],
  });
async function importReport(report = makeSkillComparisonReport()) {
  await userEvent.upload(
    screen.getByLabelText('Comparison report'),
    new File([JSON.stringify(report)], 'skill-eval-summary.json', { type: 'application/json' })
  );
  await screen.findByText('Imported Report');
}

describe('SkillComparisonRoute', () => {
  it('imports a report, shows paired results, and distinguishes missing activation from zero', async () => {
    renderComparison();
    expect(screen.getByText('No comparison report loaded')).toBeInTheDocument();
    await importReport();
    const table = screen.getByRole('table', { name: 'Metric comparison' });
    const activation = within(table).getByRole('row', { name: /skill_activation.activated/ });
    expect(within(activation).getByText('Unmeasured')).toBeInTheDocument();
    expect(within(activation).getByText('Unmeasured · 0/1 scored')).toBeInTheDocument();
    expect(within(table).getByRole('row', { name: /llm-judge.assertion_1/ })).toHaveTextContent(
      '+1'
    );
    expect(screen.getByText('Each arm is one live trial.', { exact: false })).toBeInTheDocument();
    expect(screen.getByRole('table', { name: 'Token usage' })).toHaveTextContent('1,100');
  });

  it('shows observed bounds on a with-skill mean and omits a single-trial range', async () => {
    renderComparison();
    const report = makeSkillComparisonReport();
    const scores = report.arms['with-skill'].replay!.summary.scores.scores;
    Object.assign(scores.find((score) => score.name === 'llm-judge.assertion_1')!, {
      min: 0.3,
      mean: 0.6,
      max: 0.9,
      count: 3,
    });
    Object.assign(scores.find((score) => score.name === 'harbor_reward.reward')!, {
      min: 1,
      max: 1,
    });
    await importReport(report);
    const chart = screen.getByRole('region', { name: 'Scores by metric' });
    expect(
      within(chart).getByRole('img', { name: 'Variant: 0.6, observed range 0.3 to 0.9' })
    ).toBeInTheDocument();
    expect(within(chart).getAllByRole('img', { name: /observed range/ })).toHaveLength(1);
  });

  it('formats metric values to two decimal places and keeps tool counts out of the score chart', async () => {
    renderComparison();
    const report = makeSkillComparisonReport();
    for (const [name, arm] of Object.entries(report.arms)) {
      arm.replay!.summary.scores.scores.push(
        { name: 'tool_execution_outcome.successful', mean: 1, count: 1, nan_count: 0 },
        {
          name: 'tool_execution_outcome.success_rate',
          mean: name === 'baseline' ? 0.5253 : 0.7307,
          count: 1,
          nan_count: 0,
        }
      );
    }
    await importReport(report);
    const chart = screen.getByRole('region', { name: 'Scores by metric' });
    expect(
      within(chart).queryByRole('button', { name: 'tool_execution_outcome.successful' })
    ).not.toBeInTheDocument();
    expect(
      within(chart).getByRole('button', { name: 'tool_execution_outcome.success_rate' })
    ).toBeInTheDocument();
    const table = screen.getByRole('table', { name: 'Metric comparison' });
    const row = within(table).getByRole('row', { name: /tool_execution_outcome.success_rate/ });
    expect(within(row).getByText('0.53', { exact: true })).toBeInTheDocument();
    expect(within(row).getByText('0.73', { exact: true })).toBeInTheDocument();
    expect(within(row).getByText('+0.21', { exact: true })).toBeInTheDocument();
  });

  it('filters compact package checks by a finding and opens the matching details', async () => {
    renderComparison();
    const report = makeSkillComparisonReport();
    report.package_metrics = {
      skill_structure: { outputs: [{ name: 'conforms', value: 1 }], diagnostics: [] },
      skill_security_scan: {
        outputs: [{ name: 'finding_count', value: 1 }],
        diagnostics: [
          {
            message: 'Scanner finding.',
            details: {
              rule: 'check-permissions',
              file: 'scripts/setup.py',
              line: 7,
              severity: 'low',
            },
          },
        ],
      },
    };
    await importReport(report);
    await userEvent.click(screen.getByRole('tab', { name: 'Package Checks' }));
    const packagePanel = screen.getByRole('tabpanel', { name: 'Package Checks' });
    expect(within(packagePanel).getAllByRole('group')).toHaveLength(2);
    await userEvent.type(
      screen.getByRole('searchbox', { name: 'Search package checks' }),
      'setup.py'
    );
    expect(screen.getByText('1 of 2 checks')).toBeInTheDocument();
    expect(screen.queryByText('Skill structure', { exact: true })).not.toBeInTheDocument();
    expect(within(packagePanel).getByRole('group')).not.toHaveAttribute('open');
    await userEvent.click(screen.getByText('Skill security scan', { exact: true }));
    expect(screen.getByRole('table', { name: 'Skill security scan outputs' })).toHaveTextContent(
      'Finding count'
    );
    expect(screen.getByText('scripts/setup.py')).toBeInTheDocument();
    expect(screen.getByText('check-permissions')).toBeInTheDocument();
  });

  it('selects a linked baseline outside the recent run list', async () => {
    const baseline: TasksetRunStatus = {
      run_id: 'run-baseline',
      workspace: DEFAULT_WORKSPACE,
      taskset_ref: 'default/taskset#revision',
      phase: 'completed',
      result_state: 'ready',
      accepted_at: '2026-11-19T16:00:00Z',
      timeout_seconds: 60,
      execution_deadline_at: '2026-11-19T16:01:00Z',
      finalization_deadline_at: '2026-11-19T16:02:00Z',
    };
    const treatment = { ...baseline, run_id: 'run-treatment', baseline_run_id: baseline.run_id };
    vi.mocked(evalsListTasksetRuns).mockResolvedValue({ data: [treatment] });
    vi.mocked(evalsGetTasksetRun).mockResolvedValue(baseline);
    renderComparison();
    await userEvent.click(screen.getByRole('button', { name: 'Load Saved Runs' }));
    const dialog = await screen.findByRole('dialog', { name: 'Compare Saved Executions' });
    await userEvent.click(await within(dialog).findByRole('button', { name: 'Variant' }));
    expect(await within(dialog).findByText('Control: run-baseline (linked)')).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Load Comparison' })).toBeEnabled();
    expect(evalsGetTasksetRun).toHaveBeenCalledWith(DEFAULT_WORKSPACE, baseline.run_id);
  });

  it('opens evidence from a metric and from a task heatmap cell', async () => {
    renderComparison();
    await importReport();
    await userEvent.click(screen.getByRole('button', { name: 'llm-judge.assertion_1' }));
    expect(await screen.findByRole('dialog')).toHaveTextContent('Fixture assertion evidence.');
    await userEvent.keyboard('{Escape}');
    await userEvent.click(screen.getByRole('tab', { name: 'Task Results' }));
    const heatmap = screen.getByRole('table', { name: 'Scores by task and trial' });
    await userEvent.click(
      within(heatmap).getByRole('button', {
        name: 'exact-match-task, Control, llm-judge.assertion_1: 0',
      })
    );
    expect(await screen.findByRole('dialog', { name: 'Trial Evidence' })).toHaveTextContent(
      'trial-baseline'
    );
    expect(screen.getByRole('dialog', { name: 'Trial Evidence' })).toHaveTextContent(
      'Fixture assertion evidence.'
    );
  });

  it('keeps the last valid report when a replacement is malformed', async () => {
    renderComparison();
    await importReport();
    await userEvent.upload(
      screen.getByLabelText('Comparison report'),
      new File(['{}'], 'bad.json', { type: 'application/json' })
    );
    expect(await screen.findByRole('alert')).toHaveTextContent('Invalid comparison report');
    expect(screen.getByRole('table', { name: 'Metric comparison' })).toBeInTheDocument();
  });

  it('shows incomplete scoring errors and links only to local job routes', async () => {
    renderComparison();
    const report = makeSkillComparisonReport();
    report.arms['with-skill'].replay_error = 'Judge unavailable';
    delete report.arms['with-skill'].replay;
    await userEvent.upload(
      screen.getByLabelText('Comparison report'),
      new File([JSON.stringify(report)], 'incomplete.json', { type: 'application/json' })
    );
    expect(await screen.findByText('Needs Review')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('tab', { name: 'Evidence' }));
    expect(screen.getByText('Judge unavailable')).toBeInTheDocument();
    expect(screen.getAllByRole('link', { name: 'Open Scoring Job' })[1]).toHaveAttribute(
      'href',
      `/workspaces/${DEFAULT_WORKSPACE}/agents/evaluations/skill-scoring`
    );
  });
});
