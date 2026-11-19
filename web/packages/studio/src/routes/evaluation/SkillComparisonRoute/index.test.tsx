// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { ROUTES } from '@studio/constants/routes';
import { makeSkillComparisonReport } from '@studio/mocks/evaluation/skillComparison';
import { SkillComparisonRoute } from '@studio/routes/evaluation/SkillComparisonRoute';
import { getSkillComparisonsRoute } from '@studio/routes/utils';
import { renderRoute } from '@studio/tests/util/render';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

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
        name: 'exact-match-task, Baseline, llm-judge.assertion_1: 0',
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
