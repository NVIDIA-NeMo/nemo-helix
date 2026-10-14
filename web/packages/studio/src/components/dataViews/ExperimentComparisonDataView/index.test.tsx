// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { ExperimentComparisonDataView } from '@studio/components/dataViews/ExperimentComparisonDataView';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { render, screen, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

const group = { id: 'group-id', name: 'planner' } as ExperimentResponse;

const evaluation = (
  name: string,
  createdAt: string,
  judge: number,
  tokens: number,
  durationSec?: string
): EvaluationRow =>
  ({
    id: name,
    name,
    created_at: createdAt,
    tokens: { mean: tokens },
    metadata: durationSec ? { eval_duration_sec: durationSec } : undefined,
    aggregate_scores: { judge: { mean: judge } },
  }) as unknown as EvaluationRow;

const evaluations = [
  evaluation('tuned', '2026-10-11T02:30:00', 89, 33_000, '457'),
  evaluation('weakest', '2026-10-11T02:15:00', 50, 40_000),
  evaluation('baseline', '2026-10-11T02:00:00', 55, 27_000, '582'),
];

const renderTable = () =>
  render(
    <ExperimentComparisonDataView
      workspace={DEFAULT_WORKSPACE}
      group={group}
      preloadedEvaluations={evaluations}
    />
  );

const rowFor = (name: string) => within(screen.getByRole('row', { name: new RegExp(`^${name} `) }));

describe('ExperimentComparisonDataView', () => {
  it('compares each evaluation against the oldest one by default', async () => {
    renderTable();

    expect(await rowFor('baseline').findByText('Baseline')).toBeInTheDocument();
    expect(rowFor('tuned').getByText('+34')).toHaveClass('text-feedback-success');
    expect(rowFor('weakest').getByText('-5')).toHaveClass('text-feedback-danger');
  });

  it('highlights the best value in each column by its direction', async () => {
    renderTable();

    expect(await screen.findAllByText('(best)')).toHaveLength(3);
    expect(rowFor('tuned').getByText('89')).toHaveTextContent('(best)');
    expect(rowFor('tuned').getByText('7m 37s')).toHaveTextContent('(best)');
    expect(rowFor('baseline').getByText('27,000')).toHaveTextContent('(best)');
    expect(rowFor('baseline').getByText('55')).not.toHaveTextContent('(best)');
  });

  it('highlights the worst value in each column by its direction', async () => {
    renderTable();

    expect(await screen.findAllByText('(worst)')).toHaveLength(3);
    expect(rowFor('weakest').getByText('50')).toHaveTextContent('(worst)');
    expect(rowFor('weakest').getByText('40,000')).toHaveTextContent('(worst)');
    expect(rowFor('baseline').getByText('9m 42s')).toHaveTextContent('(worst)');
    expect(rowFor('baseline').getByText('55')).not.toHaveTextContent('(worst)');
  });

  it('shows an em dash for a duration that was never recorded', async () => {
    renderTable();

    expect(await rowFor('tuned').findByText('7m 37s')).toBeInTheDocument();
    expect(rowFor('weakest').getAllByText('—').length).toBeGreaterThan(0);
  });

  it('recomputes deltas against a newly selected baseline', async () => {
    const user = userEvent.setup();
    renderTable();

    await user.click(await screen.findByRole('combobox', { name: 'Baseline' }));
    const options = await screen.findAllByRole('option');
    const tunedOption = options.find((option) => option.textContent === 'tuned');
    if (!tunedOption) throw new Error('tuned option not found');
    await user.click(tunedOption);

    expect(rowFor('tuned').getByText('Baseline')).toBeInTheDocument();
    expect(rowFor('baseline').getByText('-34')).toHaveClass('text-feedback-danger');
  });
});
