// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { GenerateJobReportRoute } from '@studio/routes/SafeSynthesizerJobReportRoute';
import { REPORT_TABS } from '@studio/routes/SafeSynthesizerJobReportRoute/reportTabs';
import {
  mockReportData,
  mockReportDataMissing,
  mockSafeSynthesizerJob,
  reportData,
} from '@studio/tests/util/mockSafeSynthesizerReport';
import { renderRoute } from '@studio/tests/util/render';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const REPORT_PATH = '/workspaces/default/safe-synthesizer/job/customers-job/report';

const renderReport = (search = '') =>
  renderRoute(undefined, {
    history: `${REPORT_PATH}${search}`,
    routes: [
      { path: ROUTES.workspace.safeSynthesizerJobReport, element: <GenerateJobReportRoute /> },
    ],
  });

const REPORT_TAB_LABELS = Object.values(REPORT_TABS).map((tab) => tab.label);

const tabNames = () =>
  REPORT_TAB_LABELS.filter((label) => screen.queryByRole('tab', { name: label }) !== null);

describe('GenerateJobReportRoute', () => {
  beforeEach(() => {
    mockSafeSynthesizerJob();
  });

  it('shows the four report tabs and an overview of scores and dataset statistics', async () => {
    renderReport();

    expect(await screen.findByTestId('dataset-statistics')).toBeInTheDocument();
    expect(tabNames()).toEqual(['Overview', 'Synthetic Quality', 'Data Privacy', 'Training Data']);
    expect(screen.getAllByTestId('overview-score-card')).toHaveLength(2);
    expect(screen.getByText('Synthetic Quality Score')).toBeInTheDocument();
    expect(screen.getByText('Data Privacy Score')).toBeInTheDocument();
    expect(screen.getByTestId('memorization-status')).toHaveTextContent(
      'Training rows duplicated in Synthetic Data: 0'
    );
    expect(screen.queryByTestId('metric-group')).not.toBeInTheDocument();
  });

  it('switches tabs and keeps the choice in the URL', async () => {
    const user = userEvent.setup();
    renderReport();

    await user.click(await screen.findByRole('tab', { name: 'Synthetic Quality' }));
    expect(await screen.findByText('Numerical and Categorical Metrics')).toBeInTheDocument();
    expect(screen.getByText('Text Metrics')).toBeInTheDocument();

    await user.click(screen.getByRole('tab', { name: 'Training Data' }));
    expect(await screen.findByTestId('training-columns-tab')).toBeInTheDocument();
  });

  it('opens the tab named in the URL', async () => {
    renderReport('?tab=data-privacy');

    expect(await screen.findByText('Protection Metrics')).toBeInTheDocument();
    expect(screen.getByTestId('metric-card-pii-replay')).toHaveTextContent('3 columns replaced');
  });

  it('expands a metric card to draw its chart', async () => {
    const user = userEvent.setup();
    renderReport('?tab=synthetic-quality');

    const card = await screen.findByTestId('metric-card-correlation-stability');
    expect(within(card).queryAllByTestId('correlation-heatmap')).toHaveLength(0);

    await user.click(
      within(card).getByLabelText('Toggle details for Column Correlation Stability')
    );

    expect(within(card).getAllByTestId('correlation-heatmap')).toHaveLength(3);
  });

  it('falls back to the summary scores when the job has no report data', async () => {
    mockReportDataMissing();
    renderReport('?tab=data-privacy');

    expect(await screen.findByTestId('score-table')).toHaveTextContent(
      'Membership Inference Protection'
    );
    expect(tabNames()).toEqual(['Overview', 'Synthetic Quality', 'Data Privacy']);
    expect(screen.queryByText('Protection Metrics')).not.toBeInTheDocument();
    expect(await screen.findByRole('button', { name: /Download Report/ })).toBeInTheDocument();
  });

  it('falls back to the summary scores when the report data does not match the schema', async () => {
    mockReportData({ schema_version: 1, scores: 'not an object' });
    renderReport();

    expect(await screen.findAllByTestId('overview-score-card')).toHaveLength(2);
    expect(screen.queryByTestId('dataset-statistics')).not.toBeInTheDocument();
    expect(tabNames()).toEqual(['Overview', 'Synthetic Quality', 'Data Privacy']);
  });

  it('hides the privacy tab and score when the report has neither a privacy score nor PII', async () => {
    mockReportData({
      ...reportData,
      scores: { ...reportData.scores, data_privacy: null },
      pii_replacement: null,
    });
    renderReport();

    expect(await screen.findByTestId('dataset-statistics')).toBeInTheDocument();
    expect(tabNames()).toEqual(['Overview', 'Synthetic Quality', 'Training Data']);
    expect(screen.getAllByTestId('overview-score-card')).toHaveLength(1);
  });
});
