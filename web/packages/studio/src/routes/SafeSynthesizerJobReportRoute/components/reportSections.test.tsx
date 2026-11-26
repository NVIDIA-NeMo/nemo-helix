// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccordionRoot } from '@nvidia/foundations-react-core';
import { MetricCard } from '@studio/routes/SafeSynthesizerJobReportRoute/components/MetricCard';
import { DataPrivacyMetrics } from '@studio/routes/SafeSynthesizerJobReportRoute/components/PrivacyMetrics';
import { SyntheticQualityMetrics } from '@studio/routes/SafeSynthesizerJobReportRoute/components/QualityMetrics';
import { TrainingColumnsTab } from '@studio/routes/SafeSynthesizerJobReportRoute/components/TrainingColumnsTab';
import { METRIC_COPY } from '@studio/routes/SafeSynthesizerJobReportRoute/reportCopy';
import { reportData, timeSeriesReportData } from '@studio/tests/util/mockSafeSynthesizerReport';
import { renderRoute } from '@studio/tests/util/render';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

describe('MetricCard', () => {
  it('mounts its details only while open', async () => {
    const user = userEvent.setup();
    renderRoute(
      <AccordionRoot multiple>
        <MetricCard id="card" copy={METRIC_COPY.deep_structure_stability} score={null}>
          <span>details</span>
        </MetricCard>
      </AccordionRoot>
    );
    const toggle = screen.getByLabelText('Toggle details for Deep Structure Stability');

    expect(screen.queryByText('details')).not.toBeInTheDocument();
    await user.click(toggle);
    expect(screen.getByText('details')).toBeInTheDocument();
    await user.click(toggle);
    expect(screen.queryByText('details')).not.toBeInTheDocument();
  });

  it('cannot be expanded without details', () => {
    renderRoute(
      <AccordionRoot multiple>
        <MetricCard id="card" copy={METRIC_COPY.text_semantic_similarity} score={null} />
      </AccordionRoot>
    );

    expect(screen.getByTestId('metric-card-card')).toHaveAttribute('data-disabled');
  });
});

describe('SyntheticQualityMetrics', () => {
  it('shows the library cards with their descriptions and scores', () => {
    renderRoute(<SyntheticQualityMetrics data={reportData} />);

    for (const copy of [
      METRIC_COPY.column_correlation_stability,
      METRIC_COPY.deep_structure_stability,
      METRIC_COPY.column_distribution_stability,
      METRIC_COPY.text_semantic_similarity,
      METRIC_COPY.text_structure_similarity,
    ]) {
      expect(screen.getByText(copy.name)).toBeInTheDocument();
      expect(screen.getByText(copy.description)).toBeInTheDocument();
    }
    expect(screen.queryByText('Time-Series Metrics')).not.toBeInTheDocument();
  });

  it('shows each card score in its gauge', () => {
    renderRoute(<SyntheticQualityMetrics data={reportData} />);
    const score = reportData.metrics.column_correlation_stability?.score.score;

    expect(
      within(screen.getByTestId('metric-card-correlation-stability')).getByRole('img', {
        name: `Score: ${score?.toFixed(1)} out of 10`,
      })
    ).toBeInTheDocument();
  });

  it('adds the time-series group for time-series data', async () => {
    const user = userEvent.setup();
    renderRoute(<SyntheticQualityMetrics data={timeSeriesReportData} />);

    expect(screen.getByText('Time-Series Metrics')).toBeInTheDocument();
    await user.click(screen.getByLabelText('Toggle details for Autocorrelation Similarity'));
    expect(screen.getByText(/^Summarizing \d+ group-column pairs/)).toBeInTheDocument();
  });
});

describe('DataPrivacyMetrics', () => {
  it('lists PII replacement and replay per column', async () => {
    const user = userEvent.setup();
    renderRoute(<DataPrivacyMetrics data={reportData} withPrivacyScore />);

    await user.click(screen.getByLabelText('Toggle details for PII Replacement and Replay'));
    const details = screen.getByTestId('pii-replacement-details');

    expect(
      within(details).getByText(
        'Sampler: Faker, locale en_US, seed 7. The test data is not replaced.'
      )
    ).toBeInTheDocument();
    expect(
      within(details).getByText('Built from first_name, keeping the original domain')
    ).toBeInTheDocument();
    expect(
      within(details).getByText('Found by GLiNER, values from replaced columns')
    ).toBeInTheDocument();
    expect(within(details).getByText('Kept as is (5):')).toBeInTheDocument();
    expect(within(details).queryByText('Value Mappings')).not.toBeInTheDocument();
  });

  it('shows value mappings when the plan has them', async () => {
    const user = userEvent.setup();
    const pii = reportData.pii_replacement;
    if (!pii) {
      throw new Error('fixture has no PII replacement');
    }
    renderRoute(
      <DataPrivacyMetrics
        data={{
          ...reportData,
          pii_replacement: { ...pii, value_mappings: { gender: { F: ['female'] } } },
        }}
        withPrivacyScore
      />
    );

    await user.click(screen.getByLabelText('Toggle details for PII Replacement and Replay'));

    expect(screen.getByText('Value Mappings')).toBeInTheDocument();
    expect(screen.getByText('female')).toBeInTheDocument();
  });

  it('shows only the PII card without a privacy score', () => {
    renderRoute(<DataPrivacyMetrics data={reportData} withPrivacyScore={false} />);

    expect(screen.getByText('PII Replacement and Replay')).toBeInTheDocument();
    expect(screen.queryByText('Protection Metrics')).not.toBeInTheDocument();
  });
});

describe('TrainingColumnsTab', () => {
  it('lists training columns with their distribution grade and PII transforms', () => {
    renderRoute(<TrainingColumnsTab columns={reportData.columns} withDistribution withTransform />);

    expect(screen.getByText('8 columns')).toBeInTheDocument();
    expect(screen.getByText('Entities (Count)')).toBeInTheDocument();
    expect(screen.getByText('email (300)')).toBeInTheDocument();
    expect(screen.getAllByText('Categorical').length).toBeGreaterThan(0);
  });
});
