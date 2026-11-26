// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable testing-library/no-container, testing-library/no-node-access -- recharts marks have no role or text, so counting them reads the SVG nodes */

import { AutocorrelationCharts } from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/AutocorrelationCharts';
import { CorrelationHeatmaps } from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/CorrelationHeatmaps';
import { DistributionChart } from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/DistributionChart';
import {
  AttributeInferenceBars,
  MembershipInferencePie,
} from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/PrivacyCharts';
import {
  DeepStructureScatter,
  TextSemanticCharts,
} from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/ProjectionCharts';
import { WindowMembershipCharts } from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/WindowMembershipCharts';
import { reportData, timeSeriesReportData } from '@studio/tests/util/mockSafeSynthesizerReport';
import { renderRoute } from '@studio/tests/util/render';
import { screen, within } from '@testing-library/react';
import { cloneElement, type ReactElement } from 'react';

vi.mock('recharts', async () => {
  const actual = await vi.importActual<typeof import('recharts')>('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: ReactElement }) =>
      cloneElement(children, { width: 800, height: 300 } as Record<string, unknown>),
  };
});

const column = (name: string) => {
  const result = reportData.columns.find((candidate) => candidate.name === name);
  if (!result) {
    throw new Error(`fixture has no column ${name}`);
  }
  return result;
};

const { metrics } = reportData;

describe('report charts', () => {
  it('draws grouped bars for a categorical distribution', () => {
    const { container } = renderRoute(
      <DistributionChart title="city" distribution={column('city').distribution} />
    );

    expect(screen.getByText('city')).toBeInTheDocument();
    expect(container.querySelectorAll('.recharts-bar-rectangle').length).toBeGreaterThan(0);
  });

  it('draws overlaid step areas for a binned distribution', () => {
    const { container } = renderRoute(
      <DistributionChart title="income" distribution={column('income').distribution} />
    );

    expect(container.querySelectorAll('.recharts-area')).toHaveLength(2);
  });

  it('says so when a numeric column has nothing to chart', () => {
    renderRoute(<DistributionChart title="empty" distribution={null} />);

    expect(screen.getByText('No numeric training values to chart')).toBeInTheDocument();
  });

  it('draws one cell per column pair in each correlation heatmap', () => {
    const matrices = metrics.column_correlation_stability?.correlations;
    if (!matrices) {
      throw new Error('fixture has no correlations');
    }
    renderRoute(<CorrelationHeatmaps matrices={matrices} />);

    const heatmaps = screen.getAllByTestId('correlation-heatmap');
    expect(
      heatmaps.map((heatmap) => within(heatmap).getByRole('img').getAttribute('aria-label'))
    ).toEqual([
      'Training Correlations',
      'Synthetic Correlations',
      'Correlation Difference (Absolute)',
    ]);
    expect(heatmaps[0].querySelectorAll('rect')).toHaveLength(matrices.columns.length ** 2);
  });

  it('draws training and synthetic projections side by side', () => {
    const projection = metrics.deep_structure_stability?.projection;
    if (!projection) {
      throw new Error('fixture has no projection');
    }
    const { container } = renderRoute(<DeepStructureScatter projection={projection} />);

    expect(screen.getByText('Training Data')).toBeInTheDocument();
    expect(screen.getByText('Synthetic Data')).toBeInTheDocument();
    expect(container.querySelectorAll('.recharts-scatter')).toHaveLength(2);
  });

  it('draws component densities and the projection for a text column', () => {
    const projection = metrics.text_semantic_similarity?.projections.notes;
    if (!projection) {
      throw new Error('fixture has no text projection');
    }
    const { container } = renderRoute(
      <TextSemanticCharts column="notes" projection={projection} />
    );

    expect(screen.getByText('notes')).toBeInTheDocument();
    expect(container.querySelectorAll('.recharts-area')).toHaveLength(4);
    expect(container.querySelectorAll('.recharts-scatter')).toHaveLength(2);
  });

  it('titles the attack outcome pie with the number of graded attacks', () => {
    const mia = metrics.membership_inference_protection;
    if (!mia) {
      throw new Error('fixture has no membership inference');
    }
    const graded = mia.outcomes.reduce((sum, outcome) => sum + (outcome.count ?? 0), 0);
    const { container } = renderRoute(<MembershipInferencePie metric={mia} />);

    expect(
      screen.getByText(`Membership inference attack outcomes (${graded} attack evaluations graded)`)
    ).toBeInTheDocument();
    const nonZero = mia.outcomes.filter((outcome) => outcome.count).length;
    expect(container.querySelectorAll('.recharts-pie-sector')).toHaveLength(nonZero);
  });

  it('draws one protection bar per column', () => {
    const aia = metrics.attribute_inference_protection;
    if (!aia) {
      throw new Error('fixture has no attribute inference');
    }
    const { container } = renderRoute(<AttributeInferenceBars metric={aia} />);

    expect(container.querySelectorAll('.recharts-bar-rectangle')).toHaveLength(aia.columns.length);
  });

  it('labels a zero protection with its grade and a missing one as unavailable', () => {
    renderRoute(
      <AttributeInferenceBars
        metric={{
          score: { raw_score: null, score: null, grade: 'Unavailable', notes: null },
          columns: [
            { column: 'notes', protection: 0, grade: 'Poor' },
            { column: 'zip', protection: null, grade: 'Unavailable' },
            { column: 'age', protection: 9.5, grade: 'Excellent' },
          ],
        }}
      />
    );

    expect(screen.getByText('0.0 Poor')).toBeInTheDocument();
    expect(screen.getByText('Unavailable')).toBeInTheDocument();
    expect(screen.getByText('9.5 Excellent')).toBeInTheDocument();
  });

  it('draws the three autocorrelation charts', () => {
    const autocorrelation = timeSeriesReportData.metrics.autocorrelation_similarity;
    if (!autocorrelation) {
      throw new Error('fixture has no autocorrelation');
    }
    renderRoute(
      <AutocorrelationCharts
        profiles={autocorrelation.details.profiles}
        overallScore={autocorrelation.score.score}
      />
    );

    expect(screen.getByText('Typical autocorrelation')).toBeInTheDocument();
    expect(screen.getByText('Difference by lag')).toBeInTheDocument();
    expect(screen.getByText('Pair scores')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Numeric column' })).toBeInTheDocument();
  });

  it('draws the window membership inference charts', () => {
    const windowMia = timeSeriesReportData.metrics.window_membership_inference_protection;
    if (!windowMia) {
      throw new Error('fixture has no window membership inference');
    }
    renderRoute(<WindowMembershipCharts details={windowMia.details} />);

    expect(screen.getByText('Distance to synthetic data')).toBeInTheDocument();
    expect(screen.getByText('Groups flagged most')).toBeInTheDocument();
    expect(screen.getByText('Closest match')).toBeInTheDocument();
  });
});
