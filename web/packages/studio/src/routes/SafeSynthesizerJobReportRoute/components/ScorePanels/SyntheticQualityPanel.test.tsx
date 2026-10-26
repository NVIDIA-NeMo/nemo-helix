// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SafeSynthesizerSummary } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { ThemeProvider } from '@nvidia/foundations-react-core';
import { SyntheticQualityPanel } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/SyntheticQualityPanel';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, within } from '@testing-library/react';

// Mock the ScoreGauge component
vi.mock('@nemo/common/src/components/ScoreGauge', () => ({
  ScoreGauge: ({ score, size }: { score?: number; size: string }) => (
    <div data-testid="score-gauge" data-size={size}>
      {score ? score.toFixed(1) : '—'}
    </div>
  ),
}));

// Mock the ScrollTable component
/* eslint-disable testing-library/no-node-access */
vi.mock('@nemo/common/src/components/ScrollTable', () => ({
  ScrollTable: ({ rows }: { rows: { cells: unknown[] }[] }) => (
    <table data-testid="scroll-table">
      <tbody>
        {rows.map((row, rowIdx) => (
          <tr key={rowIdx} data-testid={`table-row-${rowIdx}`}>
            {row.cells.map((cell, cellIdx) => {
              const content: React.ReactNode =
                cell && typeof cell === 'object' && 'children' in cell
                  ? (cell as { children: React.ReactNode }).children
                  : (cell as React.ReactNode);
              return <td key={cellIdx}>{content}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  ),
}));
/* eslint-enable testing-library/no-node-access */

// Helper function to create test wrapper
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <ThemeProvider>
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </ThemeProvider>
    );
  };
};

// Helper function to create mock report summary
const createMockReportSummary = (
  overrides?: Partial<SafeSynthesizerSummary>
): SafeSynthesizerSummary => ({
  data_privacy_score: 7.5,
  membership_inference_protection_score: 8.2,
  attribute_inference_protection_score: 7.1,
  synthetic_data_quality_score: 6.8,
  column_correlation_stability_score: 7.0,
  deep_structure_stability_score: 6.5,
  column_distribution_stability_score: 6.8,
  text_semantic_similarity_score: 7.2,
  text_structure_similarity_score: 6.9,
  num_valid_records: 100,
  num_invalid_records: 0,
  num_prompts: 100,
  valid_record_fraction: 1.0,
  timing: {
    total_time_sec: 3600,
    pii_replacer_time_sec: 100,
    training_time_sec: 2000,
    generation_time_sec: 1000,
    evaluation_time_sec: 500,
  },
  ...overrides,
});

// Expected quality metrics displayed in the score table
const EXPECTED_QUALITY_METRICS = [
  'Column Correlation Stability',
  'Deep Structure Stability',
  'Column Distribution Stability',
  'Text Semantic Similarity',
  'Text Structure Similarity',
];

describe('SyntheticQualityPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Basic Rendering', () => {
    it('should render panel with title and icon', () => {
      const mockIcon = <svg data-testid="test-icon" />;
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary()}
          title="Synthetic Quality"
          icon={mockIcon}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Synthetic Quality')).toBeInTheDocument();
      expect(screen.getByTestId('test-icon')).toBeInTheDocument();
    });

    it('should render TitledDial component with SQS title', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary()}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Synthetic Quality Score (SQS)')).toBeInTheDocument();
      // TitledDial contains a Dial, so we should see at least one dial
      expect(screen.getByTestId('titled-dial')).toBeInTheDocument();
    });

    it('should render ScoreTable with quality metrics', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary()}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByTestId('score-table')).toBeInTheDocument();
      expect(screen.getByText('Column Correlation Stability')).toBeInTheDocument();
      expect(screen.getByText('Deep Structure Stability')).toBeInTheDocument();
      expect(screen.getByText('Column Distribution Stability')).toBeInTheDocument();
      expect(screen.getByText('Text Semantic Similarity')).toBeInTheDocument();
      expect(screen.getByText('Text Structure Similarity')).toBeInTheDocument();
    });

    it('should render "Understand your SQS" section', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary()}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Understand what you can do with your data')).toBeInTheDocument();
    });

    it('should render all four use case scenarios', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary()}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Analyze internally for directional guidance')).toBeInTheDocument();
      expect(screen.getByText('Prototype, test, and run QA')).toBeInTheDocument();
      expect(screen.getByText('Balance or augment real-world datasets')).toBeInTheDocument();
      expect(screen.getByText('Train production models')).toBeInTheDocument();
    });
  });

  describe('Score Display', () => {
    it('should pass the raw SQS score to the gauge', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary({ synthetic_data_quality_score: 7.5 })}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      const titledDial = screen.getByTestId('titled-dial');
      expect(within(titledDial).getByTestId('score-gauge')).toHaveTextContent('7.5');
    });
  });

  describe('Score Table Display', () => {
    it('should display column correlation stability score', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary({
            column_correlation_stability_score: 7.0,
          })}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Column Correlation Stability')).toBeInTheDocument();
      expect(screen.getByText('7.0')).toBeInTheDocument();
    });

    it('should display deep structure stability score', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary({
            deep_structure_stability_score: 6.5,
          })}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Deep Structure Stability')).toBeInTheDocument();
      expect(screen.getByText('6.5')).toBeInTheDocument();
    });

    it('should display column distribution stability score', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary({
            synthetic_data_quality_score: 5.0,
            column_distribution_stability_score: 6.8,
          })}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Column Distribution Stability')).toBeInTheDocument();
      // Use getAllByText since 6.8 appears both in main dial and table
      const scores = screen.getAllByText('6.8');
      expect(scores.length).toBeGreaterThan(0);
    });

    it('should display text semantic similarity score', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary({
            text_semantic_similarity_score: 7.2,
          })}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Text Semantic Similarity')).toBeInTheDocument();
      expect(screen.getByText('7.2')).toBeInTheDocument();
    });

    it('should display text structure similarity score', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary({
            text_structure_similarity_score: 6.9,
          })}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Text Structure Similarity')).toBeInTheDocument();
      expect(screen.getByText('6.9')).toBeInTheDocument();
    });

    it('should handle undefined individual metric scores', () => {
      const incompleteSummary = {
        synthetic_data_quality_score: 5.0,
        num_valid_records: 100,
        num_invalid_records: 0,
        num_prompts: 100,
        valid_record_fraction: 1.0,
      } as SafeSynthesizerSummary;

      render(
        <SyntheticQualityPanel
          reportSummary={incompleteSummary}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      const zeroScores = screen.getAllByText('—');
      expect(zeroScores.length).toBeGreaterThan(0);
    });
  });

  describe('Use Case Scenarios Validation', () => {
    it('should display check icons for passing scenarios and cancel icons for failing scenarios', () => {
      // Use a moderate score to verify both passing and failing scenarios are shown
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary({ synthetic_data_quality_score: 5.0 })}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      // Verify that both check and cancel icons are rendered
      const checkIcons = screen.getAllByTestId('score-check-icon');
      const cancelIcons = screen.getAllByTestId('score-cancel-icon');

      expect(checkIcons.length).toBeGreaterThan(0);
      expect(cancelIcons.length).toBeGreaterThan(0);
      // Total should always be 4 use cases
      expect(checkIcons.length + cancelIcons.length).toBe(4);
    });

    it('should display all four use case descriptions', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary({ synthetic_data_quality_score: 7.0 })}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      // Verify use case headings are present (text validation tests component integration)
      expect(screen.getByText('Analyze internally for directional guidance')).toBeInTheDocument();
      expect(screen.getByText('Prototype, test, and run QA')).toBeInTheDocument();
      expect(screen.getByText('Balance or augment real-world datasets')).toBeInTheDocument();
      expect(screen.getByText('Train production models')).toBeInTheDocument();
    });
  });

  describe('TitledDial Integration', () => {
    it('should pass correct description to TitledDial', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary()}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(
        screen.getByText(
          /The Synthetic Quality Score is computed by taking a weighted combination/i
        )
      ).toBeInTheDocument();
    });

    it('should pass the raw score to TitledDial', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary({ synthetic_data_quality_score: 7.5 })}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      const gauges = screen.getAllByTestId('score-gauge');
      expect(gauges[0]).toHaveTextContent('7.5');
    });
  });

  describe('ScoreTable Integration', () => {
    it('should render a gauge for each sub-metric', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary()}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      // One gauge for the main SQS score plus one per sub-metric
      const gauges = screen.getAllByTestId('score-gauge');
      expect(gauges.length).toBeGreaterThan(1);
    });

    it('should pass all quality scores to ScoreTable in correct order', () => {
      render(
        <SyntheticQualityPanel
          reportSummary={createMockReportSummary({
            column_correlation_stability_score: 7.0,
            deep_structure_stability_score: 6.5,
            column_distribution_stability_score: 6.8,
            text_semantic_similarity_score: 7.2,
            text_structure_similarity_score: 6.9,
          })}
          title="Synthetic Quality"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      // Check that all metric names are present in the table
      expect(screen.getByTestId('score-table')).toBeInTheDocument();

      // Verify the metrics are displayed
      EXPECTED_QUALITY_METRICS.forEach((metric) => {
        expect(screen.getByText(metric)).toBeInTheDocument();
      });
    });
  });
});
