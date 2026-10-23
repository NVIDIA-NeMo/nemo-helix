// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SafeSynthesizerSummary } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { ThemeProvider } from '@nvidia/foundations-react-core';
import { DataPrivacyPanel } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/DataPrivacyPanel';
import { GRADE_VALUES } from '@studio/routes/SafeSynthesizerJobReportRoute/util';
import { screen } from '@studio/tests/util/render';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, within } from '@testing-library/react';

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

describe('DataPrivacyPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Basic Rendering', () => {
    it('should render panel with title and icon', () => {
      const mockIcon = <svg data-testid="test-icon" />;
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary()}
          dpEnabled
          title="Data Privacy"
          icon={mockIcon}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Data Privacy')).toBeInTheDocument();
      expect(screen.getByTestId('test-icon')).toBeInTheDocument();
    });

    it('should render TitledDial component', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary()}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      // TitledDial contains a Dial, so we should see at least one dial
      expect(screen.getByTestId('titled-dial')).toBeInTheDocument();
    });

    it('should render ScoreTable with metrics', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary()}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByTestId('score-table')).toBeInTheDocument();
      expect(screen.getByText('Membership Inference Protection')).toBeInTheDocument();
      expect(screen.getByText('Attribute Inference Protection')).toBeInTheDocument();
    });

    it('should render all four sharing scenarios', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary()}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Share internally for analytics and reporting')).toBeInTheDocument();
      expect(screen.getByText('Share externally with trusted third-parties')).toBeInTheDocument();
      expect(screen.getByText('Publish for research and community use')).toBeInTheDocument();
      expect(screen.getByText('Train production models')).toBeInTheDocument();
    });

    it('should render Differential Privacy section', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary()}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Differential Privacy')).toBeInTheDocument();
    });
  });

  describe('Data Privacy Score Display', () => {
    it('should display score value and corresponding grade label', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary({ data_privacy_score: 7.5 })}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      // Verify score is displayed
      expect(screen.getByText('7.5')).toBeInTheDocument();
      // Verify a grade label is displayed (util function handles the mapping)
      expect(screen.getByTestId('nv-tag-root')).toBeInTheDocument();
    });

    it('should handle edge cases for missing or zero scores', () => {
      const { rerender } = render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary({ data_privacy_score: 0 })}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText(GRADE_VALUES.UNAVAILABLE)).toBeInTheDocument();

      // Test undefined report summary
      rerender(
        <DataPrivacyPanel reportSummary={undefined} dpEnabled title="Data Privacy" icon={<svg />} />
      );

      expect(screen.getByText(GRADE_VALUES.UNAVAILABLE)).toBeInTheDocument();
    });
  });

  describe('Score Table Display', () => {
    it('should display membership inference protection score', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary({
            membership_inference_protection_score: 8.2,
          })}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Membership Inference Protection')).toBeInTheDocument();
      expect(screen.getByText('8.2')).toBeInTheDocument();
    });

    it('should display attribute inference protection score', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary({
            attribute_inference_protection_score: 7.1,
          })}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Attribute Inference Protection')).toBeInTheDocument();
      expect(screen.getByText('7.1')).toBeInTheDocument();
    });

    it('should render zero scores as unavailable in table', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary({
            membership_inference_protection_score: 0,
            attribute_inference_protection_score: 0,
          })}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      const unavailable = screen.getAllByText('—');
      expect(unavailable.length).toBeGreaterThan(0);
    });
  });

  describe('Differential Privacy Status', () => {
    it('should show Differential Privacy as enabled when dpEnabled is true', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary()}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Differential Privacy')).toBeInTheDocument();
      expect(within(screen.getByTestId('dp-status')).getByText('On')).toBeInTheDocument();
    });

    it('should show Differential Privacy as disabled when dpEnabled is false', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary()}
          dpEnabled={false}
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Differential Privacy')).toBeInTheDocument();
      expect(within(screen.getByTestId('dp-status')).getByText('Off')).toBeInTheDocument();
    });

    it('should render Differential Privacy status indicator', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary()}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      // Verify that Differential Privacy status is displayed correctly
      expect(screen.getByText('Differential Privacy')).toBeInTheDocument();
      expect(screen.getByTestId('dp-status')).toBeInTheDocument();
    });
  });

  describe('Sharing Scenarios Validation', () => {
    it('should display check icons for passing scenarios and cancel icons for failing scenarios', () => {
      // Score 3.0 → MODERATE grade: POOR and MODERATE thresholds pass,
      // GOOD and VERY_GOOD thresholds fail → 2 check icons, 2 cancel icons
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary({ data_privacy_score: 3.0 })}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      const checkIcons = screen.getAllByTestId('circle-check-icon');
      const cancelIcons = screen.getAllByTestId('ban-icon');

      expect(checkIcons).toHaveLength(2);
      expect(cancelIcons).toHaveLength(2);
      expect(checkIcons.length + cancelIcons.length).toBe(4);
    });
  });

  describe('Score Gauge Props', () => {
    it('should pass the raw DPS score to the main gauge', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary({ data_privacy_score: 7.5 })}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      const gauges = screen.getAllByTestId('score-gauge');
      expect(gauges[0]).toHaveTextContent('7.5');
    });

    it('should pass correct props to ScoreTable gauges', () => {
      render(
        <DataPrivacyPanel
          reportSummary={createMockReportSummary({
            membership_inference_protection_score: 8.2,
            attribute_inference_protection_score: 7.1,
          })}
          dpEnabled
          title="Data Privacy"
          icon={<svg />}
        />,
        { wrapper: createWrapper() }
      );

      // Should have dials for the scores in the table
      expect(screen.getByText('8.2')).toBeInTheDocument();
      expect(screen.getByText('7.1')).toBeInTheDocument();
    });
  });
});
