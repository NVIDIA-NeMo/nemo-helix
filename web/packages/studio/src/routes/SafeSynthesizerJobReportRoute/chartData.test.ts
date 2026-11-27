// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  autoBinEdges,
  binnedRows,
  formatNumber,
  histogram,
  lagDifferenceRows,
  orderAutocorrelationColumns,
  piiColumnGroups,
  piiHeadline,
  piiReplayNote,
  quantile,
  shortenLabel,
} from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import type {
  AutocorrelationProfile,
  PIIReplayMetric,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import { parseEvaluationReportData } from '@studio/routes/SafeSynthesizerJobReportRoute/useEvaluationReportData';
import { reportData, timeSeriesReportData } from '@studio/tests/util/mockSafeSynthesizerReport';

const profile = (column: string, similarity: number, training: number[], synthetic: number[]) =>
  ({
    group: 'g',
    column,
    lags: training.map((_, index) => index + 1),
    effective_max_lag: training.length,
    evaluated_lags: training.length,
    error: 1 - similarity,
    similarity,
    reason: null,
    training_acf: training,
    synthetic_acf: synthetic,
  }) satisfies AutocorrelationProfile;

const replay = (overrides: Partial<PIIReplayMetric>): PIIReplayMetric => ({
  score: { raw_score: null, score: null, grade: 'Unavailable', notes: null },
  enabled: true,
  training_total_records: 0,
  synthetic_total_records: 0,
  entries: [],
  skipped_reason: null,
  ...overrides,
});

describe('parseEvaluationReportData', () => {
  it('accepts the library fixtures', () => {
    expect(parseEvaluationReportData(JSON.stringify(reportData))?.scores).toEqual(
      reportData.scores
    );
    expect(parseEvaluationReportData(JSON.stringify(timeSeriesReportData))).not.toBeNull();
  });

  it.each(['not json', 'null', '7', '{"schema_version": 1}'])('rejects %s', (text) => {
    expect(parseEvaluationReportData(text)).toBeNull();
  });
});

describe('chart data', () => {
  it('closes binned rows with the last edge', () => {
    const rows = binnedRows({
      labels: ['[0, 10)', '[10, 20)'],
      bin_edges: [0, 10, 20],
      training_percent: [40, 60],
      synthetic_percent: [50, 50],
    });

    expect(rows.map((row) => row.edge)).toEqual([0, 10, 20]);
    expect(rows[2].training).toBe(60);
  });

  it('counts the top edge in the last bin, like numpy', () => {
    expect(histogram([0, 5, 10, 10], [0, 5, 10], 'percent')).toEqual([25, 75]);
    expect(histogram([0, 1], [0, 1], 'density')).toEqual([1]);
  });

  it('interpolates quantiles linearly', () => {
    expect(quantile([1, 2, 3, 4], 0.5)).toBe(2.5);
  });

  it('spans both samples with automatic bin edges', () => {
    const edges = autoBinEdges([0, 1, 2], [5]);

    expect(edges[0]).toBe(0);
    expect(edges.at(-1)).toBe(5);
    expect(edges.length).toBeGreaterThan(1);
  });

  it('orders autocorrelation columns from least to most similar', () => {
    const profiles = [profile('b', 0.9, [0.5], [0.4]), profile('a', 0.2, [0.5], [0.1])];

    expect(orderAutocorrelationColumns(profiles)).toEqual(['a', 'b']);
    expect(lagDifferenceRows(profiles)[0].difference).toBeCloseTo(0.25);
  });

  it('shortens long labels to their first and last three characters', () => {
    expect(shortenLabel('transaction_amount')).toBe('tra...unt');
    expect(shortenLabel('amount')).toBe('amount');
  });

  it('formats tick values without float noise', () => {
    expect(formatNumber(5.6000000000000005, 4)).toBe('5.6');
    expect(formatNumber(0.1 + 0.2)).toBe('0.3');
    expect(formatNumber(40)).toBe('40');
  });
});

describe('PII replacement', () => {
  it('joins replaced columns with their replay results, most frequent type first', () => {
    const pii = reportData.pii_replacement;
    if (!pii) {
      throw new Error('fixture has no PII replacement');
    }
    const groups = piiColumnGroups(pii, reportData.metrics.pii_replay);

    expect(groups.map((group) => group.columnName)).toEqual(['first_name', 'email', 'notes']);
    expect(groups[2].entities.map((entity) => entity.piiType)).toEqual(['first_name', 'city']);
    expect(groups[1].entities[0].replayPercentage).toBe(4.1);
    expect(piiHeadline(pii, groups, null)).toBe('3 columns replaced');
  });

  it('lists replay results for columns outside the plan when replacement was off', () => {
    const groups = piiColumnGroups(
      {
        replaced: false,
        columns: [],
        kept_columns: [],
        sampler: null,
        locale: null,
        seed: null,
        value_mappings: {},
      },
      replay({ entries: reportData.metrics.pii_replay?.entries.slice(0, 1) ?? [] })
    );

    expect(groups).toHaveLength(1);
    expect(groups[0].replacement).toBeNull();
  });

  it('explains missing or partial replay results', () => {
    expect(piiReplayNote(replay({ enabled: false }))).toBe(
      'PII replay was turned off with evaluation.pii_replay_enabled.'
    );
    expect(piiReplayNote(replay({ skipped_reason: 'too many rows' }))).toBe(
      'PII replay was not run: too many rows.'
    );
    expect(piiReplayNote(replay({}))).toBeNull();
  });
});
