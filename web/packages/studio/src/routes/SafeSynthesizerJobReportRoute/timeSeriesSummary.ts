// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { plural } from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import type {
  AutocorrelationProfile,
  WindowMembershipInferenceDetails,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';

export const PAIR_SCORE_MAX_COLUMNS = 8;

const count = (value: number) => value.toLocaleString();
const percent = (rate: number) => `${(100 * rate).toFixed(1)}%`;
const noun = (value: number, singular: string, pluralForm: string) =>
  value === 1 ? singular : pluralForm;

/** The library report's sentences above the autocorrelation charts. */
export const autocorrelationSummary = (profiles: readonly AutocorrelationProfile[]): string[] => {
  const groups = new Set(profiles.map((profile) => String(profile.group))).size;
  const columns = new Set(profiles.map((profile) => profile.column)).size;
  let pairs = `Summarizing ${profiles.length} group-column pairs (${plural(groups, 'group')}, ${columns} numeric ${noun(columns, 'column', 'columns')}).`;
  if (columns > PAIR_SCORE_MAX_COLUMNS) {
    pairs += ` Pair scores shows the ${PAIR_SCORE_MAX_COLUMNS} lowest-scoring columns.`;
  }
  const lines = [pairs];
  const constant = profiles.filter((profile) => profile.evaluated_lags === 0).length;
  if (constant === 1) {
    lines.push(
      '1 pair has constant synthetic values. It scores 0 in Pair scores but has no synthetic autocorrelation, so the other two charts leave it out.'
    );
  }
  if (constant > 1) {
    lines.push(
      `${constant} pairs have constant synthetic values. They score 0 in Pair scores but have no synthetic autocorrelation, so the other two charts leave them out.`
    );
  }
  return lines;
};

/** The library report's sentences above the window membership inference charts. */
export const windowMembershipSummary = (details: WindowMembershipInferenceDetails): string[] => {
  const lines: string[] = [];
  const { counts } = details;
  const full = details.evaluation_mode === 'full';
  if (counts) {
    const sampled = (['training', 'holdout', 'synthetic'] as const)
      .filter((split) => split !== 'holdout' || full)
      .filter((split) => counts[`${split}_windows`] < counts[`${split}_windows_total`])
      .map((split) => `${count(counts[`${split}_windows_total`])} ${split}`);
    const holdout = full
      ? ` and ${count(counts.holdout_windows)} holdout ${noun(counts.holdout_windows, 'window', 'windows')} from ${counts.holdout_groups_with_windows} holdout ${noun(counts.holdout_groups_with_windows, 'group', 'groups')}`
      : '';
    let line = `Compared ${count(counts.training_windows)} training ${noun(counts.training_windows, 'window', 'windows')} from ${plural(details.training_group_count, 'group')}${holdout} with ${count(counts.synthetic_windows)} synthetic ${noun(counts.synthetic_windows, 'window', 'windows')} (${details.window_size} steps across ${details.columns.length} numeric ${noun(details.columns.length, 'column', 'columns')}).`;
    if (sampled.length) {
      line += ` Windows were sampled reproducibly from ${sampled.join(' and ')} windows.`;
    }
    lines.push(line);
  }
  if (details.exact_copy_rate !== null) {
    const flagged = details.attack
      ? `${percent(details.attack.training_flag_rate)} of training windows and ${percent(details.attack.holdout_flag_rate)} of holdout windows are flagged as close to a synthetic window. `
      : '';
    lines.push(
      `${flagged}${percent(details.exact_copy_rate)} of training windows have an exact copy in the synthetic data.`
    );
  }
  const match = details.closest_match;
  if (match && !match.exact_copy) {
    const values = (details.window_size ?? 0) * details.columns.length;
    const perValue =
      match.distance !== null && values > 0
        ? ` It differs from that synthetic window by ${(match.distance / Math.sqrt(values)).toFixed(2)} training standard deviations per value.`
        : '';
    lines.push(
      `Closest match shows only the synthetic window. The closest training window is hidden because it is not an exact copy, so the report shows no training values that the synthetic data doesn't contain.${perValue}`
    );
  }
  return lines;
};
